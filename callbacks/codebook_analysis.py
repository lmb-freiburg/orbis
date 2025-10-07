from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback


class CodebookUsageLogger(Callback):
    """
    Track unique codebook indices used by a vector-quantizer during training/validation.
    """
    def __init__(self, log_batches_training: bool = False):
        super().__init__()
        self.log_batches_training = log_batches_training
        self.used_indices_train = set()
        self.used_indices_val = set()
        self.hook_dict = {}
        self.hook_handle = None

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # setup fwd hook to collect indices
        def quantizer_hook_fn(module, x, y):
            _, _, info = y
            if isinstance(info[2], tuple):
                self.hook_dict["indices"] = info[2][0].cpu().numpy()
                self.hook_dict["indices2"] = info[2][1].cpu().numpy()
            else:
                self.hook_dict["indices"] = info[2].cpu().numpy()
        
        if isinstance(pl_module.quantize, torch.nn.ModuleList):
            self.hook_handle = pl_module.quantize[0].register_forward_hook(quantizer_hook_fn)
        else:
            self.hook_handle = pl_module.quantize.register_forward_hook(quantizer_hook_fn)

    # update logger's counter after iter, log for batch if requested
    def on_train_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        used_batch_set = set(self.hook_dict['indices'].flatten())
        self.used_indices_train.update(used_batch_set)
        if self.log_batches_training:
            pl_module.log("train/indices_used_batch_avg", len(used_batch_set), prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule, outputs:Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        used_batch_set = set(self.hook_dict['indices'].flatten())
        self.used_indices_val.update(used_batch_set)

    # log and reset counter
    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("train/total_indices_used", len(self.used_indices_train), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.used_indices_train = set()
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("val/total_indices_used", len(self.used_indices_val), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.used_indices_val = set()
    