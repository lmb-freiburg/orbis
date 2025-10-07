import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    """
    Custom checkpoint callback for PyTorch Lightning.

    Saves checkpoints every `every` epochs and optionally 
    stops training after `end` epochs.

    Args:
        every (int): Frequency (in epochs) to save checkpoints.
        end (int, optional): Epoch after which training stops. Defaults to None.
        dirpath (str, optional): Directory path to save checkpoints. Must be set.
    """
    def __init__(self, every: int, end: int = None, dirpath: str = None):
        super().__init__()
        self.every = every
        self.dirpath = dirpath
        self.end = end

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0 and pl_module.current_epoch>0:
            assert self.dirpath is not None
            current = f"{self.dirpath}/epoch-{pl_module.current_epoch}.ckpt"
            prev = (
                f"{self.dirpath}/epoch-{pl_module.current_epoch - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            # prev.unlink(missing_ok=True)
        if self.end is not None and pl_module.current_epoch >= self.end:
            trainer.should_stop = True
            return
