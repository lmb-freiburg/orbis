import os
from PIL import Image
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

class ImageLogger(Callback):
    """
    Log example image batches during training/validation.

    Features:
    - Logs every `batch_frequency` steps (and on early warmup steps: 1, 2, 4, 8, ...),
      unless `increase_log_steps=False`.
    - Saves PNG grids locally under: <logger.save_dir>/images/{train|val}/...
    - Supports TensorBoard out of the box; others can be added via `logger_log_images`.

    Args:
        batch_frequency: Log frequency in training/validation steps.
        max_images: Max number of images per key to log.
        clamp: Clamp tensors to [-1, 1] before visualization.
        increase_log_steps: If True, additionally log at 1,2,4,... up to batch_frequency.
        nrow: Grid row size for the saved image grids.
    """
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def _testensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = grid.clamp(-1, 1.)
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w            
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    # This code will only be executed by the process with rank 0
    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid.clamp(-1., 1.)  # -1,1 -> -1,1; c,h,w
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")
