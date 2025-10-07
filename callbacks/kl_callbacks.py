import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from data.utils import custom_collate



class RunningStatsTensors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 0  # Count of data points seen so far
        # Initialize mean and variance tensors as parameters to enable device transfer
        self.mean = torch.nn.Parameter(torch.zeros(1))
        self.S = torch.nn.Parameter(torch.zeros(1))

    def update(self, x):
        batch_size = x.numel()  # Get the number of elements in the new tensor

        if self.n == 0:
            # First batch initialization
            self.mean.data = x.mean()  # Initialize the mean with the batch mean
            # Initialize variance with the sum of squared differences from the batch mean
            self.S.data = ((x - self.mean) ** 2).sum()
        else:
            # Calculate the new total number of elements
            new_n = self.n + batch_size
            delta = x.mean() - self.mean
            new_mean = self.mean + delta * batch_size / new_n

            # Update the variance (S) using Welford's method
            self.S.data += ((x - self.mean)**2).sum()

            # Update the running mean
            self.mean.data = new_mean

        # Update the count of elements
        self.n += batch_size

    def get_mean(self):
        return self.mean
    
    def get_std(self):
        # Unbiased estimate: divide by n - 1
        if self.n > 1:
            return torch.sqrt(self.S / (self.n - 1))
        else:
            return torch.tensor(0.0, device=self.mean.device)

class DatasetStatisticsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.running_stats = RunningStatsTensors()
        self.running_stats2 = RunningStatsTensors()
    
    def on_train_start(self, trainer, pl_module):
        print("Calculating dataset statistics...")
        # Access the training data loader
        train_loader = trainer.train_dataloader
        # create a copy of the train_loader with batch size 256
        train_loader = DataLoader(train_loader.dataset, batch_size=64, num_workers=16, collate_fn=custom_collate, pin_memory=True, drop_last=True)

        autoencoder = pl_module
        self.running_stats= self.running_stats.to(autoencoder.device) if hasattr(autoencoder, "device") else self.running_stats
        self.running_stats2= self.running_stats2.to(autoencoder.device) if hasattr(autoencoder, "device") else self.running_stats2
        # Iterate over the training data loader

        for idx, batch in tqdm(enumerate(train_loader), desc="Processing Batches"):
            # Compute the statistics for the current batch
            with torch.no_grad():
                # Forward pass through the autoencoder

                x = batch
                x = x.cuda()
                x= x.float()

                # Get the latent representation
                (h, h2, _, _) = autoencoder.encode(x)[-1]
                
                h = h.view(-1)
                h2 = h2.view(-1)
                # Update the running statistics
                self.running_stats.update(h)
                self.running_stats2.update(h2)

                if idx % 100 == 0:
                    mean = self.running_stats.get_mean()
                    std = self.running_stats.get_std()
                    print(f"Latent mean: {mean.mean()}, Latent std: {std.mean()}")
                    mean2 = self.running_stats2.get_mean()
                    std2 = self.running_stats2.get_std()
                    print(f"Latent mean: {mean2.mean()}, Latent std: {std2.mean()}")

        # Compute the final mean and standard deviation 
        mean = self.running_stats.get_mean()
        std = self.running_stats2.get_std()
        # Log the mean and standard deviation
        trainer.logger.log_metrics({"train/latent_mean": mean.mean(), "train/latent_std": std.mean()}, step=0)
        print(f"Latent mean: {mean.mean()}, Latent std: {std.mean()}")