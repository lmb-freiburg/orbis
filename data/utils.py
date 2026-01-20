import collections
import os
import tarfile
import urllib
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from data.helper_types import Annotation
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from tqdm import tqdm


def unpack(path):
    if path.endswith("tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("tar"):
        with tarfile.open(path, "r:") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("zip"):
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(path=os.path.split(path)[0])
    else:
        raise NotImplementedError(
            "Unknown file extension: {}".format(os.path.splitext(path)[1])
        )


def reporthook(bar):
    """tqdm progress bar for downloads."""

    def hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            bar.total = tsize
        bar.update(b * bsize - bar.n)

    return hook


def get_root(name):
    base = "data/"
    root = os.path.join(base, name)
    os.makedirs(root, exist_ok=True)
    return root


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()


def prompt_download(file_, source, target_dir, content_dir=None):
    targetpath = os.path.join(target_dir, file_)
    while not os.path.exists(targetpath):
        if content_dir is not None and os.path.exists(
            os.path.join(target_dir, content_dir)
        ):
            break
        print(
            "Please download '{}' from '{}' to '{}'.".format(file_, source, targetpath)
        )
        if content_dir is not None:
            print(
                "Or place its content into '{}'.".format(
                    os.path.join(target_dir, content_dir)
                )
            )
        input("Press Enter when done...")
    return targetpath


def download_url(file_, url, target_dir):
    targetpath = os.path.join(target_dir, file_)
    os.makedirs(target_dir, exist_ok=True)
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=file_
    ) as bar:
        urllib.request.urlretrieve(url, targetpath, reporthook=reporthook(bar))
    return targetpath


def download_urls(urls, target_dir):
    paths = dict()
    for fname, url in urls.items():
        outpath = download_url(fname, url, target_dir)
        paths[fname] = outpath
    return paths


def quadratic_crop(x, bbox, alpha=1.0):
    """bbox is xmin, ymin, xmax, ymax"""
    im_h, im_w = x.shape[:2]
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.clip(bbox, 0, max(im_h, im_w))
    center = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    l = int(alpha * max(w, h))
    l = max(l, 2)

    required_padding = -1 * min(
        center[0] - l, center[1] - l, im_w - (center[0] + l), im_h - (center[1] + l)
    )
    required_padding = int(np.ceil(required_padding))
    if required_padding > 0:
        padding = [
            [required_padding, required_padding],
            [required_padding, required_padding],
        ]
        padding += [[0, 0]] * (len(x.shape) - 2)
        x = np.pad(x, padding, "reflect")
        center = center[0] + required_padding, center[1] + required_padding
    xmin = int(center[0] - l / 2)
    ymin = int(center[1] - l / 2)
    return np.array(x[ymin : ymin + l, xmin : xmin + l, ...])


def custom_collate(batch):
    r"""source: pytorch 1.9.0, only one modification to original code """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
        return batch  # added
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_trajectory_from_speeds_and_yaw_rates(speeds, yaw_rates, dt):
    headings = np.cumsum(yaw_rates) * dt  # integrate yaw rates to get headings
    dx = speeds * np.cos(headings) * dt
    dy = speeds * np.sin(headings) * dt
    
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    traj = np.stack([x, y], axis=1)
    
    # Transform to local coordinates (first position is origin, first heading is along x-axis)
    traj -= traj[0]  # translate to origin
    initial_heading = headings[0]
    rotation_matrix = np.array([[np.cos(-initial_heading), -np.sin(-initial_heading)],
                                    [np.sin(-initial_heading),  np.cos(-initial_heading)]])
    local_traj = traj @ rotation_matrix.T  # rotate to align with initial heading
    
    return local_traj.astype(np.float32), headings.astype(np.float32)

def get_trajectory_from_speeds_and_yaw_rates_batch(speeds, yaw_rates, dt):
    """
    Args:
        speeds: Tensor of shape (B, N)
        yaw_rates: Tensor of shape (B, N)
        dt: Time step (scalar)
    Returns:
        local_traj: Tensor of shape (B, N, 2)
        headings: Tensor of shape (B, N)
    """
    assert speeds.shape == yaw_rates.shape, f"Speeds shape {speeds.shape} and yaw rates shape {yaw_rates.shape} do not match"
    B, N = speeds.shape
    
    # if dt is a scalar, ok, if dt is a tensor, make sure it has shape (B) and expand to (B, 1)
    if isinstance(dt, torch.Tensor):
        assert dt.shape == (B,), f"dt shape {dt.shape} does not match batch size {B}"
        dt = dt.view(B, 1)  # Shape: (B, 1)
    
    # Integrate yaw rates to get headings for each batch
    headings = torch.cumsum(yaw_rates, dim=1) * dt  # Shape: (B, N)

    # Calculate dx and dy for each batch
    dx = speeds * torch.cos(headings) * dt  # Shape: (B, N)
    dy = speeds * torch.sin(headings) * dt  # Shape: (B, N)

    # Calculate x and y for each batch
    x = torch.cumsum(dx, dim=1)  # Shape: (B, N)
    y = torch.cumsum(dy, dim=1)  # Shape: (B, N)

    # Stack x and y to form the trajectory for each batch
    traj = torch.stack([x, y], dim=2)  # Shape: (B, N, 2)

    # Transform to local coordinates for each batch
    traj = traj- traj[:, 0:1, :]  # Translate to origin for each batch
    initial_heading = headings[:, 0]  # Shape: (B,)

    # Create rotation matrices for each batch
    cos_theta = torch.cos(-initial_heading)  # Shape: (B,)
    sin_theta = torch.sin(-initial_heading)  # Shape: (B,)

    # Rotation matrix for each batch
    rotation_matrix = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta,  cos_theta], dim=1)
    ], dim=1)  # Shape: (B, 2, 2)

    # Rotate to align with initial heading for each batch
    local_traj = torch.einsum('bni,bij->bnj', traj, rotation_matrix)  # Shape: (B, N, 2)

    return torch.cat([local_traj, headings.unsqueeze(-1)], dim=-1).float()  # Return (B, N, 3)


class RunningNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))

    def _reduce_dims(self, x):
        # Dimensions to reduce: batch and spatial (leave channel/features alone)
        return [0] + list(range(2, x.dim()))

    def update_stats(self, x):
        dims = self._reduce_dims(x)
        batch_mean = x.mean(dim=dims)
        batch_std = x.std(dim=dims, unbiased=False)

        # Update running stats
        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

    def normalize(self, x):
        mean = self.running_mean.view(1, -1, *[1] * (x.dim() - 2))
        std = self.running_std.view(1, -1, *[1] * (x.dim() - 2))
        return (x - mean) / (std + self.eps)

    def denormalize(self, x):
        mean = self.running_mean.view(1, -1, *[1] * (x.dim() - 2))
        std = self.running_std.view(1, -1, *[1] * (x.dim() - 2))
        return x * (std + self.eps) + mean

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return self.normalize(x)
