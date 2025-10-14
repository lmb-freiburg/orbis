from torch.utils.data import Dataset
import json
import os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.v2.functional as FV2
import cv2
from PIL import Image
import sqlite3
import pandas as pd


class MultiFrameValidationDataset(Dataset):
    """
    A base class for datasets that load multiple frames from a sequence of images.
    This dataset is intended for validation purposes, since loading images from disk for training is typically avoided.
    This class is designed to be subclassed for specific datasets.
    """
    
    def __init__(self, *, size, num_frames=None):
        """
        Initializes the dataset with the specified size, number of frames, frame rate multiplier, and sample indices.
        Args:
            size (int or tuple): The size to which the images will be resized. If an int is provided, it will be used for both width and height.
            num_frames (int, optional): The number of frames to return. If None, all frames will be returned.
            sample_indices (list, optional): A list of indices to sample from the dataset. If None, all samples will be used.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.frame_paths = None
        
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                                transforms.CenterCrop(self.size),
                                                transforms.ToTensor()])

    def init_frame_paths(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the frame paths for a given sample.")

    def __getitem__(self, index):
        assert self.frame_paths is not None, "Frame paths have not been initialized. Call init_frame_paths() before accessing items."
        frame_paths = self.frame_paths[index]
        if self.num_frames is not None:
            assert len(frame_paths) == self.num_frames, f"Expected {self.num_frames} frames, but got {len(frame_paths)} frames."
        images = [cv2.imread(frame_path) for frame_path in frame_paths]
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        images = [self.transform(image)*2-1 for image in images]
        images = torch.stack(images, dim=0)
        return images
        
    def __len__(self):
        return len(self.frame_paths)


class MultiFrameFromPaths(MultiFrameValidationDataset):
    """
    A dataset for loading frame sequences from a list of image paths.
    """
    def __init__(self, *, size, image_paths, num_frames=None):
        super().__init__(size=size)
        self.image_paths = image_paths
        self.num_frames = num_frames or len(image_paths)
        
        self.init_frame_paths()

    def init_frame_paths(self):
        """
        Initializes the frame paths from the list of image paths.
        """
        if self.num_frames is not None:
            if len(self.image_paths) < self.num_frames:
                raise ValueError(f"Number of frames {len(self.image_paths)} is less than the required {self.num_frames}")
            self.image_paths = self.image_paths[:self.num_frames]
            
        self.frame_paths = [self.image_paths]


class JSONFramesListLoader(MultiFrameValidationDataset):
    """
    A dataset for loading frame sequences from a JSON file, as used in the Vista codebase.
    Each entry in the JSON file contains a list of frame paths, and the dataset returns a sequence of frames as a tensor.
    The frames are resized and cropped to a specified size, and normalized to the range [-1, 1].
    """
    
    def __init__(self, *, size, json_path, images_root, num_frames=None, frame_rate_multiplier=1, sample_indices=None):
        super().__init__(size=size, num_frames=num_frames, sample_indices=sample_indices)
        self.json_path = json_path
        self.images_root = images_root
        
        assert frame_rate_multiplier <= 1, "Frame rate multiplier should be less than or equal to 1"
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        self.init_frame_paths()
        
    
    def init_frame_paths(self):
        """
        Initializes the frame paths from the JSON file.
        """
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        self.frame_paths = []
        for sample in self.data:
            frames = sample['frames'][::self.frame_interval]
            if self.num_frames is not None:
                if len(frames) < self.num_frames:
                    raise ValueError(f"Number of frames {len(frames)} is less than the required {self.num_frames}")
                frames = frames[:self.num_frames]
            self.frame_paths.append([os.path.join(self.images_root, frame) for frame in frames])
        
        if self.sample_indices is not None:
            # If sample_indices is provided, filter the data to only include the specified indices
            self.frame_paths = [self.frame_paths[i] for i in self.sample_indices]
        
        
    
    
    
