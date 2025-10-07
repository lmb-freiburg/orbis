import os
import json
import random
import importlib

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from data.base import ImagePaths

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation
        self.size = size
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            area = height * width
            aspect_ratio = width / height

            target_area = random.uniform(*self.scale) * area

            new_width = int(round((target_area * aspect_ratio) ** 0.5))
            new_height = int(round((target_area / aspect_ratio) ** 0.5))
            x1 = (new_width - self.size) // 2
            y1 = (new_height - self.size) // 2
            self.fixed_params = (new_width, new_height, x1, y1)
        return self.fixed_params    

    def __call__(self, img):
        new_width, new_height, x1, y1 = self.get_params(img)
        img = img.resize((new_width, new_height), self.interpolation)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))

    def reset(self):
        self.fixed_params = None


class RandomShiftCrop(object):
    def __init__(self, size, max_shift_horizontal=60, max_shift_vertical=60):
        """
        size: Crop size. If an int is provided, the crop will be (size, size). 
              If a tuple is provided, it should be (crop_width, crop_height).
        max_shift_horizontal: Maximum horizontal shift (in pixels) from the center of the crop.
        max_shift_vertical: Maximum vertical shift (in pixels) from the center of the crop.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            crop_height, crop_width = self.size

            # Calculate the center coordinates for the crop
            center_left = (width - crop_width) // 2
            center_top = (height - crop_height) // 2

            # Apply random horizontal and vertical shifts
            shift_horizontal = random.randint(-self.max_shift_horizontal, self.max_shift_horizontal)
            shift_vertical = random.randint(-self.max_shift_vertical, self.max_shift_vertical)

            left = center_left + shift_horizontal
            top = center_top + shift_vertical

            # Clamp the values to ensure the crop is entirely within the image boundaries
            left = max(0, min(left, width - crop_width))
            top = max(0, min(top, height - crop_height))

            self.fixed_params = (left, top)
        return self.fixed_params

    def __call__(self, img):
        left, top = self.get_params(img)
        crop_height, crop_width = self.size
        return img.crop((left, top, left + crop_width, top + crop_height))

    def reset(self):
        self.fixed_params = None


class NumpyToTensor:
    def __call__(self, x):
        assert isinstance(x, np.ndarray), f'input must be a numpy array, got {type(x)}'
        assert x.ndim == 3, 'input must be a 3D array'
        return torch.from_numpy(x).permute(2, 0, 1)



class CustomMultiFrame(Dataset):
    def __init__(self, size, num_frames, images_list_file, crop_size=None, scale=False):
        super().__init__()

        self.num_frames = num_frames
        # get FIRST frame paths (i.e. for each video, get TOT_FRAMES-NUM_FRAMES)
        with open(images_list_file, 'r') as f:
            paths = json.load(f)
        
        # data: first frame paths
        self.data = []
        for _, paths_list in paths.items():
            self.data.extend(paths_list[:len(paths_list)-num_frames+1]) # we want to handle 1 frame as well
        
        self.dummy_imagepaths = ImagePaths([], size=size, crop_size=crop_size, random_crop=False, scale=scale)
        self.transform = NumpyToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # get all frames for starting frame i
        first_frame_path = self.data[i]
        # split folder and file name
        dirname, first_frame_filename = os.path.split(first_frame_path)
        # get first frame index number from file name of format 'xxxx.jpg'
        first_frame_idx = int(first_frame_filename.split('.')[0])
        # filenames with format '0000.jpg', replace with inclremental index
        frame_paths = [os.path.join(dirname, f"{f_i:04d}.jpg") for f_i in range(first_frame_idx, first_frame_idx+self.num_frames)]
        
        frames = [self.dummy_imagepaths.preprocess_image(f) for f in frame_paths]
        frames = torch.stack([self.transform(f) for f in frames], 1)
        return frames

class MultiHDF5DatasetMultiFrame(Dataset):
    """
    This dataset reads multiple HDF5 files, each containing multiple videos.
    Each video is stored as a key in the HDF5 file, and each key contains a sequence of frames.
    The dataset returns a random sub-sequence of frames from a randomly selected video.
    """
    def __init__(self, size, hdf5_paths_file, num_frames, frame_rate_multiplier=1):
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames

        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        #self.files = [h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*8, rdcc_nslots=1000) for path in self.hdf5_paths]
        # self.files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.files = []
        for path in self.hdf5_paths:
            try:
                file = h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
                self.files.append(file)
            except Exception as e:
                print(f'Error opening file {path}: {e}')
        
        self.lengths = []
        self.file_keys = []
        for file in self.files:
            keys = [k for k in file.keys() if 'meta_data' not in k]
            self.file_keys.append(keys)
            self.lengths.append({key: len(file[key]) for key in keys})

        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)
        print(f'Total length: {self.total_length}, {len(self.files)} files.')
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
    
    def __len__(self):
        return self.total_length
    
    def get_indices(self):
        file_index = random.randint(0, len(self.files) - 1)
        key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
        key = self.file_keys[file_index][key_index]
        try:
            video_length = self.lengths[file_index][key]
            frames_needed_after = (self.num_frames+1)*self.frame_interval
            if video_length <= frames_needed_after:
                raise ValueError(f'file_index: {file_index}, key_index: {key_index}/{len(self.file_keys[file_index])}, video length: {video_length}, frames_needed_after: {frames_needed_after}')
            img_start_index = random.randint(0, video_length - frames_needed_after)
            if 'meta_data' in key:
                key = key.replace('_meta_data', '')
            indices = [img_start_index+i*self.frame_interval for i in range(self.num_frames)]
        except ValueError as e:
            print(e)
            return self.get_indices()
        return file_index, key, indices
    
    def __getitem__(self, idx):
        file_index, key, indices = self.get_indices()
        h5_file = h5py.File(self.hdf5_paths[file_index], 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
        images = torch.stack([self.transform(Image.fromarray(h5_file[key][i]))*2-1 for i in indices], dim=0)
        return images
    
    def close(self):
        for file in self.files:
            file.close()


class MultiHDF5DatasetMultiFrameIdxMapping(Dataset):
    '''
    This dataset maps each index to a specific frame in a specific video. Useful for validation and selecting subsets of frames.
    num_frames: number of frames to return for each index

    '''
    def __init__(self, size, hdf5_paths_file, num_frames, frame_rate_multiplier=1, aug='resize_center', scale_min=0.15, scale_max=0.5):

        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.hdf5_paths_file = hdf5_paths_file
        # expand environment variables in path
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.hdf5_files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.index_to_starting_frame_map = []
        for file in self.hdf5_files:
            keys = list(file.keys())
            for key in keys:
                video_length = len(file[key])
                # we take every nth frame, as long as we can get num_frames frames after that
                max_frame_index = video_length - num_frames*self.frame_interval-1
                for i in range(0, max_frame_index + 1):
                    self.index_to_starting_frame_map.append((file, key, i))
        
        self.aug = aug
        if self.aug == 'resize_center':
            self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
        elif self.aug == 'random_resize_center':
            self.custom_crop = RandomResizedCenterCrop(size=self.size, scale=(scale_min, scale_max))
            self.transform = transforms.Compose([
                                        self.custom_crop,
                                        transforms.ToTensor(),
                                        ])
        elif self.aug == 'random_shift':
            self.custom_crop = RandomShiftCrop(size=self.size, max_shift_horizontal=60, max_shift_vertical=30)
            self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                        self.custom_crop,
                                        transforms.ToTensor(),
                                        ])
    
    def apply_same_transform_to_all(self, frames, transform):
        return [transform(frame) for frame in frames]
    
    def __len__(self):
        return len(self.index_to_starting_frame_map)
    
    def __str__(self):
        s = f'MultiHDF5DatasetMultiFrameIdxMapping({self.hdf5_paths_file}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})'
        return s
    
    def get_images_and_indices(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f'Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}')
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        images = [Image.fromarray(file[key][start_frame+i*self.frame_interval]) for i in range(self.num_frames)]
        return images, (file.filename, key, start_frame)

    def apply_transforms(self, images):
        if self.aug == 'random_resize_center' or self.aug == 'random_shift':
            self.custom_crop.reset()
        images = self.apply_same_transform_to_all(images, self.transform)
        return torch.stack(images, dim=0)*2-1

    def __getitem__(self, idx):
        images, _ = self.get_images_and_indices(idx)
        images = self.apply_transforms(images)
        return images
        
    def close(self):
        for file in self.hdf5_files:
            file.close()


class MultiHDF5DatasetMultiFrameFromJSON(MultiHDF5DatasetMultiFrameIdxMapping):
    """
    The structure of the JSON file should be as follows:
    [
        {
            "h5_path": <PATH TO THE H5 FILE CONTAINING THE VIDEO>, 
            "video_key": <KEY/NAME OF THE VIDEO, e.g. 53773fdf-311fd624>
            "start_frame": <STARTING FRAME INDEX>
        }
    ]
    """
    def __init__(self, size, samples_json, num_frames, frame_rate_multiplier=1, num_samples=800):

        # if frame_rate_multiplier != 1:
        #     raise NotImplementedError
        
        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)

        self.samples_json = samples_json
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        
        # read json
        with open(os.path.expandvars(samples_json), 'r') as f:
            self.samples = json.load(f)[:num_samples]
        
        # get all h5 file paths
        h5_paths = list(set([sample['h5_path'] for sample in self.samples]))
        self.hdf5_files = {h5_path: h5py.File(h5_path, 'r') for h5_path in h5_paths}
        self.index_to_starting_frame_map = []
        for sample in self.samples:
            file = self.hdf5_files[sample['h5_path']]
            key = sample['video_key']
            start_frame = sample['start_frame']
            self.index_to_starting_frame_map.append((file, key, start_frame))

        self.aug = 'resize_center'
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
    
    def get_images_and_indices(self, idx):
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        if len(file[key])<=start_frame+self.num_frames:
            start_frame = len(file[key]) - self.num_frames
        images = [Image.fromarray(file[key][start_frame+i*self.frame_interval]) for i in range(self.num_frames)]
        return images, (file.filename, key, start_frame)
    
    def close(self):
        for file in self.hdf5_files.values():
            file.close()
    
    def __str__(self):
        s = f'MultiHDF5DatasetMultiFrameIdxMapping({self.samples_json}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})'
        return s


class MultiHDF5DatasetMultiFrameFromJSONFrameRateWrapper(MultiHDF5DatasetMultiFrameFromJSON):
    def __init__(self, size, samples_json, num_frames, stored_data_frame_rate, frame_rate, num_samples=800):
        frame_rate_multiplier = frame_rate/stored_data_frame_rate
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        super().__init__(size, samples_json, num_frames, frame_rate_multiplier, num_samples)
    
    def __getitem__(self, idx):
        return {'images': super().__getitem__(idx), 'frame_rate': self.frame_rate}


class MultiHDF5DatasetMultiFrameRandomizeFrameRate(MultiHDF5DatasetMultiFrameIdxMapping):
    """
    Enable the model to sample from multiple frame intervals (with weights). It also returns the frame rate and frame interval used for each sample.
    """
    def __init__(self, *, size, hdf5_paths_file, num_frames, stored_data_frame_rate, frame_rates_and_weights, **kwargs):
        """
        frame_intervals_and_weights: list of tuples containing (frame_interval, weight)
        e.g. [(1, 0.5), (2, 0.5)] means that 50% of the time the model will keep the original frame rate, and 50% of the time it will sample every 2nd frame.
        """
        self.hdf5_paths_file = hdf5_paths_file
        self.stored_data_frame_rate = stored_data_frame_rate
        self.frame_rates, self.frame_rate_weights = zip(*frame_rates_and_weights)
        # weights must sum to 1
        assert sum(self.frame_rate_weights) == 1, 'weights must sum to 1'
        assert all([weight >= 0 for weight in self.frame_rate_weights]), 'weights must be non-negative'
        
        # how many frames to skip for each frame rate
        frame_intervals = [self.stored_data_frame_rate/frame_rate for frame_rate in self.frame_rates]
        assert all([np.abs(frame_interval-round(frame_interval))<0.001 for frame_interval in frame_intervals]), f'frame intervals should be (close to) integers, got {frame_intervals}'
        # round to nearest integer
        self.frame_intervals = list(map(round, frame_intervals))
        
        super().__init__(size, hdf5_paths_file, num_frames, frame_rate_multiplier=1/max(self.frame_intervals), **kwargs)
        del self.frame_interval
    
    def get_images_and_indices(self, idx, frame_interval):
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        if len(file[key])<=start_frame+self.num_frames*frame_interval:
            raise ValueError(f'file_index: {idx}, key: {key}, start_frame: {start_frame}, frame_interval: {frame_interval}, video length: {len(file[key])}, frames_needed_after: {self.num_frames*frame_interval}')
        images = [Image.fromarray(file[key][start_frame+i*frame_interval]) for i in range(self.num_frames)]
        return images, (file.filename, key, start_frame)
    
    def __getitem__(self, idx):
        # get random frame interval
        frame_rate_idx = np.random.choice(range(len(self.frame_intervals)), p=self.frame_rate_weights)
        frame_interval = self.frame_intervals[frame_rate_idx]
        try:
            images, _ = self.get_images_and_indices(idx, frame_interval)
        except ValueError as e:
            # if the video is too short, try again with a random sample
            rnd_idx = np.random.randint(0, len(self.index_to_starting_frame_map))
            return self.__getitem__(rnd_idx)
        # return images and frame interval
        images = self.apply_transforms(images)
        return {'images': images, 'frame_rate': self.frame_rates[frame_rate_idx], 'frame_interval': frame_interval}

    def __str__(self):
        return f'''{self.__class__.__name__}({self.hdf5_paths_file},
                    num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, 
                    stored_data_frame_rate={self.stored_data_frame_rate}, frame_rates={self.frame_rates}, frame_intervals={self.frame_intervals}, frame_rate_weights={self.frame_rate_weights})'''



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
