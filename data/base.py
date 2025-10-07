import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, crop_size=None, random_crop=False, labels=None, scale=False):
        self.size = size
        self.scale = scale
        self.crop_size = crop_size
        self.random_crop = random_crop
        if self.crop_size is not None and self.size is not None:
            assert self.crop_size <= self.size

        if self.crop_size is None:
            self.crop_size = self.size

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        if self.size is not None and self.size > 0:
            self.resizer = albumentations.SmallestMaxSize(max_size = 720)
            self.rescaler = albumentations.RandomScale(scale_limit=[-0.5, 0.0])
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size,width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size,width=self.crop_size)
                
            if self.scale:
                self.preprocessor = albumentations.Compose([self.resizer, self.rescaler, self.cropper])
            else:
                self.preprocessor = albumentations.Compose([self.cropper])
            #self.preprocessor = albumentations.Compose([self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
