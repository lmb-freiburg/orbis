from torch.utils.data import Dataset

from data.base import ImagePaths
from PIL import Image
from torchvision import transforms
import h5py
import random
from omegaconf import ListConfig


class CustomBase(Dataset):
    def __init__(self):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=False, scale=False, crop_size=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, crop_size=crop_size,
                               random_crop=random_crop, scale=scale)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, scale=False):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, scale=scale)

        
        
class RandomResizedCenterCrop:
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        self.size = size
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img):
        width, height = img.size
        area = height * width
        aspect_ratio = width / height
        target_area = random.uniform(*self.scale) * area

        new_width = int(round((target_area * aspect_ratio) ** 0.5))
        new_height = int(round((target_area / aspect_ratio) ** 0.5))

        if isinstance(self.size, ListConfig):
            crop_h, crop_w = self.size[0], self.size[1]
        else:
            crop_h = crop_w = self.size

        if new_width < crop_w or new_height < crop_h:
            scale = max(crop_w / new_width, crop_h / new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)

        img = img.resize((new_width, new_height), self.interpolation)

        x1 = (new_width - crop_w) // 2
        y1 = (new_height - crop_h) // 2

        return img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

class MultiHDF5Dataset(Dataset):
    def __init__(self, size, hdf5_paths_file, aug='resize_center', scale_min=0.15, scale_max=0.5):
        self.size = size

        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = [h5py.File(p, 'r') for p in self.hdf5_paths]
        self.file_keys = [list(f.keys()) for f in self.files]
        self.lengths = [{k: len(f[k]) for k in keys} for f, keys in zip(self.files, self.file_keys)]

        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)
        print(f'Total dataset length: {self.total_length}')

        if aug == 'resize_center':
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])
        elif aug == 'random_resize_center':
            self.transform = transforms.Compose([
                RandomResizedCenterCrop(size=size, scale=(scale_min, scale_max)),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unsupported augmentation type: {aug}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Deterministic seed for reproducibility in multithreaded loaders
        random.seed(idx)

        file_idx = random.randint(0, len(self.files) - 1)
        file = self.files[file_idx]
        keys = self.file_keys[file_idx]

        key = random.choice(keys)
        if 'meta_data' in key:
            key = key.replace('_meta_data', '')

        max_index = self.lengths[file_idx][key]
        img_idx = random.randint(0, max_index - 1)

        image_data = file[key][img_idx]
        image = Image.fromarray(image_data)
        image = self.transform(image)
        return image * 2 - 1  # normalize to [-1, 1]

    def close(self):
        for f in self.files:
            f.close()
