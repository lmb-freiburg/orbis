import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class DoTAClipDataset(Dataset):
    def __init__(self, sequence_dir, annotation_dir, transform=None):
        """
        Args:
            sequence_dir (str): Path to 'DoTA_Sequences' directory containing video folders.
            annotation_dir (str): Path to 'DOTA_annotations' directory containing JSONs.
            transform (callable, optional): Transforms applied to each frame.
        """
        self.sequence_dir = sequence_dir
        self.transform = transform
        self.samples = []

        # Helper function to ensure all frames in a clip exist on disk
        def is_valid_clip(clip_paths):
            return all(os.path.exists(path) for path in clip_paths)

        # Iterate over all folders in the sequence directory
        ct = 0
        for video_name in os.listdir(sequence_dir):
            ct +=1
            video_folder = os.path.join(sequence_dir, video_name, 'images')
            
            # Ensure we are only looking at directories
            if not os.path.isdir(video_folder):
                continue
             
            json_path = os.path.join(annotation_dir, f"{video_name}.json")
            # If the annotation doesn't exist, ignore this sample
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                anno = json.load(f)
                
            if (str(anno.get('ignore', 'false')).lower() != 'false') or anno.get('anomaly_start', -1) == -1:
                continue

            num_frames = anno['num_frames']
            anomaly_start = anno['anomaly_start']
            anomaly_end = anno['anomaly_end']
            frames_meta = anno['labels']

            # Ensure we have enough frames to extract a 5-frame clip
            if num_frames < 5 or (anomaly_start + 4) >= num_frames:
                continue

            # ----------------------------------------------------
            # 1. Extract Anomaly Clip (Positive Class: Label 1)
            # ----------------------------------------------------
            # Modified to replace 'frames' with 'DOTA_sequences'
            anomaly_clip = [
                os.path.join(self.sequence_dir, frames_meta[i]['image_path'].replace('frames', 'DOTA_sequences'))
                for i in range(anomaly_start, anomaly_start + 5)
            ]
            
            if is_valid_clip(anomaly_clip):
                self.samples.append({
                    'video_name': video_name,
                    'clip_paths': anomaly_clip,
                    'label': 1,
                    'clip_type': 'anomaly_start'
                })

            # ----------------------------------------------------
            # 2. Extract Normal Clip (Negative Class: Label 0)
            # ----------------------------------------------------
            left_distance = anomaly_start
            right_distance = num_frames - 1 - anomaly_end

            # Pick the furthest continuous 5 frames from the anomaly
            if left_distance >= right_distance and left_distance >= 5:
                normal_start = 0
            elif right_distance > left_distance and right_distance >= 5:
                normal_start = num_frames - 5
            else:
                continue 

            # Modified to replace 'frames' with 'DOTA_sequences'
            normal_clip = [
                os.path.join(self.sequence_dir, frames_meta[i]['image_path'].replace('frames/', ''))
                for i in range(normal_start, normal_start + 5)
            ]
            
            if is_valid_clip(normal_clip):
                self.samples.append({
                    'video_name': video_name,
                    'clip_paths': normal_clip,
                    'label': 0,
                    'clip_type': 'furthest_normal'
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_paths = sample['clip_paths']
        label = sample['label']
        
        frames = []
        for img_path in clip_paths:
            img = Image.open(img_path).convert('RGB')
                
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)
            
        # Stack frames to shape (Channels, Time, Height, Width)
        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        
        return clip_tensor, torch.tensor(label, dtype=torch.float32)


def get_dota_dataloaders(sequence_dir, annotation_dir, batch_size=8, num_workers=4):
    """
    Initializes the dataset, splits it 80/20, and returns Train/Val DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the full dataset
    full_dataset = DoTAClipDataset(sequence_dir, annotation_dir, transform=transform)
    
    # Calculate 80-20 split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Print dataset statistics
    print("-" * 30)
    print(f"Total valid clips found: {total_size}")
    print(f"Training set size:       {train_size}")
    print(f"Validation set size:     {val_size}")
    print("-" * 30)
    
    # Handle edge case where dataset might be empty or too small
    if total_size == 0:
        raise ValueError("No valid clips were found. Check your directory paths and JSON structure.")
        
    # Randomly split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Define your paths here (adjust relative to where you run the script)
    # Since this will be inside the 'data' folder, use relative paths to the repo root
    SEQ_DIR = "../DoTA_sequences"
    ANNO_DIR = "../DOTA_annotations"
    
    print("Testing DoTA Dataset Pipeline...")
    
    try:
        # Create dataloaders
        train_loader, val_loader = get_dota_dataloaders(SEQ_DIR, ANNO_DIR, batch_size=2, num_workers=0)
        
        # Fetch one batch to verify the tensor shapes and labels
        for clips, labels in train_loader:
            print("\n--- Batch Verification ---")
            print(f"Clip tensor shape: {clips.shape}  --> Expected: [Batch=2, Channels=3, Time=5, Height=288, Width=512]")
            print(f"Labels shape:      {labels.shape}  --> Expected: [Batch=2]")
            print(f"Labels values:     {labels.tolist()}")
            break
            
        print("\nSuccess! Pipeline is ready.")
        
    except FileNotFoundError:
        print(f"\nError: Could not find directories at {SEQ_DIR} or {ANNO_DIR}.")
        print("Please verify your folder paths.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")