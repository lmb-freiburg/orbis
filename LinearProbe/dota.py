import os
import json
import shutil
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import Counter

class DoTAClipDataset(Dataset):
    def __init__(self, sequence_dir, annotation_dir, transform=None, return_multiclass_labels=False, num_frames_per_clip=6, max_samples=None):
        """
        Args:
            sequence_dir (str): Path to 'DoTA_Sequences' directory containing video folders.
            annotation_dir (str): Path to 'DOTA_annotations' directory containing JSONs.
            transform (callable, optional): Transforms applied to each frame.
            return_multiclass_labels (bool): Whether to return detailed multiclass labels.
            num_frames_per_clip (int): Target number of frames to extract per clip.
            max_samples (int, optional): Cap the dataset size before calculating statistics.
        """
        self.sequence_dir = sequence_dir
        self.transform = transform
        self.return_multiclass_labels = return_multiclass_labels
        self.num_frames_per_clip = num_frames_per_clip
        self.max_samples = max_samples
        self.samples = []
        self.class_to_idx = {}

        multiclass_keys = (
            'accident_id',
            'accident_name'
        )

        def normalize_multiclass_label(raw_label):
            if raw_label is None:
                return None

            if isinstance(raw_label, int):
                if 1 <= raw_label <= 18:
                    return raw_label
                return 0

            if isinstance(raw_label, str):
                label = int(raw_label) 
                return normalize_multiclass_label(label)

            return int(raw_label)

        def resolve_multiclass_label(annotation, frames_meta, frame_idx):
            candidate_sources = [annotation]
            if frame_idx is not None and frames_meta and 0 <= frame_idx < len(frames_meta):
                candidate_sources.append(frames_meta[frame_idx])

            for source in candidate_sources:
                if not isinstance(source, dict):
                    continue

                for key in multiclass_keys:
                    if key in source and source[key] is not None:
                        label = normalize_multiclass_label(source[key])
                        if label is not None:
                            return label

            return None

        # Helper function to ensure all frames in a clip exist on disk
        def is_valid_clip(clip_paths):
            return all(os.path.exists(path) for path in clip_paths)

        # ----------------------------------------------------
        # Window Calculation Setup
        # ----------------------------------------------------
        raw_window = (self.num_frames_per_clip * 2) - 1

        # Iterate over all folders in the sequence directory
        for idx, video_name in enumerate(os.listdir(sequence_dir)):
            video_folder = os.path.join(sequence_dir, video_name, 'images')
            
            if not os.path.isdir(video_folder):
                continue
             
            json_path = os.path.join(annotation_dir, f"{video_name}.json")
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

            if num_frames < raw_window or (anomaly_start + raw_window) > num_frames:
                continue

            # 1. Extract Anomaly Clip (Positive Class: Label 1)
            anomaly_clip = [
                os.path.join(self.sequence_dir, frames_meta[i]['image_path'].replace('frames/', ''))
                for i in range(anomaly_start, anomaly_start + raw_window, 2)
            ]
            anomaly_class_label = resolve_multiclass_label(anno, frames_meta, anomaly_start)
            
            # --- SKIP CONDITION FOR LABELS MAPPED TO 9 ---
            if anomaly_class_label >= 9: #Class is Unknown
                continue
                
            if is_valid_clip(anomaly_clip):
                self.samples.append({
                    'video_name': video_name,
                    'clip_paths': anomaly_clip,
                    'label': 1,
                    'class_label': anomaly_class_label,
                    'clip_type': 'anomaly_start'
                })

            # 2. Extract Normal Clip (Negative Class: Label 0)
            if anomaly_start >= raw_window:
                normal_start = 0
                normal_clip = [
                    os.path.join(self.sequence_dir, frames_meta[i]['image_path'].replace('frames/', ''))
                    for i in range(normal_start, normal_start + raw_window, 2)
                ]
                
                if is_valid_clip(normal_clip):
                    self.samples.append({
                        'video_name': video_name,
                        'clip_paths': normal_clip,
                        'label': 0,
                        'class_label': 0,
                        'clip_type': 'video_start'
                    })
            elif (num_frames - raw_window) > anomaly_end:
                normal_start = num_frames - raw_window
                normal_clip = [
                    os.path.join(self.sequence_dir, frames_meta[i]['image_path'].replace('frames/', ''))
                    for i in range(normal_start, normal_start + raw_window, 2)
                ]
                
                if is_valid_clip(normal_clip):
                    self.samples.append({
                        'video_name': video_name,
                        'clip_paths': normal_clip,
                        'label': 0,
                        'class_label': 0,
                        'clip_type': 'video_end'
                    })

        # ----------------------------------------------------
        # Apply max_samples cap BEFORE statistics calculation
        # ----------------------------------------------------
        if self.max_samples is not None:
            random.seed(43)
            self.samples = random.sample(self.samples, self.max_samples)
            # self.samples = self.samples[:self.max_samples]

        # Process multiclass labels maps (required internally regardless of mode)
        multiclass_labels = sorted({sample['class_label'] for sample in self.samples if sample['class_label'] is not None}, key=lambda value: str(value))
        if multiclass_labels and any(isinstance(label, str) for label in multiclass_labels):
            self.class_to_idx = {label: idx for idx, label in enumerate(multiclass_labels)}
        elif multiclass_labels:
            self.class_to_idx = {label: int(label) for label in multiclass_labels}

        for sample in self.samples:
            if sample['class_label'] is None:
                continue

            if isinstance(sample['class_label'], str):
                sample['class_label'] = self.class_to_idx[sample['class_label']]
            else:
                sample['class_label'] = int(sample['class_label'])

        # ----------------------------------------------------
        # Conditionally print statistics based on current mode
        # ----------------------------------------------------
        if self.return_multiclass_labels:
            print(f"\n--- Multiclass Distribution (Capped at {len(self.samples)} samples) ---")
            valid_class_labels = [sample['class_label'] for sample in self.samples if sample['class_label'] is not None]
            idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
            
            mc_counts = Counter(valid_class_labels)
            for class_idx, count in sorted(mc_counts.items()):
                class_name = idx_to_class.get(class_idx, f"Class_{class_idx}")
                print(f" {str(class_name):<20} | ID: {class_idx:<3} | Count: {count}")
            print("-------------------------------\n")
        else:
            labels = [sample['label'] for sample in self.samples]
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            counts = torch.bincount(labels_tensor)
            
            print(f"\n--- Binary Label Distribution (Capped at {len(self.samples)} samples) ---")
            if len(counts) >= 2:
                print(f"Normal (0): {counts[0].item()} | Anomalous (1): {counts[1].item()}")
            else:
                print(f"Counts: {counts.tolist()}")
            print("--------------------------------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_paths = sample['clip_paths']
        label = sample['label']
        mc_label = sample.get('class_label', None)
        video_id = sample['video_name']

        frames = []
        for img_path in clip_paths:
            img = Image.open(img_path).convert('RGB')
                
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)
            
        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        if self.return_multiclass_labels:
            mc_label_tensor = torch.tensor(-1 if mc_label is None else mc_label, dtype=torch.long)
            return clip_tensor, torch.tensor(label, dtype=torch.long), mc_label_tensor, video_id

        return clip_tensor, torch.tensor(label, dtype=torch.long), video_id


def get_dota_dataloaders(sequence_dir, annotation_dir, batch_size=8, num_workers=4, max_samples=900, return_multiclass_labels=False, num_frames_per_clip=6):
    """
    Initializes the dataset, splits it 80/20, empties target export directories, and returns Train/Val DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = DoTAClipDataset(
        sequence_dir,
        annotation_dir,
        transform=transform,
        return_multiclass_labels=return_multiclass_labels,
        num_frames_per_clip=num_frames_per_clip,
        max_samples=max_samples # Handled inside the class now
    )
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    print("-" * 30)
    print(f"Total valid clips generated: {total_size}")
    print(f"Training set size:           {train_size}")
    print(f"Validation set size:         {val_size}")
    print("-" * 30)
    
    if total_size == 0:
        raise ValueError("No valid clips were found. Check your directory paths and JSON structure.")
        
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    good_indices = [i for i, sample in enumerate(full_dataset.samples) if sample["label"] == 0]
    anomalous_indices = [i for i, sample in enumerate(full_dataset.samples) if sample["label"] == 1]
    
    sampled_good = random.sample(good_indices, min(100, len(good_indices)))
    sampled_anomalous = random.sample(anomalous_indices, min(100, len(anomalous_indices)))
    
    gif_eligible_indices = set(sampled_good + sampled_anomalous)
    print(f"Selected {len(sampled_good)} good and {len(sampled_anomalous)} anomalous clips for GIF generation.")

    export_root_dir = "../DOTA_training"
    if os.path.exists(export_root_dir):
        print(f"Cleaning out old export directory: {export_root_dir}")
        shutil.rmtree(export_root_dir)
        
    data_dir = os.path.join(export_root_dir, "data")
    gif_dir = os.path.join(export_root_dir, "gifs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    def export_split(subset, split_name):
        for sample_idx in subset.indices:
            sample = full_dataset.samples[sample_idx]
            video_id = sample["video_name"]
            class_name = "anomalous" if sample["label"] == 1 else "good"
            
            target_data_dir = os.path.join(data_dir, split_name, f"{video_id}_{class_name}")
            os.makedirs(target_data_dir, exist_ok=True)

            target_gif_dir = os.path.join(gif_dir, class_name)
            os.makedirs(target_gif_dir, exist_ok=True)

            for frame_idx, src_path in enumerate(sample["clip_paths"]):
                _, ext = os.path.splitext(src_path)
                dst_path = os.path.join(target_data_dir, f"frame_{frame_idx:04d}{ext}")
                shutil.copy2(src_path, dst_path)

    export_split(train_dataset, "train")
    export_split(val_dataset, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


if __name__ == "__main__":
    SEQ_DIR = "../DoTA_sequences"
    ANNO_DIR = "../DOTA_annotations"
    
    # Global Parameters
    TARGET_FRAMES = 6 
    RETURN_MULTICLASS = True  # <-- Toggle multiclass output here
    MAX_SAMPLES = 900       # <-- Global control of the subset size

    print("Testing DoTA Dataset Pipeline...")
    
    try:
        train_loader, val_loader = get_dota_dataloaders(
            SEQ_DIR, 
            ANNO_DIR, 
            batch_size=10, 
            num_workers=0, 
            max_samples=MAX_SAMPLES,
            num_frames_per_clip=TARGET_FRAMES,
            return_multiclass_labels=RETURN_MULTICLASS
        )
        
        print("\n--- Batch Verification ---")
        if RETURN_MULTICLASS:
            for clips, labels, mc_labels, video_ids in train_loader:
                print(f"Clip tensor shape:      {clips.shape}  --> Expected: [Batch=10, Channels=3, Time={TARGET_FRAMES}, Height=288, Width=512]")
                print(f"Binary Labels shape:    {labels.shape}  --> Expected: [Batch=10]")
                print(f"Binary Labels values:   {labels.tolist()}")
                print(f"Multiclass shape:       {mc_labels.shape} --> Expected: [Batch=10]")
                print(f"Multiclass values:      {mc_labels.tolist()}")
                print(f"Video IDs:              {video_ids}")
                break
        else:
            for clips, labels, video_ids in train_loader:
                print(f"Clip tensor shape:      {clips.shape}  --> Expected: [Batch=10, Channels=3, Time={TARGET_FRAMES}, Height=288, Width=512]")
                print(f"Binary Labels shape:    {labels.shape}  --> Expected: [Batch=10]")
                print(f"Binary Labels values:   {labels.tolist()}")
                print(f"Video IDs:              {video_ids}")
                break
            
        print("\nSuccess! Pipeline is ready.")
        
    except FileNotFoundError:
        print(f"\nError: Could not find directories at {SEQ_DIR} or {ANNO_DIR}.")
        print("Please verify your folder paths.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")