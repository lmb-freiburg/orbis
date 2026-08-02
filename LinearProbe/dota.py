import cv2
import os
import json
import shutil
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm

DOTA_CLASS_NAMES = {
    0: "normal",
    1: "start_stop_or_stationary",
    2: "moving_ahead_or_waiting",
    3: "lateral",
    4: "oncoming",
    5: "turning",
    6: "pedestrian",
    7: "obstacle",
    8: "leave_to_right",
    9: "leave_to_left",
    10: "unknown",
}


class DoTAClipDataset(Dataset):
    def __init__(self, sequence_dir, annotation_dir, transform=None, return_multiclass_labels=False, num_frames_per_clip=6, max_samples=None, skip_night=True):
        """
        Args:
            sequence_dir (str): Path to 'DoTA_Sequences' directory containing video folders.
            annotation_dir (str): Path to 'DOTA_annotations' directory containing JSONs.
            transform (callable, optional): Transforms applied to each frame.
            return_multiclass_labels (bool): Whether to return detailed multiclass labels.
            num_frames_per_clip (int): Target number of frames to extract per clip.
            max_samples (int, optional): Cap the dataset size before calculating statistics.
            skip_night (bool, optional): Skip samples where "night": true in annotations.
        """
        self.sequence_dir = sequence_dir
        self.transform = transform
        self.return_multiclass_labels = return_multiclass_labels
        self.num_frames_per_clip = num_frames_per_clip
        self.max_samples = max_samples
        self.skip_night = skip_night
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
                if 0 <= raw_label <= 18:
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

            if self.skip_night and (str(anno.get('night', 'false')).lower() == 'true' or anno.get('night') is True):
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
            
            # --- SKIP CONDITION FOR LABELS MAPPED TO UNKNOWN class ---
            if anomaly_class_label > 9: #Class is Unknown
                continue
                
            raw_ego = anno.get('ego_involve', anno.get('ego_involved', False))
            if isinstance(raw_ego, bool):
                ego_label = 1 if raw_ego else 0
            elif isinstance(raw_ego, (int, float)):
                ego_label = 1 if raw_ego != 0 else 0
            elif isinstance(raw_ego, str):
                ego_label = 1 if raw_ego.lower() in ('true', '1', 'yes') else 0
            else:
                ego_label = 0

            if is_valid_clip(anomaly_clip):
                self.samples.append({
                    'video_name': video_name,
                    'clip_paths': anomaly_clip,
                    'label': 1,
                    'class_label': anomaly_class_label,        # Target multiclass label (1..9)
                    'source_class_label': anomaly_class_label, # Source video sequence label
                    'ego_label': ego_label,                    # Ego involvement: 1 (ego) vs 0 (non-ego)
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
                        'class_label': 0,                          # Target multiclass label: 0 ("normal")
                        'source_class_label': anomaly_class_label, # Source video sequence label
                        'ego_label': ego_label,                    # Ego involvement: 1 (ego) vs 0 (non-ego)
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
                        'class_label': 0,                          # Target multiclass label: 0 ("normal")
                        'source_class_label': anomaly_class_label, # Source video sequence label
                        'ego_label': ego_label,                    # Ego involvement: 1 (ego) vs 0 (non-ego)
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
            
            mc_counts = Counter(valid_class_labels)
            for class_idx, count in sorted(mc_counts.items()):
                class_name = DOTA_CLASS_NAMES.get(class_idx, f"Class_{class_idx}")
                print(f" {str(class_name):<30} | ID: {class_idx:<3} | Count: {count}")
            
            ego_counts = Counter(s.get('ego_label', 0) for s in self.samples)
            print(f" Ego Involved (1): {ego_counts.get(1, 0)} | Non-Ego / 3rd Party (0): {ego_counts.get(0, 0)}")
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
        target_frame_id = os.path.basename(clip_paths[-1])

        frames = []
        for img_path in clip_paths:
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img) * 2 - 1
            frames.append(img)
            
        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        source_mc_label = sample.get('source_class_label', mc_label)
        ego_label = sample.get('ego_label', 0)
        if self.return_multiclass_labels:
            mc_label_tensor = torch.tensor(-1 if mc_label is None else mc_label, dtype=torch.long)
            source_mc_label_tensor = torch.tensor(-1 if source_mc_label is None else source_mc_label, dtype=torch.long)
            ego_label_tensor = torch.tensor(ego_label, dtype=torch.long)
            return clip_tensor, torch.tensor(label, dtype=torch.long), mc_label_tensor, source_mc_label_tensor, ego_label_tensor, video_id, target_frame_id

        return clip_tensor, torch.tensor(label, dtype=torch.long), video_id, target_frame_id


class DotaCloudDataset(Dataset):
    """
    Dataset class that loads clip image paths and metadata directly from
    the lightweight metadata file ('DoTA_training.pt') inside the DOTA_training folder.
    Frames are loaded on-the-fly from disk in __getitem__.
    """
    def __init__(self, export_dir="DOTA_training", cache_file="DoTA_training.pt", transform=None):
        cache_path = os.path.join(export_dir, cache_file) if not os.path.isabs(cache_file) and not os.path.exists(cache_file) else cache_file
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"DotaCloudDataset cache file not found at '{cache_path}'. Please run get_dota_dataloaders(use_cloud_dataset=False) first to generate it.")

        print(f"Loading DotaCloudDataset from '{cache_path}'...")
        data = torch.load(cache_path, map_location="cpu")
        self.clip_paths = data["clip_paths"]                 # List of list of exported frame image paths
        self.labels = data["labels"].long()                   # Tensor [N]
        self.mc_labels = data.get("mc_labels")                 # Tensor [N]
        self.source_mc_labels = data.get("source_mc_labels")     # Tensor [N]
        self.ego_labels = data.get("ego_labels")               # Tensor [N]
        self.video_ids = data.get("video_ids", [])
        self.target_frame_ids = data.get("target_frame_ids", [])
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop((288, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        self.samples = [
            {"label": int(self.labels[i].item()), "video_name": self.video_ids[i] if i < len(self.video_ids) else ""}
            for i in range(len(self.clip_paths))
        ]
        print(f"Successfully loaded DotaCloudDataset: {len(self.clip_paths)} samples.")

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        frame_paths = self.clip_paths[idx]
        frames = []
        for img_path in frame_paths:
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [3, 6, 288, 512]

        label = self.labels[idx]
        mc_label = self.mc_labels[idx] if self.mc_labels is not None else torch.tensor(-1, dtype=torch.long)
        source_mc_label = self.source_mc_labels[idx] if self.source_mc_labels is not None else torch.tensor(-1, dtype=torch.long)
        ego_label = self.ego_labels[idx] if self.ego_labels is not None else torch.tensor(-1, dtype=torch.long)
        video_id = self.video_ids[idx] if idx < len(self.video_ids) else ""
        target_frame_id = self.target_frame_ids[idx] if idx < len(self.target_frame_ids) else ""

        return clip_tensor, label, mc_label, source_mc_label, ego_label, video_id, target_frame_id


def get_dota_dataloaders(
    sequence_dir, 
    annotation_dir, 
    batch_size=8, 
    num_workers=4, 
    max_samples=None, 
    return_multiclass_labels=False, 
    num_frames_per_clip=6,
    use_cloud_dataset=False,
    cloud_dir="DOTA_training",
    cloud_file="DoTA_training.pt"
):
    """
    Initializes dataset (either DoTAClipDataset or DotaCloudDataset), performs 80/20 train/val split,
    exports images/cache to DOTA_training folder, and returns Train/Val DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Maps to [-1, 1], matching multiframe_val
    ])
    
    if use_cloud_dataset:
        full_dataset = DotaCloudDataset(export_dir=cloud_dir, cache_file=cloud_file, transform=transform)
    else:
        full_dataset = DoTAClipDataset(
            sequence_dir,
            annotation_dir,
            transform=transform,
            return_multiclass_labels=return_multiclass_labels,
            num_frames_per_clip=num_frames_per_clip,
            max_samples=max_samples
        )
    
    total_size = len(full_dataset)
    subset_size = min(total_size, max_samples) if max_samples is not None else total_size
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size
    unused_size = total_size - subset_size
    
    print("-" * 30)
    print(f"Total valid clips available: {total_size}")
    if subset_size < total_size:
        print(f"Max samples cap applied:     {subset_size}")
    print(f"Training set size:           {train_size}")
    print(f"Validation set size:         {val_size}")
    print("-" * 30)
    
    if subset_size == 0:
        raise ValueError("No valid clips were found. Check your directory paths and JSON structure.")
        
    if unused_size > 0:
        train_dataset, val_dataset, _ = random_split(
            full_dataset, 
            [train_size, val_size, unused_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    if not use_cloud_dataset:
        export_root_dir = cloud_dir
        if os.path.exists(export_root_dir):
            print(f"Cleaning out old export directory: {export_root_dir}")
            shutil.rmtree(export_root_dir)
            
        data_dir = os.path.join(export_root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        train_indices_set = set(train_dataset.indices)
        all_clip_paths, all_labels, all_mc_labels, all_src_mc_labels, all_ego_labels, all_video_ids, all_target_frame_ids = [], [], [], [], [], [], []

        print("Exporting frame images to 'DOTA_training/data' and building metadata cache...")
        for i in tqdm(range(len(full_dataset))):
            sample = full_dataset.samples[i]
            video_id = sample["video_name"]
            label = sample["label"]
            mc_label = sample.get("class_label", None)
            source_mc_label = sample.get("source_class_label", mc_label)
            ego_label = sample.get("ego_label", 0)

            split_name = "train" if i in train_indices_set else "val"
            class_name = "anomalous" if label == 1 else "good"
            target_data_dir = os.path.join(data_dir, split_name, f"{video_id}_{class_name}")
            os.makedirs(target_data_dir, exist_ok=True)

            sample_exported_paths = []
            for frame_idx, src_path in enumerate(sample["clip_paths"]):
                _, ext = os.path.splitext(src_path)
                dst_path = os.path.join(target_data_dir, f"frame_{frame_idx:04d}{ext}")
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                sample_exported_paths.append(dst_path)

            all_clip_paths.append(sample_exported_paths)
            all_labels.append(label)
            all_mc_labels.append(-1 if mc_label is None else mc_label)
            all_src_mc_labels.append(-1 if source_mc_label is None else source_mc_label)
            all_ego_labels.append(ego_label)
            all_video_ids.append(video_id)
            all_target_frame_ids.append(os.path.basename(sample["clip_paths"][-1]))

        cache_dict = {
            'clip_paths': all_clip_paths,
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'mc_labels': torch.tensor(all_mc_labels, dtype=torch.long),
            'source_mc_labels': torch.tensor(all_src_mc_labels, dtype=torch.long),
            'ego_labels': torch.tensor(all_ego_labels, dtype=torch.long),
            'video_ids': all_video_ids,
            'target_frame_ids': all_target_frame_ids
        }
        cache_file_path = os.path.join(export_root_dir, cloud_file)
        torch.save(cache_dict, cache_file_path)
        print(f"Successfully saved lightweight dataset metadata cache ({len(all_clip_paths)} samples) to '{cache_file_path}'")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


if __name__ == "__main__":
    SEQ_DIR = "../DoTA_sequences"
    ANNO_DIR = "../DOTA_annotations"
    
    # Global Parameters
    TARGET_FRAMES = 6 
    RETURN_MULTICLASS = True  # <-- Toggle multiclass output here
    MAX_SAMPLES = None      # <-- Global control of the subset size

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
        for clips, labels, mc_labels, source_mc_labels, ego_labels, video_ids, target_frame_ids in train_loader:
            print(f"Clip tensor shape:      {clips.shape}  --> Expected: [Batch=10, Channels=3, Time={TARGET_FRAMES}, Height=288, Width=512]")
            print(f"Binary Labels shape:    {labels.shape}  --> Expected: [Batch=10]")
            print(f"Binary Labels values:   {labels.tolist()}")
            print(f"Multiclass shape:       {mc_labels.shape} --> Expected: [Batch=10]")
            print(f"Source MC Labels:       {source_mc_labels.tolist()}")
            print(f"Ego Labels:             {ego_labels.tolist()}")
            print(f"Video IDs:              {video_ids[:3]}")
            print(f"Target Frame IDs:       {target_frame_ids[:3]}")
            break
            
        print("\nSuccess! Pipeline is ready.")
        
    except FileNotFoundError:
        print(f"\nError: Could not find directories at {SEQ_DIR} or {ANNO_DIR}.")
        print("Please verify your folder paths.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")