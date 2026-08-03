import os
import json
import torch

def normalize_multiclass_label(raw_label):
    if raw_label is None:
        return None
    if isinstance(raw_label, int):
        return raw_label if 0 <= raw_label <= 18 else 0
    if isinstance(raw_label, str):
        try:
            return normalize_multiclass_label(int(raw_label))
        except ValueError:
            return 0
    return int(raw_label)


def resolve_multiclass_label(annotation, frames_meta, frame_idx):
    candidate_sources = [annotation]
    if frame_idx is not None and frames_meta and 0 <= frame_idx < len(frames_meta):
        candidate_sources.append(frames_meta[frame_idx])

    for source in candidate_sources:
        if not isinstance(source, dict):
            continue
        for key in ('accident_id', 'accident_name'):
            if key in source and source[key] is not None:
                label = normalize_multiclass_label(source[key])
                if label is not None:
                    return label
    return 0


def get_target_frame_id_from_disk(video_id, label, anomaly_start, anomaly_end, seq_dir='../DoTA_sequences', raw_window=12):
    img_dir = os.path.join(seq_dir, video_id, 'images')
    if not os.path.exists(img_dir):
        short_id = video_id.rsplit('_', 2)[0] if '_' in video_id else video_id
        img_dir = os.path.join(seq_dir, short_id, 'images')

    if not os.path.exists(img_dir):
        return '000000.jpg'

    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    if not frame_files:
        return '000000.jpg'

    if label == 1:
        indices = list(range(anomaly_start, anomaly_start + raw_window, 2))
    else:
        if anomaly_start >= raw_window:
            normal_start = 0
        elif (len(frame_files) - raw_window) > anomaly_end:
            normal_start = max(0, len(frame_files) - raw_window)
        else:
            normal_start = 0
        indices = list(range(normal_start, normal_start + raw_window, 2))

    last_idx = min(indices[-1], len(frame_files) - 1)
    return frame_files[last_idx]


def update_cache_file(cache_path, seq_dir='../DoTA_sequences', anno_dir='../DOTA_annotations', raw_window=12):
    if not os.path.exists(cache_path):
        print(f"Skipping update for missing file: {cache_path}")
        return False

    print(f"\n==================================================")
    print(f"Updating metadata for cached feature file:\n  -> '{cache_path}'")
    print(f"==================================================")

    cache = torch.load(cache_path)
    features = cache['features']
    labels = cache['labels']
    video_ids = cache['video_ids']
    N = len(video_ids)

    new_mc_labels = []
    new_source_mc_labels = []
    new_ego_labels = []
    new_target_frame_ids = []
    missing_anno = 0

    for i in range(N):
        vid = video_ids[i]
        label = int(labels[i].item())
        json_path = os.path.join(anno_dir, f"{vid}.json")

        if not os.path.exists(json_path):
            missing_anno += 1
            new_mc_labels.append(0 if label == 0 else 1)
            new_source_mc_labels.append(1)
            new_ego_labels.append(0)
            new_target_frame_ids.append('000000.jpg')
            continue

        with open(json_path, 'r') as f:
            anno = json.load(f)

        raw_ego = anno.get('ego_involve', anno.get('ego_involved', False))
        if isinstance(raw_ego, bool):
            ego_val = 1 if raw_ego else 0
        elif isinstance(raw_ego, (int, float)):
            ego_val = 1 if raw_ego != 0 else 0
        elif isinstance(raw_ego, str):
            ego_val = 1 if raw_ego.lower() in ('true', '1', 'yes') else 0
        else:
            ego_val = 0

        frames_meta = anno.get('images', [])
        num_frames = min(anno.get('num_frames', len(frames_meta)), len(frames_meta))
        anomaly_start = anno.get('accident_start', anno.get('anomaly_start', 0))
        anomaly_end = anno.get('accident_end', anno.get('anomaly_end', num_frames - 1))
        anomaly_class_label = resolve_multiclass_label(anno, frames_meta, anomaly_start)

        source_mc_label = anomaly_class_label
        mc_label = 0 if label == 0 else anomaly_class_label

        target_frame_id = get_target_frame_id_from_disk(
            vid, label, anomaly_start, anomaly_end, seq_dir=seq_dir, raw_window=raw_window
        )

        new_mc_labels.append(mc_label)
        new_source_mc_labels.append(source_mc_label)
        new_ego_labels.append(ego_val)
        new_target_frame_ids.append(target_frame_id)

    tensor_mc_labels = torch.tensor(new_mc_labels, dtype=torch.long)
    tensor_source_mc_labels = torch.tensor(new_source_mc_labels, dtype=torch.long)
    tensor_ego_labels = torch.tensor(new_ego_labels, dtype=torch.long)

    updated_dict = {
        'features': features,
        'labels': labels,
        'mc_labels': tensor_mc_labels,
        'source_mc_labels': tensor_source_mc_labels,
        'ego_labels': tensor_ego_labels,
        'video_ids': video_ids,
        'target_frame_ids': new_target_frame_ids
    }

    torch.save(updated_dict, cache_path)
    print(f"Successfully updated '{cache_path}':")
    print(f"  - Total samples:        {N}")
    print(f"  - Missing annotations:  {missing_anno}")
    print(f"  - Key 'features':       tensor {features.shape}, {features.dtype}")
    print(f"  - Key 'labels':         tensor {labels.shape}, {labels.dtype}")
    print(f"  - Key 'mc_labels':      tensor {tensor_mc_labels.shape}, {tensor_mc_labels.dtype}")
    print(f"  - Key 'source_mc_labels': tensor {tensor_source_mc_labels.shape}, {tensor_source_mc_labels.dtype}")
    print(f"  - Key 'ego_labels':     tensor {tensor_ego_labels.shape}, {tensor_ego_labels.dtype}")
    print(f"  - Key 'video_ids':       list of len {len(video_ids)}")
    print(f"  - Key 'target_frame_ids': list of len {len(new_target_frame_ids)}")
    return True


if __name__ == "__main__":
    cache_dir = "./cached_features"
    target_files = [
        os.path.join(cache_dir, "train_block18_4000_unpooled_mc.pt"),
        os.path.join(cache_dir, "val_block18_4000_unpooled_mc.pt"),
        os.path.join(cache_dir, "train_block20_4000_unpooled_mc.pt"),
        os.path.join(cache_dir, "val_block20_4000_unpooled_mc.pt"),
    ]

    for c_path in target_files:
        update_cache_file(c_path)
