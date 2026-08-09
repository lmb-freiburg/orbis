import torch
import cv2
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from PIL import Image

# Orbis imports (Ensure these are reachable in your environment)
import sys
from pathlib import Path
from omegaconf import OmegaConf

PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROBE_DIR)

for path_to_add in [PROJECT_ROOT, PROBE_DIR]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

def resolve_path(p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    p1 = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PROBE_DIR, p)
    if os.path.exists(p2):
        return p2
    return p1

from util import instantiate_from_config
from linear_attention_probe_binary import AttentionProbe

try:
    from dota import DOTA_CLASS_NAMES
except ImportError:
    try:
        from LinearProbe.dota import DOTA_CLASS_NAMES
    except ImportError:
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

def process_attention_map(attn_1d, img_bgr, W, H):
    """Helper function to normalize, resize, and overlay a single attention tensor."""
    attn_2d = attn_1d.reshape(18, 32).numpy()
    
    # Normalize attention weights to [0, 1]
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    
    # Resize and Color
    attn_resized = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    
    # Overlay
    overlay_bgr = cv2.addWeighted(img_bgr, 0.5, heatmap_bgr, 0.5, 0)
    
    # Convert BGR to RGB for matplotlib
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def generate_comparative_attention_heatmap(
    img_path_good, 
    attn_good, 
    img_path_anom, 
    attn_anom, 
    save_path,
    class_id_good=0,
    class_label_good="normal",
    class_id_anom=None,
    class_label_anom=None,
    video_name=None
):
    """
    Creates a 2x5 grid comparing Good (Top) vs Anomalous (Bottom) for a single sequence.
    Columns: Raw | Head 5 (Pure) | Head 5 (Overlay) | Mean (Pure) | Mean (Overlay)
    Displays Class ID and Class Label on the plots.
    """
    # Extract tensor, class_id, class_label if passed as a dictionary (e.g. from saved weights file)
    if isinstance(attn_good, dict):
        if 'class_id' in attn_good:
            class_id_good = attn_good.get('class_id', class_id_good)
        if 'class_label' in attn_good:
            class_label_good = attn_good.get('class_label', class_label_good)
        attn_good = attn_good['attn_weights']

    if isinstance(attn_anom, dict):
        if 'class_id' in attn_anom:
            class_id_anom = attn_anom.get('class_id', class_id_anom)
        if 'class_label' in attn_anom:
            class_label_anom = attn_anom.get('class_label', class_label_anom)
        attn_anom = attn_anom['attn_weights']

    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    
    def plot_row(row_idx, img_path, attn_tensor, prefix, class_id=None, class_label=None):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Build class label string
        label_info = ""
        if class_id is not None and class_label is not None:
            label_info = f" [Class {class_id}: {class_label}]"
        elif class_label is not None:
            label_info = f" [{class_label}]"
        elif class_id is not None:
            label_info = f" [Class ID: {class_id}]"

        row_title_prefix = f"{prefix}{label_info}"

        # Extract Head 5 (Index 4, assuming 0-indexed) and Mean
        head5_attn = attn_tensor[4] 
        mean_attn = attn_tensor.mean(dim=0)
        
        h5_heat, h5_over = process_attention_map(head5_attn, img, W, H)
        mean_heat, mean_over = process_attention_map(mean_attn, img, W, H)
        
        # Col 0: Raw Image
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(img_rgb)
        ax.set_title(f'{row_title_prefix} - Raw', fontsize=16, pad=15)
        ax.axis('off')
        
        # Col 1: Head 5 Pure Attention
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(h5_heat)
        ax.set_title(f'{row_title_prefix} - Head 5 (Pure)', fontsize=16, pad=15)
        ax.axis('off')
        
        # Col 2: Head 5 Overlay
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(h5_over)
        ax.set_title(f'{row_title_prefix} - Head 5 (Overlay)', fontsize=16, pad=15)
        ax.axis('off')
        
        # Col 3: Mean Pure Attention
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(mean_heat)
        ax.set_title(f'{row_title_prefix} - Mean (Pure)', fontsize=16, pad=15)
        ax.axis('off')
        
        # Col 4: Mean Overlay
        ax = fig.add_subplot(gs[row_idx, 4])
        ax.imshow(mean_over)
        ax.set_title(f'{row_title_prefix} - Mean (Overlay)', fontsize=16, pad=15)
        ax.axis('off')

    # Plot Good on Top (Row 0), Anomalous on Bottom (Row 1)
    plot_row(0, img_path_good, attn_good, "Good", class_id=class_id_good, class_label=class_label_good)
    plot_row(1, img_path_anom, attn_anom, "Anomalous", class_id=class_id_anom, class_label=class_label_anom)

    suptitle_parts = []
    if video_name:
        suptitle_parts.append(f"Sequence: {video_name}")
    if class_id_anom is not None and class_label_anom is not None:
        suptitle_parts.append(f"Anomaly Class ID {class_id_anom}: {class_label_anom}")
    elif class_id_anom is not None:
        suptitle_parts.append(f"Anomaly Class ID: {class_id_anom}")

    if suptitle_parts:
        fig.suptitle(" | ".join(suptitle_parts), fontsize=20, y=0.99)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) 
    print(f"Saved comparative plot to {save_path}")

# ==========================================
# Relative Sampling & Forward Pass Logic
# ==========================================

def load_orbis_model(exp_dir, config_path, ckpt_path, device):
    """Loads the base Orbis ViT model to extract unpooled latents."""
    from util import instantiate_from_config 
    
    config_file = (Path(exp_dir) / config_path).resolve()
    ckpt_file = (Path(exp_dir) / ckpt_path).resolve()

    cfg = OmegaConf.load(config_file)
    model = instantiate_from_config(cfg.model)
    state_dict = torch.load(str(ckpt_file), map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()

def sample_and_generate_relative_heatmaps(
    sequence_dir, 
    annotation_dir, 
    orbis_exp_dir,
    orbis_config_path,
    orbis_ckpt_path,
    probe_weights_path, 
    base_output_dir, 
    device,
    num_samples=10, 
    num_frames_per_clip=6
):
    """
    1. Randomly samples 10 sequences directly from DoTA sequence dir.
    2. Extracts both 'good' and 'anomalous' frames for each based on DoTA JSON.
    3. Runs them through Orbis -> Probe to extract live multi-head attention.
    4. Saves a single 2x5 comparative heatmap per sequence in target_dir/relative/.
    """
    
    print(f"--- Initializing Relative Heatmap Generation ({num_samples} sequences) ---")
    
    # 1. Load Orbis Backbone
    print("Loading Orbis Base Model...")
    orbis_model = load_orbis_model(orbis_exp_dir, orbis_config_path, orbis_ckpt_path, device)
    
    # Set up the forward hook on Block 18
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0] if isinstance(output, tuple) else output
        return hook
    
    backbone = getattr(orbis_model, "vit", orbis_model)
    hook_handle = backbone.blocks[17].register_forward_hook(get_activation('block18'))
    
    # 2. Load Trained Linear Probe
    print("Loading Attention Probe...")
    probe_model = AttentionProbe(input_dim=768, num_classes=2, num_heads=8).to(device) 
    probe_model.load_state_dict(torch.load(probe_weights_path, map_location=device))
    probe_model.eval()
    
    # 3. Setup Image Transforms
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Maps to [-1, 1], matching multiframe_val
    ])
    
    # 4. Filter and Sample Valid Sequences from DoTA
    raw_window = (num_frames_per_clip * 2) - 1
    valid_sequences = []
    
    for video_name in os.listdir(sequence_dir):
        video_folder = os.path.join(sequence_dir, video_name, 'images')
        json_path = os.path.join(annotation_dir, f"{video_name}.json")
        
        if not os.path.isdir(video_folder) or not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            anno = json.load(f)
            
        if (str(anno.get('ignore', 'false')).lower() != 'false') or anno.get('anomaly_start', -1) == -1:
            continue
            
        num_frames = anno['num_frames']
        anomaly_start = anno['anomaly_start']
        anomaly_end = anno['anomaly_end']
        
        # Check if the sequence is long enough to have BOTH a good and an anomalous window
        has_anomalous = num_frames >= raw_window and (anomaly_start + raw_window) <= num_frames
        has_good = anomaly_start >= raw_window or (num_frames - raw_window) > anomaly_end
        
        if has_anomalous and has_good:
            valid_sequences.append((video_name, anno))
            
    if len(valid_sequences) == 0:
        print("Error: Found zero sequences that contain both a good and anomalous window.")
        return
        
    sampled_sequences = random.sample(valid_sequences, min(num_samples, len(valid_sequences)))
    relative_dir = os.path.join(base_output_dir, "relative")
    os.makedirs(relative_dir, exist_ok=True)
    
    # 5. Process Each Sampled Sequence
    for video_name, anno in sampled_sequences:
        print(f"\nProcessing Sequence: {video_name}")
        frames_meta = anno['labels']
        anomaly_start = anno['anomaly_start']
        anomaly_end = anno['anomaly_end']
        num_frames = anno['num_frames']
        
        # 5a. Determine Window Indices
        anomaly_indices = list(range(anomaly_start, anomaly_start + raw_window, 2))
        
        if anomaly_start >= raw_window:
            good_start = 0
        elif (num_frames - raw_window) > anomaly_end:
            good_start = num_frames - raw_window
        else:
            print(f"Skipping {video_name}: No valid non-overlapping good window available.")
            continue
            
        good_indices = list(range(good_start, good_start + raw_window, 2))
        
        clips_to_process = {
            "good": good_indices,
            "anomalous": anomaly_indices
        }
        
        sequence_data = {}
        
        for condition, indices in clips_to_process.items():
            clip_paths = [os.path.join(sequence_dir, frames_meta[i]['image_path'].replace('frames/', '')) for i in indices]
            
            # Verify images exist
            if not all(os.path.exists(p) for p in clip_paths):
                print(f"Skipping {video_name} ({condition}): Missing frame on disk.")
                break
                
            # Load and Transform Frames
            frames = []
            for p in clip_paths:
                img = cv2.imread(p)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                frames.append(transform(img))
                
            # Shape: [Batch=1, Channels=3, Time=6, Height=288, Width=512]
            clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            # --- Forward Pass through Orbis Backbone ---
            with torch.no_grad():
                if hasattr(orbis_model, "encode_frames") and hasattr(orbis_model, "vit"):
                    clips = clip_tensor.permute(0, 2, 1, 3, 4).contiguous()
                    latents = orbis_model.encode_frames(clips)
                    context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
                    target = latents[:, -1:].contiguous()
                    
                    t = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
                    frame_rate = torch.full((clips.shape[0],), 5.0, dtype=torch.float32, device=device)
                    _ = orbis_model.vit(target, context, t, frame_rate=frame_rate)
                else:
                    t = torch.zeros(clip_tensor.shape[0], dtype=torch.float32, device=device)
                    _ = orbis_model(clip_tensor, t)
                
                # Retrieve intercepted unpooled features (skip pooling)
                features = activation['block18']
                if features.dim() == 4:
                    features = features[:, -1, :, :] # Shape: [1, 576, 768]
                
                # --- Forward Pass through Attention Probe ---
                _, attn_weights = probe_model(features) # Shape: [1, 8, 1, 576]
                
                # Squeeze to [8, 576]
                final_attn_tensor = attn_weights.squeeze(0).squeeze(1).cpu() 
            
            # Target frame for the heatmap is the *last* frame in the window
            sequence_data[condition] = {
                "img_path": clip_paths[-1],
                "attn_tensor": final_attn_tensor
            }
            
        # If we successfully processed both Good and Anomalous clips, generate the combined plot
        if "good" in sequence_data and "anomalous" in sequence_data:
            # Extract class ID and label for anomalous clip from annotation JSON
            anom_class_id = None
            for key in ('accident_id', 'accident_name'):
                if key in anno and anno[key] is not None:
                    try:
                        anom_class_id = int(anno[key])
                        break
                    except (ValueError, TypeError):
                        pass
            if anom_class_id is None:
                for f_meta in anno.get('labels', []):
                    for key in ('accident_id', 'accident_name'):
                        if key in f_meta and f_meta[key] is not None:
                            try:
                                anom_class_id = int(f_meta[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    if anom_class_id is not None:
                        break

            if anom_class_id is None:
                anom_class_id = 0

            anom_class_label = DOTA_CLASS_NAMES.get(anom_class_id, f"Class_{anom_class_id}")

            save_path = os.path.join(relative_dir, f"{video_name}_comparison.jpg")
            generate_comparative_attention_heatmap(
                img_path_good=sequence_data["good"]["img_path"],
                attn_good=sequence_data["good"]["attn_tensor"],
                img_path_anom=sequence_data["anomalous"]["img_path"],
                attn_anom=sequence_data["anomalous"]["attn_tensor"],
                save_path=save_path,
                class_id_good=0,
                class_label_good="normal",
                class_id_anom=anom_class_id,
                class_label_anom=anom_class_label,
                video_name=video_name
            )
            
    # Clean up the hook
    hook_handle.remove()
    print("\n--- Relative Heatmap Generation Complete ---")


def generate_heatmaps_from_saved_weights(
    weights_path="best_val_attention_weights.pt",
    sequence_dir="../DoTA_sequences",
    output_dir="DiagnosticProbes/heatmaps/saved",
    num_worst_fp=5,
    num_worst_fn=5,
    only_worst_mistakes=True
):
    weights_path = resolve_path(weights_path)
    sequence_dir = resolve_path(sequence_dir)
    output_dir = resolve_path(output_dir)

    if not weights_path or not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return

    ckpt = torch.load(weights_path)
    if isinstance(ckpt, dict) and "sequences" in ckpt:
        sequences = ckpt["sequences"]
        fp_ids = ckpt.get("fps", [])
        fn_ids = ckpt.get("fns", [])
    else:
        sequences = ckpt
        fp_ids = [vid for vid, data in sequences.items() if isinstance(data, dict) and data.get("binary_label") == 0 and data.get("pred_label") == 1]
        fn_ids = [vid for vid, data in sequences.items() if isinstance(data, dict) and data.get("binary_label") == 1 and data.get("pred_label") == 0]

    print(f"Loaded {len(sequences)} sequence attention weights from {weights_path}")

    target_samples = []
    if only_worst_mistakes:
        print(f"Found {len(fp_ids)} False Positives and {len(fn_ids)} False Negatives.")
        print(f"Extracting Top {min(num_worst_fp, len(fp_ids))} Worst FPs and Top {min(num_worst_fn, len(fn_ids))} Worst FNs...")
        for vid in fp_ids[:num_worst_fp]:
            if vid in sequences:
                target_samples.append((vid, sequences[vid], "FP"))
        for vid in fn_ids[:num_worst_fn]:
            if vid in sequences:
                target_samples.append((vid, sequences[vid], "FN"))

    if not target_samples:
        print("No worst mistake entries found or only_worst_mistakes is False. Processing standard samples...")
        for vid, data in list(sequences.items())[:10]:
            target_samples.append((vid, data, "SAMPLE"))

    os.makedirs(output_dir, exist_ok=True)

    for video_id, data, category in target_samples:
        if isinstance(data, dict):
            attn_weights = data['attn_weights']  # Shape: [8, 576]
            class_id = data.get('class_id')
            class_label = data.get('class_label', 'Unknown')
            source_class_id = data.get('source_class_id')
            source_class_label = data.get('source_class_label')
            prob_anom = data.get('prob_anom')
            target_frame_id = data.get('target_frame_id')
        else:
            attn_weights = data
            class_id = None
            class_label = "Unknown"
            source_class_id = None
            source_class_label = None
            prob_anom = None
            target_frame_id = None

        # Resolve video folder path
        video_name = video_id.rsplit('_', 2)[0] if '_' in video_id else video_id
        video_folder = os.path.join(sequence_dir, video_name, 'images')
        
        if not os.path.exists(video_folder):
            video_folder = os.path.join(sequence_dir, video_id, 'images')
            if not os.path.exists(video_folder):
                print(f"Warning: Image folder for {video_id} not found at {video_folder}")
                continue

        if target_frame_id and os.path.exists(os.path.join(video_folder, target_frame_id)):
            target_img_path = os.path.join(video_folder, target_frame_id)
        else:
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
            if not frame_files:
                continue
            target_img_path = os.path.join(video_folder, frame_files[-1])

        target_frame_name = os.path.basename(target_img_path)
        print(f"[{category}] Video: {video_id} --> Overlaying attention onto Target Frame ID: {target_frame_name} ({target_img_path})")

        img = cv2.imread(target_img_path)
        if img is None:
            print(f"Error: Could not load image at {target_img_path}")
            continue
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot 2x5 Grid: Raw Image, Mean Attention, and all 8 Heads individually
        fig = plt.figure(figsize=(25, 10))
        gs = gridspec.GridSpec(2, 5, figure=fig)

        # Row 0, Col 0: Raw RGB Image (with Target Frame ID)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_rgb)
        ax.set_title(f"Raw Image ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        # Row 0, Cols 1 to 4: Heads 1, 2, 3, 4
        num_heads = attn_weights.shape[0] if attn_weights.dim() > 1 else 8
        for h in range(min(4, num_heads)):
            head_attn = attn_weights[h]
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[0, h + 1])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        # Row 1, Col 0: Mean Attention Across All Heads
        mean_attn = attn_weights.mean(dim=0) if attn_weights.dim() > 1 else attn_weights
        _, mean_overlay = process_attention_map(mean_attn, img, W, H)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(mean_overlay)
        ax.set_title(f"Mean Attention ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        # Row 1, Cols 1 to 4: Heads 5, 6, 7, 8
        for h in range(4, min(8, num_heads)):
            head_attn = attn_weights[h]
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[1, h - 3])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        # Header Title detailing worst case, frame ID, target & source class probability
        frame_str = f" | Target Frame: {target_frame_name}"
        src_str = f" [Source Seq: {source_class_label}]" if source_class_label and source_class_label != class_label else ""
        if category == "FP":
            prob_str = f" | P(Anomalous)={prob_anom:.4f}" if prob_anom is not None else ""
            header_str = f"[WORST FALSE POSITIVE] Video: {video_id}{frame_str} | Class: {class_label} (ID: {class_id}){src_str}{prob_str}"
        elif category == "FN":
            prob_str = f" | P(Anomalous)={prob_anom:.4f}" if prob_anom is not None else ""
            header_str = f"[WORST FALSE NEGATIVE] Video: {video_id}{frame_str} | Class: {class_label} (ID: {class_id}){src_str}{prob_str}"
        else:
            label_str = f" [Class {class_id}: {class_label}]" if class_id is not None and class_label is not None else ""
            header_str = f"Video: {video_id}{frame_str}{label_str}{src_str}"

        fig.suptitle(header_str, fontsize=18, y=1.02, fontweight='bold')
        plt.tight_layout()

        frame_stem = os.path.splitext(target_frame_name)[0]
        save_filename = f"{category}_{video_id}_frame_{frame_stem}_heatmap.jpg" if category != "SAMPLE" else f"{video_id}_frame_{frame_stem}_heatmap.jpg"
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved attention heatmap for [{category}] {video_id} (Target Frame: {target_frame_name}) to {save_path}\n")

    print(f"\nCompleted generating heatmaps in: {output_dir}")


def generate_attention_heatmaps_binary(
    weights_path="best_val_attention_weights.pt",
    sequence_dir="../DoTA_sequences",
    output_dir="DiagnosticProbes/heatmaps/binary"
):
    weights_path = resolve_path(weights_path)
    sequence_dir = resolve_path(sequence_dir)
    output_dir = resolve_path(output_dir)

    if not weights_path or not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return

    ckpt = torch.load(weights_path)
    if isinstance(ckpt, dict) and "sequences" in ckpt:
        sequences = ckpt["sequences"]
    else:
        sequences = ckpt

    import shutil
    if os.path.exists(output_dir):
        print(f"Cleaning out stale heatmaps from previous runs in '{output_dir}'...")
        shutil.rmtree(output_dir)

    print(f"\n==================================================")
    print(f"Generating binary attention heatmaps in: '{output_dir}'")
    print(f"Total validation sequences: {len(sequences)}")
    print(f"==================================================")

    category_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for key_id, data in sequences.items():
        if not isinstance(data, dict):
            continue

        video_id = data.get('video_id', key_id)
        binary_label = data.get('binary_label', 0)
        pred_label = data.get('pred_label', 0)

        if binary_label == 1 and pred_label == 1:
            category = "TP"
        elif binary_label == 0 and pred_label == 0:
            category = "TN"
        elif binary_label == 0 and pred_label == 1:
            category = "FP"
        elif binary_label == 1 and pred_label == 0:
            category = "FN"
        else:
            category = "UNKNOWN"

        category_counts[category] = category_counts.get(category, 0) + 1

        # Source class label for folder naming
        src_id = data.get('source_class_id', -1)
        src_label = data.get('source_class_label')
        if not src_label or src_label == 'Unknown':
            class_id = data.get('class_id', 0)
            src_label = DOTA_CLASS_NAMES.get(src_id if src_id >= 0 else class_id, f"Class_{src_id}")

        # Sanitize folder name
        source_folder = str(src_label).lower().replace(' ', '_')
        target_dir = os.path.join(output_dir, source_folder, category)
        os.makedirs(target_dir, exist_ok=True)

        attn_weights = data.get('attn_weights')  # Shape: [8, 576] or [576]
        if attn_weights is None:
            continue

        target_frame_id = data.get('target_frame_id')
        class_id = data.get('class_id')
        class_label = data.get('class_label', 'Unknown')
        prob_anom = data.get('prob_anom')

        # Resolve image file
        video_name = video_id.rsplit('_', 2)[0] if '_' in video_id and not os.path.exists(os.path.join(sequence_dir, video_id, 'images')) else video_id
        video_folder = os.path.join(sequence_dir, video_id, 'images')
        if not os.path.exists(video_folder):
            video_folder = os.path.join(sequence_dir, video_name, 'images')

        if not os.path.exists(video_folder):
            print(f"Warning: Image folder for {video_id} not found at {video_folder}")
            continue

        if target_frame_id and os.path.exists(os.path.join(video_folder, target_frame_id)):
            target_img_path = os.path.join(video_folder, target_frame_id)
        else:
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
            if not frame_files:
                continue
            target_img_path = os.path.join(video_folder, frame_files[-1])

        target_frame_name = os.path.basename(target_img_path)
        img = cv2.imread(target_img_path)
        if img is None:
            continue

        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Render 2x5 Grid: Raw RGB, Mean Attention, and 8 individual heads
        fig = plt.figure(figsize=(25, 10))
        gs = gridspec.GridSpec(2, 5, figure=fig)

        # Row 0, Col 0: Raw RGB Image
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_rgb)
        ax.set_title(f"Raw Image ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        num_heads = attn_weights.shape[0] if attn_weights.dim() > 1 else 8
        for h in range(min(4, num_heads)):
            head_attn = attn_weights[h] if attn_weights.dim() > 1 else attn_weights
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[0, h + 1])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        mean_attn = attn_weights.mean(dim=0) if attn_weights.dim() > 1 else attn_weights
        _, mean_overlay = process_attention_map(mean_attn, img, W, H)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(mean_overlay)
        ax.set_title(f"Mean Attention ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        for h in range(4, min(8, num_heads)):
            head_attn = attn_weights[h]
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[1, h - 3])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        frame_str = f" | Target Frame: {target_frame_name}"
        src_str = f" [Source Seq: {src_label}]" if src_label and src_label != class_label else ""
        prob_str = f" | P(Anomalous)={prob_anom:.4f}" if prob_anom is not None else ""
        header_str = f"[{category}] Video: {video_id}{frame_str} | Class: {class_label} (ID: {class_id}){src_str}{prob_str}"

        fig.suptitle(header_str, fontsize=18, y=1.02, fontweight='bold')
        plt.tight_layout()

        frame_stem = os.path.splitext(target_frame_name)[0]
        save_filename = f"{category}_{video_id}_frame_{frame_stem}_heatmap.jpg"
        save_path = os.path.join(target_dir, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"\nHeatmaps summary by category:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} heatmaps saved")
    print(f"Completed generating binary attention heatmaps in: '{output_dir}'")


# ==========================================
# Execution Entry Point:
# ==========================================
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    generate_attention_heatmaps_binary(
        weights_path="checkpoints/binary/best_binary_val_attention_weights.pt",
        sequence_dir="../DoTA_sequences",
        output_dir="DiagnosticProbes/heatmaps/binary"
    )