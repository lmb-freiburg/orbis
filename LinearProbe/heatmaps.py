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
sys.path.append(str(Path(__file__).resolve().parents[1])) # Uncomment if needed
from util import instantiate_from_config
from linear_attention_probe_binary import AttentionProbe # Assuming this is the name of your probe file

def process_attention_map(attn_1d, img_bgr, W, H):
    """Helper function to normalize, resize, and overlay a single attention tensor."""
    attn_2d = attn_1d.reshape(18, 32).numpy()
    
    # Normalize attention weights to [0, 1]
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    
    # Resize and Color
    attn_resized = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    
    # Overlay
    overlay_bgr = cv2.addWeighted(img_bgr, 0.7, heatmap_bgr, 0.3, 0)
    
    # Convert BGR to RGB for matplotlib
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def generate_attention_heatmap(image_path, attn_tensor, save_path):
    """
    Overlays multi-head attention tensors onto the original image.
    Uses a GridSpec layout: 
    - Row 0 (3 columns): Raw Image | Mean Pure Attention | Mean Overlay
    - Row 1 (N columns): Pure attention for Head 1, 2, ..., N
    - Row 2 (N columns): Overlay for Head 1, 2, ..., N
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    num_heads = attn_tensor.shape[0]
    total_cols = num_heads * 3  
    total_rows = 3
    
    fig = plt.figure(figsize=(6 * num_heads, 18))
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig)
    
    mean_attn = attn_tensor.mean(dim=0)
    mean_heat, mean_over = process_attention_map(mean_attn, img, W, H)
    
    ax_raw = fig.add_subplot(gs[0, 0:num_heads])
    ax_raw.imshow(img_rgb)
    ax_raw.set_title('Raw Image', fontsize=24, pad=20)
    ax_raw.axis('off')
    
    ax_mean_heat = fig.add_subplot(gs[0, num_heads:2*num_heads])
    ax_mean_heat.imshow(mean_heat)
    ax_mean_heat.set_title('Mean Attention (Pure)', fontsize=24, pad=20)
    ax_mean_heat.axis('off')
    
    ax_mean_over = fig.add_subplot(gs[0, 2*num_heads:3*num_heads])
    ax_mean_over.imshow(mean_over)
    ax_mean_over.set_title('Mean Attention (Overlay)', fontsize=24, pad=20)
    ax_mean_over.axis('off')
    
    for head_idx in range(num_heads):
        head_heat, head_over = process_attention_map(attn_tensor[head_idx], img, W, H)
        
        col_start = head_idx * 3
        col_end = (head_idx + 1) * 3
        
        ax_head_pure = fig.add_subplot(gs[1, col_start:col_end])
        ax_head_pure.imshow(head_heat)
        ax_head_pure.set_title(f'Head {head_idx+1}\nPure Attention', fontsize=18)
        ax_head_pure.axis('off')

        ax_head_over = fig.add_subplot(gs[2, col_start:col_end])
        ax_head_over.imshow(head_over)
        ax_head_over.set_title(f'Head {head_idx+1}\nOverlay', fontsize=18)
        ax_head_over.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) 
    print(f"Saved plot to {save_path}")

# ==========================================
# New Relative Sampling & Forward Pass Logic
# ==========================================

def load_orbis_model(exp_dir, config_path, ckpt_path, device):
    """Loads the base Orbis ViT model to extract unpooled latents."""
    # Note: Replace `instantiate_from_config` with your actual import
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
    4. Saves heatmaps in target_dir/relative/.
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
    probe_model = AttentionProbe(input_dim=768, num_classes=2, num_heads=8).to(device) # Change num_classes to 18 if you moved to multi-class
    probe_model.load_state_dict(torch.load(probe_weights_path, map_location=device))
    probe_model.eval()
    
    # 3. Setup Image Transforms
    transform = transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        else:
            good_start = num_frames - raw_window
        good_indices = list(range(good_start, good_start + raw_window, 2))
        
        # We will process both clips iteratively
        clips_to_process = {
            "anomalous": anomaly_indices,
            "good": good_indices
        }
        
        for condition, indices in clips_to_process.items():
            clip_paths = [os.path.join(sequence_dir, frames_meta[i]['image_path'].replace('frames/', '')) for i in indices]
            
            # Verify images exist
            if not all(os.path.exists(p) for p in clip_paths):
                print(f"Skipping {video_name} ({condition}): Missing frame on disk.")
                continue
                
            # Load and Transform Frames
            frames = []
            for p in clip_paths:
                img = Image.open(p).convert('RGB')
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
            
            # Target frame for the heatmap is the *last* frame in the window (index 5)
            target_image_path = clip_paths[-1]
            save_path = os.path.join(relative_dir, f"{video_name}_{condition}.jpg")
            
            generate_attention_heatmap(target_image_path, final_attn_tensor, save_path)
            
    # Clean up the hook
    hook_handle.remove()
    print("\n--- Relative Heatmap Generation Complete ---")


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Call the new function
    sample_and_generate_relative_heatmaps(
        sequence_dir="../DoTA_sequences",
        annotation_dir="../DOTA_annotations",
        orbis_exp_dir="./logs_wm/orbis_288x512",
        orbis_config_path="config.yaml",
        orbis_ckpt_path="checkpoints/last.ckpt",
        probe_weights_path="best_attention_probe.pt", # Path to your saved probe state dict
        base_output_dir="./attention_heatmaps",
        device=DEVICE,
        num_samples=10,
        num_frames_per_clip=6
    )