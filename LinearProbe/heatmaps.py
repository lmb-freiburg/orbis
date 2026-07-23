import torch
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    # 1. Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    num_heads = attn_tensor.shape[0]
    
    # 2. Setup the Grid - (3 * num_heads) columns to perfectly align 3 top items over 8 bottom items
    total_cols = num_heads * 3  
    total_rows = 3
    
    fig = plt.figure(figsize=(6 * num_heads, 18))
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig)
    
    # 3. Process and Plot the Top Row (Mean Attention)
    mean_attn = attn_tensor.mean(dim=0)
    mean_heat, mean_over = process_attention_map(mean_attn, img, W, H)
    
    # Raw Image (spans first third)
    ax_raw = fig.add_subplot(gs[0, 0:num_heads])
    ax_raw.imshow(img_rgb)
    ax_raw.set_title('Raw Image', fontsize=24, pad=20)
    ax_raw.axis('off')
    
    # Mean Pure Attention (spans middle third)
    ax_mean_heat = fig.add_subplot(gs[0, num_heads:2*num_heads])
    ax_mean_heat.imshow(mean_heat)
    ax_mean_heat.set_title('Mean Attention (Pure)', fontsize=24, pad=20)
    ax_mean_heat.axis('off')
    
    # Mean Overlay (spans last third)
    ax_mean_over = fig.add_subplot(gs[0, 2*num_heads:3*num_heads])
    ax_mean_over.imshow(mean_over)
    ax_mean_over.set_title('Mean Attention (Overlay)', fontsize=24, pad=20)
    ax_mean_over.axis('off')
    
    # 4. Process and Plot the Individual Heads (Rows 1 & 2)
    for head_idx in range(num_heads):
        head_heat, head_over = process_attention_map(attn_tensor[head_idx], img, W, H)
        
        # Calculate column span for this specific head (each spans 3 units in the grid)
        col_start = head_idx * 3
        col_end = (head_idx + 1) * 3
        
        # Plot Row 1: Pure Attention for Head
        ax_head_pure = fig.add_subplot(gs[1, col_start:col_end])
        ax_head_pure.imshow(head_heat)
        ax_head_pure.set_title(f'Head {head_idx+1}\nPure Attention', fontsize=18)
        ax_head_pure.axis('off')

        # Plot Row 2: Overlay for Head
        ax_head_over = fig.add_subplot(gs[2, col_start:col_end])
        ax_head_over.imshow(head_over)
        ax_head_over.set_title(f'Head {head_idx+1}\nOverlay', fontsize=18)
        ax_head_over.axis('off')

    plt.tight_layout()
    
    # 5. Save output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # Free up memory
    
    print(f"Saved plot to {save_path}")


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Load your saved attention weights lookup map
    attn_map = torch.load("best_val_attention_weights.pt", map_location="cpu")

    print('Length of attention maps:', len(attn_map))

    # Assuming you have a folder of raw validation frames mapping to these IDs
    raw_frames_dir = "../DOTA_training/data/val"
    base_output_dir = "./attention_heatmaps"

    # Extract all sequence IDs from the dictionary
    all_sequence_ids = list(attn_map.keys())
    
    # Randomly sample up to 100 IDs
    num_samples = min(100, len(all_sequence_ids))
    sampled_ids = random.sample(all_sequence_ids, num_samples)
    
    print(f"Randomly selected {num_samples} sequences for heatmap generation.")

    for sequence_id in sampled_ids:
        attn_weights = attn_map[sequence_id]
        
        # Check for _anomalous vs _good folders
        path_anomalous = os.path.join(raw_frames_dir, f"{sequence_id}_anomalous", "frame_0005.jpg")
        path_good = os.path.join(raw_frames_dir, f"{sequence_id}_good", "frame_0005.jpg")
        
        if os.path.exists(path_anomalous):
            original_img_path = path_anomalous
            suffix = "anomalous"
        elif os.path.exists(path_good):
            original_img_path = path_good
            suffix = "good"
        else:
            print(f"Skipping {sequence_id}: Original image not found at either anomalous or good paths.")
            continue
            
        # Create a dynamic target directory based on the class (good/anomalous)
        target_dir = os.path.join(base_output_dir, suffix)
        
        # Dynamically name the file without the word 'heatmap'
        save_path = os.path.join(target_dir, f"{sequence_id}_{suffix}.jpg")
        
        generate_attention_heatmap(original_img_path, attn_weights, save_path)