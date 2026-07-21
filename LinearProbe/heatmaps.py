import torch
import cv2
import numpy as np
import os

def generate_attention_heatmap(image_path, attn_tensor, save_path):
    """
    Overlays a 1D attention tensor (576 tokens) onto the original image.
    
    Args:
        image_path: Path to the original video frame (e.g., the target frame).
        attn_tensor: 1D torch Tensor of size [576].
        save_path: Where to save the resulting heatmap image.
    """
    # 1. Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    H, W, _ = img.shape

    # 2. Reshape attention to spatial dimensions (18x32)
    # Assumes your tokens are a flattened 18 (height) x 32 (width) grid
    attn_2d = attn_tensor.reshape(18, 32).numpy()
    
    # 3. Normalize attention weights to [0, 1] for coloring
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    
    # 4. Resize attention map to match original image dimensions
    attn_resized = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # 5. Apply Jet colormap (Red = High attention, Blue = Low attention)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    
    # 6. Overlay heatmap onto the original image (50% transparency)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # 7. Save output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    print(f"Saved heatmap to {save_path}")

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Load your saved attention weights lookup map
    attn_map = torch.load("best_val_attention_weights.pt", map_location="cpu")

    # Assuming you have a folder of raw validation frames mapping to these IDs
    raw_frames_dir = "../DOTA_traing/data/val"
    output_dir = "./attention_heatmaps"

    for sequence_id, attn_weights in attn_map.items():
        # Adjust this path logic based on how your dataset stores images
        original_img_path = os.path.join(raw_frames_dir, f"{sequence_id}_anamolous, frame_0005.jpg")
        save_path = os.path.join(output_dir, f"{sequence_id}_heatmap.jpg")
        
        if os.path.exists(original_img_path):
            generate_attention_heatmap(original_img_path, attn_weights, save_path)
        else:
            print(f"Skipping {sequence_id}: Original image not found.")