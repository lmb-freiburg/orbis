import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


#NEEDS FURTHER CHANGES 

def create_heatmap_from_activations(
    pt_file_path, 
    output_dir, 
    num_frames=5,       # Total frames (T) processed by the model (e.g., 4 context + 1 target)
    target_frame_idx=-2,# Index of the 'last context frame' (e.g., -2 if the last is the target)
    latent_h=36,        # STDiT spatial grid height (e.g., 288 / 8 VAE downsample)
    latent_w=64,        # STDiT spatial grid width (e.g., 512 / 8 VAE downsample)
    num_samples=20      # How many samples to visualize
):
    """
    Loads unpooled STDiT activations, extracts the specified frame, 
    collates the channels, and saves spatial heatmaps.
    """
    print(f"Loading activations from {pt_file_path}...")
    data = torch.load(pt_file_path, map_location="cpu")
    
    features = data['features'] # Shape: [B, ...]
    labels = data['labels']
    
    # ---------------------------------------------------------
    # 1. Reshape the Sequence back to Spatio-Temporal Dimensions
    # ---------------------------------------------------------
    B = features.shape[0]
    C = features.shape[-1]
    
    # Orbis/STDiT flattens tokens. We must reshape back to [B, T, H, W, C]
    # NOTE: Adjust latent_h and latent_w if your model uses an additional 
    # patchification step (e.g., 2x2 spatial patches would make it 18x32).
    expected_tokens = num_frames * latent_h * latent_w
    
    if features.dim() == 3: # Format is [B, Sequence_Length, C]
        actual_tokens = features.shape[1]
        if actual_tokens != expected_tokens:
            raise ValueError(f"Token count mismatch. Expected {expected_tokens}, got {actual_tokens}. Check your H, W, and T dimensions.")
            
        features = features.view(B, num_frames, latent_h, latent_w, C)
        
    elif features.dim() == 4: # Format is [B, T, N, C]
        features = features.view(B, num_frames, latent_h, latent_w, C)
        
    print(f"Reshaped features to: {features.shape} -> [B, T, H, W, C]")

    os.makedirs(output_dir, exist_ok=True)
    
    # Determine how many to process
    process_count = min(num_samples, B)
    print(f"Generating heatmaps for {process_count} samples...")
    
    for i in tqdm(range(process_count)):
        label = labels[i].item()
        class_name = "anomalous" if label == 1 else "good"
        
        # ---------------------------------------------------------
        # 2. Extract the Last Context Frame
        # ---------------------------------------------------------
        # If your model passes 5 frames where [0, 1, 2, 3] are context and [4] is target,
        # the "last context frame" is index 3 (or -2).
        frame_features = features[i, target_frame_idx] # Shape: [H, W, C]
        
        # ---------------------------------------------------------
        # 3. Collate Activations across the Channel Dimension (C)
        # ---------------------------------------------------------
        # Using L2 Norm across channels calculates the "magnitude" of attention/activation
        heatmap_raw = torch.norm(frame_features, p=2, dim=-1).numpy() # Shape: [H, W]
        
        # Alternatively, use Mean Pooling:
        # heatmap_raw = torch.mean(frame_features, dim=-1).numpy()
        
        # ---------------------------------------------------------
        # 4. Normalize for Visualization [0, 1]
        # ---------------------------------------------------------
        heatmap_min = heatmap_raw.min()
        heatmap_max = heatmap_raw.max()
        heatmap_norm = (heatmap_raw - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # ---------------------------------------------------------
        # 5. Resize and Apply Colormap
        # ---------------------------------------------------------
        # Resize from latent space (36x64) back to original pixel space (288x512) for clarity
        heatmap_resized = cv2.resize(heatmap_norm, (512, 288), interpolation=cv2.INTER_CUBIC)
        
        # Convert to 8-bit image for colormap
        heatmap_8bit = np.uint8(255 * heatmap_resized)
        
        # Apply Jet colormap (Red = High Activation, Blue = Low Activation)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # ---------------------------------------------------------
        # 6. Save the Image
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap_color)
        plt.title(f"Block 18 Activations\nSample {i} ({class_name}) | Frame Index: {target_frame_idx}")
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f"sample_{i:03d}_{class_name}_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

if __name__ == "__main__":
    PT_FILE = "./cached_features/val_block18.pt"
    OUTPUT_DIR = "./cached_features/heatmaps_block18"
    
    # Adjust `latent_h` and `latent_w` based on your exact spatial patch size
    create_heatmap_from_activations(
        pt_file_path=PT_FILE,
        output_dir=OUTPUT_DIR,
        num_frames=5,
        target_frame_idx=-1, # Grabs the 4th frame (last context frame before target)
        latent_h=36, 
        latent_w=64,
        num_samples=50
    )
    print("Done! Heatmaps saved to", OUTPUT_DIR)