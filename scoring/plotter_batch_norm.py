import os
import sys
import glob
import warnings
from pathlib import Path

# 1. Suppress torchvision C-extension warning on macOS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 2. Add project root to sys.path so modules can import each other reliably
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. Direct imports from scorer.py and util.py
from util import instantiate_from_config
from scoring.scorer_batch_norm import (
    surprise_score,
    get_sorted_frame_paths,
    load_clip_as_window,
    get_device,
)

RESULTS_DIR = "results"

## changes for new normalization method

def normalize_map_with_calib(raw_map, t_val, calib_stats, head="combined", eps=1e-8):
    """
    Normalizes a 3D score map [C, H, W] per-channel first using 3D calib_stats [C, H, W],
    then averages across channels to produce a 2D spatial score map [H, W].
    """
    # Remove extra batch dimension if raw_map is [1, C, H, W] or [1, 1, C, H, W]
    while raw_map.dim() > 3:
        raw_map = raw_map.squeeze(0)

    # Move calibration stats to the map's device and ensure floating type matching
    c_mean = calib_stats[head][t_val]["mean"].to(raw_map.device).float()  # [C, H, W]
    c_std = calib_stats[head][t_val]["std"].to(raw_map.device).float()    # [C, H, W]

    # Step 1: Normalize PER CHANNEL first -> Shape: [C, H, W]
    z_score_per_channel = (raw_map.float() - c_mean) / (c_std + eps)

    # Step 2: Take the mean across channels -> Shape: [H, W]
    z_score_combined = z_score_per_channel.mean(dim=0)

    return z_score_combined


def plot_and_overlay(
    model, 
    clip_id, 
    folder_path, 
    sample_label, 
    t_grid, 
    calib_stats, 
    device, 
    heads=["detailed", "semantic"],
    n_noise_samples=2, 
    z_vmax=3.0,
    z_threshold=1.0,
    use_minmax=False,
    output_dir_override=None,
    eps=1e-8
):
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    # 1. Compute raw error maps for requested heads
    per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=heads, n_noise_samples=n_noise_samples)
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
    output_dir = output_dir_override if output_dir_override else f"results/batch_norm/{clip_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup figure
    n_rows = len(heads)
    n_cols = len(t_grid) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, head in enumerate(heads):
        z_maps_list = []
        z_up_np_dict = {}

        for t_val in t_grid:
            raw_map = per_t_maps[head][t_val]
            # Normalizes each channel [C, H, W] standardizing against c_mean & c_std first, then takes mean(dim=0) -> [H, W]
            z_map = normalize_map_with_calib(raw_map, t_val, calib_stats, head=head, eps=eps)
            z_maps_list.append(z_map)
            
            # Upsample 2D map [H, W] to full frame resolution [288, 512]
            z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
            z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
            z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

        # Average normalized maps across timesteps (t_grid)
        z_map_avg = torch.stack(z_maps_list).mean(dim=0)
        z_map_avg_4d = z_map_avg.unsqueeze(0).unsqueeze(0).float()
        z_up_avg = F.interpolate(z_map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
        z_up_avg_np = z_up_avg.squeeze().cpu().numpy()

        # Col 0: Raw Frame
        axes[row_idx, 0].imshow(target_frame)
        axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame")
        axes[row_idx, 0].axis("off")

        # Col 1: Average Map Across t
        if use_minmax:
            avg_min, avg_max = z_up_avg_np.min(), z_up_avg_np.max()
            z_avg_proc = (z_up_avg_np - avg_min) / (avg_max - avg_min + eps)
            z_avg_masked = np.ma.masked_where(z_avg_proc < 0.2, z_avg_proc)
            curr_vmin, curr_vmax = 0.2, 1.0
        else:
            z_avg_masked = np.ma.masked_where(z_up_avg_np < z_threshold, z_up_avg_np)
            curr_vmin, curr_vmax = z_threshold, z_vmax

        axes[row_idx, 1].imshow(target_frame)
        im_avg = axes[row_idx, 1].imshow(z_avg_masked, cmap="jet", alpha=0.6, vmin=curr_vmin, vmax=curr_vmax)
        axes[row_idx, 1].set_title("Avg Across t", fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Cols 2..N: Timesteps
        for i, t_val in enumerate(t_grid):
            z_up_np = z_up_np_dict[t_val]
            if use_minmax:
                z_min, z_max = z_up_np.min(), z_up_np.max()
                z_proc = (z_up_np - z_min) / (z_max - z_min + eps)
                z_masked = np.ma.masked_where(z_proc < 0.2, z_proc)
                curr_vmin, curr_vmax = 0.2, 1.0
            else:
                z_masked = np.ma.masked_where(z_up_np < z_threshold, z_up_np)
                curr_vmin, curr_vmax = z_threshold, z_vmax

            col_idx = i + 2
            axes[row_idx, col_idx].imshow(target_frame)
            axes[row_idx, col_idx].imshow(z_masked, cmap="jet", alpha=0.55, vmin=curr_vmin, vmax=curr_vmax)
            axes[row_idx, col_idx].set_title(f"t={t_val}")
            axes[row_idx, col_idx].axis("off")

    mode_str = "MinMax" if use_minmax else "ZScore"
    fig.suptitle(f"{clip_id} ({sample_label}) — Detailed & Semantic [{mode_str}]", fontsize=14)
    
    save_path = f"{output_dir}/overlay_{sample_label.lower()}_combined_{mode_str.lower()}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# def normalize_map_with_calib(raw_map, t_val, calib_stats, head, eps=1e-8):
#     """
#     raw_map: [H, W] PyTorch tensor.
#     calib_stats: loaded nested dict from calib_stats_detailed_semantic.pt
#     head: string ("detailed" or "semantic")
#     """
#     # Extract head-specific calibration stats
#     mean = calib_stats[head][t_val]["mean"].to(raw_map.device)
#     std = calib_stats[head][t_val]["std"].to(raw_map.device)
    
#     # Standardize (Z-Score)
#     z_map = (raw_map - mean) / (std + eps)
    
#     # ReLU clamp to ignore below-average error (keep positive anomalies)
#     return torch.clamp(z_map, min=0.0)


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


# def plot_and_overlay(
#     model, 
#     clip_id, 
#     folder_path, 
#     sample_label, 
#     t_grid, 
#     calib_stats, 
#     device, 
#     heads=["detailed", "semantic"],
#     n_noise_samples=2, 
#     z_vmax=3.0,
#     z_threshold=1.0,
#     use_minmax=False,
#     output_dir_override=None,
#     eps=1e-8
# ):
#     paths = get_sorted_frame_paths(folder_path)
#     window = load_clip_as_window(paths).unsqueeze(0).to(device)
#     frame_rate = torch.full((1,), 5.0).to(device)

#     # 1. Compute raw error maps for requested heads
#     per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=heads, n_noise_samples=n_noise_samples)
#     target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
#     output_dir = output_dir_override if output_dir_override else f"results/batch_norm/{clip_id}"
#     os.makedirs(output_dir, exist_ok=True)

#     # 2. Setup 2-row figure (Row 0 = Detailed, Row 1 = Semantic)
#     n_rows = len(heads)
#     n_cols = len(t_grid) + 2
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))

#     if n_rows == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for row_idx, head in enumerate(heads):
#         z_maps_list = []
#         z_up_np_dict = {}

#         for t_val in t_grid:
#             raw_map = per_t_maps[head][t_val]
#             z_map = normalize_map_with_calib(raw_map, t_val, calib_stats, head=head, eps=eps)
#             z_maps_list.append(z_map)
            
#             z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
#             z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
#             z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

#         z_map_avg = torch.stack(z_maps_list).mean(dim=0)
#         z_map_avg_4d = z_map_avg.unsqueeze(0).unsqueeze(0).float()
#         z_up_avg = F.interpolate(z_map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
#         z_up_avg_np = z_up_avg.squeeze().cpu().numpy()

#         # Col 0: Raw Frame
#         axes[row_idx, 0].imshow(target_frame)
#         axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame")
#         axes[row_idx, 0].axis("off")

#         # Col 1: Average Map Across t
#         if use_minmax:
#             avg_min, avg_max = z_up_avg_np.min(), z_up_avg_np.max()
#             z_avg_proc = (z_up_avg_np - avg_min) / (avg_max - avg_min + eps)
#             z_avg_masked = np.ma.masked_where(z_avg_proc < 0.2, z_avg_proc)
#             curr_vmin, curr_vmax = 0.2, 1.0
#         else:
#             z_avg_masked = np.ma.masked_where(z_up_avg_np < z_threshold, z_up_avg_np)
#             curr_vmin, curr_vmax = z_threshold, z_vmax

#         axes[row_idx, 1].imshow(target_frame)
#         im_avg = axes[row_idx, 1].imshow(z_avg_masked, cmap="jet", alpha=0.6, vmin=curr_vmin, vmax=curr_vmax)
#         axes[row_idx, 1].set_title("Avg Across t", fontweight="bold")
#         axes[row_idx, 1].axis("off")

#         # Cols 2..N: Timesteps
#         for i, t_val in enumerate(t_grid):
#             z_up_np = z_up_np_dict[t_val]
#             if use_minmax:
#                 z_min, z_max = z_up_np.min(), z_up_np.max()
#                 z_proc = (z_up_np - z_min) / (z_max - z_min + eps)
#                 z_masked = np.ma.masked_where(z_proc < 0.2, z_proc)
#                 curr_vmin, curr_vmax = 0.2, 1.0
#             else:
#                 z_masked = np.ma.masked_where(z_up_np < z_threshold, z_up_np)
#                 curr_vmin, curr_vmax = z_threshold, z_vmax

#             col_idx = i + 2
#             axes[row_idx, col_idx].imshow(target_frame)
#             axes[row_idx, col_idx].imshow(z_masked, cmap="jet", alpha=0.55, vmin=curr_vmin, vmax=curr_vmax)
#             axes[row_idx, col_idx].set_title(f"t={t_val}")
#             axes[row_idx, col_idx].axis("off")

#     mode_str = "MinMax" if use_minmax else "ZScore"
#     fig.suptitle(f"{clip_id} ({sample_label}) — Detailed & Semantic [{mode_str}]", fontsize=14)
    
#     save_path = f"{output_dir}/overlay_{sample_label.lower()}_combined_{mode_str.lower()}.png"
#     fig.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    t_grid = [0.2, 0.4, 0.6, 0.8]
    HEADS = ["detailed", "semantic"]
    
    # Load model
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load multi-head calibration statistics
    calib_stats_path = "results/calib_stats_detailed_semantic.pt"
    if not os.path.exists(calib_stats_path):
        calib_stats_path = f"{RESULTS_DIR}/calib_stats.pt"
        if not os.path.exists(calib_stats_path):
            raise FileNotFoundError("Missing calibration stats file. Please run the calibration script first.")

    calib_stats = torch.load(calib_stats_path, weights_only=True)

    test_clip_id = "D_pyFV4nKd4_003993"

    # Plot Normal Sample for both Detailed & Semantic
    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/non-ood",
        sample_label="Normal",
        t_grid=t_grid,
        calib_stats=calib_stats,
        device=device,
        heads=HEADS,
        use_minmax=False,
        z_vmax=3.0,
    )

    # Plot Anomaly Sample for both Detailed & Semantic
    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/ood",
        sample_label="Anomaly",
        t_grid=t_grid,
        calib_stats=calib_stats,
        device=device,
        heads=HEADS,
        use_minmax=False,
        z_vmax=3.0,
    )