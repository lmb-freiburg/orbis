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
    surprise_score_semantic,
    get_sorted_frame_paths,
    load_clip_as_window,
    get_device,
)

RESULTS_DIR = "results"


def normalize_map_with_calib(raw_map, t_val, calib_stats, eps=1e-8):
    """
    raw_map: [H, W] PyTorch tensor.
    calib_stats: loaded dict from calib_stats.pt
    """
    # Ensure mean and std match the device of raw_map (e.g., mps or cuda or cpu)
    mean = calib_stats[t_val]["mean"].to(raw_map.device)
    std = calib_stats[t_val]["std"].to(raw_map.device)
    
    # Standardize (Z-Score)
    z_map = (raw_map - mean) / (std + eps)
    
    # ReLU clamp to ignore below-average error (keep positive anomalies)
    return torch.clamp(z_map, min=0.0)


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def plot_and_overlay(
    model, 
    clip_id, 
    folder_path, 
    sample_label, 
    t_grid, 
    calib_stats, 
    device, 
    n_noise_samples=4, 
    z_vmax=3.0,
    z_threshold=1.0,
    use_minmax=False,
    output_dir_override=None,
    eps=1e-8
):
    """
    Computes error map, normalizes with calib_stats, and plots heatmaps alongside raw target frame.
    Layout Order: [Original Frame] -> [Avg Across t] -> [t1, t2, ...]
    """
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    # 1. Compute raw error maps on the fly
    per_t_map = surprise_score_semantic(model, window, frame_rate, t_grid, n_noise_samples=n_noise_samples)

    # 2. Load background target frame
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
    # 3. Handle output directory override
    if output_dir_override:
        output_dir = output_dir_override
    else:
        output_dir = f"{RESULTS_DIR}/batch_norm/{clip_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Standardize maps across all timesteps first
    z_maps_list = []
    z_up_np_dict = {}

    for t_val in t_grid:
        raw_map = per_t_map[t_val]
        z_map = normalize_map_with_calib(raw_map, t_val, calib_stats)
        z_maps_list.append(z_map)
        
        # Upsample to full image resolution [288, 512]
        z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
        z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
        z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

    # Compute Average Z-map across all t
    z_map_avg = torch.stack(z_maps_list).mean(dim=0)
    z_map_avg_4d = z_map_avg.unsqueeze(0).unsqueeze(0).float()
    z_up_avg = F.interpolate(z_map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
    z_up_avg_np = z_up_avg.squeeze().cpu().numpy()

    # Columns: [Original Frame] + [Avg Across t] + [t1, t2, ...]
    n_cols = len(t_grid) + 2
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3))

    axes[0].imshow(target_frame)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    if use_minmax:
        avg_min, avg_max = z_up_avg_np.min(), z_up_avg_np.max()
        z_avg_proc = (z_up_avg_np - avg_min) / (avg_max - avg_min + eps)
        z_avg_masked = np.ma.masked_where(z_avg_proc < 0.2, z_avg_proc)
        cbar_label = "Relative Intensity [0, 1]"
        curr_vmin, curr_vmax = 0.2, 1.0
    else:
        z_avg_masked = np.ma.masked_where(z_up_avg_np < z_threshold, z_up_avg_np)
        cbar_label = "Std Deviations ($\sigma$)"
        curr_vmin, curr_vmax = z_threshold, z_vmax

    axes[1].imshow(target_frame)
    im_avg = axes[1].imshow(z_avg_masked, cmap="jet", alpha=0.6, vmin=curr_vmin, vmax=curr_vmax)
    axes[1].set_title("Avg Across t", fontweight="bold")
    axes[1].axis("off")

    # --- PANELS 3..N: Individual t noise levels ---
    for i, t_val in enumerate(t_grid):
        z_up_np = z_up_np_dict[t_val]

        if use_minmax:
            z_min, z_max = z_up_np.min(), z_up_np.max()
            z_proc = (z_up_np - z_min) / (z_max - z_min + eps)
            z_masked = np.ma.masked_where(z_proc < 0.2, z_proc)
            curr_vmin, curr_vmax = 0.2, 1.0
            title_suffix = "(Min-Max)"
        else:
            z_masked = np.ma.masked_where(z_up_np < z_threshold, z_up_np)
            curr_vmin, curr_vmax = z_threshold, z_vmax
            title_suffix = ""

        ax_idx = i + 2  # Offset by 2 (Original + Avg)
        axes[ax_idx].imshow(target_frame)
        im = axes[ax_idx].imshow(z_masked, cmap="jet", alpha=0.55, vmin=curr_vmin, vmax=curr_vmax)
        axes[ax_idx].set_title(f"t={t_val} {title_suffix}".strip())
        axes[ax_idx].axis("off")

    # Colorbar attached to the summary map
    cbar = fig.colorbar(im_avg, ax=axes, orientation="vertical", fraction=0.012, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    
    mode_str = "MinMax" if use_minmax else "ZScore"
    fig.suptitle(f"{clip_id} ({sample_label}) — Calibrated Heatmap [{mode_str}]", fontsize=13)
    
    save_path = f"{output_dir}/overlay_{sample_label.lower()}_{mode_str.lower()}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    t_grid = [0.2, 0.4, 0.6, 0.8]
    
    # Load model
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load calibration statistics produced by scorer.py
    calib_stats_path = f"{RESULTS_DIR}_pt/calib_stats.pt"  # Fixed folder name to match previous script output
    if not os.path.exists(calib_stats_path):
        # Fallback check
        calib_stats_path = "results/calib_stats.pt"
        if not os.path.exists(calib_stats_path):
            raise FileNotFoundError("Missing calib_stats.pt. Please run calibration script first.")

    calib_stats = torch.load(calib_stats_path, weights_only=True)

    test_clip_id = "D_pyFV4nKd4_003993"

    # Plot Non-OOD Sample
    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/non-ood",
        sample_label="Normal",
        t_grid=t_grid,
        calib_stats=calib_stats,
        device=device,
        use_minmax=True,
        z_vmax=3.0,  # Max intensity scaling for matplotlib
    )

    # Plot OOD Sample
    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/ood",
        sample_label="Anomaly",
        t_grid=t_grid,
        calib_stats=calib_stats,
        use_minmax=True,
        device=device,
        z_vmax=3.0,
    )