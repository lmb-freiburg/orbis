import os
import sys
import glob
import warnings
from pathlib import Path

# 1. Suppress torchvision C-extension & fallback warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", message=".*_upsample_bicubic2d_aa.*")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 2. Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. Direct imports
from util import instantiate_from_config
from scoring.scorer_batch_norm import (
    surprise_score,
    get_sorted_frame_paths,
    load_clip_as_window,
    get_device,
)


def normalize_map_with_calib(raw_map, t_val, calib_stats, head, eps=1e-8):
    mean = calib_stats["combined"][t_val]["mean"].to(raw_map.device)
    std = calib_stats["combined"][t_val]["std"].to(raw_map.device)
    half = mean.shape[0] // 2  # 16 channels

    if head == "detailed":
        mean = mean[:half]
        std = std[:half]
    elif head == "semantic":
        mean = mean[half:]
        std = std[half:]

    z_map = torch.clamp((raw_map - mean) / (std + eps), min=0.0)
    return z_map


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def save_single_plot(target_frame, masked_or_raw_map, title, save_path, curr_vmin=None, curr_vmax=None, is_raw_frame=False):
    """Helper to render and save standalone individual plots."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.12, 2.88), dpi=150)
    
    ax.imshow(target_frame)
    if not is_raw_frame and masked_or_raw_map is not None:
        ax.imshow(masked_or_raw_map, cmap="jet", alpha=0.5, vmin=curr_vmin, vmax=curr_vmax)
    
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_and_overlay(
    model, 
    clip_id, 
    folder_path, 
    sample_label, 
    t_grid, 
    calib_stats, 
    device, 
    heads=["detailed", "semantic", "combined"],
    n_noise_samples=2, 
    z_vmax=3.0,
    z_threshold=1.0,
    use_minmax=False,
    use_normalized=True,
    eps=1e-8
):
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    # 1. Compute raw error maps for requested heads
    per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=heads, n_noise_samples=n_noise_samples)
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
    base_poster_dir = f"results/poster/{clip_id}"

    # Setup multi-row layout for 'all' variant
    n_rows = len(heads)
    n_cols = len(t_grid) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.0 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, head in enumerate(heads):
        z_maps_list = []
        z_up_np_dict = {}

        for t_val in t_grid:
            raw_map = per_t_maps[head][t_val]
            if use_normalized:
                z_map = normalize_map_with_calib(raw_map, t_val, calib_stats, head=head, eps=eps)
                z_map = z_map.mean(dim=0)
            else:
                z_map = raw_map.mean(dim=0)
            z_maps_list.append(z_map)

            z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
            z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
            z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

        # Compute average across timesteps t
        z_map_avg = torch.stack(z_maps_list).mean(dim=0)
        z_map_avg_4d = z_map_avg.unsqueeze(0).unsqueeze(0).float()
        z_up_avg = F.interpolate(z_map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
        z_up_avg_np = z_up_avg.squeeze().cpu().numpy()

        # --- Combined Plot: Col 0 (Raw Frame) ---
        axes[row_idx, 0].imshow(target_frame)
        axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame", fontsize=9)
        axes[row_idx, 0].axis("off")

        # Save individual raw frame plot
        frame_save_path = f"{base_poster_dir}/frame/frame.png"
        if not os.path.exists(frame_save_path):
            save_single_plot(target_frame, None, "Target Frame", frame_save_path, is_raw_frame=True)

        # --- Combined Plot: Col 1 (Avg Across t) ---
        if use_minmax:
            avg_min, avg_max = z_up_avg_np.min(), z_up_avg_np.max()
            z_avg_proc = (z_up_avg_np - avg_min) / (avg_max - avg_min + eps)
            z_avg_masked = np.ma.masked_where(z_avg_proc < 0.2, z_avg_proc)
            curr_vmin, curr_vmax = 0.2, 1.0
        else:
            z_avg_masked = np.ma.masked_where(z_up_avg_np < z_threshold, z_up_avg_np)
            curr_vmin, curr_vmax = z_threshold, z_vmax

        axes[row_idx, 1].imshow(target_frame)
        axes[row_idx, 1].imshow(z_avg_masked, cmap="jet", alpha=0.5, vmin=curr_vmin, vmax=curr_vmax)
        axes[row_idx, 1].set_title("Avg Across t", fontweight="bold", fontsize=9)
        axes[row_idx, 1].axis("off")

        # Save individual plot for Avg
        save_single_plot(
            target_frame, 
            z_avg_masked, 
            f"{head.capitalize()} — Avg Across t", 
            f"{base_poster_dir}/combined_avg/{head}_avg.png", 
            curr_vmin=curr_vmin, 
            curr_vmax=curr_vmax
        )

        # --- Combined Plot: Cols 2..N (Individual Timesteps t) ---
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
            if use_normalized or use_minmax:
                axes[row_idx, col_idx].imshow(z_masked, cmap="jet", alpha=0.5, vmin=curr_vmin, vmax=curr_vmax)
                plot_map = z_masked
            else:
                axes[row_idx, col_idx].imshow(z_up_np, cmap="jet", alpha=0.5)
                plot_map = z_up_np

            axes[row_idx, col_idx].set_title(f"t={t_val}", fontsize=9)
            axes[row_idx, col_idx].axis("off")

            # Save individual scenario plot for timestep t
            save_single_plot(
                target_frame, 
                plot_map, 
                f"{head.capitalize()} — t={t_val}", 
                f"{base_poster_dir}/combined_{t_val}/{head}_{t_val}.png", 
                curr_vmin=curr_vmin, 
                curr_vmax=curr_vmax
            )

    # Tighten up spacing between subplots for the 'all' combined plot
    plt.subplots_adjust(wspace=0.02, hspace=0.08)

    if use_normalized:
        mode_str = "MinMax" if use_minmax else "ZScore"
    elif use_minmax:
        mode_str = "Unnormalized MinMax"
    else:
        mode_str = "Unnormalized"

    fig.suptitle(f"{clip_id} ({sample_label}) — Detailed, Semantic & Combined [{mode_str}]", fontsize=12, y=0.98)
    
    # Save the all-in-one plot
    all_output_dir = f"{base_poster_dir}/all"
    os.makedirs(all_output_dir, exist_ok=True)
    fig.savefig(f"{all_output_dir}/overlay_{sample_label.lower()}_all.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    t_grid = [0.2, 0.4, 0.6, 0.8]
    HEADS = ["detailed", "semantic", "combined"]
    
    # Load model
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load multi-head calibration statistics
    calib_stats_path = "results/calib_stats_combined3000.pt"
    if not os.path.exists(calib_stats_path):
        raise FileNotFoundError("Missing calibration stats file. Please run the calibration script first.")

    calib_stats = torch.load(calib_stats_path, weights_only=True)

    test_clip_ids = ["d2SCftR5sWc_002095"]

    for test_clip_id in test_clip_ids:
        print(f"Processing clip: {test_clip_id}")
        plot_and_overlay(
            model=model,
            clip_id=test_clip_id,
            folder_path=f"DoTA_class/pedestrian/{test_clip_id}/ood",
            sample_label="Anomaly",
            t_grid=t_grid,
            calib_stats=calib_stats,
            device=device,
            heads=HEADS,
            use_minmax=False,
            z_threshold=1.0,
            use_normalized=True,
            z_vmax=3.0,
        )