import os
import sys
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import instantiate_from_config
from scorer_batch_norm import (
    surprise_score,
    get_sorted_frame_paths,
    load_clip_as_window,
    get_device,
)

RESULTS_DIR = "results"


def raw_map_to_spatial(raw_map):
    """
    Takes raw channel error map [C, H, W] and computes channel average 
    to get an unnormalized 2D spatial score map [H, W].
    """
    while raw_map.dim() > 3:
        raw_map = raw_map.squeeze(0)
    return raw_map.float().mean(dim=0)


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
    calib_stats=None, 
    device=None, 
    heads=["detailed", "semantic"],
    n_noise_samples=2, 
    output_dir_override=None,
    eps=1e-8
):
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    # 1. Compute raw error maps for requested heads (prediction done first, then split)
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
        raw_maps_list = []
        raw_up_np_dict = {}

        for t_val in t_grid:
            raw_map = per_t_maps[head][t_val]
            # Unnormalized 2D spatial map [H, W]
            spatial_map = raw_map_to_spatial(raw_map)
            raw_maps_list.append(spatial_map)
            
            # Upsample 2D map [H, W] to full frame resolution [288, 512]
            map_4d = spatial_map.unsqueeze(0).unsqueeze(0).float()
            map_up = F.interpolate(map_4d, size=(288, 512), mode="bilinear", align_corners=False)
            raw_up_np_dict[t_val] = map_up.squeeze().cpu().numpy()

        # Average unnormalized maps across timesteps (t_grid)
        map_avg = torch.stack(raw_maps_list).mean(dim=0)
        map_avg_4d = map_avg.unsqueeze(0).unsqueeze(0).float()
        map_avg_up = F.interpolate(map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
        map_avg_up_np = map_avg_up.squeeze().cpu().numpy()

        # Col 0: Raw Frame
        axes[row_idx, 0].imshow(target_frame)
        axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame")
        axes[row_idx, 0].axis("off")

        # Col 1: Average Unnormalized Map Across t (Min-Max scaled visually)
        avg_min, avg_max = map_avg_up_np.min(), map_avg_up_np.max()
        proc_avg = (map_avg_up_np - avg_min) / (avg_max - avg_min + eps)
        masked_avg = np.ma.masked_where(proc_avg < 0.2, proc_avg)

        axes[row_idx, 1].imshow(target_frame)
        axes[row_idx, 1].imshow(masked_avg, cmap="jet", alpha=0.6, vmin=0.2, vmax=1.0)
        axes[row_idx, 1].set_title("Avg Across t (Unnorm)", fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Cols 2..N: Timesteps
        for i, t_val in enumerate(t_grid):
            raw_up_np = raw_up_np_dict[t_val]
            min_val, max_val = raw_up_np.min(), raw_up_np.max()
            proc_val = (raw_up_np - min_val) / (max_val - min_val + eps)
            masked_val = np.ma.masked_where(proc_val < 0.2, proc_val)

            col_idx = i + 2
            axes[row_idx, col_idx].imshow(target_frame)
            axes[row_idx, col_idx].imshow(masked_val, cmap="jet", alpha=0.55, vmin=0.2, vmax=1.0)
            axes[row_idx, col_idx].set_title(f"t={t_val}")
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(f"{clip_id} ({sample_label}) — Detailed & Semantic [Unnormalized]", fontsize=14)
    
    save_path = f"{output_dir}/overlay_{sample_label.lower()}_combined_unnormalized.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    t_grid = [0.2, 0.4, 0.6, 0.8]
    HEADS = ["detailed", "semantic"]
    
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    test_clip_id = "D_pyFV4nKd4_003993"

    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/non-ood",
        sample_label="Normal",
        t_grid=t_grid,
        device=device,
        heads=HEADS,
    )

    plot_and_overlay(
        model=model,
        clip_id=test_clip_id,
        folder_path=f"DoTA_pedestrian/{test_clip_id}/ood",
        sample_label="Anomaly",
        t_grid=t_grid,
        device=device,
        heads=HEADS,
    )