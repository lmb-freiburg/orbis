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


def normalize_map_with_calib(raw_map, t_val, calib_stats, head, eps=1e-8):
    """
    raw_map: [H, W] PyTorch tensor.
    calib_stats: loaded nested dict from calib_stats_combined.pt
    head: string ("detailed" or "semantic")
    """
    # Extract head-specific calibration stats
    mean = calib_stats[head][t_val]["mean"].to(raw_map.device)
    std = calib_stats[head][t_val]["std"].to(raw_map.device)
    
    # Standardize (Z-Score)
    z_map = (raw_map - mean) / (std + eps)
    
    # ReLU clamp to ignore below-average error (keep positive anomalies)
    return torch.clamp(z_map, min=0.0)


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG frames found in folder: '{folder}'")
    idx = min(frame_index, len(paths) - 1)
    img = cv2.cvtColor(cv2.imread(paths[idx]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def get_sorted_frame_paths_safe(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images found in folder: '{folder}'")
    return paths


def load_clip_as_window_safe(frame_paths, size=(512, 288)):
    n = len(frame_paths)
    if n >= 11:
        idxs = [0, 2, 4, 6, 8, 10]
    elif n == 6:
        idxs = [0, 1, 2, 3, 4, 5]
    else:
        idxs = np.linspace(0, n - 1, 6, dtype=int)

    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)


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
    paths = get_sorted_frame_paths_safe(folder_path)
    window = load_clip_as_window_safe(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    # 1. Compute raw error maps for requested heads
    per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=heads, n_noise_samples=n_noise_samples)
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
    output_dir = output_dir_override if output_dir_override else f"results/batch_norm/{clip_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup 2-row figure (Row 0 = Detailed, Row 1 = Semantic)
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
            z_map = normalize_map_with_calib(raw_map, t_val, calib_stats, head=head, eps=eps)
            z_maps_list.append(z_map)
            
            z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
            z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
            z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

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

def find_clip_folders(clip_id):
    """
    Finds normal (good/non-ood) and anomaly (anomalous/ood) folder paths for clip_id.
    First checks DoTA_training/DoTA_training.pt, then standard folder structures.
    """
    good_path = None
    anom_path = None

    dota_pt_path = "./DOTA_training/DoTA_training.pt"
    if os.path.exists(dota_pt_path):
        try:
            dota_data = torch.load(dota_pt_path, map_location='cpu')
            vids = dota_data.get('video_ids', [])
            paths = dota_data.get('clip_paths', [])
            labels = dota_data.get('labels', [])

            for i, vid in enumerate(vids):
                if vid == clip_id:
                    lbl = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                    frame_paths = paths[i]
                    if len(frame_paths) > 0:
                        folder = os.path.dirname(frame_paths[0])
                        if lbl == 0 and not good_path:
                            good_path = folder
                        elif lbl == 1 and not anom_path:
                            anom_path = folder
        except Exception as e:
            print(f"Warning reading DoTA_training.pt: {e}")

    # Fallbacks if not found in pt or pt file not present
    if not good_path:
        for candidate in [
            f"DOTA_training/data/train/{clip_id}_good",
            f"DOTA_training/data/val/{clip_id}_good",
            f"DoTA_pedestrian/{clip_id}/non-ood"
        ]:
            if os.path.exists(candidate):
                good_path = candidate
                break

    if not anom_path:
        for candidate in [
            f"DOTA_training/data/train/{clip_id}_anomalous",
            f"DOTA_training/data/val/{clip_id}_anomalous",
            f"DoTA_pedestrian/{clip_id}/ood"
        ]:
            if os.path.exists(candidate):
                anom_path = candidate
                break

    return good_path, anom_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run surprise score batch norm plotter for sequence.")
    parser.add_argument("--clip_id", type=str, default="h55PiQMnlJY_003552", help="Clip ID to plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    calib_stats_candidates = [
        "./results_pt/calib_stats_detailed_semantic.pt",
        "./cached_features/calib_stats_detailed_semantic.pt",
        "./cached_features/calib_stats_combined.pt",
        f"{RESULTS_DIR}/calib_stats.pt"
    ]
    calib_stats_path = None
    for cand in calib_stats_candidates:
        if os.path.exists(cand):
            calib_stats_path = cand
            break

    if not calib_stats_path:
        raise FileNotFoundError("Missing calibration stats file. Please run calibration script first.")

    print(f"Loaded calibration stats from '{calib_stats_path}'")
    calib_stats = torch.load(calib_stats_path, weights_only=True)

    test_clip_id = args.clip_id
    good_folder, anom_folder = find_clip_folders(test_clip_id)

    if good_folder and os.path.exists(good_folder):
        print(f"Plotting Normal sample for clip '{test_clip_id}' from '{good_folder}'...")
        plot_and_overlay(
            model=model,
            clip_id=test_clip_id,
            folder_path=good_folder,
            sample_label="Normal",
            t_grid=t_grid,
            calib_stats=calib_stats,
            device=device,
            heads=HEADS,
            use_minmax=False,
            z_vmax=3.0,
        )
    else:
        print(f"Warning: Normal folder for '{test_clip_id}' not found.")

    if anom_folder and os.path.exists(anom_folder):
        print(f"Plotting Anomaly sample for clip '{test_clip_id}' from '{anom_folder}'...")
        plot_and_overlay(
            model=model,
            clip_id=test_clip_id,
            folder_path=anom_folder,
            sample_label="Anomaly",
            t_grid=t_grid,
            calib_stats=calib_stats,
            device=device,
            heads=HEADS,
            use_minmax=False,
            z_vmax=3.0,
        )
    else:
        print(f"Warning: Anomaly folder for '{test_clip_id}' not found.")