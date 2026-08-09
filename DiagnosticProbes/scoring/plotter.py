"""
Plots for visualizing surprise scores across different heads and time steps.
parameters:
- minmax: if True, the scores are normalized to [0, 1] using min-max normalization. If False, the scores are normalized using z-score normalization.
- z_threshold: the threshold for z-score normalization. Scores below this threshold are masked out in the overlay plots.
- z_vmax: the maximum value for z-score normalization. Scores above this value are clipped in the overlay plots.
- use_normalized: if True, the scores are normalized using z-score normalization. If False, the scores are not normalized.
- plot_name_suffix: a string to append to the output plot file names for differentiation.
"""

import os
import sys
import glob
import warnings
from pathlib import Path

# Suppress torchvision C-extension warning on macOS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Suppress torchvision & bicubic MPS fallback warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", message=".*_upsample_bicubic2d_aa.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import instantiate_from_config

RESULTS_DIR = "results"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_sorted_frame_paths(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    assert len(paths) == 11, f"expected 11 frames in {folder}, found {len(paths)}"
    return paths


def load_clip_as_window(frame_paths, size=(512, 288)):
    idxs = [0, 2, 4, 6, 8, 10]
    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)

def normalize_map_with_calib(raw_map, t_val, calib_stats, head, eps=1e-8):
    mean = calib_stats["combined"][t_val]["mean"].to(raw_map.device)
    std = calib_stats["combined"][t_val]["std"].to(raw_map.device)

    half = mean.shape[0] // 2 

    # detailed
    if head == "detailed":
        mean= mean[:half]
        std = std[:half] 

    # semantic
    if head == "semantic":
        mean = mean[half:]
        std = std[half:]

    # combined
    if head == "combined":
        mean = mean
        std = std

    z_map = torch.clamp((raw_map - mean) / (std + eps), min=0.0)
    return z_map


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

@torch.no_grad()
def surprise_score(model, images, frame_rate, t_grid, heads, n_noise_samples=2, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    
    x = model.encode_frames(images)
    context, target = x[:, :-1].clone(), x[:, -1:]
    b = x.shape[0]
    n_channels = target.shape[2]
    half = n_channels // 2

    context_exp = context.repeat_interleave(n_noise_samples, dim=0)
    target_exp = target.repeat_interleave(n_noise_samples, dim=0)

    per_t_maps = {h: {} for h in heads}
    is_mps = x.device.type == "mps"

    for t_val in t_grid:
        t = torch.full((b * n_noise_samples,), t_val, device=x.device)
        fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate

        def run_inference_pass(ctx, tgt):
            tgt_t, noise = model.add_noise(tgt, t)
            
            if is_mps:
                pred = net(tgt_t, ctx, t, frame_rate=fr)
            else:
                pred = net(tgt_t, ctx, t, frame_rate=fr)
            
            true_v = model.A(t) * tgt + model.B(t) * noise
            err = (pred.float() - true_v.float()) ** 2
            
            err_avg = err.view(b, n_noise_samples, 1, n_channels, err.shape[-2], err.shape[-1]).mean(dim=1)
        
            return err_avg.squeeze(0).squeeze(0)

        combined_scores = run_inference_pass(
            context_exp, target_exp
        )
        
        # detailed
        if "detailed" in heads:
            per_t_maps["detailed"][t_val] =  combined_scores[:half, :, :]  

        # semantic
        if "semantic" in heads:
            per_t_maps["semantic"][t_val] =  combined_scores[half:, :, :]

        # ---------------------------------------------------------
        # PASS 3: COMBINED
        # ---------------------------------------------------------
        if "combined" in heads:
            per_t_maps["combined"][t_val] = combined_scores

    return per_t_maps


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
    output_dir_override=None,
    plot_name_suffix="", 
    eps=1e-8
):
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=heads, n_noise_samples=n_noise_samples)
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))
    
    output_dir = output_dir_override if output_dir_override else f"results/heatmaps_poster/{clip_id}/{plot_name_suffix}"
    os.makedirs(output_dir, exist_ok=True)

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
            if use_normalized:
                z_map = normalize_map_with_calib(raw_map, t_val, calib_stats, head=head, eps=eps)
                z_map = z_map.mean(dim=0)  
            else:
                z_map = raw_map.mean(dim=0) 
            z_maps_list.append(z_map)

            
            z_map_4d = z_map.unsqueeze(0).unsqueeze(0).float()
            z_up = F.interpolate(z_map_4d, size=(288, 512), mode="bilinear", align_corners=False)
            z_up_np_dict[t_val] = z_up.squeeze().cpu().numpy()

        z_map_avg = torch.stack(z_maps_list).mean(dim=0)
        z_map_avg_4d = z_map_avg.unsqueeze(0).unsqueeze(0).float()
        z_up_avg = F.interpolate(z_map_avg_4d, size=(288, 512), mode="bilinear", align_corners=False)
        z_up_avg_np = z_up_avg.squeeze().cpu().numpy()

        axes[row_idx, 0].imshow(target_frame)
        axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame")
        axes[row_idx, 0].axis("off")

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
        axes[row_idx, 1].set_title("Avg Across t", fontweight="bold")
        axes[row_idx, 1].axis("off")

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
            else:
                axes[row_idx, col_idx].imshow(z_up_np, cmap="jet", alpha=0.5)
            axes[row_idx, col_idx].set_title(f"t={t_val}")
            axes[row_idx, col_idx].axis("off")

    if use_normalized:
        mode_str = "MinMax" if use_minmax else "ZScore"
    elif use_minmax:
        mode_str = "Unnormalized MinMax"
    else:
        mode_str = "Unnormalized"
    fig.suptitle(f"{clip_id} ({sample_label}) — Detailed, Semantic and Combined [{mode_str}]", fontsize=14)
    
    save_path = f"{output_dir}/overlay_{sample_label.lower()}_all_{mode_str.lower()}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    t_grid = [0.2, 0.4, 0.6, 0.8]
    HEADS = ["detailed", "semantic", "combined"]
    
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    calib_stats_path = "results/calib_stats_combined3000.pt"
    if not os.path.exists(calib_stats_path):
        raise FileNotFoundError("Missing calibration stats file. Please run the calibration script first.")

    calib_stats = torch.load(calib_stats_path, weights_only=True)

    # provide the clip ids to plot
    test_clip_ids = ["1u69z-wsDIc_004195", "5vKPYV5w6pw_005653"]

    for test_clip_id in test_clip_ids:
        print(f"\nProcessing clip: {test_clip_id}")
        plot_and_overlay(
            model=model,
            clip_id=test_clip_id,
            folder_path=f"DoTA_class/{test_clip_id}/ood",
            sample_label="Anomaly",
            t_grid=t_grid,
            calib_stats=calib_stats,
            device=device,
            heads=HEADS,
            use_minmax=False,
            z_threshold = 1.0,
            use_normalized=True,
            plot_name_suffix="1to3std",
            z_vmax=3.0,
        )
