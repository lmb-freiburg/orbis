"""
Unified scorer/plotter utilities.

Key design change vs scorer_batch_norm.py / scorer_batch_norm_combined.py:
  - There is now only ONE inference pass per t_val, run on the FULL,
    unmasked context/target (i.e. what the old code called "combined").
  - "detailed" and "semantic" are never produced by zeroing input channels
    before the network call. They are produced by slicing the resulting
    per-channel squared-error tensor AFTER it has been computed.
  - This makes calib_stats_combined.pt (built from the unmasked pass)
    a valid source for normalizing detailed/semantic too: just slice its
    per-channel mean/std the same way you slice the error map.
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

HEADS = ["detailed", "semantic", "combined"]


# ---------------------------------------------------------------------------
# Device / IO helpers
# ---------------------------------------------------------------------------
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


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def find_clip_folder(project_root, clip_id, split):
    """
    Locate <clip_id>/<split> under whichever dataset root actually has it.
    Tries the flat layouts first (DoTA_prepared, DoTA_pedestrian), then
    falls back to the class-nested layout (DoTA_class/<class_name>/<clip_id>/<split>).
    """
    project_root = Path(project_root)
    flat_candidates = [
        project_root / "DoTA_prepared" / clip_id / split,
        project_root / "DoTA_pedestrian" / clip_id / split,
    ]
    for c in flat_candidates:
        if c.exists():
            return c

    matches = glob.glob(str(project_root / "DoTA_class" / "*" / clip_id / split))
    if matches:
        return Path(matches[0])

    raise FileNotFoundError(
        f"Could not locate folder for clip_id={clip_id!r} split={split!r} "
        f"under {project_root}"
    )


# ---------------------------------------------------------------------------
# Core scorer: ONE pass per t, full channels, no masking
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_full_channel_error(model, images, frame_rate, t_grid, n_noise_samples=2, use_ema=False):
    """
    Runs a single inference pass per t_val on the full, unmasked
    context/target. Returns {t_val: err_tensor [C, H, W]} plus n_channels.

    No channel masking happens here — head separation is a pure
    post-processing step (see split_into_heads / channel_slice_for_head).
    """
    net = model.ema_vit if use_ema else model.vit
    net.eval()

    x = model.encode_frames(images)
    context, target = x[:, :-1].clone(), x[:, -1:]
    b = x.shape[0]
    n_channels = target.shape[2]

    context_exp = context.repeat_interleave(n_noise_samples, dim=0)
    target_exp = target.repeat_interleave(n_noise_samples, dim=0)

    full_err_per_t = {}
    for t_val in t_grid:
        t = torch.full((b * n_noise_samples,), t_val, device=x.device)
        fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate

        tgt_t, noise = model.add_noise(target_exp, t)
        pred = net(tgt_t, context_exp, t, frame_rate=fr)
        true_v = model.A(t) * target_exp + model.B(t) * noise
        err = (pred.float() - true_v.float()) ** 2  # [B*N, 1, C, H, W]

        err_avg = err.view(
            b, n_noise_samples, 1, n_channels, err.shape[-2], err.shape[-1]
        ).mean(dim=1)
        full_err_per_t[t_val] = err_avg.squeeze(0).squeeze(0)  # [C, H, W]

    return full_err_per_t, n_channels


def channel_slice_for_head(head, half):
    """The single source of truth for which channels belong to which head."""
    if head == "detailed":
        return slice(0, half)
    if head == "semantic":
        return slice(half, None)
    if head == "combined":
        return slice(None)
    raise ValueError(f"Unknown head: {head}")


def split_into_heads(full_err_per_t, n_channels, heads=HEADS):
    """
    Post-hoc channel slicing of the already-computed error tensor.
    This is the ONLY place detailed/semantic/combined diverge.
    """
    half = n_channels // 2
    per_t_maps = {h: {} for h in heads}
    for t_val, err in full_err_per_t.items():
        for h in heads:
            per_t_maps[h][t_val] = err[channel_slice_for_head(h, half)]
    return per_t_maps, half


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def raw_map_to_spatial(raw_map):
    """[C, H, W] -> [H, W] via channel mean."""
    return raw_map.float().mean(dim=0)


def zscore_normalize(raw_map, calib_mean, calib_std, eps=1e-8):
    """
    (map - calib_mean) / (calib_std + eps), channel-wise.
    calib_mean/calib_std must already be sliced to the same channel
    range as raw_map (see channel_slice_for_head).
    """
    return (raw_map - calib_mean) / (calib_std + eps)


def to_display(spatial_map_up_np, eps=1e-8, threshold=0.2):
    """Per-frame min-max stretch, used ONLY for the jet-colormap overlay
    (this is a visualization step, not a normalization method — it is the
    same for both the unnormalized and the calib-normalized variants)."""
    vmin, vmax = spatial_map_up_np.min(), spatial_map_up_np.max()
    proc = (spatial_map_up_np - vmin) / (vmax - vmin + eps)
    return np.ma.masked_where(proc < threshold, proc)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_figure(target_frame, per_t_maps, heads, t_grid, title, save_path,
                  upsample_size=(288, 512), eps=1e-8):
    n_rows = len(heads)
    n_cols = len(t_grid) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, head in enumerate(heads):
        raw_maps_list = []
        up_np_dict = {}
        for t_val in t_grid:
            spatial = raw_map_to_spatial(per_t_maps[head][t_val])
            raw_maps_list.append(spatial)
            up = F.interpolate(
                spatial.unsqueeze(0).unsqueeze(0).float(),
                size=upsample_size, mode="bilinear", align_corners=False,
            )
            up_np_dict[t_val] = up.squeeze().cpu().numpy()

        map_avg = torch.stack(raw_maps_list).mean(dim=0)
        avg_up = F.interpolate(
            map_avg.unsqueeze(0).unsqueeze(0).float(),
            size=upsample_size, mode="bilinear", align_corners=False,
        )
        avg_up_np = avg_up.squeeze().cpu().numpy()

        axes[row_idx, 0].imshow(target_frame)
        axes[row_idx, 0].set_title(f"{head.capitalize()} — Frame")
        axes[row_idx, 0].axis("off")

        masked_avg = to_display(avg_up_np, eps)
        axes[row_idx, 1].imshow(target_frame)
        axes[row_idx, 1].imshow(masked_avg, cmap="jet", alpha=0.6, vmin=0.2, vmax=1.0)
        axes[row_idx, 1].set_title("Avg Across t", fontweight="bold")
        axes[row_idx, 1].axis("off")

        for i, t_val in enumerate(t_grid):
            masked_val = to_display(up_np_dict[t_val], eps)
            col = i + 2
            axes[row_idx, col].imshow(target_frame)
            axes[row_idx, col].imshow(masked_val, cmap="jet", alpha=0.55, vmin=0.2, vmax=1.0)
            axes[row_idx, col].set_title(f"t={t_val}")
            axes[row_idx, col].axis("off")

    fig.suptitle(title, fontsize=14)
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level: generate BOTH variants (unnormalized + calib-normalized) for one clip/split
# ---------------------------------------------------------------------------
def generate_all_variants(model, clip_id, folder_path, sample_label, t_grid,
                           calib_stats_combined_path, device, output_dir,
                           heads=HEADS, n_noise_samples=2, eps=1e-8):
    paths = get_sorted_frame_paths(folder_path)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)
    target_frame = load_target_frame(folder_path, frame_index=10, size=(512, 288))

    # ---- ONE pass, full channels ----
    full_err_per_t, n_channels = compute_full_channel_error(
        model, window, frame_rate, t_grid, n_noise_samples=n_noise_samples
    )
    per_t_maps, half = split_into_heads(full_err_per_t, n_channels, heads)

    # ---- Variant 1: UN-NORMALIZED (raw squared error; no calib stats used at all) ----
    build_figure(
        target_frame, per_t_maps, heads, t_grid,
        title=f"{clip_id} ({sample_label}) — Unnormalized",
        save_path=f"{output_dir}/overlay_{sample_label.lower()}_unnormalized.png",
        eps=eps,
    )

    # ---- Variant 2: NORMALIZED (z-score vs combined calib stats, sliced per head) ----
    calib_stats = torch.load(calib_stats_combined_path, map_location="cpu")["combined"]
    normalized_maps = {h: {} for h in heads}
    for t_val in t_grid:
        calib_mean_full = calib_stats[t_val]["mean"].to(device)  # [C, H, W], all channels
        calib_std_full = calib_stats[t_val]["std"].to(device)
        for head in heads:
            ch_slice = channel_slice_for_head(head, half)
            calib_mean = calib_mean_full[ch_slice]
            calib_std = calib_std_full[ch_slice]
            raw = per_t_maps[head][t_val]  # already sliced to this head's channels
            normalized_maps[head][t_val] = zscore_normalize(raw, calib_mean, calib_std, eps)

    n_calib = calib_stats[t_grid[0]]["n"]
    build_figure(
        target_frame, normalized_maps, heads, t_grid,
        title=f"{clip_id} ({sample_label}) — Normalized (calib n={n_calib})",
        save_path=f"{output_dir}/overlay_{sample_label.lower()}_normalized.png",
        eps=eps,
    )