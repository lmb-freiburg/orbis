"""
score_dataset.py

Computes retrospective-surprise scores (detailed / semantic / combined heads)
for every non-ood and ood sample produced by dota.py, averages across the
noise-level grid (t_grid), and stores BOTH the raw and the calib-normalized
version -- per-head, single map + single pooled scalar per sample. Use
build_feature_matrix.py to decide (empirically) whether raw or normalized
features make better classifier inputs -- see the note below.

Assumptions (change these if they don't match your setup):
  - dota.py has already been run, so DoTA_prepared_new/<clip_id>/{non-ood,ood}/
    each contain NUM_FRAMES (6) already-subsampled frames named 000000..000005.jpg.
    Because dota.py already picked out every 2nd frame, this script does NOT
    subsample again (unlike load_clip_as_window in the calibration script,
    which expects 11 raw frames).
  - Calib stats are loaded from two files and merged: CALIB_STATS_PATHS below.
    Adjust the paths/filenames to match what's actually in your results/ dir.
  - frame_rate is fixed at 5.0 fps for every clip, matching the calibration
    run. Change FRAME_RATE_HZ below if your clips actually vary.

On normalization (raw vs. calib z-score) as a classifier feature:
  Averaging is done PER t_val first using that t_val's own calib mean/std,
  THEN averaged across t_grid -- normalizing after averaging across t would
  mix distributions with different scales and isn't meaningful, since each
  t_val has its own noise level and thus its own error distribution.
  Both raw_pooled and norm_pooled are stored so you can compare -- see the
  reasoning in the chat reply for why normalized is probably the better
  default, but this isn't free to assume, so both are kept.
"""

import json
import sys
from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util import instantiate_from_config
from dota import ANNOTATIONS_DIR, OUTPUT_DIR, NUM_FRAMES  # reuse the same constants/paths

# ----------------------------------------------------
# Config
# ----------------------------------------------------
EXP_DIR = "logs_wm/orbis_288x512"
CALIB_STATS_PATHS = [
    "results/calib_stats_detailed_semantic.pt",
    "results/calib_stats_combined.pt",
]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "sample_scores.pt"

HEADS = ["detailed", "semantic", "combined"]
T_GRID = [0.2, 0.4, 0.6, 0.8]
N_NOISE_SAMPLES = 2
FRAME_RATE_HZ = 5.0
IMG_SIZE = (512, 288)  # (W, H), matches training resolution


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_prepared_frame_paths(folder):
    paths = sorted(Path(folder).glob("*.jpg"))
    assert len(paths) == NUM_FRAMES, f"expected {NUM_FRAMES} frames in {folder}, found {len(paths)}"
    return paths


def load_prepared_clip(frame_paths, size=IMG_SIZE):
    """dota.py already selected the right every-2nd frames and renamed them
    000000..000005, so we just load them in order -- no re-subsampling here."""
    frames = []
    for p in frame_paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    print("shape ",t.shape)
    return torch.stack(frames)


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

        def run_inference_pass(ctx, tgt, err_start_idx, err_end_idx):
            tgt_t, noise = model.add_noise(tgt, t)
            pred = net(tgt_t, ctx, t, frame_rate=fr)
            true_v = model.A(t) * tgt + model.B(t) * noise
            err = (pred.float() - true_v.float()) ** 2
            err_avg = err.view(b, n_noise_samples, 1, n_channels, err.shape[-2], err.shape[-1]).mean(dim=1)
            active_err = err_avg[:, :, err_start_idx:err_end_idx]
            return active_err.mean(dim=2).squeeze(0).squeeze(0)

        if "detailed" in heads:
            ctx_d, tgt_d = context_exp.clone(), target_exp.clone()
            ctx_d[:, :, half:, :, :] = 0
            tgt_d[:, :, half:, :, :] = 0
            per_t_maps["detailed"][t_val] = run_inference_pass(ctx_d, tgt_d, 0, half)

        if "semantic" in heads:
            ctx_s, tgt_s = context_exp.clone(), target_exp.clone()
            ctx_s[:, :, :half, :, :] = 0
            tgt_s[:, :, :half, :, :] = 0
            per_t_maps["semantic"][t_val] = run_inference_pass(ctx_s, tgt_s, half, n_channels)

        if "combined" in heads:
            per_t_maps["combined"][t_val] = run_inference_pass(context_exp, target_exp, 0, n_channels)

    return per_t_maps


def load_calib_stats(paths):
    merged = {}
    for p in paths:
        stats = torch.load(p, map_location="cpu")
        merged.update(stats)
    return merged


def average_over_t(per_t_maps, calib_stats, heads, t_grid, eps=1e-8):
    """For each head, returns (raw_avg, norm_avg): [18,32] tensors averaged
    across t_grid. Normalization happens PER t_val (against that t_val's own
    calib mean/std) before averaging across t -- averaging raw errors from
    different noise levels first and normalizing after would be meaningless,
    since each t_val has its own error scale."""
    raw_avg, norm_avg = {}, {}
    for h in heads:
        raw_stack, norm_stack = [], []
        for t_val in t_grid:
            raw_map = per_t_maps[h][t_val]
            mean = calib_stats[h][t_val]["mean"].to(raw_map.device)
            std = calib_stats[h][t_val]["std"].to(raw_map.device)
            norm_map = (raw_map - mean) / (std + eps)
            raw_stack.append(raw_map)
            norm_stack.append(norm_map)
        raw_avg[h] = torch.stack(raw_stack).mean(dim=0).cpu()
        norm_avg[h] = torch.stack(norm_stack).mean(dim=0).cpu()
    return raw_avg, norm_avg


def build_valid_clips():
    json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
    valid_clips = []
    for jf in json_files:
        with open(jf) as f:
            meta = json.load(f)
        clip_id = meta["video_name"]

        if meta.get("ignore", False) or str(meta.get("ignore")).lower() == "true":
            continue

        non_ood_dir = Path(OUTPUT_DIR) / clip_id / "non-ood"
        ood_dir = Path(OUTPUT_DIR) / clip_id / "ood"
        if not (non_ood_dir.is_dir() and ood_dir.is_dir()):
            continue
        if len(list(non_ood_dir.glob("*.jpg"))) != NUM_FRAMES:
            continue
        if len(list(ood_dir.glob("*.jpg"))) != NUM_FRAMES:
            continue

        anomaly_start = meta.get("anomaly_start", -1)
        accident_name = "unknown"
        if "labels" in meta and 0 <= anomaly_start < len(meta["labels"]):
            accident_name = meta["labels"][anomaly_start].get("accident_name", "unknown")

        valid_clips.append({
            "clip_id": clip_id,
            "accident_name": accident_name,
            "night": meta.get("night", False),
        })
    return valid_clips


def main():
    device = get_device()
    print(f"Using device: {device}")

    cfg = OmegaConf.load(f"{EXP_DIR}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{EXP_DIR}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    calib_stats = load_calib_stats(CALIB_STATS_PATHS)
    missing = [h for h in HEADS if h not in calib_stats]
    if missing:
        raise ValueError(
            f"calib stats are missing heads {missing} across {CALIB_STATS_PATHS}. "
            f"Check the filenames/paths, or rerun calibration with HEADS = {HEADS}."
        )

    valid_clips = build_valid_clips()


    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples = MAX_SAMPLES,
        return_multiclass_labels=MULTI_CLASS,
        num_frames_per_clip=6
    )


    print(f"Found {len(valid_clips)} clips with prepared non-ood/ood frames.")

    frame_rate = torch.full((1,), FRAME_RATE_HZ, device=device)
    all_samples = []

    for clip in tqdm(valid_clips, desc="Scoring clips"):
        clip_id = clip["clip_id"]
        clip_dir = Path(OUTPUT_DIR) / clip_id

        for split, label in (("non-ood", 0), ("ood", 1)):
            folder = clip_dir / split
            paths = get_prepared_frame_paths(folder)
            window = load_prepared_clip(paths).unsqueeze(0).to(device)

            per_t_maps = surprise_score(
                model, window, frame_rate, T_GRID, heads=HEADS, n_noise_samples=N_NOISE_SAMPLES
            )
            raw_avg, norm_avg = average_over_t(per_t_maps, calib_stats, HEADS, T_GRID)

            raw_pooled = {h: raw_avg[h].max().item() for h in HEADS}
            norm_pooled = {h: norm_avg[h].max().item() for h in HEADS}

            all_samples.append({
                "clip_id": clip_id,
                "split": split,
                "label": label,
                "accident_name": clip["accident_name"] if label == 1 else "normal",
                "night": clip["night"],
                "raw_map": raw_avg,
                "norm_map": norm_avg,
                "raw_pooled": raw_pooled,
                "norm_pooled": norm_pooled,
            })

        if device.type == "mps":
            torch.mps.empty_cache()

    torch.save(all_samples, OUT_PATH)
    print(f"Saved {len(all_samples)} scored samples to {OUT_PATH}")


if __name__ == "__main__":
    main()