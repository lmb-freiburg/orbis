"""
score_dataset.py

Computes retrospective-surprise scores (detailed / semantic / combined heads)
for every clip produced by the new DoTAClipDataset (dota.py), averages across
the noise-level grid (t_grid), and stores BOTH the raw and the calib-normalized
version -- per-head, single map + single pooled scalar per sample. Use
build_feature_matrix.py to decide (empirically) whether raw or normalized
features make better classifier inputs -- see the note below.

Changes vs. the old version:
  - dota.py no longer materializes non-ood/ood jpg folders via a separate
    prep step. It now exposes get_dota_dataloaders(..., normalize_for_world_model=True),
    which builds (label 0/1) clips directly from DoTA_Sequences +
    DOTA_annotations, applies the [-1, 1] pixel scaling the world model
    expects, and returns the exact same 80/20 train/val split (seed=42) used
    by cache_features.py for the linear-probe pipeline -- so we call it
    directly instead of re-deriving the split ourselves. Each scored sample
    is tagged with "split": "train" or "val".
  - NOTE: get_dota_dataloaders can wipe and re-populate ../DOTA_training
    (export_split) on every call. This script now passes skip_export=True by
    default (--rebuild_export to override) since scoring reads frames
    straight from --seq_dir regardless -- the export is a side artifact for
    other tooling, not something the loaders themselves need.
  - --max_samples defaults to 900, applied by DoTAClipDataset BEFORE the
    80/20 split (so you get ~720 train / ~180 val out of the 900).
  - Scores are checkpointed periodically (every --checkpoint_interval
    clips), same pattern as cache_features.py, instead of only being saved
    once at the very end -- so a crash partway through a long scoring run
    doesn't lose everything.

Assumptions (change these if they don't match your setup):
  - DoTA_Sequences/<video>/images and DOTA_annotations/<video>.json exist and
    are laid out the way DoTAClipDataset expects (see dota.py).
  - Calib stats are loaded from two files and merged: --calib_stats below.
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

import argparse
import json
import sys
from pathlib import Path

PYTORCH_ENABLE_MPS_FALLBACK=1

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util import instantiate_from_config
from dota import get_dota_dataloaders, DOTA_CLASS_NAMES

# ----------------------------------------------------
# Config
# ----------------------------------------------------
HEADS = ["detailed", "semantic", "combined"]
T_GRID = [0.2, 0.4, 0.6, 0.8]
N_NOISE_SAMPLES = 2
FRAME_RATE_HZ = 5.0
NUM_FRAMES = 6

_NIGHT_CACHE = {}



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
            return active_err.squeeze(0).squeeze(0) # change

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


def load_night_flag(anno_dir, video_name):
    """DoTAClipDataset samples no longer carry the 'night' flag (it's only
    used internally to optionally filter). Look it up from the annotation
    JSON on demand, cached per video so we don't re-read the same file for
    both the normal and anomalous clip of the same video."""
    if video_name in _NIGHT_CACHE:
        return _NIGHT_CACHE[video_name]
    json_path = Path(anno_dir) / f"{video_name}.json"
    night = False
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        night = bool(meta.get("night", False))
    _NIGHT_CACHE[video_name] = night
    return night


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score DoTA clips for retrospective surprise (raw + calib-normalized)."
    )
    parser.add_argument("--exp_dir", type=str, default="logs_wm/orbis_288x512")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--seq_dir", type=str, default="/Volumes/maccbeast/frames/")
    parser.add_argument("--anno_dir", type=str, default="annotations/")
    parser.add_argument(
        "--calib_stats",
        type=str,
        nargs="+",
        default=[
            "results/calib_stats_detailed_semantic.pt",
            "results/calib_stats_combined.pt",
        ],
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--out_name", type=str, default="sample_scores.pt")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save a partial checkpoint every N scored clips.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=900,
        help="Cap the dataset to this many clips before the 80/20 train/val split (None = use all).",
    )
    parser.add_argument(
        "--rebuild_export",
        action="store_true",
        help=(
            "Also have get_dota_dataloaders (re)build the on-disk ../DOTA_training export. "
            "Off by default: scoring reads frames straight from --seq_dir regardless, and if "
            "you've already generated that export for the full dataset, skipping it here avoids "
            "wiping/recopying it for what would otherwise be this run's (smaller) --max_samples subset."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    out_path = save_dir / args.out_name
    partial_path = save_dir / f"{Path(args.out_name).stem}_partial.pt"

    cfg = OmegaConf.load(f"{args.exp_dir}/{args.config}")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{args.exp_dir}/{args.ckpt}", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    calib_stats = load_calib_stats(args.calib_stats)
    missing = [h for h in HEADS if h not in calib_stats]
    if missing:
        raise ValueError(
            f"calib stats are missing heads {missing} across {args.calib_stats}. "
            f"Check the filenames/paths, or rerun calibration with HEADS = {HEADS}."
        )

    # get_dota_dataloaders now handles everything we were doing by hand:
    # builds the dataset, applies the [-1, 1] world-model normalization
    # (normalize_for_world_model=True), and returns the same 80/20 train/val
    # split (seed=42) cache_features.py uses -- so the scores here line up
    # with the cached ViT features' train/val membership for free.
    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=1,  # surprise_score's t-loop assumes one clip per call
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        return_multiclass_labels=True,
        num_frames_per_clip=NUM_FRAMES,
        normalize_for_world_model=True,
        skip_export=not args.rebuild_export,
    )

    frame_rate = torch.full((1,), FRAME_RATE_HZ, device=device)
    all_samples = []
    scored_count = 0

    for split_name, loader in (("train", train_loader), ("val", val_loader)):
        for batch_data in tqdm(loader, desc=f"Scoring clips ({split_name})"):
            clip_tensor, label, mc_label, source_mc_label, video_id, target_frame_id = batch_data

            # clip_tensor from DoTAClipDataset is (B=1, C, T, H, W); the model
            # (encode_frames / vit) expects (B, T, C, H, W), same layout
            # cache_features.py permutes to before calling encode_frames.
            window = clip_tensor.permute(0, 2, 1, 3, 4).to(device)

            per_t_maps = surprise_score(
                model, window, frame_rate, T_GRID, heads=HEADS, n_noise_samples=N_NOISE_SAMPLES
            )
            raw_avg, norm_avg = average_over_t(per_t_maps, calib_stats, HEADS, T_GRID)

            # raw_pooled = {h: raw_avg[h].max().item() for h in HEADS}
            # norm_pooled = {h: norm_avg[h].max().item() for h in HEADS}

            label_val = int(label.item())
            class_idx = int(mc_label.item())
            source_class_idx = int(source_mc_label.item())
            video_name = video_id[0]

            all_samples.append({
                "clip_id": video_name,
                "target_frame_id": target_frame_id[0],
                "split": split_name,
                "label": label_val,
                "accident_name": DOTA_CLASS_NAMES.get(class_idx, "unknown"),
                "source_accident_name": DOTA_CLASS_NAMES.get(source_class_idx, "unknown"),
                "night": load_night_flag(args.anno_dir, video_name),
                "raw_map": raw_avg,
                "norm_map": norm_avg,
                # "raw_pooled": raw_pooled,
                # "norm_pooled": norm_pooled,
            })

            if device.type == "mps":
                torch.mps.empty_cache()

            scored_count += 1
            if scored_count % args.checkpoint_interval == 0:
                torch.save(all_samples, partial_path)
                print(f"Checkpoint: saved {len(all_samples)} scored samples to {partial_path}")

    torch.save(all_samples, out_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"Saved {len(all_samples)} scored samples to {out_path}")


if __name__ == "__main__":
    main()