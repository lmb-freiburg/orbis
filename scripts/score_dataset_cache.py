"""
score_dataset.py

Computes retrospective-surprise scores (detailed / semantic / combined heads)
for every clip produced by DoTAClipDataset (dota.py) across all noise levels
in the grid (t_grid) WITHOUT channel mean reduction or calibration normalization.

Fully optimized for cloud CUDA runs on NVIDIA T4 GPUs + 8 vCPU instances.

Output feature shapes per clip entry (stored as torch.float16):
  - 'combined': [T, 32, H, W]
  - 'detailed': [T, 16, H, W]
  - 'semantic': [T, 16, H, W]
  where T = len(t_grid) (4 by default).
"""

import argparse
import json
import sys
from pathlib import Path

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
# Config & CUDA Tuning Flags
# ----------------------------------------------------
HEADS = ["detailed", "semantic", "combined"]
T_GRID = [0.2, 0.4, 0.6, 0.8]
N_NOISE_SAMPLES = 2
FRAME_RATE_HZ = 5.0
NUM_FRAMES = 6

_NIGHT_CACHE = {}

# Optimize CUDA kernel selection for fixed image shapes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def surprise_score_optimized(model, images, frame_rate, t_grid, heads, n_noise_samples=2, use_ema=False):
    """
    Optimized inference pass for CUDA Tensor Cores (NVIDIA T4).
    Batches 'detailed', 'semantic', and 'combined' passes into a SINGLE
    forward call per t_val rather than 3 separate sequential passes.
    """
    net = model.ema_vit if use_ema else model.vit
    net.eval()

    device = images.device

    # Use Automatic Mixed Precision (FP16) to leverage T4 Tensor Cores
    with torch.amp.autocast("cuda", dtype=torch.float16):
        x = model.encode_frames(images)
        context, target = x[:, :-1], x[:, -1:]
        b = x.shape[0]  # Assumed B=1
        n_channels = target.shape[2]
        half = n_channels // 2

        context_exp = context.repeat_interleave(n_noise_samples, dim=0)
        target_exp = target.repeat_interleave(n_noise_samples, dim=0)

        per_head_maps = {h: [] for h in heads}

        for t_val in t_grid:
            t = torch.full((b * n_noise_samples,), t_val, device=device)
            fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate

            batch_ctx_list, batch_tgt_list, head_order = [], [], []

            # 1. Prepare masked context and target inputs for requested heads
            if "detailed" in heads:
                ctx_d, tgt_d = context_exp.clone(), target_exp.clone()
                ctx_d[:, :, half:, :, :] = 0
                tgt_d[:, :, half:, :, :] = 0
                batch_ctx_list.append(ctx_d)
                batch_tgt_list.append(tgt_d)
                head_order.append("detailed")

            if "semantic" in heads:
                ctx_s, tgt_s = context_exp.clone(), target_exp.clone()
                ctx_s[:, :, :half, :, :] = 0
                tgt_s[:, :, :half, :, :] = 0
                batch_ctx_list.append(ctx_s)
                batch_tgt_list.append(tgt_s)
                head_order.append("semantic")

            if "combined" in heads:
                batch_ctx_list.append(context_exp)
                batch_tgt_list.append(target_exp)
                head_order.append("combined")

            # 2. Concat all active heads into ONE single forward pass
            concat_ctx = torch.cat(batch_ctx_list, dim=0)
            concat_tgt = torch.cat(batch_tgt_list, dim=0)
            concat_t = t.repeat(len(head_order))
            concat_fr = fr.repeat(len(head_order))

            total_batch_size = concat_ctx.shape[0]
            if frame_rate.numel() == 1:
                concat_fr = frame_rate.repeat(total_batch_size)
            else:
                concat_fr = frame_rate.repeat_interleave(n_noise_samples, dim=0).repeat(len(head_order))

            tgt_t, noise = model.add_noise(concat_tgt, concat_t)
            pred = net(tgt_t, concat_ctx, concat_t, frame_rate=concat_fr)
            true_v = model.A(concat_t) * concat_tgt + model.B(concat_t) * noise

            # Compute squared residual error
            err = (pred.float() - true_v.float()) ** 2  # [N_heads * b * n_noise_samples, 1, C, H, W]

            # Unstack heads and process results
            err_split = err.chunk(len(head_order), dim=0)

            for idx, h_name in enumerate(head_order):
                # Unroll noise dimension and average across n_noise_samples
                h_err = err_split[idx].squeeze(1).view(b, n_noise_samples, n_channels, err.shape[-2], err.shape[-1])
                h_err_avg = h_err.mean(dim=1)  # [B, C, H, W]

                start_idx = half if h_name == "semantic" else 0
                end_idx = half if h_name == "detailed" else n_channels

                # Extract target channels, squeeze batch dim, convert to FP16 CPU tensor
                active_err = h_err_avg[:, start_idx:end_idx].squeeze(0).cpu().half()
                per_head_maps[h_name].append(active_err)

    # Stack along time dimension T -> shapes: [T, C, H, W] in torch.float16
    stacked_maps = {h: torch.stack(per_head_maps[h], dim=0) for h in heads}
    return stacked_maps


def load_night_flag(anno_dir, video_name):
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
        description="Score DoTA clips for retrospective surprise optimized for GCP T4 GPU."
    )
    parser.add_argument("--exp_dir", type=str, default="logs_wm/orbis_288x512")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--seq_dir", type=str, default="/Volumes/maccbeast/frames/")
    parser.add_argument("--anno_dir", type=str, default="annotations/")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Optimal CPU worker threads for n1-standard-8 (8 vCPUs).",
    )
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
        help="Cap the dataset to this many clips before the 80/20 train/val split.",
    )
    parser.add_argument(
        "--rebuild_export",
        action="store_true",
        help="Also have get_dota_dataloaders (re)build the on-disk export.",
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

    train_loader, val_loader = get_dota_dataloaders(
            args.seq_dir,
            args.anno_dir,
            batch_size=1,  # Keep B=1 per loader call (handled internally by surprise_score_optimized)
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            return_multiclass_labels=True,
            num_frames_per_clip=NUM_FRAMES,
            use_cloud_dataset=True,
            cloud_dir="DOTA_training",
            cloud_file="DoTA_training.pt"
        )

    cfg = OmegaConf.load(f"{args.exp_dir}/{args.config}")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{args.exp_dir}/{args.ckpt}", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    

    frame_rate = torch.full((1,), FRAME_RATE_HZ, device=device)
    all_samples = []
    scored_count = 0

    for split_name, loader in (("train", train_loader), ("val", val_loader)):
        for batch_data in tqdm(loader, desc=f"Scoring clips ({split_name})"):
            clip_tensor, label, mc_label, source_mc_label, ego_label, video_id, target_frame_id = batch_data

            # Use non_blocking=True for fast pinned-memory transfer over PCIe
            window = clip_tensor.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)

            head_maps = surprise_score_optimized(
                model, window, frame_rate, T_GRID, heads=HEADS, n_noise_samples=N_NOISE_SAMPLES
            )

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
                # "night": load_night_flag(args.anno_dir, video_name),
                "head_maps": head_maps,  # Dict of [T, C, H, W] tensors stored in float16
            })

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