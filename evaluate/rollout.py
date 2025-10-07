#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image  # noqa: F401  # kept in case downstream uses it
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

# Ensure project root (one level up from this file) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util import instantiate_from_config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_ckpt_epoch_step(ckpt_path: Path) -> Tuple[int, int]:
    """Return (epoch, global_step) from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor in [-1, 1] or [0, 1] range to uint8 HWC.
    Accepts CHW.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")
    # If in [-1, 1], map to [0, 1]
    if t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    # CHW -> HWC
    t = t.detach().float().cpu()
    t = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return t.numpy()


def gif_from_frames(frames: List[torch.Tensor], fps: int = 7) -> List[np.ndarray]:
    """Convert a list of CHW image tensors to a list of uint8 HWC frames."""
    return [to_uint8_image(frm) for frm in frames]


def length_of(loader: Iterable) -> Optional[int]:
    """Return len(loader) if available; otherwise None."""
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return None


@torch.inference_mode()
def generate_images(args: argparse.Namespace, unknown_args: List[str]) -> None:
    frames_dir = Path(args.frames_dir)
    if frames_dir.exists():
        logger.warning(
            "Output folder exists. New images will be added to the same folder. "
            "Delete it if you want to start from scratch."
        )
    else:
        ensure_dir(frames_dir)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and patch model config
    base_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(unknown_args))

    # Build & load model
    model = instantiate_from_config(cfg.model)
    state = torch.load(str(args.ckpt), map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Validation data config (optional)
    if args.val_config is not None:
        data_cfg = OmegaConf.merge(
            OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args)
        )
    else:
        data_cfg = cfg

    # Dataset setup (for conditioning)
    num_condition_frames = (
        data_cfg.data.params.validation.params.num_frames - 1
    )
    num_frames_total = num_condition_frames + args.num_gen_frames

    # If saving real frames too, request longer sequence from the datamodule
    if args.save_real:
        data_cfg.data.params.validation.params.num_frames = num_frames_total

    # Ensure we don't drag along training config by accident
    if hasattr(data_cfg.data.params, "train"):
        del data_cfg.data.params.train

    data_module = instantiate_from_config(data_cfg.data)
    data_module.prepare_data()
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Progress handling
    total_batches = length_of(val_loader)
    pbar = tqdm(
        total=total_batches,
        desc="Generating",
        dynamic_ncols=True,
    )

    # Iterate
    sample_idx = 0
    for batch in val_loader:
        if args.num_videos is not None and sample_idx >= args.num_videos:
            break

        # Get images tensor from dict batches or tensor-like
        if isinstance(batch, dict):
            x = batch["images"].to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)

        # Conditioning: take first K frames as input
        cond_x = x[:, :num_condition_frames]

        # autocast only on CUDA (float16) to avoid CPU AMP dtype quirks
        use_amp = device.type == "cuda"
        ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if use_amp
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        ) if False else torch.autocast(enabled=False, device_type="cuda" if device.type == "cuda" else "cpu")

        # We keep AMP disabled by default; uncomment above to enable on CUDA if desired.

        with torch.no_grad():
            # model.roll_out returns latents, gen_frames; assume gen_frames: [B, T, C, H, W]
            latents, gen_frames = model.roll_out(
                x_0=cond_x,
                num_gen_frames=args.num_gen_frames,
                latent_input=False,
                eta=args.eta,
                NFE=args.num_steps,
                sample_with_ema=args.evaluate_ema,
                num_samples=cond_x.size(0),
            )

        # Save generated and (optionally) real frames
        for b in range(x.size(0)):
            if args.num_videos is not None and sample_idx >= args.num_videos:
                break

            seq_name = f"sequence_{sample_idx:04d}"
            fake_dir = frames_dir / "fake_images" / seq_name
            gif_dir = frames_dir / "gen_gifs"
            ensure_dir(fake_dir)
            ensure_dir(gif_dir)

            # gen_frames[b] is a sequence [T, C, H, W]
            seq_frames = gen_frames[b]  # (T, C, H, W)
            T = seq_frames.shape[0]

            # Save individual frames as jpg
            for f_idx in range(T):
                frame = (seq_frames[f_idx] + 1.0) / 2.0  # [0,1]
                save_image(frame, fake_dir / f"frame_{f_idx:04d}.jpg")

            # Save GIF directly from tensors (avoid re-reading from disk)
            gif_frames = gif_from_frames([seq_frames[f] for f in range(T)], fps=7)
            imageio.mimsave(gif_dir / f"{seq_name}.gif", gif_frames, fps=7, loop=0)

            # Optionally save real frames
            if args.save_real:
                real_dir = frames_dir / "real_images" / seq_name
                ensure_dir(real_dir)
                for f_idx in range(x.shape[1]):
                    real_frame = (x[b, f_idx] + 1.0) / 2.0
                    save_image(real_frame, real_dir / f"frame_{f_idx:04d}.jpg")

            sample_idx += 1

        # Update progress bar
        if total_batches is not None:
            pbar.update(1)
        else:
            # Unknown length: set a descriptive postfix each iteration
            pbar.set_postfix_str(f"samples={sample_idx}")

    pbar.close()
    if device.type == "cuda":
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Max CUDA memory: {max_mem_gb:.02f} GB")


def build_output_dir(
    exp_dir: Path,
    frames_dir_arg: Optional[str],
    ckpt_path: Path,
    val_config: Optional[str],
    num_steps: int,
) -> Path:
    if frames_dir_arg is not None:
        return (exp_dir / frames_dir_arg).resolve()
    epoch, global_step = get_ckpt_epoch_step(ckpt_path)
    data_tag = (
        Path(val_config).stem if val_config is not None else "default_data"
    )
    rel = Path("gen_rollout") / data_tag / f"ep{epoch}iter{global_step}_{num_steps}steps"
    return (exp_dir / rel).resolve()


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Generate rollouts (frames + GIFs) from a trained model."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory (contains config and checkpoints).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/last.ckpt",
        help="Checkpoint path, relative to exp_dir.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config path, relative to exp_dir.",
    )
    parser.add_argument(
        "--val_config",
        type=str,
        default=None,
        help="Optional validation data config path (absolute or relative to exp_dir).",
    )
    parser.add_argument(
        "--num_gen_frames",
        type=int,
        default=1,
        help="Number of frames to generate (roll-out length).",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Output directory for frames/GIFs (relative to exp_dir if relative).",
    )
    parser.add_argument(
        "--save_real",
        type=str2bool,
        default=True,
        help="Also save ground-truth frames next to generated ones.",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=None,
        help="Generate at most this many sequences (None = all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device string (e.g., "cuda", "cuda:0", or "cpu").',
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="Sampler steps (passed to roll_out as NFE).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Stochasticity for sampling (passed to roll_out).",
    )
    parser.add_argument(
        "--evaluate_ema",
        type=str2bool,
        default=True,
        help="Evaluate with EMA weights if available.",
    )
    args, unknown = parser.parse_known_args(argv)

    # Resolve paths relative to exp_dir
    exp_dir = Path(args.exp_dir).resolve()
    ckpt = (exp_dir / args.ckpt).resolve()
    config = (exp_dir / args.config).resolve()
    val_config = (
        (exp_dir / args.val_config).resolve()
        if args.val_config and not Path(args.val_config).is_absolute()
        else (Path(args.val_config).resolve() if args.val_config else None)
    )

    # Basic checks
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    if val_config is not None and not val_config.exists():
        raise FileNotFoundError(f"val_config not found: {val_config}")

    # Compute frames_dir
    frames_dir = build_output_dir(
        exp_dir=exp_dir,
        frames_dir_arg=args.frames_dir,
        ckpt_path=ckpt,
        val_config=str(val_config) if val_config is not None else None,
        num_steps=args.num_steps,
    )

    # Store resolved paths back
    args.exp_dir = str(exp_dir)
    args.ckpt = str(ckpt)
    args.config = str(config)
    args.val_config = str(val_config) if val_config is not None else None
    args.frames_dir = str(frames_dir)

    return args, unknown


def main(argv: Optional[List[str]] = None) -> None:
    args, unknown = parse_args(argv)
    logger.info(f"Saving outputs to: {args.frames_dir}")
    generate_images(args, unknown)


if __name__ == "__main__":
    main()
