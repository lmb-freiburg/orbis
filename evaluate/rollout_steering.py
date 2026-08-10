import argparse
import os
import sys
from typing import List, Optional, Tuple
from pathlib import Path
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import cv2

import torch
from torchvision.utils import save_image

from pytorch_lightning import seed_everything

from util import instantiate_from_config
from main import logger


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_output_dir(
    exp_dir: Path,
    frames_dir_arg: Optional[str],
    ckpt_path: Path,
    val_config: Optional[str],
    steering: str,
    num_steps: int,
) -> Path:
    if frames_dir_arg is not None:
        return (exp_dir / frames_dir_arg).resolve()
    epoch, global_step = get_ckpt_epoch_step(ckpt_path)
    data_tag = (
        Path(val_config).stem if val_config is not None else "default_data"
    )
    steering_tag = get_steering_string(steering)
    rel = Path("gen_rollout") / data_tag / f"ep{epoch}iter{global_step}_{num_steps}steps" / f"steering_{steering_tag}"
    return (exp_dir / rel).resolve()


def get_ckpt_epoch_step(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    return epoch, global_step


def get_steering_string(steering: str) -> str:
    if steering.endswith('.npy') or steering.endswith('.csv'):
        # file name minus path and extension
        steering = Path(steering).stem
    return steering


def draw_arrow(img, start, end, color=(0, 1, 0), thickness=2):
    img = img.permute(1, 2, 0).cpu().numpy().copy()  # still [-1,1]

    # Convert color from [-1,1] space
    color = tuple(c * 2 - 1 for c in color)

    cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.1)

    return torch.from_numpy(img).permute(2, 0, 1)


def overlay_goals_on_images(images, goals, draw_goal_arrows=False):
    # images: (B, T, C, H, W)    # T: num frames with context
    # goals:  (B, t, 2)          # t: num gen frames
    
    ctx_size = images.shape[1] - goals.shape[1]
    
    # draw goals
    for b in range(images.shape[0]):
        for t in range(goals.shape[1]):
            frame_idx = ctx_size + t
            goal = goals[b, t]
            if torch.isnan(goal).any():
                continue
            if draw_goal_arrows:
                start = (int(images.shape[4]*0.2), int(images.shape[3]*0.8))
                end = (int(start[0] - goal[1].item() * images.shape[4]//30), int(start[1] - goal[0].item() * images.shape[3]//30))
                images[b, frame_idx] = draw_arrow(images[b, frame_idx], start, end, color=(255, 0, 0), thickness=2)
    return images


def apply_steering(data_batch, steering: str):
    if steering == "from_data":
        pass
    elif steering.endswith(".npy") or steering.endswith(".csv"):
        assert os.path.isfile(steering), f"Steering file {steering} does not exist"
        loaded_steering = np.load(steering) if steering.endswith(".npy") else np.loadtxt(steering, delimiter=',')  # (num_frames, 2 or 3)
        assert loaded_steering.shape[1] in [2, 3], f"Steering file should have shape (num_frames, 2) or (num_frames, 3), got {loaded_steering.shape}"
        assert loaded_steering.shape[0] >= data_batch['steering'].shape[1], f"Steering file should have at least {data_batch['steering'].shape[1]} frames, got {loaded_steering.shape[0]}"
        # data_batch['steering']: (B, T, 3), loaded_steering: (F, 2 or 3), with F<=T
        for b_idx in range(data_batch['steering'].shape[0]):
            data_batch['steering'][b_idx,:,:loaded_steering.shape[1]] = torch.from_numpy(loaded_steering[:data_batch['steering'].shape[1], :])
    else:
        raise ValueError(f"Unknown steering mode: {steering}")

    return data_batch


@torch.no_grad()
def generate_images(args, unknown_args):
    if os.path.exists(args.frames_dir):
        print("Folder exist, new images will be saved to the same folder, delete it if you want to start from scratch")
    if not os.path.exists(args.frames_dir):
        os.makedirs(args.frames_dir)

    if args.seed > 0:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)
        
    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(args.ckpt)["state_dict"], strict=True)
    model = model.cuda()
    _ = model.eval()
    
    # Validation data config, if provided
    if args.val_config is not None:
        config = OmegaConf.merge(OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args))
    
    # Get the dataset (for conditioning)
    # if we want to save real frames for evaluation, we need to tell the dataloader
    config.data.params.validation.params.num_frames = model.num_context_frames + getattr(model, 'goal_distance_frames', 0) + args.num_gen_frames
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = data.val_dataloader()
    
    roll_out_args = {
        "num_gen_frames": args.num_gen_frames,
        "eta": args.eta,
        "NFE": args.num_steps,
        "sample_with_ema": args.evaluate_ema,
    }
    
    logger.info(f"Steering mode: {args.steering}") 
    logger.info(f"Saving generated images to {args.frames_dir}")
    sample_idx = 0
    progress_bar = tqdm(range(len(val_loader.dataset)//val_loader.batch_size))
    loader_iter = iter(val_loader)
    for batch_idx, _ in enumerate(progress_bar):
        # try:
        data_batch = next(loader_iter)
        
        data_batch = {k: v.cuda() for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        data_batch = apply_steering(data_batch, args.steering)

        gen_out = model.roll_out(data_batch=data_batch, latent_input=False, **roll_out_args)
        gen_frames = gen_out['images']

        gen_frames = overlay_goals_on_images(gen_frames, gen_out['goals'])

        # save generated images and gifs
        for sample_in_batch_idx in range(gen_frames.shape[0]):
            subfolder_path_fake = os.path.join(args.frames_dir, "fake_images", f"sequence_{sample_idx:04d}")
            subfolder_path_gifs = os.path.join(args.frames_dir, f"gen_gifs")
            if not os.path.exists(subfolder_path_fake): os.makedirs(subfolder_path_fake)
            if not os.path.exists(subfolder_path_gifs): os.makedirs(subfolder_path_gifs)
            for f in range(len(gen_frames[sample_in_batch_idx])):
                save_image((gen_frames[sample_in_batch_idx, f]+1.0)/2.0, os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg"))
            # gifs
            imageio.mimsave(os.path.join(subfolder_path_gifs, f"sequence_{sample_idx:04d}.gif"), [np.array(Image.open(os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg"))) for f in range(len(gen_frames[sample_in_batch_idx]))], fps=7, loop=0)

            if args.save_real:
                subfolder_path_real = os.path.join(args.frames_dir, "real_images", f"sequence_{sample_idx:04d}")
                if not os.path.exists(subfolder_path_real): os.makedirs(subfolder_path_real)
                for f in range(data_batch['images'].shape[1]):
                    save_image((data_batch['images'][sample_in_batch_idx, f]+1.0)/2.0, os.path.join(subfolder_path_real, f"frame_{f:04d}.jpg"))
            sample_idx += 1
        
        if args.num_videos is not None and sample_idx >= args.num_videos:
            break
        
    progress_bar.set_description(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.02f} GB")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Generate rollouts (frames + GIFs) from a trained model."
    )
    parser.add_argument(
        "--steering",
        type=str,
        default="from_data",
        help="Options: 'from_data', or path/to/file.npy or path/to/file.csv to load steering values from file.",
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
        default=False,
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
        steering=args.steering,
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
    generate_images(args, unknown)


if __name__ == "__main__":
    main()