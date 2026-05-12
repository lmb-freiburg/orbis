#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio
import numpy as np
import onnxruntime as ort
import torch
import onnx
from PIL import Image  # noqa: F401  # kept in case downstream uses it
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

# Ensure project root (one level up from this file) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from external.orbis.util import instantiate_from_config  # noqa: E402
except ModuleNotFoundError:
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
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")
    if t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    t = t.detach().float().cpu()
    t = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return t.numpy()


def gif_from_frames(frames: List[torch.Tensor], fps: int = 7) -> List[np.ndarray]:
    return [to_uint8_image(frm) for frm in frames]


def length_of(loader: Iterable) -> Optional[int]:
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return None


def resolve_input_hw(input_size: object) -> Tuple[int, int]:
    if isinstance(input_size, (list, tuple, ListConfig)):
        return int(input_size[0]), int(input_size[1])
    size = int(input_size)
    return size, size


def build_ort_session(args: argparse.Namespace) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if args.device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    if not args.enable_ort_optimizations:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=session_options,
        providers=providers,
    )
    logger.info(f"ONNX Runtime providers: {session.get_providers()}")
    logger.info(f"ORT graph optimizations enabled: {args.enable_ort_optimizations}")
    return session


def ensure_tokenizer_env(config_path: Path, exp_dir: Path) -> None:
    config_text = config_path.read_text(encoding="utf-8")
    if "$TK_WORK_DIR" not in config_text:
        return
    if os.getenv("TK_WORK_DIR"):
        return

    candidate_roots = [exp_dir.parent, exp_dir, PROJECT_ROOT.parent.parent / "checkpoints"]
    for candidate_root in candidate_roots:
        tokenizer_dir = candidate_root / "tokenizer_288x512"
        if tokenizer_dir.exists():
            os.environ["TK_WORK_DIR"] = str(candidate_root)
            logger.info(f"Set TK_WORK_DIR to: {candidate_root}")
            return

    logger.warning("Config references $TK_WORK_DIR but no tokenizer_288x512 folder was found to infer it.")


def export_vit_to_onnx(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    use_ema: bool,
    do_constant_folding: bool,
) -> None:
    class VitWrapper(torch.nn.Module):
        def __init__(self, vit: torch.nn.Module):
            super().__init__()
            self.vit = vit

        def forward(self, target_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor, frame_rate: torch.Tensor) -> torch.Tensor:
            return self.vit(target_t, context, t, frame_rate=frame_rate)

    if use_ema and hasattr(model, "ema_vit"):
        logger.info("Applying EMA weights before ONNX export")
        ema_params = dict(model.ema_vit.named_parameters())
        for name, param in model.vit.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name].data)

    input_h, input_w = resolve_input_hw(model.vit.input_size)

    dummy_target_t = torch.randn(1, 1, model.vit.in_channels, input_h, input_w, device=next(model.parameters()).device)
    dummy_context_frames = max(1, int(getattr(model.vit, "max_num_frames", 2)) - 1)
    dummy_context = torch.randn(
        1,
        dummy_context_frames,
        model.vit.in_channels,
        input_h,
        input_w,
        device=dummy_target_t.device,
    )
    dummy_t = torch.rand(1, device=dummy_target_t.device)
    dummy_frame_rate = torch.ones(1, device=dummy_target_t.device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = VitWrapper(model.vit).eval()
    torch.onnx.export(
        wrapper,
        (dummy_target_t, dummy_context, dummy_t, dummy_frame_rate),
        str(onnx_path),
        opset_version=17,
        do_constant_folding=do_constant_folding,
        input_names=["target_t", "context", "t", "frame_rate"],
        output_names=["output"],
        dynamic_axes={
            "target_t": {0: "batch"},
            "context": {0: "batch", 1: "context_frames"},
            "t": {0: "batch"},
            "frame_rate": {0: "batch"},
            "output": {0: "batch"},
        },
        verbose=False,
    )
    onnx.checker.check_model(str(onnx_path))
    logger.info(f"Exported ONNX model to: {onnx_path}")


def _build_ort_inputs(
    session: ort.InferenceSession,
    target_t: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    frame_rate: torch.Tensor,
) -> dict[str, np.ndarray]:
    named_inputs = {
        "target_t": np.ascontiguousarray(target_t.detach().float().cpu().numpy()),
        "context": np.ascontiguousarray(context.detach().float().cpu().numpy()),
        "t": np.ascontiguousarray(t.detach().float().cpu().numpy()),
        "frame_rate": np.ascontiguousarray(frame_rate.detach().float().cpu().numpy()),
    }
    session_input_names = [session_input.name for session_input in session.get_inputs()]
    if all(name in named_inputs for name in session_input_names):
        return {name: named_inputs[name] for name in session_input_names}

    ordered_arrays = [named_inputs["target_t"], named_inputs["context"], named_inputs["t"], named_inputs["frame_rate"]]
    return {
        session_input.name: ordered_arrays[index]
        for index, session_input in enumerate(session.get_inputs())
    }


def vit_forward_onnx(
    session: ort.InferenceSession,
    target_t: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    frame_rate: torch.Tensor,
) -> torch.Tensor:
    onnx_inputs = _build_ort_inputs(session, target_t, context, t, frame_rate)
    output = session.run(None, onnx_inputs)[0]
    return torch.from_numpy(output).to(device=target_t.device, dtype=target_t.dtype)


@torch.no_grad()
def sample_onnx(
    *,
    model: torch.nn.Module,
    ort_session: ort.InferenceSession,
    images: torch.Tensor,
    eta: float,
    nfe: int,
    num_samples: int,
    frame_rate: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    context = images.clone()

    if frame_rate is None:
        frame_rate = torch.full((num_samples,), 5.0, device=device, dtype=torch.float32)
    else:
        frame_rate = frame_rate.to(device=device, dtype=torch.float32)

    input_h, input_w = resolve_input_hw(model.vit.input_size)

    target_t = torch.randn(num_samples, 1, model.vit.in_channels, input_h, input_w, device=device)
    t_steps = torch.linspace(1, 0, nfe + 1, device=device)

    for i in range(nfe):
        t = t_steps[i].repeat(target_t.shape[0])
        neg_v = vit_forward_onnx(
            ort_session,
            target_t,
            context,
            t=t * model.timescale,
            frame_rate=frame_rate,
        )
        dt = t_steps[i] - t_steps[i + 1]
        dw = torch.randn_like(target_t) * torch.sqrt(dt)
        diffusion = dt
        target_t = target_t + neg_v * dt + eta * torch.sqrt(2 * diffusion) * dw

    last_frame = target_t.clone()
    images_out = model.decode_frames(last_frame)
    return target_t.squeeze(1), images_out


@torch.no_grad()
def roll_out_onnx(
    *,
    model: torch.nn.Module,
    ort_session: ort.InferenceSession,
    x_0: torch.Tensor,
    num_gen_frames: int,
    eta: float,
    nfe: int,
    num_samples: int,
    frame_rate: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_c = model.encode_frames(x_0)
    x_all = x_c.clone()
    samples = []

    for _ in tqdm(range(num_gen_frames), desc="Rolling out frames", leave=False):
        x_last, sample = sample_onnx(
            model=model,
            ort_session=ort_session,
            images=x_c,
            eta=eta,
            nfe=nfe,
            num_samples=num_samples,
            frame_rate=frame_rate,
        )
        x_all = torch.cat([x_all, x_last.unsqueeze(1)], dim=1)
        x_c = torch.cat([x_c[:, 1:], x_last.unsqueeze(1)], dim=1)
        samples.append(sample)

    return x_all, torch.cat(samples, dim=1)


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

    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ensure_tokenizer_env(Path(args.config), Path(args.exp_dir))

    base_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(unknown_args))

    model = instantiate_from_config(cfg.model)
    state = torch.load(str(args.ckpt), map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        logger.warning(f"ONNX model not found at {onnx_path}. Exporting it now.")
        export_vit_to_onnx(
            model=model,
            onnx_path=onnx_path,
            use_ema=args.evaluate_ema,
            do_constant_folding=args.enable_constant_folding,
        )

    ort_session = build_ort_session(args)
    if args.evaluate_ema:
        logger.info("Using weights baked into the ONNX model; evaluate_ema does not switch weights at runtime.")

    if args.val_config is not None:
        data_cfg = OmegaConf.merge(
            OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args)
        )
    else:
        data_cfg = cfg

    num_condition_frames = None
    if args.save_real:
        num_condition_frames = data_cfg.data.params.validation.params.num_frames - 1
        num_frames_total = num_condition_frames + args.num_gen_frames
        data_cfg.data.params.validation.params.num_frames = num_frames_total

    if hasattr(data_cfg.data.params, "train"):
        del data_cfg.data.params.train

    data_module = instantiate_from_config(data_cfg.data)
    data_module.prepare_data()
    data_module.setup()
    val_loader = data_module.val_dataloader()

    logger.info(f"Saving outputs to: {args.frames_dir}")
    total_batches = length_of(val_loader)
    pbar = tqdm(total=total_batches, desc="Generating", dynamic_ncols=True)

    sample_idx = 0
    for batch in val_loader:
        if args.num_videos is not None and sample_idx >= args.num_videos:
            break

        if isinstance(batch, dict):
            x = batch["images"].to(device, non_blocking=True)
            frame_rate = batch.get("frame_rate")
            if frame_rate is not None:
                frame_rate = frame_rate.to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            frame_rate = None

        cond_x = x[:, :num_condition_frames]

        with torch.no_grad():
            _, gen_frames = roll_out_onnx(
                model=model,
                ort_session=ort_session,
                x_0=cond_x,
                num_gen_frames=args.num_gen_frames,
                eta=args.eta,
                nfe=args.num_steps,
                num_samples=cond_x.size(0),
                frame_rate=frame_rate,
            )

        for b in range(x.size(0)):
            if args.num_videos is not None and sample_idx >= args.num_videos:
                break

            seq_name = f"sequence_{sample_idx:04d}"
            fake_dir = frames_dir / "fake_images" / seq_name
            gif_dir = frames_dir / "gen_gifs"
            ensure_dir(fake_dir)
            ensure_dir(gif_dir)

            seq_frames = gen_frames[b]
            time_steps = seq_frames.shape[0]

            for f_idx in range(time_steps):
                frame = (seq_frames[f_idx] + 1.0) / 2.0
                save_image(frame, fake_dir / f"frame_{f_idx:04d}.jpg")

            gif_frames = gif_from_frames([seq_frames[f] for f in range(time_steps)], fps=7)
            imageio.mimsave(gif_dir / f"{seq_name}.gif", gif_frames, fps=7, loop=0)

            if args.save_real:
                real_dir = frames_dir / "real_images" / seq_name
                ensure_dir(real_dir)
                for f_idx in range(x.shape[1]):
                    real_frame = (x[b, f_idx] + 1.0) / 2.0
                    save_image(real_frame, real_dir / f"frame_{f_idx:04d}.jpg")

            sample_idx += 1

        if total_batches is not None:
            pbar.update(1)
        else:
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
    data_tag = Path(val_config).stem if val_config is not None else "default_data"
    rel = Path("gen_rollout") / data_tag / f"ep{epoch}iter{global_step}_{num_steps}steps"
    return (exp_dir / rel).resolve()


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Generate rollouts (frames + GIFs) from an exported ONNX model."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory (contains config and checkpoints).")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path, relative to exp_dir.")
    parser.add_argument("--onnx", type=str, default="onnx/last_enhanced.onnx", help="ONNX model path, relative to exp_dir.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path, relative to exp_dir.")
    parser.add_argument("--val_config", type=str, default=None, help="Optional validation data config path (absolute or relative to exp_dir).")
    parser.add_argument("--num_gen_frames", type=int, default=1, help="Number of frames to generate (roll-out length).")
    parser.add_argument("--frames_dir", type=str, default=None, help="Output directory for frames/GIFs (relative to exp_dir if relative).")
    parser.add_argument("--save_real", type=str2bool, default=False, help="Also save ground-truth frames next to generated ones.")
    parser.add_argument("--num_videos", type=int, default=None, help="Generate at most this many sequences (None = all).")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--device", type=str, default="cuda", help='Device string (e.g., "cuda", "cuda:0", or "cpu").')
    parser.add_argument("--num_steps", type=int, default=30, help="Sampler steps (passed to roll_out as NFE).")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity for sampling (passed to roll_out).")
    parser.add_argument("--evaluate_ema", type=str2bool, default=True, help="Kept for CLI compatibility; ONNX uses the weights baked into the exported model.")
    parser.add_argument("--enable_ort_optimizations", type=str2bool, default=False, help="Enable ONNX Runtime graph optimizations. Disabled by default for this model.")
    parser.add_argument("--enable_constant_folding", type=str2bool, default=False, help="Enable constant folding during ONNX export when auto-exporting a missing model.")
    args, unknown = parser.parse_known_args(argv)

    exp_dir = Path(args.exp_dir).resolve()
    ckpt = (exp_dir / args.ckpt).resolve()
    onnx_path = (exp_dir / args.onnx).resolve() if not Path(args.onnx).is_absolute() else Path(args.onnx).resolve()
    config = (exp_dir / args.config).resolve()
    val_config = (
        (exp_dir / args.val_config).resolve()
        if args.val_config and not Path(args.val_config).is_absolute()
        else (Path(args.val_config).resolve() if args.val_config else None)
    )

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    if val_config is not None and not val_config.exists():
        raise FileNotFoundError(f"val_config not found: {val_config}")

    frames_dir = build_output_dir(
        exp_dir=exp_dir,
        frames_dir_arg=args.frames_dir,
        ckpt_path=ckpt,
        val_config=str(val_config) if val_config is not None else None,
        num_steps=args.num_steps,
    )

    args.exp_dir = str(exp_dir)
    args.ckpt = str(ckpt)
    args.onnx = str(onnx_path)
    args.config = str(config)
    args.val_config = str(val_config) if val_config is not None else None
    args.frames_dir = str(frames_dir)
    return args, unknown


def main(argv: Optional[List[str]] = None) -> None:
    args, unknown = parse_args(argv)
    generate_images(args, unknown)


if __name__ == "__main__":
    main()