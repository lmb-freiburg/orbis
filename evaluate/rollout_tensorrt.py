#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

from rollout_onnx import (
    build_output_dir,
    ensure_dir,
    ensure_tokenizer_env,
    export_vit_to_onnx,
    get_ckpt_epoch_step,
    gif_from_frames,
    length_of,
    logger,
    resolve_input_hw,
    str2bool,
)


try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = None


# Ensure project root (one level up from this file) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from external.orbis.util import instantiate_from_config  # noqa: E402
except ModuleNotFoundError:
    from util import instantiate_from_config  # noqa: E402


def _require_tensorrt() -> None:
    if trt is None:
        raise ModuleNotFoundError(
            "TensorRT Python bindings are not installed in the active environment. "
            "Install TensorRT and ensure `import tensorrt` works before using rollout_tensorrt.py."
        )


def build_engine_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Path,
    input_hw: Tuple[int, int],
    in_channels: int,
    max_context_frames: int,
    opt_batch_size: int,
    max_batch_size: int,
    workspace_gb: float,
    enable_fp16: bool,
) -> None:
    _require_tensorrt()

    logger_trt = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger_trt)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger_trt)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    if enable_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_h, input_w = input_hw
    max_context_frames = max(1, int(max_context_frames))
    opt_batch_size = max(1, int(opt_batch_size))
    max_batch_size = max(opt_batch_size, int(max_batch_size))

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "target_t",
        min=(1, 1, in_channels, input_h, input_w),
        opt=(opt_batch_size, 1, in_channels, input_h, input_w),
        max=(max_batch_size, 1, in_channels, input_h, input_w),
    )
    profile.set_shape(
        "context",
        min=(1, 1, in_channels, input_h, input_w),
        opt=(opt_batch_size, max_context_frames, in_channels, input_h, input_w),
        max=(max_batch_size, max_context_frames, in_channels, input_h, input_w),
    )
    profile.set_shape("t", min=(1,), opt=(opt_batch_size,), max=(max_batch_size,))
    profile.set_shape("frame_rate", min=(1,), opt=(opt_batch_size,), max=(max_batch_size,))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    logger.info(f"Exported TensorRT engine to: {engine_path}")


class TensorRTRunner:
    def __init__(self, engine_path: Path):
        _require_tensorrt()
        runtime_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(runtime_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self._input_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        ]
        self._output_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        if self._output_names != ["output"]:
            logger.warning(f"Unexpected TensorRT output tensors: {self._output_names}")

    def forward(
        self,
        target_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        frame_rate: torch.Tensor,
    ) -> torch.Tensor:
        if target_t.device.type != "cuda":
            raise ValueError("TensorRT inference requires CUDA tensors")

        inputs = {
            "target_t": target_t.contiguous().float(),
            "context": context.contiguous().float(),
            "t": t.contiguous().float(),
            "frame_rate": frame_rate.contiguous().float(),
        }
        for name, tensor in inputs.items():
            self._context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))

        output_name = self._output_names[0]
        output_shape = tuple(int(dim) for dim in self._context.get_tensor_shape(output_name))
        output = torch.empty(output_shape, device=target_t.device, dtype=torch.float32)

        for name, tensor in inputs.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._context.set_tensor_address(output_name, int(output.data_ptr()))

        stream = torch.cuda.current_stream(device=target_t.device)
        success = self._context.execute_async_v3(stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        return output.to(dtype=target_t.dtype)


@torch.no_grad()
def sample_tensorrt(
    *,
    model: torch.nn.Module,
    runner: TensorRTRunner,
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
        neg_v = runner.forward(
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
def roll_out_tensorrt(
    *,
    model: torch.nn.Module,
    runner: TensorRTRunner,
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
        x_last, sample = sample_tensorrt(
            model=model,
            runner=runner,
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
    _require_tensorrt()

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

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT rollout requires CUDA, but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("TensorRT rollout requires a CUDA device")
    torch.cuda.set_device(device)
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
        logger.warning(f"ONNX model not found at {onnx_path}. Exporting it now as TensorRT build input.")
        export_vit_to_onnx(
            model=model,
            onnx_path=onnx_path,
            use_ema=args.evaluate_ema,
            do_constant_folding=args.enable_constant_folding,
        )

    engine_path = Path(args.engine)
    if not engine_path.exists():
        logger.warning(f"TensorRT engine not found at {engine_path}. Building it now.")
        input_hw = resolve_input_hw(model.vit.input_size)
        max_context_frames = max(1, int(getattr(model.vit, "max_num_frames", 2)) - 1)
        build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_hw=input_hw,
            in_channels=int(model.vit.in_channels),
            max_context_frames=max_context_frames,
            opt_batch_size=args.opt_batch_size,
            max_batch_size=args.max_batch_size,
            workspace_gb=args.trt_workspace_gb,
            enable_fp16=args.enable_fp16,
        )

    runner = TensorRTRunner(engine_path)
    if args.evaluate_ema:
        logger.info("Using weights baked into the TensorRT engine; evaluate_ema does not switch weights at runtime.")

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

        _, gen_frames = roll_out_tensorrt(
            model=model,
            runner=runner,
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
    max_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(f"Max CUDA memory: {max_mem_gb:.02f} GB")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Generate rollouts (frames + GIFs) from a TensorRT engine."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory (contains config and checkpoints).")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path, relative to exp_dir.")
    parser.add_argument("--onnx", type=str, default="onnx/last_enhanced.onnx", help="ONNX path used as TensorRT build input, relative to exp_dir.")
    parser.add_argument("--engine", type=str, default="tensorrt/last_enhanced_fp16.engine", help="TensorRT engine path, relative to exp_dir.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path, relative to exp_dir.")
    parser.add_argument("--val_config", type=str, default=None, help="Optional validation data config path (absolute or relative to exp_dir).")
    parser.add_argument("--num_gen_frames", type=int, default=1, help="Number of frames to generate (roll-out length).")
    parser.add_argument("--frames_dir", type=str, default=None, help="Output directory for frames/GIFs (relative to exp_dir if relative).")
    parser.add_argument("--save_real", type=str2bool, default=False, help="Also save ground-truth frames next to generated ones.")
    parser.add_argument("--num_videos", type=int, default=None, help="Generate at most this many sequences (None = all).")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help='CUDA device string, for example "cuda:0".')
    parser.add_argument("--num_steps", type=int, default=30, help="Sampler steps (passed to roll_out as NFE).")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity for sampling (passed to roll_out).")
    parser.add_argument("--evaluate_ema", type=str2bool, default=True, help="Kept for CLI compatibility; TensorRT uses the weights baked into the engine.")
    parser.add_argument("--enable_constant_folding", type=str2bool, default=False, help="Enable constant folding during ONNX export when auto-exporting a missing model.")
    parser.add_argument("--enable_fp16", type=str2bool, default=True, help="Enable FP16 TensorRT engine building when supported.")
    parser.add_argument("--trt_workspace_gb", type=float, default=8.0, help="TensorRT workspace memory pool size in GB.")
    parser.add_argument("--opt_batch_size", type=int, default=1, help="TensorRT optimization profile batch size.")
    parser.add_argument("--max_batch_size", type=int, default=4, help="TensorRT optimization profile maximum batch size.")
    args, unknown = parser.parse_known_args(argv)

    exp_dir = Path(args.exp_dir).resolve()
    ckpt = (exp_dir / args.ckpt).resolve()
    onnx_path = (exp_dir / args.onnx).resolve() if not Path(args.onnx).is_absolute() else Path(args.onnx).resolve()
    engine_path = (exp_dir / args.engine).resolve() if not Path(args.engine).is_absolute() else Path(args.engine).resolve()
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
    args.engine = str(engine_path)
    args.config = str(config)
    args.val_config = str(val_config) if val_config is not None else None
    args.frames_dir = str(frames_dir)
    return args, unknown


def main(argv: Optional[List[str]] = None) -> None:
    args, unknown = parse_args(argv)
    generate_images(args, unknown)


if __name__ == "__main__":
    main()