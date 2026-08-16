# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export the two spatial subpaths inside one model.vit.blocks[i] block to ONNX
and compare PyTorch vs ONNX Runtime outputs:
1. space_attn + residual
2. space_mlp + residual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from einops import rearrange

from export.common import ArtifactResolver, ModelLoader, OnnxExporter, OnnxSessionFactory, make_vit_inputs
from export.debug_runner import SingleOutputParityRunner
from networks.DiT.dit import modulate


ARTIFACT_RESOLVER = ArtifactResolver()
MODEL_LOADER = ModelLoader()
ONNX_EXPORTER = OnnxExporter()
ORT_SESSION_FACTORY = OnnxSessionFactory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ARTIFACT_RESOLVER.add_common_arguments(parser)
    parser.add_argument("--block-index", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    return args


def load_model(config_path: Path, ckpt_path: Path) -> nn.Module:
    return MODEL_LOADER.load(config_path, ckpt_path)


def _spatial_modulation(block: nn.Module, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
    shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s, *_ = block.adaLN_modulation(c).chunk(9, dim=1)
    return shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s


class SpaceAttnResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, _, _ = x.shape
        shift_msa_s, scale_msa_s, gate_msa_s, *_ = _spatial_modulation(self.block, c)
        x_modulated = modulate(self.block.norm1(x), shift_msa_s, scale_msa_s)
        x_modulated = rearrange(x_modulated, "b f n d -> (b f) n d", b=batch_size, f=frame_count)
        x_attn = self.block.space_attn(x_modulated)
        x_attn = rearrange(x_attn, "(b f) n d -> b f n d", b=batch_size, f=frame_count)
        return x + gate_msa_s.unsqueeze(1).unsqueeze(1) * x_attn


class SpaceMlpResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x_after_attn: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        *_, shift_mlp_s, scale_mlp_s, gate_mlp_s = _spatial_modulation(self.block, c)
        x_modulated = modulate(self.block.norm2(x_after_attn), shift_mlp_s, scale_mlp_s)
        return x_after_attn + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.block.space_mlp(x_modulated)


def capture_spatial_subpath_io(
    vit: nn.Module,
    block_index: int,
    target_t: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    frame_rate: torch.Tensor,
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        raise ValueError("model.vit does not expose blocks")
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"block_index must be in [0, {len(blocks) - 1}]")

    block = blocks[block_index]
    captured: dict[str, torch.Tensor] = {}

    attn_wrapper = SpaceAttnResidualWrapper(block)
    mlp_wrapper = SpaceMlpResidualWrapper(block)

    def block_pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["x_in"] = inputs[0].detach().cpu().clone()
        captured["c"] = inputs[1].detach().cpu().clone()

    def norm_time_attn_pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["x_after_spatial"] = inputs[0].detach().cpu().clone()

    handles = [
        block.register_forward_pre_hook(block_pre_hook),
        block.norm_time_attn.register_forward_pre_hook(norm_time_attn_pre_hook),
    ]

    try:
        with torch.no_grad():
            vit(target_t, context, t, frame_rate=frame_rate)
    finally:
        for handle in handles:
            handle.remove()

    if "x_in" not in captured or "c" not in captured or "x_after_spatial" not in captured:
        raise RuntimeError("Failed to capture spatial subpath tensors")

    with torch.no_grad():
        captured["x_after_attn"] = attn_wrapper(captured["x_in"], captured["c"]).detach().cpu().clone()
        captured["x_after_mlp"] = mlp_wrapper(captured["x_after_attn"], captured["c"]).detach().cpu().clone()

    return block, captured


def run_subpath_test(
    *,
    title: str,
    wrapper: nn.Module,
    captured_x: torch.Tensor,
    captured_c: torch.Tensor,
    expected_output: torch.Tensor,
    num_samples: int,
    onnx_path: Path,
    opset: int,
    atol: float,
    rtol: float,
) -> None:
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=atol, rtol=rtol)
    print(f"\n[{title}]")
    with torch.no_grad():
        wrapped_output = wrapper(captured_x, captured_c)
    verify_diff = (wrapped_output - expected_output).abs()
    print(
        f"  Wrapper vs captured output: max_abs_diff={float(verify_diff.max()):.3e} "
        f"mean_abs_diff={float(verify_diff.mean()):.3e}"
    )

    exported_onnx_path, session = parity_runner.export_and_create_session(
        wrapper,
        (captured_x, captured_c),
        onnx_path,
        input_names=["input_0", "input_1"],
        output_names=["output"],
        opset=opset,
    )
    summary = parity_runner.run_same_shape_suite(
        wrapper,
        session,
        (captured_x, captured_c),
        num_samples=num_samples,
        random_input_factory=lambda: (
            torch.randn_like(captured_x),
            torch.randn_like(captured_c),
        ),
    )
    print(f"  Total mismatches including captured sample: {summary.mismatches}")
    print(f"  Worst max abs diff: {summary.worst_max_abs_diff:.3e}")
    print(f"  Worst mean abs diff: {summary.worst_mean_abs_diff:.3e}")


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Spatial Subpaths ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Capturing real spatial subpath tensors...")
    vit_inputs = make_vit_inputs(
        model.vit,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        target_frames=args.target_frames,
    )
    block, captured = capture_spatial_subpath_io(
        model.vit,
        args.block_index,
        *vit_inputs,
    )
    print(f"  Block index: {args.block_index}")
    print(f"  Block type: {type(block).__name__}")
    print(f"  shift_size: {getattr(block.space_attn, 'shift_size', None)}")
    print(f"  x_in shape: {tuple(captured['x_in'].shape)}")
    print(f"  c shape: {tuple(captured['c'].shape)}")
    print(f"  x_after_attn shape: {tuple(captured['x_after_attn'].shape)}")
    print(f"  x_after_mlp shape: {tuple(captured['x_after_mlp'].shape)}")
    print(f"  captured full spatial output shape: {tuple(captured['x_after_spatial'].shape)}")

    run_subpath_test(
        title="3. Testing space_attn + residual",
        wrapper=SpaceAttnResidualWrapper(block).eval(),
        captured_x=captured["x_in"],
        captured_c=captured["c"],
        expected_output=captured["x_after_attn"],
        num_samples=args.num_samples,
        onnx_path=artifacts.onnx_path,
        opset=args.opset,
        atol=args.atol,
        rtol=args.rtol,
    )

    run_subpath_test(
        title="4. Testing space_mlp + residual",
        wrapper=SpaceMlpResidualWrapper(block).eval(),
        captured_x=captured["x_after_attn"],
        captured_c=captured["c"],
        expected_output=captured["x_after_mlp"],
        num_samples=args.num_samples,
        onnx_path=artifacts.onnx_path,
        opset=args.opset,
        atol=args.atol,
        rtol=args.rtol,
    )

    spatial_reconstruction_diff = (captured["x_after_mlp"] - captured["x_after_spatial"]).abs()
    print("\n[5] Captured pipeline consistency")
    print(
        f"  Reconstructed spatial output vs captured continuation: "
        f"max_abs_diff={float(spatial_reconstruction_diff.max()):.3e} "
        f"mean_abs_diff={float(spatial_reconstruction_diff.mean()):.3e}"
    )


if __name__ == "__main__":
    main()