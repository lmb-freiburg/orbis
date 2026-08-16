# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export the spatial path inside one model.vit.blocks[i] block to ONNX and
compare PyTorch vs ONNX Runtime outputs on captured and random inputs.
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
    parser.add_argument("--block-index", type=int, default=0)
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


class SpatialPathWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, token_count, _ = x.shape
        (
            shift_msa_s,
            scale_msa_s,
            gate_msa_s,
            shift_mlp_s,
            scale_mlp_s,
            gate_mlp_s,
            _shift_mlp_t,
            _scale_mlp_t,
            _gate_mlp_t,
        ) = self.block.adaLN_modulation(c).chunk(9, dim=1)

        x_modulated = modulate(self.block.norm1(x), shift_msa_s, scale_msa_s)
        x_modulated = rearrange(x_modulated, "b f n d -> (b f) n d", b=batch_size, f=frame_count)
        x_attn = self.block.space_attn(x_modulated)
        x_attn = rearrange(x_attn, "(b f) n d -> b f n d", b=batch_size, f=frame_count)
        x = x + gate_msa_s.unsqueeze(1).unsqueeze(1) * x_attn

        x_modulated = modulate(self.block.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.block.space_mlp(x_modulated)
        return x


def capture_spatial_io(
    vit: nn.Module,
    block_index: int,
    target_t: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    frame_rate: torch.Tensor,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        raise ValueError("model.vit does not expose blocks")

    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"block_index must be in [0, {len(blocks) - 1}]")

    block = blocks[block_index]
    captured: dict[str, torch.Tensor] = {}

    def block_pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["x"] = inputs[0].detach().cpu().clone()
        captured["c"] = inputs[1].detach().cpu().clone()

    def post_spatial_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["spatial_output"] = inputs[0].detach().cpu().clone()

    handles = [
        block.register_forward_pre_hook(block_pre_hook),
        block.norm_time_attn.register_forward_pre_hook(post_spatial_hook),
    ]

    try:
        with torch.no_grad():
            vit(target_t, context, t, frame_rate=frame_rate)
    finally:
        for handle in handles:
            handle.remove()

    if "x" not in captured or "c" not in captured or "spatial_output" not in captured:
        raise RuntimeError("Failed to capture spatial path tensors from model.vit forward")

    return block, captured["x"], captured["c"], captured["spatial_output"]


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("Spatial Path ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Capturing real spatial-path inputs...")
    vit_inputs = make_vit_inputs(
        model.vit,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        target_frames=args.target_frames,
    )
    block, captured_x, captured_c, captured_spatial_output = capture_spatial_io(
        model.vit,
        args.block_index,
        *vit_inputs,
    )
    wrapper = SpatialPathWrapper(block)
    wrapper.eval()

    print(f"  Block index: {args.block_index}")
    print(f"  Block type: {type(block).__name__}")
    print(f"  Spatial-path x shape: {tuple(captured_x.shape)}")
    print(f"  Conditioning c shape: {tuple(captured_c.shape)}")
    print(f"  Spatial-path output shape: {tuple(captured_spatial_output.shape)}")
    print(f"  shift_size: {getattr(block.space_attn, 'shift_size', None)}")

    print("\n[3] Verifying wrapper matches captured spatial-path continuation...")
    with torch.no_grad():
        wrapped_output = wrapper(captured_x, captured_c)
    wrapper_vs_captured_diff = (wrapped_output - captured_spatial_output).abs()
    print(
        f"  Wrapper vs captured spatial output: max_abs_diff={float(wrapper_vs_captured_diff.max()):.3e} "
        f"mean_abs_diff={float(wrapper_vs_captured_diff.mean()):.3e}"
    )

    print("\n[4] Exporting spatial path to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        wrapper,
        (captured_x, captured_c),
        artifacts.onnx_path,
        input_names=["input_0", "input_1"],
        output_names=["output"],
        opset=args.opset,
    )
    print(f"  Exported: {onnx_path}")

    print("\n[5] Creating ONNX Runtime session...")
    print("  Session ready")

    print("\n[6] Comparing captured and random same-shape samples...")
    summary = parity_runner.run_same_shape_suite(
        wrapper,
        session,
        (captured_x, captured_c),
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            torch.randn_like(captured_x),
            torch.randn_like(captured_c),
        ),
    )

    print("\n[7] Summary")
    print(f"  Random samples tested: {args.num_samples}")
    print(f"  Total mismatches including captured sample: {summary.mismatches}")
    print(f"  Worst max abs diff: {summary.worst_max_abs_diff:.3e}")
    print(f"  Worst mean abs diff: {summary.worst_mean_abs_diff:.3e}")
    if summary.mismatches == 0:
        print("  Result: spatial path exports correctly and matches ONNX Runtime within tolerance")
    else:
        print("  Result: spatial path exports, but ONNX Runtime output diverges from PyTorch")


if __name__ == "__main__":
    main()