# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export the full spatial path inside one model.vit.blocks[i] block to ONNX,
returning both intermediate stage outputs:
1. after space_attn + residual
2. after space_mlp + residual

This lets us compare PyTorch vs ONNX Runtime at both points inside one graph.
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
from export.debug_runner import NamedOutputParityRunner
from networks.DiT.dit import modulate


ARTIFACT_RESOLVER = ArtifactResolver()
MODEL_LOADER = ModelLoader()
ONNX_EXPORTER = OnnxExporter()
ORT_SESSION_FACTORY = OnnxSessionFactory()
OUTPUT_NAMES = ["after_attn", "after_mlp"]


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
    parser.add_argument("--disable-constant-folding", action="store_true")
    parser.add_argument("--disable-ort-optimizations", action="store_true")
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    return args


def load_model(config_path: Path, ckpt_path: Path) -> nn.Module:
    return MODEL_LOADER.load(config_path, ckpt_path)


def _spatial_modulation(block: nn.Module, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
    shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s, *_ = block.adaLN_modulation(c).chunk(9, dim=1)
    return shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s


class SpatialStagesWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frame_count, _, _ = x.shape
        shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s = _spatial_modulation(self.block, c)

        x_modulated = modulate(self.block.norm1(x), shift_msa_s, scale_msa_s)
        x_modulated = rearrange(x_modulated, "b f n d -> (b f) n d", b=batch_size, f=frame_count)
        x_attn = self.block.space_attn(x_modulated)
        x_attn = rearrange(x_attn, "(b f) n d -> b f n d", b=batch_size, f=frame_count)
        x_after_attn = x + gate_msa_s.unsqueeze(1).unsqueeze(1) * x_attn

        x_modulated = modulate(self.block.norm2(x_after_attn), shift_mlp_s, scale_mlp_s)
        x_after_mlp = x_after_attn + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.block.space_mlp(x_modulated)
        return x_after_attn, x_after_mlp


def capture_spatial_stage_io(
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
    wrapper = SpatialStagesWrapper(block)

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
        raise RuntimeError("Failed to capture spatial stage tensors")

    with torch.no_grad():
        x_after_attn, x_after_mlp = wrapper(captured["x_in"], captured["c"])
        captured["x_after_attn"] = x_after_attn.detach().cpu().clone()
        captured["x_after_mlp"] = x_after_mlp.detach().cpu().clone()

    return block, captured


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = NamedOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("Spatial Path Stages ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Capturing real spatial stage tensors...")
    vit_inputs = make_vit_inputs(
        model.vit,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        target_frames=args.target_frames,
    )
    block, captured = capture_spatial_stage_io(
        model.vit,
        args.block_index,
        *vit_inputs,
    )
    wrapper = SpatialStagesWrapper(block).eval()

    print(f"  Block index: {args.block_index}")
    print(f"  Block type: {type(block).__name__}")
    print(f"  shift_size: {getattr(block.space_attn, 'shift_size', None)}")
    print(f"  x_in shape: {tuple(captured['x_in'].shape)}")
    print(f"  c shape: {tuple(captured['c'].shape)}")
    print(f"  x_after_attn shape: {tuple(captured['x_after_attn'].shape)}")
    print(f"  x_after_mlp shape: {tuple(captured['x_after_mlp'].shape)}")

    stage_reconstruction_diff = (captured["x_after_mlp"] - captured["x_after_spatial"]).abs()
    print(
        f"  Captured reconstruction vs spatial continuation: max_abs_diff={float(stage_reconstruction_diff.max()):.3e} "
        f"mean_abs_diff={float(stage_reconstruction_diff.mean()):.3e}"
    )

    print("\n[3] Exporting staged spatial graph to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        wrapper,
        (captured["x_in"], captured["c"]),
        artifacts.onnx_path,
        input_names=["input_0", "input_1"],
        output_names=OUTPUT_NAMES,
        opset=args.opset,
        do_constant_folding=not args.disable_constant_folding,
        disable_optimizations=args.disable_ort_optimizations,
    )
    print(f"  Exported: {onnx_path}")
    print(f"  Constant folding enabled: {not args.disable_constant_folding}")

    print("\n[4] Creating ONNX Runtime session...")
    print("  Session ready")
    print(f"  ORT graph optimizations enabled: {not args.disable_ort_optimizations}")

    print("\n[5] Comparing captured and random same-shape samples...")
    summary = parity_runner.run_named_same_shape_suite(
        wrapper,
        session,
        (captured["x_in"], captured["c"]),
        output_names=OUTPUT_NAMES,
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            torch.randn_like(captured["x_in"]),
            torch.randn_like(captured["c"]),
        ),
    )

    print("\n[6] Summary")
    for key in OUTPUT_NAMES:
        print(
            f"  {key}: mismatches={summary.mismatches[key]} worst_max_abs_diff={summary.worst_max_abs_diff[key]:.3e} "
            f"worst_mean_abs_diff={summary.worst_mean_abs_diff[key]:.3e}"
        )


if __name__ == "__main__":
    main()