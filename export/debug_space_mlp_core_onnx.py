# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export only the norm2 -> modulate -> space_mlp core path inside one
model.vit.blocks[i] block to ONNX and compare PyTorch vs ONNX Runtime.

This excludes the final residual addition so we can determine whether the
drift originates inside the MLP path itself or only when adding it back.
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


def _space_mlp_modulation(block: nn.Module, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, _, shift_mlp_s, scale_mlp_s, gate_mlp_s, *_ = block.adaLN_modulation(c).chunk(9, dim=1)
    return shift_mlp_s, scale_mlp_s, gate_mlp_s


class SpaceMlpCoreWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x_after_attn: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_mlp_s, scale_mlp_s, _gate_mlp_s = _space_mlp_modulation(self.block, c)
        x_modulated = modulate(self.block.norm2(x_after_attn), shift_mlp_s, scale_mlp_s)
        return self.block.space_mlp(x_modulated)


def capture_space_mlp_core_io(
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
        raise RuntimeError("Failed to capture spatial tensors for MLP core")

    shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s, *_ = block.adaLN_modulation(captured["c"]).chunk(9, dim=1)
    batch_size, frame_count, _, _ = captured["x_in"].shape
    with torch.no_grad():
        x_modulated = modulate(block.norm1(captured["x_in"]), shift_msa_s, scale_msa_s)
        x_modulated = x_modulated.reshape(batch_size * frame_count, x_modulated.shape[2], x_modulated.shape[3])
        x_attn = block.space_attn(x_modulated)
        x_attn = x_attn.reshape(batch_size, frame_count, x_attn.shape[1], x_attn.shape[2])
        x_after_attn = captured["x_in"] + gate_msa_s.unsqueeze(1).unsqueeze(1) * x_attn
        x_core = block.space_mlp(modulate(block.norm2(x_after_attn), shift_mlp_s, scale_mlp_s))

    captured["x_after_attn"] = x_after_attn.detach().cpu().clone()
    captured["x_core"] = x_core.detach().cpu().clone()
    captured["x_reconstructed_after_spatial"] = (x_after_attn + gate_mlp_s.unsqueeze(1).unsqueeze(1) * x_core).detach().cpu().clone()
    return block, captured


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("Space MLP Core ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Capturing real MLP-core tensors...")
    vit_inputs = make_vit_inputs(
        model.vit,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        target_frames=args.target_frames,
    )
    block, captured = capture_space_mlp_core_io(
        model.vit,
        args.block_index,
        *vit_inputs,
    )
    wrapper = SpaceMlpCoreWrapper(block).eval()

    print(f"  Block index: {args.block_index}")
    print(f"  Block type: {type(block).__name__}")
    print(f"  x_after_attn shape: {tuple(captured['x_after_attn'].shape)}")
    print(f"  c shape: {tuple(captured['c'].shape)}")
    print(f"  x_core shape: {tuple(captured['x_core'].shape)}")

    with torch.no_grad():
        wrapped_output = wrapper(captured["x_after_attn"], captured["c"])
    verify_diff = (wrapped_output - captured["x_core"]).abs()
    print(
        f"  Wrapper vs captured MLP core: max_abs_diff={float(verify_diff.max()):.3e} "
        f"mean_abs_diff={float(verify_diff.mean()):.3e}"
    )

    reconstruction_diff = (captured["x_reconstructed_after_spatial"] - captured["x_after_spatial"]).abs()
    print(
        f"  Reconstructed full spatial output vs captured: max_abs_diff={float(reconstruction_diff.max()):.3e} "
        f"mean_abs_diff={float(reconstruction_diff.mean()):.3e}"
    )

    print("\n[3] Exporting MLP core to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        wrapper,
        (captured["x_after_attn"], captured["c"]),
        artifacts.onnx_path,
        input_names=["input_0", "input_1"],
        output_names=["output"],
        opset=args.opset,
    )
    print(f"  Exported: {onnx_path}")

    print("\n[4] Creating ONNX Runtime session...")
    print("  Session ready")

    print("\n[5] Comparing captured and random same-shape samples...")
    summary = parity_runner.run_same_shape_suite(
        wrapper,
        session,
        (captured["x_after_attn"], captured["c"]),
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            torch.randn_like(captured["x_after_attn"]),
            torch.randn_like(captured["c"]),
        ),
    )

    print("\n[6] Summary")
    print(f"  Total mismatches including captured sample: {summary.mismatches}")
    print(f"  Worst max abs diff: {summary.worst_max_abs_diff:.3e}")
    print(f"  Worst mean abs diff: {summary.worst_mean_abs_diff:.3e}")


if __name__ == "__main__":
    main()