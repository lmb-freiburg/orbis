# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export one full model.vit.blocks[i] block to ONNX and compare PyTorch vs
ONNX Runtime outputs on captured and random same-shape inputs.
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


def capture_block_io(
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

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["x"] = inputs[0].detach().cpu().clone()
        captured["c"] = inputs[1].detach().cpu().clone()
        captured["output"] = output.detach().cpu().clone()

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            vit(target_t, context, t, frame_rate=frame_rate)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Failed to capture block inputs from model.vit forward")

    return block, captured["x"], captured["c"], captured["output"]


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("STDiT Block ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Capturing real inputs for model.vit.blocks[i]...")
    vit_inputs = make_vit_inputs(
        model.vit,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        target_frames=args.target_frames,
    )
    block, captured_x, captured_c, captured_output = capture_block_io(
        model.vit,
        args.block_index,
        *vit_inputs,
    )
    block.eval()
    print(f"  Block index: {args.block_index}")
    print(f"  Block type: {type(block).__name__}")
    print(f"  Captured x shape: {tuple(captured_x.shape)}")
    print(f"  Captured c shape: {tuple(captured_c.shape)}")
    print(f"  Captured output shape: {tuple(captured_output.shape)}")

    print("\n[3] Exporting block to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        block,
        (captured_x, captured_c),
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
        block,
        session,
        (captured_x, captured_c),
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            torch.randn_like(captured_x),
            torch.randn_like(captured_c),
        ),
    )

    print("\n[6] Summary")
    print(f"  Random samples tested: {args.num_samples}")
    print(f"  Total mismatches including captured sample: {summary.mismatches}")
    print(f"  Worst max abs diff: {summary.worst_max_abs_diff:.3e}")
    print(f"  Worst mean abs diff: {summary.worst_mean_abs_diff:.3e}")
    if summary.mismatches == 0:
        print("  Result: model.vit.blocks[i] exports correctly and matches ONNX Runtime within tolerance")
    else:
        print("  Result: model.vit.blocks[i] exports, but ONNX Runtime output diverges from PyTorch")


if __name__ == "__main__":
    main()