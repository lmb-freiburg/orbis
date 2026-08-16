# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export a single SwinTransformerBlock to ONNX and compare PyTorch vs
ONNX Runtime outputs on multiple random samples.
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

from export.common import ArtifactResolver, ModelLoader, OnnxExporter, OnnxSessionFactory
from export.debug_runner import SingleOutputParityRunner


ARTIFACT_RESOLVER = ArtifactResolver()
MODEL_LOADER = ModelLoader()
ONNX_EXPORTER = OnnxExporter()
ORT_SESSION_FACTORY = OnnxSessionFactory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ARTIFACT_RESOLVER.add_common_arguments(parser)
    parser.add_argument("--module-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    return args


def load_model(config_path: Path, ckpt_path: Path) -> nn.Module:
    return MODEL_LOADER.load(config_path, ckpt_path)


def find_swin_block(model: nn.Module, module_index: int) -> tuple[str, nn.Module]:
    matches: list[tuple[str, nn.Module]] = []
    for name, module in model.vit.named_modules():
        if type(module).__name__ == "SwinTransformerBlock":
            matches.append((name, module))

    if not matches:
        raise RuntimeError("No SwinTransformerBlock modules found")

    if module_index < 0 or module_index >= len(matches):
        raise IndexError(f"module_index must be in [0, {len(matches) - 1}]")

    return matches[module_index]


def derive_input_shape(module: nn.Module, batch_size: int) -> tuple[int, int, int, tuple[int, int]]:
    input_resolution = getattr(module, "input_resolution", None)
    if input_resolution is None:
        raise ValueError("SwinTransformerBlock does not expose input_resolution")

    channel_count = getattr(module, "dim", None)
    if channel_count is None:
        raise ValueError("SwinTransformerBlock does not expose dim")

    token_count = int(input_resolution[0] * input_resolution[1])
    return batch_size, token_count, channel_count, tuple(input_resolution)


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("SwinTransformerBlock ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Selecting SwinTransformerBlock module...")
    module_name, block = find_swin_block(model, args.module_index)
    block.eval()
    print(f"  Module: {module_name}")

    batch_size, token_count, channel_count, input_resolution = derive_input_shape(block, args.batch_size)
    print(
        "  Derived shape: "
        f"batch={batch_size}, resolution={input_resolution}, tokens={token_count}, channels={channel_count}"
    )
    print(f"  Shift size: {getattr(block, 'shift_size', None)}")
    print(f"  Window size: {getattr(block, 'window_size', None)}")

    export_x = torch.randn(batch_size, token_count, channel_count, dtype=torch.float32)

    print("\n[3] Exporting to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        block,
        (export_x,),
        artifacts.onnx_path,
        input_names=["input_0"],
        output_names=["output"],
        opset=args.opset,
    )
    print(f"  Exported: {onnx_path}")

    print("\n[4] Creating ONNX Runtime session...")
    print("  Session ready")

    print("\n[5] Comparing PyTorch vs ONNX Runtime...")
    summary = parity_runner.run_same_shape_suite(
        block,
        session,
        (export_x,),
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            torch.randn(batch_size, token_count, channel_count, dtype=torch.float32),
        ),
        captured_label="Export sample",
        random_label="Sample",
    )

    print("\n[6] Summary")
    print(f"  Samples tested: {args.num_samples}")
    print(f"  Mismatches including export sample: {summary.mismatches}")
    print(f"  Worst max abs diff: {summary.worst_max_abs_diff:.3e}")
    print(f"  Worst mean abs diff: {summary.worst_mean_abs_diff:.3e}")
    if summary.mismatches == 0:
        print("  Result: SwinTransformerBlock exports correctly and matches ONNX Runtime within tolerance")
    else:
        print("  Result: SwinTransformerBlock exports, but ONNX Runtime output diverges from PyTorch")


if __name__ == "__main__":
    main()