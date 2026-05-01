# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

"""
Export a single WindowAttention module to ONNX and compare PyTorch vs
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
    parser.add_argument("--batch-windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--with-mask", action="store_true")
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    return args


def load_model(config_path: Path, ckpt_path: Path) -> nn.Module:
    return MODEL_LOADER.load(config_path, ckpt_path)


def find_window_attention(model: nn.Module, module_index: int) -> tuple[str, nn.Module]:
    matches: list[tuple[str, nn.Module]] = []
    for name, module in model.vit.named_modules():
        if type(module).__name__ == "WindowAttention":
            matches.append((name, module))

    if not matches:
        raise RuntimeError("No WindowAttention modules found")

    if module_index < 0 or module_index >= len(matches):
        raise IndexError(f"module_index must be in [0, {len(matches) - 1}]")

    return matches[module_index]


def derive_input_shape(module: nn.Module, batch_windows: int) -> tuple[int, int, int]:
    window_size = getattr(module, "window_size", None)
    if window_size is None:
        raise ValueError("WindowAttention module does not expose window_size")

    if isinstance(window_size, int):
        token_count = window_size * window_size
    else:
        token_count = int(window_size[0] * window_size[1])

    channel_count = getattr(module, "dim", None)
    if channel_count is None and hasattr(module, "qkv"):
        channel_count = module.qkv.in_features
    if channel_count is None:
        raise ValueError("WindowAttention module does not expose an input channel size")

    return batch_windows, token_count, channel_count


def build_mask(token_count: int) -> torch.Tensor:
    mask = torch.zeros(1, token_count, token_count, dtype=torch.float32)
    half = token_count // 2
    mask[:, :half, half:] = -100.0
    mask[:, half:, :half] = -100.0
    return mask


class WindowAttentionWrapper(nn.Module):
    def __init__(self, module: nn.Module, with_mask: bool):
        super().__init__()
        self.module = module
        self.with_mask = with_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.with_mask:
            if mask is None:
                raise ValueError("mask is required when with_mask=True")
            return self.module(x, mask=mask)
        return self.module(x)


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts
    torch.manual_seed(args.seed)
    parity_runner = SingleOutputParityRunner(ONNX_EXPORTER, ORT_SESSION_FACTORY, atol=args.atol, rtol=args.rtol)

    print("=" * 60)
    print("WindowAttention ONNX Parity Test")
    print("=" * 60)
    print(f"Config: {artifacts.config_path}")
    print(f"Checkpoint: {artifacts.ckpt_path}")
    print(f"ONNX: {artifacts.onnx_path}")

    print("\n[1] Loading model...")
    model = load_model(artifacts.config_path, artifacts.ckpt_path)

    print("\n[2] Selecting WindowAttention module...")
    module_name, module = find_window_attention(model, args.module_index)
    module.eval()
    print(f"  Module: {module_name}")

    batch_windows, token_count, channel_count = derive_input_shape(module, args.batch_windows)
    print(
        "  Derived shape: "
        f"windows={batch_windows}, tokens={token_count}, channels={channel_count}"
    )
    print(f"  Mask branch enabled: {args.with_mask}")

    wrapper = WindowAttentionWrapper(module, with_mask=args.with_mask)
    wrapper.eval()

    export_x = torch.randn(batch_windows, token_count, channel_count, dtype=torch.float32)
    export_inputs: tuple[torch.Tensor, ...]
    if args.with_mask:
        export_mask = build_mask(token_count)
        export_inputs = (export_x, export_mask)
    else:
        export_inputs = (export_x,)

    print("\n[3] Exporting to ONNX...")
    onnx_path, session = parity_runner.export_and_create_session(
        wrapper,
        export_inputs,
        artifacts.onnx_path,
        input_names=[f"input_{index}" for index in range(len(export_inputs))],
        output_names=["output"],
        opset=args.opset,
    )
    print(f"  Exported: {onnx_path}")

    print("\n[4] Creating ONNX Runtime session...")
    print("  Session ready")

    print("\n[5] Comparing PyTorch vs ONNX Runtime...")
    summary = parity_runner.run_same_shape_suite(
        wrapper,
        session,
        export_inputs,
        num_samples=args.num_samples,
        random_input_factory=lambda: (
            (lambda x, mask: (x, mask) if args.with_mask else (x,))(
                torch.randn(batch_windows, token_count, channel_count, dtype=torch.float32),
                build_mask(token_count),
            )
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
        print("  Result: WindowAttention exports correctly and matches ONNX Runtime within tolerance")
    else:
        print("  Result: WindowAttention exports, but ONNX Runtime output diverges from PyTorch")


if __name__ == "__main__":
    main()