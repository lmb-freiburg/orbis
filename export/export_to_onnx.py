# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from export.common import ArtifactResolver, ModelLoader, OnnxExporter, OnnxSessionFactory, VitOnnxExporter
from export.debug_runner import SingleOutputParityRunner
from export.export_workflow import ExportVerificationWorkflow, ExportWorkflowConfig


ARTIFACT_RESOLVER = ArtifactResolver()
MODEL_LOADER = ModelLoader()
ONNX_EXPORTER = OnnxExporter()
ORT_SESSION_FACTORY = OnnxSessionFactory()
VIT_ONNX_EXPORTER = VitOnnxExporter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model.vit to ONNX using repo-local defaults.")
    ARTIFACT_RESOLVER.add_common_arguments(parser)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--swin-module-index", type=int, default=0)
    parser.add_argument("--window-attention-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--disable-constant-folding", action="store_true")
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    return args


def main() -> None:
    args = parse_args()
    workflow = ExportVerificationWorkflow(
        config=ExportWorkflowConfig(
            artifacts=args.artifacts,
            block_index=args.block_index,
            swin_module_index=args.swin_module_index,
            window_attention_index=args.window_attention_index,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            num_samples=args.num_samples,
            opset=args.opset,
            atol=args.atol,
            rtol=args.rtol,
            do_constant_folding=not args.disable_constant_folding,
        ),
        model_loader=MODEL_LOADER,
        parity_runner=SingleOutputParityRunner(
            ONNX_EXPORTER,
            ORT_SESSION_FACTORY,
            atol=args.atol,
            rtol=args.rtol,
        ),
        session_factory=ORT_SESSION_FACTORY,
        vit_exporter=VIT_ONNX_EXPORTER,
    )
    raise SystemExit(workflow.run())


if __name__ == "__main__":
    main()