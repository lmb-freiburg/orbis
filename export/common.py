# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf

from util import instantiate_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "logs_wm" / "orbis_288x512"


@dataclass(frozen=True)
class ResolvedArtifactPaths:
    config_path: Path
    ckpt_path: Path
    onnx_path: Path | None = None


class ArtifactResolver:
    def __init__(self, run_dir: Path = DEFAULT_RUN_DIR):
        self.run_dir = run_dir
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.onnx_dir = self.run_dir / "onnx"

    def add_common_arguments(self, parser: argparse.ArgumentParser, *, include_onnx: bool = True) -> None:
        parser.add_argument("--config-path", type=str, default=None)
        parser.add_argument("--ckpt-path", type=str, default=None)
        if include_onnx:
            parser.add_argument("--onnx-path", type=str, default=None)

    def expand_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    def latest_matching_file(self, directory: Path, pattern: str) -> Path:
        matches = [path for path in directory.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
        return max(matches, key=lambda path: path.stat().st_mtime)

    def resolve_existing_file(self, path: Path, description: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
        return path

    def default_config_path(self) -> Path:
        return self.resolve_existing_file(self.run_dir / "config.yaml", "Config file")

    def default_ckpt_path(self) -> Path:
        return self.latest_matching_file(self.ckpt_dir, "*.ckpt")

    def default_onnx_path(self) -> Path:
        return self.latest_matching_file(self.onnx_dir, "*.onnx")

    def resolve_from_args(self, args: argparse.Namespace, *, include_onnx: bool = True) -> ResolvedArtifactPaths:
        config_path = self.resolve_existing_file(
            self.expand_path(args.config_path) if args.config_path else self.default_config_path(),
            "Config file",
        )
        ckpt_path = self.resolve_existing_file(
            self.expand_path(args.ckpt_path) if args.ckpt_path else self.default_ckpt_path(),
            "Checkpoint file",
        )
        onnx_path: Path | None = None
        if include_onnx:
            onnx_path = self.expand_path(args.onnx_path) if args.onnx_path else self.default_onnx_path()
        return ResolvedArtifactPaths(config_path=config_path, ckpt_path=ckpt_path, onnx_path=onnx_path)


class ModelLoader:
    def __init__(self, *, strict: bool = False, use_ema: bool = True):
        self.strict = strict
        self.use_ema = use_ema

    def load(self, config_path: Path, ckpt_path: Path) -> nn.Module:
        self._ensure_tokenizer_env(config_path)
        cfg = OmegaConf.load(config_path)
        model = instantiate_from_config(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=self.strict)
        model.eval()

        if self.use_ema and hasattr(model, "ema_vit"):
            ema_params = dict(model.ema_vit.named_parameters())
            for name, param in model.vit.named_parameters():
                if name in ema_params:
                    param.data.copy_(ema_params[name].data)

        return model

    def _ensure_tokenizer_env(self, config_path: Path) -> None:
        config_text = config_path.read_text(encoding="utf-8")
        if "$TK_WORK_DIR" not in config_text:
            return
        if os.getenv("TK_WORK_DIR"):
            return

        candidate_roots = [
            REPO_ROOT / "logs_tk",
            REPO_ROOT,
            config_path.parent,
            config_path.parent.parent,
        ]
        for candidate_root in candidate_roots:
            tokenizer_dir = candidate_root / "tokenizer_288x512"
            if tokenizer_dir.exists():
                os.environ["TK_WORK_DIR"] = str(candidate_root)
                return


class OnnxExporter:
    def export(
        self,
        module: nn.Module,
        inputs: Sequence[torch.Tensor] | tuple[torch.Tensor, ...],
        onnx_path: Path,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        opset: int = 17,
        do_constant_folding: bool = True,
        dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    ) -> Path:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"onnx(\..*)?",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Constant folding - Only steps=1 can be constant folded.*",
                category=UserWarning,
                module=r"torch\.onnx(\..*)?",
            )
            torch.onnx.export(
                module,
                tuple(inputs),
                str(onnx_path),
                opset_version=opset,
                do_constant_folding=do_constant_folding,
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
            onnx.checker.check_model(str(onnx_path))
        return onnx_path


class OnnxSessionFactory:
    def create_cpu_session(self, onnx_path: Path | str, *, disable_optimizations: bool = False) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        if disable_optimizations:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )


class VitOnnxWrapper(nn.Module):
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(
        self,
        target_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        frame_rate: torch.Tensor,
    ) -> torch.Tensor:
        return self.vit(target_t, context, t, frame_rate=frame_rate)


class VitOnnxExporter:
    def __init__(self, exporter: OnnxExporter | None = None):
        self.exporter = exporter or OnnxExporter()

    def export(
        self,
        model: nn.Module,
        onnx_path: Path,
        *,
        opset: int = 17,
        do_constant_folding: bool = True,
    ) -> Path:
        input_h, input_w = resolve_input_hw(model.vit.input_size)
        device = next(model.parameters()).device
        dummy_target_t = torch.randn(1, 1, model.vit.in_channels, input_h, input_w, device=device)
        dummy_context_frames = max(1, int(getattr(model.vit, "max_num_frames", 2)) - 1)
        dummy_context = torch.randn(
            1,
            dummy_context_frames,
            model.vit.in_channels,
            input_h,
            input_w,
            device=device,
        )
        dummy_t = torch.rand(1, device=device)
        dummy_frame_rate = torch.ones(1, device=device)
        wrapper = VitOnnxWrapper(model.vit).eval()
        return self.exporter.export(
            wrapper,
            (dummy_target_t, dummy_context, dummy_t, dummy_frame_rate),
            onnx_path,
            input_names=["target_t", "context", "t", "frame_rate"],
            output_names=["output"],
            opset=opset,
            do_constant_folding=do_constant_folding,
            dynamic_axes={
                "target_t": {0: "batch"},
                "context": {0: "batch", 1: "context_frames"},
                "t": {0: "batch"},
                "frame_rate": {0: "batch"},
                "output": {0: "batch"},
            },
        )


def resolve_input_hw(input_size: object) -> tuple[int, int]:
    if isinstance(input_size, (list, tuple, ListConfig)):
        return int(input_size[0]), int(input_size[1])
    size = int(input_size)
    return size, size


def make_vit_inputs(
    vit: nn.Module,
    batch_size: int,
    context_frames: int,
    target_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_size = getattr(vit, "input_size", None)
    in_channels = getattr(vit, "in_channels", None)
    if input_size is None or in_channels is None:
        raise ValueError("model.vit must expose input_size and in_channels")

    height, width = resolve_input_hw(input_size)
    target_t = torch.randn(batch_size, target_frames, in_channels, height, width, dtype=torch.float32)
    context = torch.randn(batch_size, context_frames, in_channels, height, width, dtype=torch.float32)
    t = torch.rand(batch_size, dtype=torch.float32)
    frame_rate = torch.ones(batch_size, dtype=torch.float32)
    return target_t, context, t, frame_rate