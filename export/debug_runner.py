# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import onnxruntime as ort
import torch
import torch.nn as nn

from export.common import OnnxExporter, OnnxSessionFactory


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def status_badge(ok: bool) -> str:
    return style(" PASS ", BOLD, GREEN) if ok else style(" FAIL ", BOLD, RED)


@dataclass(frozen=True)
class SampleComparison:
    ok: bool
    max_abs_diff: float
    mean_abs_diff: float


@dataclass(frozen=True)
class ParitySuiteSummary:
    captured: SampleComparison
    mismatches: int
    worst_max_abs_diff: float
    worst_mean_abs_diff: float


@dataclass(frozen=True)
class NamedSampleComparison:
    results: Mapping[str, SampleComparison]


@dataclass(frozen=True)
class NamedParitySuiteSummary:
    captured: NamedSampleComparison
    mismatches: Mapping[str, int]
    worst_max_abs_diff: Mapping[str, float]
    worst_mean_abs_diff: Mapping[str, float]


class SingleOutputParityRunner:
    def __init__(
        self,
        exporter: OnnxExporter,
        session_factory: OnnxSessionFactory,
        *,
        atol: float,
        rtol: float,
    ):
        self.exporter = exporter
        self.session_factory = session_factory
        self.atol = atol
        self.rtol = rtol

    def build_indexed_ort_inputs(self, inputs: Sequence[torch.Tensor]) -> dict[str, object]:
        return {
            f"input_{index}": tensor.cpu().numpy()
            for index, tensor in enumerate(inputs)
        }

    def export_and_create_session(
        self,
        module: nn.Module,
        sample_inputs: Sequence[torch.Tensor],
        onnx_path: Path,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        opset: int,
        do_constant_folding: bool = True,
        disable_optimizations: bool = False,
    ) -> tuple[str, ort.InferenceSession]:
        exported_path = self.exporter.export(
            module,
            tuple(sample_inputs),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset=opset,
            do_constant_folding=do_constant_folding,
        )
        session = self.session_factory.create_cpu_session(
            exported_path,
            disable_optimizations=disable_optimizations,
        )
        return str(exported_path), session

    def compare_sample(
        self,
        module: nn.Module,
        session: ort.InferenceSession,
        module_inputs: Sequence[torch.Tensor],
        ort_inputs: dict[str, object] | None = None,
    ) -> SampleComparison:
        with torch.no_grad():
            pytorch_output = module(*tuple(module_inputs))

        effective_ort_inputs = ort_inputs or self.build_indexed_ort_inputs(module_inputs)
        ort_output = session.run(None, effective_ort_inputs)[0]
        pytorch_output_np = pytorch_output.cpu().numpy()
        abs_diff = abs(pytorch_output_np - ort_output)
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
        ok = torch.allclose(
            torch.from_numpy(ort_output),
            torch.from_numpy(pytorch_output_np),
            atol=self.atol,
            rtol=self.rtol,
        )
        return SampleComparison(ok=ok, max_abs_diff=max_abs_diff, mean_abs_diff=mean_abs_diff)

    def print_sample(self, label: str, result: SampleComparison) -> None:
        print(
            f"  {style(label, CYAN)} {status_badge(result.ok)} "
            f"{style('max', DIM)}={result.max_abs_diff:.3e} "
            f"{style('mean', DIM)}={result.mean_abs_diff:.3e}"
        )

    def run_same_shape_suite(
        self,
        module: nn.Module,
        session: ort.InferenceSession,
        captured_inputs: Sequence[torch.Tensor],
        *,
        num_samples: int,
        random_input_factory: Callable[[], Sequence[torch.Tensor]],
        captured_label: str = "Captured sample",
        random_label: str = "Random sample",
    ) -> ParitySuiteSummary:
        captured_result = self.compare_sample(module, session, captured_inputs)
        self.print_sample(captured_label, captured_result)

        worst_max_abs_diff = captured_result.max_abs_diff
        worst_mean_abs_diff = captured_result.mean_abs_diff
        mismatches = 0 if captured_result.ok else 1

        for sample_index in range(num_samples):
            sample_inputs = tuple(random_input_factory())
            sample_result = self.compare_sample(module, session, sample_inputs)
            if not sample_result.ok:
                mismatches += 1
            worst_max_abs_diff = max(worst_max_abs_diff, sample_result.max_abs_diff)
            worst_mean_abs_diff = max(worst_mean_abs_diff, sample_result.mean_abs_diff)
            self.print_sample(f"{random_label} {sample_index + 1:02d}/{num_samples}", sample_result)

        return ParitySuiteSummary(
            captured=captured_result,
            mismatches=mismatches,
            worst_max_abs_diff=worst_max_abs_diff,
            worst_mean_abs_diff=worst_mean_abs_diff,
        )


class NamedOutputParityRunner(SingleOutputParityRunner):
    def compare_named_sample(
        self,
        module: nn.Module,
        session: ort.InferenceSession,
        module_inputs: Sequence[torch.Tensor],
        *,
        output_names: Sequence[str],
        ort_inputs: dict[str, object] | None = None,
    ) -> NamedSampleComparison:
        with torch.no_grad():
            pytorch_outputs = module(*tuple(module_inputs))

        if not isinstance(pytorch_outputs, tuple):
            pytorch_outputs = (pytorch_outputs,)

        effective_ort_inputs = ort_inputs or self.build_indexed_ort_inputs(module_inputs)
        ort_outputs = session.run(None, effective_ort_inputs)
        results: dict[str, SampleComparison] = {}

        for output_name, pytorch_output, ort_output in zip(output_names, pytorch_outputs, ort_outputs):
            pytorch_output_np = pytorch_output.cpu().numpy()
            abs_diff = abs(pytorch_output_np - ort_output)
            max_abs_diff = float(abs_diff.max())
            mean_abs_diff = float(abs_diff.mean())
            ok = torch.allclose(
                torch.from_numpy(ort_output),
                torch.from_numpy(pytorch_output_np),
                atol=self.atol,
                rtol=self.rtol,
            )
            results[output_name] = SampleComparison(
                ok=ok,
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
            )

        return NamedSampleComparison(results=results)

    def print_named_sample(self, label: str, result: NamedSampleComparison) -> None:
        print(f"  {style(label, CYAN)}")
        for output_name, comparison in result.results.items():
            print(
                f"    {output_name} {status_badge(comparison.ok)} "
                f"{style('max', DIM)}={comparison.max_abs_diff:.3e} "
                f"{style('mean', DIM)}={comparison.mean_abs_diff:.3e}"
            )

    def run_named_same_shape_suite(
        self,
        module: nn.Module,
        session: ort.InferenceSession,
        captured_inputs: Sequence[torch.Tensor],
        *,
        output_names: Sequence[str],
        num_samples: int,
        random_input_factory: Callable[[], Sequence[torch.Tensor]],
        captured_label: str = "Captured sample",
        random_label: str = "Random sample",
    ) -> NamedParitySuiteSummary:
        captured_result = self.compare_named_sample(
            module,
            session,
            captured_inputs,
            output_names=output_names,
        )
        self.print_named_sample(captured_label, captured_result)

        mismatches = {
            output_name: 0 if comparison.ok else 1
            for output_name, comparison in captured_result.results.items()
        }
        worst_max_abs_diff = {
            output_name: comparison.max_abs_diff
            for output_name, comparison in captured_result.results.items()
        }
        worst_mean_abs_diff = {
            output_name: comparison.mean_abs_diff
            for output_name, comparison in captured_result.results.items()
        }

        for sample_index in range(num_samples):
            sample_inputs = tuple(random_input_factory())
            sample_result = self.compare_named_sample(
                module,
                session,
                sample_inputs,
                output_names=output_names,
            )
            self.print_named_sample(f"{random_label} {sample_index + 1:02d}/{num_samples}", sample_result)
            for output_name, comparison in sample_result.results.items():
                if not comparison.ok:
                    mismatches[output_name] += 1
                worst_max_abs_diff[output_name] = max(worst_max_abs_diff[output_name], comparison.max_abs_diff)
                worst_mean_abs_diff[output_name] = max(worst_mean_abs_diff[output_name], comparison.mean_abs_diff)

        return NamedParitySuiteSummary(
            captured=captured_result,
            mismatches=mismatches,
            worst_max_abs_diff=worst_max_abs_diff,
            worst_mean_abs_diff=worst_mean_abs_diff,
        )