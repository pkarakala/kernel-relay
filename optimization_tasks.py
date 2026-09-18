"""Task registry wrapping, rather than replacing, the verified V1 substrate."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

import agentic_kernel_compiler as compiler
from agent_schema import HardwareFingerprint


@dataclass(frozen=True)
class TaskSettings:
    rows: int = 8192
    hidden_size: int = 128
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if type(self.rows) is not int or type(self.hidden_size) is not int:
            raise ValueError("rows and hidden_size must be integers")
        if self.rows <= 0 or self.hidden_size <= 0:
            raise ValueError("rows and hidden_size must be positive")
        if self.dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")


@dataclass(frozen=True)
class OptimizationTask:
    task_id: str
    description: str
    module_factory: Callable[..., torch.nn.Module]
    input_factory: Callable[..., tuple[torch.Tensor, ...]]
    target_constraints: dict[str, Any]
    validation_factory: Callable[..., compiler.ValidationSuite]

    def prepare(self, settings: TaskSettings, device: torch.device) -> compiler.ValidationSuite:
        module = self.module_factory(settings.hidden_size).to(device).eval()
        inputs = self.input_factory(settings.rows, settings.hidden_size, getattr(torch, settings.dtype), device)
        return self.validation_factory(module, inputs, device)

    @torch.no_grad()
    def describe(self, settings: TaskSettings, device: torch.device) -> dict[str, Any]:
        suite = self.prepare(settings, device)
        check = suite.validate(suite.module)
        if not check.passed:
            raise compiler.ReferenceValidationError(check.reason)
        graph, error = compiler.capture_and_propagate(suite.module, suite.inputs)
        fusion = compiler.find_fusion_region(graph, suite.inputs, settings.hidden_size, error)
        return {
            "task_id": self.task_id, "description": self.description,
            "pytorch_source": inspect.getsource(self.module_factory),
            "fx_graph": str(graph.graph), "fx_code": graph.code,
            "graph_nodes": compiler.graph_rows(graph, fusion.nodes),
            "shape_propagation_error": error, "fusion_eligible": bool(fusion.nodes),
            "fusion_reason": fusion.reason,
            "inputs": [{"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "stride": list(tensor.stride())} for tensor in suite.inputs],
            "dtype": settings.dtype, "eps": suite.module.eps,
            "target_constraints": self.target_constraints,
            "reference_behavior": "eager PyTorch; inference only; inputs must not be mutated",
            "validation_cases": list(compiler.VALIDATION_CASES),
            "tolerances": dict(zip(("atol", "rtol"), compiler.dtype_tolerances(suite.inputs[0].dtype))),
            "input_seed": compiler.INPUT_SEED,
        }


TASKS = {
    "bias_layernorm_silu": OptimizationTask(
        "bias_layernorm_silu", "bias_add -> LayerNorm -> SiLU, four contiguous tensor inputs",
        compiler.BiasAddLayerNormSiLU, compiler.make_inputs,
        {"devices": ["cpu", "cuda"], "dtypes": ["float16", "float32"],
         "triton_device": "NVIDIA CUDA", "example_triton_max_hidden_size": 256},
        compiler.ValidationSuite,
    )
}


def load_task(task_id: str) -> OptimizationTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown task {task_id!r}; available: {', '.join(sorted(TASKS))}") from exc


def fingerprint(device: torch.device) -> HardwareFingerprint:
    status = "not imported (CPU)"
    if device.type == "cuda":
        _, _, status = compiler.maybe_load_triton(device)
    return HardwareFingerprint(str(device), dict(compiler.target_fingerprint(device, status)))
