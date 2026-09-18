#!/usr/bin/env python3
"""A small, closed-loop compiler POC for bias-add + LayerNorm + SiLU.

The program captures and analyzes a PyTorch graph, generates a bounded set of
Triton launch configurations on supported CUDA systems, verifies every executed
candidate, benchmarks only verified paths, and falls back safely elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import operator
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


SEARCH_BLOCK_SIZES = (32, 64, 128, 256)
SEARCH_NUM_WARPS = (2, 4, 8)
SEARCH_NUM_STAGES = (2, 3, 4)
MAX_TRITON_HIDDEN_SIZE = max(SEARCH_BLOCK_SIZES)
DEFAULT_EPS = 1.0e-5
INPUT_SEED = 20260915
TUNING_SEED = 20260916
VALIDATION_CASES = (
    "primary_random", "random_seed_plus_1", "random_seed_plus_2", "all_zero",
    "constant_rows", "near_zero", "alternating_large", "nontrivial_affine",
)
TRITON_KERNEL: Any | None = None


class BiasAddLayerNormSiLU(nn.Module):
    """y = silu(layer_norm(x + input_bias, gamma, beta))."""

    def __init__(self, hidden_size: int, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        input_bias: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        biased = x + input_bias
        normalized = F.layer_norm(
            biased,
            (self.hidden_size,),
            weight=gamma,
            bias=beta,
            eps=self.eps,
        )
        return F.silu(normalized)


@dataclass(frozen=True)
class KernelConfig:
    block_size: int
    num_warps: int
    num_stages: int

    def compact(self) -> str:
        return f"B{self.block_size}/W{self.num_warps}/S{self.num_stages}"


@dataclass
class VerificationResult:
    passed: bool
    max_abs: float
    max_rel: float
    reason: str
    max_normalized: float = math.inf
    mismatched: int = 0
    elements: int = 0
    atol: float = 0.0
    rtol: float = 0.0

    @property
    def mismatch_percent(self) -> float:
        return 100.0 * self.mismatched / self.elements if self.elements else 0.0


@dataclass(frozen=True)
class FusionMatch:
    nodes: tuple[Node, ...] = ()
    eps: float | None = None
    rejections: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.nodes:
            return f"validated functional add -> layer_norm -> non-inplace silu; eps={self.eps:g}"
        return "; ".join(self.rejections)


@dataclass
class CandidateResult:
    config: KernelConfig
    decision: str
    reason: str
    initial_median_ms: float | None = None
    verification: VerificationResult | None = None
    finalist_trials: list[float] = field(default_factory=list)
    is_finalist: bool = False
    failure: dict[str, str] | None = None

    @property
    def finalist_median_ms(self) -> float:
        return statistics.median(self.finalist_trials) if self.finalist_trials else math.inf


@dataclass
class CompiledPath:
    label: str
    function: Callable[[], torch.Tensor]
    verification: VerificationResult
    used_compile: bool
    detail: str
    failure: dict[str, str] | None = None
    attempt_verification: VerificationResult | None = None


def ascii_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = "| " + " | ".join(
        name.ljust(widths[index]) for index, name in enumerate(headers)
    ) + " |"
    lines = [border, header, border]
    lines.extend(
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rendered
    )
    lines.append(border)
    return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def one_line_error(exc: BaseException, limit: int = 180) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return f"{type(exc).__name__}: {text}"


def failure_record(exc: BaseException, phase: str) -> dict[str, str]:
    """Classify known compiler exceptions without importing optional compilers."""
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    compiler_errors = {
        "CompilationError", "CompileTimeAssertionFailure", "OutOfResources", "PTXASError",
        "BackendCompilerFailed", "InvalidCxxCompiler", "CppCompileError", "Unsupported",
    }
    return {
        "classification": "compilation" if compiler_errors.intersection(chain) else "runtime",
        "phase": phase,
        "exception_chain": " -> ".join(chain),
        "summary": one_line_error(exc),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop micro-compiler POC for bias-add + LayerNorm + SiLU."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json-output", type=Path, help="atomically write a completed evaluation as JSON")
    parser.add_argument("--peak-tflops", type=float, help="user-supplied target roofline peak")
    parser.add_argument("--peak-bandwidth-gbps", type=float, help="user-supplied target bandwidth")
    parser.add_argument("--deployment-margin", type=float, default=0.02,
                        help="required fractional Triton latency reduction (default: 0.02)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cap warm-up iterations at 3 and measured iterations at 10",
    )
    args = parser.parse_args(argv)
    if args.rows <= 0:
        parser.error("--rows must be greater than zero")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.iters <= 0:
        parser.error("--iters must be greater than zero")
    if (args.peak_tflops is None) != (args.peak_bandwidth_gbps is None):
        parser.error("--peak-tflops and --peak-bandwidth-gbps require both or neither")
    for name in ("peak_tflops", "peak_bandwidth_gbps"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if not math.isfinite(args.deployment_margin) or not 0 <= args.deployment_margin < 1:
        parser.error("--deployment-margin must be finite and in [0, 1)")
    return args


def resolve_device(requested: str) -> torch.device:
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise ValueError("--device cuda was explicitly requested, but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    return torch.device(requested)


def maybe_load_triton(device: torch.device) -> tuple[Any | None, Any | None, str]:
    """Import Triton only for an active CUDA target."""
    global TRITON_KERNEL
    if device.type != "cuda":
        return None, None, "not imported (active target is not CUDA)"
    if importlib.util.find_spec("triton") is None:
        return None, None, "not installed"
    try:
        import triton as triton_module
        import triton.language as tl_module
    except Exception as exc:  # optional dependency: preserve the safe path
        return None, None, f"import failed: {one_line_error(exc)}"

    # Triton resolves language intrinsics through the function's module globals.
    globals()["tl"] = tl_module
    try:
        _triton_fused_impl.__annotations__["BLOCK_SIZE"] = tl_module.constexpr
        TRITON_KERNEL = triton_module.jit(_triton_fused_impl)
    except Exception as exc:
        return triton_module, None, f"kernel construction failed: {one_line_error(exc)}"
    return triton_module, tl_module, str(getattr(triton_module, "__version__", "unknown"))


def _triton_fused_impl(
    x_ptr,
    input_bias_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    hidden_size,
    eps,
    BLOCK_SIZE,
):
    """One Triton program per row; decorated lazily by maybe_load_triton()."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size
    row_offsets = row * hidden_size + columns

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    input_bias = tl.load(input_bias_ptr + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    # Preserve eager's materialized add dtype without a global-memory intermediate.
    values = (x + input_bias).to(x_ptr.dtype.element_ty).to(tl.float32)

    mean = tl.sum(values, axis=0) / hidden_size
    centered = values - mean
    squared = tl.where(mask, centered * centered, 0.0)
    variance = tl.sum(squared, axis=0) / hidden_size
    inv_std = tl.rsqrt(variance + eps)

    gamma = tl.load(gamma_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    affine = centered * inv_std * gamma + beta
    # Eager LayerNorm returns the input dtype before SiLU, including for FP16.
    affine = affine.to(output_ptr.dtype.element_ty).to(tl.float32)

    # Stable sigmoid: the sole exponential has a non-positive argument.
    exp_neg_abs = tl.exp(-tl.abs(affine))
    numerator = tl.where(affine >= 0.0, 1.0, exp_neg_abs)
    silu = affine * numerator / (1.0 + exp_neg_abs)
    tl.store(output_ptr + row_offsets, silu, mask=mask)


def target_fingerprint(device: torch.device, triton_status: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Active target", str(device)),
        ("Python", platform.python_version()),
        ("PyTorch", str(torch.__version__)),
    ]
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        rows.extend(
            [
                ("Device", torch.cuda.get_device_name(index)),
                ("Compute capability", f"{major}.{minor}"),
                ("Total memory", f"{props.total_memory / (1024 ** 3):.2f} GiB"),
                ("CUDA build", str(torch.version.cuda)),
                ("Triton", triton_status),
            ]
        )
    else:
        rows.extend(
            [
                ("Platform", platform.platform()),
                ("Processor", platform.processor() or platform.machine() or "unknown"),
                ("Triton", triton_status),
            ]
        )
    return rows


def make_inputs(
    rows: int, hidden_size: int, dtype: torch.dtype, device: torch.device,
    case: str = "primary_random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    seed_offset = {"random_seed_plus_1": 1, "random_seed_plus_2": 2}.get(case, 0)
    if case not in VALIDATION_CASES:
        raise ValueError(f"unknown validation case: {case}")
    generator.manual_seed(INPUT_SEED + seed_offset)
    x_cpu = torch.randn((rows, hidden_size), generator=generator, dtype=torch.float32)
    input_bias_cpu = 0.2 * torch.randn(
        (hidden_size,), generator=generator, dtype=torch.float32
    )
    gamma_cpu = 1.0 + 0.1 * torch.randn(
        (hidden_size,), generator=generator, dtype=torch.float32
    )
    beta_cpu = 0.1 * torch.randn(
        (hidden_size,), generator=generator, dtype=torch.float32
    )
    if case == "all_zero":
        x_cpu.zero_()
        input_bias_cpu.zero_()
    elif case == "constant_rows":
        x_cpu.copy_(torch.linspace(-3.0, 3.0, rows).unsqueeze(1))
        input_bias_cpu.zero_()  # post-add rows have exactly zero variance
    elif case == "near_zero":
        x_cpu.mul_(1.0e-5)
        input_bias_cpu.mul_(1.0e-6)
    elif case == "alternating_large":
        signs = torch.where(torch.arange(hidden_size) % 2 == 0, 1.0, -1.0)
        x_cpu.copy_(128.0 * signs)  # large but finite in both supported dtypes
    if case in ("all_zero", "constant_rows", "near_zero", "alternating_large", "nontrivial_affine"):
        gamma_cpu = torch.linspace(-1.75, 1.5, hidden_size)
        beta_cpu = 0.7 * torch.cos(torch.arange(hidden_size, dtype=torch.float32))
    return tuple(
        tensor.to(device=device, dtype=dtype).contiguous()
        for tensor in (x_cpu, input_bias_cpu, gamma_cpu, beta_cpu)
    )  # type: ignore[return-value]


def format_target(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", str(target))
    if module in ("", "builtins"):
        return name
    return f"{module}.{name}"


def capture_and_propagate(
    module: nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[GraphModule, str | None]:
    graph_module = symbolic_trace(module)
    try:
        ShapeProp(graph_module).propagate(*inputs)
        return graph_module, None
    except Exception as exc:
        return graph_module, one_line_error(exc)


def is_function_target(node: Node, candidates: Sequence[Any]) -> bool:
    return node.op == "call_function" and any(node.target is item for item in candidates)


def node_argument(node: Node, index: int, name: str, default: Any = None) -> Any:
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def find_fusion_region(
    graph_module: GraphModule,
    inputs: tuple[torch.Tensor, ...],
    hidden_size: int,
    shape_error: str | None = None,
) -> FusionMatch:
    """Validate only the fixed four-input functional workload, not arbitrary graphs.

    Supported spellings: operator.add/torch.add, F.layer_norm/torch.layer_norm,
    and F.silu, with positional or keyword arguments. Module/method/ATen forms
    are deliberately ineligible. Binding operands to the four placeholders
    ensures that the fixed kernel consumes exactly the matched tensors.
    """
    if shape_error is not None:
        return FusionMatch(rejections=(f"ShapeProp failed: {shape_error}",))
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != 4 or len(inputs) != 4:
        return FusionMatch(rejections=("expected exactly four workload input placeholders",))
    x_node, bias_node, gamma_node, beta_node = placeholders
    x = inputs[0]
    expected_shape, expected_dtype = tuple(x.shape), x.dtype
    output = next(node for node in graph_module.graph.nodes if node.op == "output")
    rejections: list[str] = []

    def metadata_error(operand: Any, shape: tuple[int, ...], label: str) -> str | None:
        meta = operand.meta.get("tensor_meta") if isinstance(operand, Node) else None
        if meta is None:
            return f"{label} tensor metadata is missing"
        if tuple(meta.shape) != shape:
            return f"{label} shape {tuple(meta.shape)} != {shape}"
        if meta.dtype != expected_dtype:
            return f"{label} dtype {meta.dtype} != {expected_dtype}"
        return None

    def check(silu: Node) -> tuple[tuple[Node, ...], float | None, str | None]:
        if node_argument(silu, 1, "inplace", False) is not False:
            return (), None, "SiLU must be non-inplace"
        layer_norm = node_argument(silu, 0, "input")
        if not isinstance(layer_norm, Node) or not is_function_target(
            layer_norm, (F.layer_norm, torch.layer_norm)
        ):
            return (), None, "SiLU input edge is not a supported functional LayerNorm"
        add = node_argument(layer_norm, 0, "input")
        if not isinstance(add, Node) or not is_function_target(add, (operator.add, torch.add)):
            return (), None, "LayerNorm input edge is not a supported functional add"
        if set(add.users) != {layer_norm}:
            return (), None, "add intermediate has external users"
        if set(layer_norm.users) != {silu}:
            return (), None, "LayerNorm intermediate has external users"
        left = node_argument(add, 0, "input")
        right = node_argument(add, 1, "other")
        if not ((left is x_node and right is bias_node) or
                (right is x_node and left is bias_node)):
            return (), None, "add operands must be main input and last-dimension input_bias"
        if add.kwargs.get("alpha", 1) != 1 or add.kwargs.get("out") is not None:
            return (), None, "add requires alpha=1 and no out argument"
        if not expected_shape or expected_shape[-1] != hidden_size:
            return (), None, "main input last dimension does not equal hidden_size"
        normalized_shape = node_argument(layer_norm, 1, "normalized_shape")
        if not isinstance(normalized_shape, (tuple, list)) or tuple(normalized_shape) != (hidden_size,):
            return (), None, "LayerNorm normalized_shape must be exactly (hidden_size,)"
        gamma = node_argument(layer_norm, 2, "weight")
        beta = node_argument(layer_norm, 3, "bias")
        if gamma is None or beta is None:
            return (), None, "LayerNorm weight and bias operands must both exist"
        if gamma is not gamma_node or beta is not beta_node:
            return (), None, "LayerNorm weight/bias must bind to workload gamma/beta inputs"
        # Both supported functional LayerNorm forms default to eps=1e-5.
        eps = node_argument(layer_norm, 4, "eps", 1.0e-5)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            return (), None, "LayerNorm epsilon must be a finite positive scalar constant"
        for operand, shape, label in (
            (x_node, expected_shape, "main input"),
            (bias_node, (hidden_size,), "input_bias"),
            (add, expected_shape, "add output"),
            (gamma, (hidden_size,), "LayerNorm weight"),
            (beta, (hidden_size,), "LayerNorm bias"),
            (layer_norm, expected_shape, "LayerNorm output"),
            (silu, expected_shape, "matched output"),
        ):
            error = metadata_error(operand, shape, label)
            if error:
                return (), None, error
        if output.args != (silu,):
            return (), None, "matched SiLU must be the sole workload output"
        return (add, layer_norm, silu), float(eps), None

    for silu in graph_module.graph.nodes:
        if not is_function_target(silu, (F.silu,)):
            continue
        nodes, eps, error = check(silu)
        if error:
            rejections.append(f"{silu.name}: {error}")
            continue  # an invalid earlier chain must not hide a later valid chain
        return FusionMatch(nodes, eps, tuple(rejections))
    return FusionMatch(rejections=tuple(rejections) or (
        "no supported functional add -> layer_norm -> silu chain found (module/method/ATen forms unsupported)",
    ))


def tensor_metadata(node: Node) -> tuple[str, str]:
    metadata = node.meta.get("tensor_meta")
    if metadata is None:
        return "-", "-"
    shape = str(tuple(metadata.shape))
    return shape, str(metadata.dtype).replace("torch.", "")


def graph_rows(graph_module: GraphModule, fusion_nodes: Sequence[Node]) -> list[list[str]]:
    fused = set(fusion_nodes)
    rows: list[list[str]] = []
    for node in graph_module.graph.nodes:
        shape, dtype = tensor_metadata(node)
        rows.append(
            [
                node.name,
                node.op,
                format_target(node.target),
                shape,
                dtype,
                "yes" if node in fused else "no",
            ]
        )
    return rows


def analytical_costs(rows: int, hidden_size: int, element_size: int) -> dict[str, float]:
    elements = rows * hidden_size
    # Approximation per row: bias add H; LayerNorm 8H+2; stable SiLU 5H.
    # exp and rsqrt each count as one scalar operation in this disclosed model.
    flops = rows * (14 * hidden_size + 2)
    eager_elements = 6 * elements + 3 * hidden_size
    fused_elements = 2 * elements + 3 * hidden_size
    eager_bytes = eager_elements * element_size
    fused_bytes = fused_elements * element_size
    return {
        "flops": float(flops),
        "eager_bytes": float(eager_bytes),
        "fused_bytes": float(fused_bytes),
        "avoided_bytes": float(eager_bytes - fused_bytes),
        "eager_ai": flops / eager_bytes,
        "fused_ai": flops / fused_bytes,
        "useful_bytes": float(fused_bytes),
    }


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Explicit (atol, rtol) policy; do not silently relax on a failed case.

    FP16 intentionally allows fused reduction plus transcendental fast math.
    These are the existing tolerances, not a new claim of bitwise equivalence.
    """
    if dtype == torch.float16:
        return 5.0e-3, 5.0e-3
    if dtype == torch.float32:
        return 1.0e-4, 1.0e-4
    raise ValueError(f"no tolerance policy for {dtype}")


def verify_output(candidate: torch.Tensor, reference: torch.Tensor) -> VerificationResult:
    atol, rtol = dtype_tolerances(reference.dtype)

    def rejected(reason: str) -> VerificationResult:
        return VerificationResult(False, math.inf, math.inf, reason, math.inf,
                                  reference.numel(), reference.numel(), atol, rtol)

    if candidate.shape != reference.shape:
        return rejected(f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}")
    if candidate.dtype != reference.dtype:
        return rejected(f"dtype {candidate.dtype} != {reference.dtype}")
    if not bool(torch.isfinite(reference).all().item()):
        return rejected("reference contains non-finite values")
    if not bool(torch.isfinite(candidate).all().item()):
        return rejected("candidate contains non-finite values")

    # Diagnostics use FP64 to avoid overflow/cancellation in finite FP32 extremes.
    ref_abs = reference.double().abs()
    difference = (candidate.double() - reference.double()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    # Conventional relative error: nonzero / zero = inf; exact zero / zero = 0.
    relative = torch.where(difference == 0, 0.0, difference / ref_abs)
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    threshold = atol + rtol * ref_abs
    normalized = difference / threshold
    max_normalized = float(normalized.max().item()) if normalized.numel() else 0.0
    mismatched = int((difference > threshold).sum().item())
    passed = max_normalized <= 1.0 and mismatched == 0
    reason = f"within atol={atol:g}, rtol={rtol:g}" if passed else (
        f"tolerance failure (atol={atol:g}, rtol={rtol:g})"
    )
    return VerificationResult(passed, max_abs, max_rel, reason, max_normalized,
                              mismatched, reference.numel(), atol, rtol)


class ReferenceValidationError(RuntimeError):
    """An invalid oracle is fatal, not an optimization fallback."""


@dataclass
class ValidationSuite:
    module: nn.Module
    inputs: tuple[torch.Tensor, ...]
    device: torch.device

    def validate(self, function: Callable[..., torch.Tensor]) -> VerificationResult:
        x = self.inputs[0]
        atol, rtol = dtype_tolerances(x.dtype)
        aggregate = VerificationResult(True, 0.0, 0.0, f"all {len(VALIDATION_CASES)} validation cases passed",
                                       0.0, 0, 0, atol, rtol)
        # Only primary inputs and one additional case are retained at a time.
        for name in VALIDATION_CASES:
            inputs = self.inputs if name == "primary_random" else make_inputs(
                x.shape[0], x.shape[-1], x.dtype, self.device, name
            )
            reference = self.module(*inputs)
            if reference.shape != x.shape or reference.dtype != x.dtype or not bool(
                torch.isfinite(reference).all().item()
            ):
                raise ReferenceValidationError(f"{name}: eager reference shape/dtype/finite gate failed")
            try:
                candidate = function(*inputs)
                synchronize(self.device)
            except Exception as exc:
                raise RuntimeError(f"validation case {name}: {one_line_error(exc)}") from exc
            result = verify_output(candidate, reference)
            del candidate, reference, inputs
            if not result.passed:
                result.reason = f"{name}: {result.reason}"
                return result
            aggregate.max_abs = max(aggregate.max_abs, result.max_abs)
            aggregate.max_rel = max(aggregate.max_rel, result.max_rel)
            aggregate.max_normalized = max(aggregate.max_normalized, result.max_normalized)
            aggregate.elements += result.elements
        return aggregate


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    # One unconditional untimed call absorbs deferred first-use setup even when
    # the user explicitly requests zero warm-up iterations.
    function()
    for _ in range(warmup):
        function()
    synchronize(device)
    samples_ms: list[float] = []
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        synchronize(device)
        for start, end in zip(starts, ends):
            start.record()
            function()
            end.record()
        synchronize(device)
        samples_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    else:
        for _ in range(iterations):
            start_ns = time.perf_counter_ns()
            function()
            end_ns = time.perf_counter_ns()
            samples_ms.append((end_ns - start_ns) / 1.0e6)
    return float(statistics.median(samples_ms))


def prepare_compiled_path(suite: ValidationSuite, eager_check: VerificationResult) -> CompiledPath:
    eager_call = lambda: suite.module(*suite.inputs)
    detail = "torch.compile is unavailable"
    failure = None
    verification = None
    if hasattr(torch, "compile"):
        try:
            compiled_module = torch.compile(suite.module, fullgraph=True, dynamic=False)
            # First execution and the entire correctness suite are untimed.
            verification = suite.validate(compiled_module)
            if verification.passed:
                return CompiledPath(
                    "torch.compile / Inductor", lambda: compiled_module(*suite.inputs),
                    verification, True, "compiled; " + verification.reason,
                )
            detail = (f"compiled output rejected: {verification.reason}; "
                      f"max normalized={verification.max_normalized:.4g}, "
                      f"mismatches={verification.mismatched} ({verification.mismatch_percent:.4g}%)")
            failure = {"classification": "correctness", "phase": "validation", "summary": detail}
        except ReferenceValidationError:
            raise
        except Exception as exc:
            detail = f"compile/execute failed: {one_line_error(exc)}"
            failure = failure_record(exc, "compilation/validation")
    return CompiledPath("torch.compile fallback: eager", eager_call, eager_check, False,
                        detail, failure, verification)


def all_configs() -> list[KernelConfig]:
    return [
        KernelConfig(block_size, num_warps, num_stages)
        for block_size in SEARCH_BLOCK_SIZES
        for num_warps in SEARCH_NUM_WARPS
        for num_stages in SEARCH_NUM_STAGES
    ]


def triton_eligibility(
    fusion_nodes: Sequence[Node],
    device: torch.device,
    triton_language: Any | None,
    inputs: tuple[torch.Tensor, ...],
    hidden_size: int,
) -> list[str]:
    reasons: list[str] = []
    if not fusion_nodes:
        reasons.append("FX fusion pattern was not eligible")
    if device.type != "cuda":
        reasons.append("active target is CPU, not CUDA")
    if triton_language is None or TRITON_KERNEL is None:
        reasons.append("Triton kernel is unavailable")
    if hidden_size > MAX_TRITON_HIDDEN_SIZE:
        reasons.append(
            f"hidden size {hidden_size} exceeds bounded kernel limit {MAX_TRITON_HIDDEN_SIZE}"
        )
    x, input_bias, gamma, beta = inputs
    if x.dtype not in (torch.float16, torch.float32):
        reasons.append(f"dtype {x.dtype} is unsupported by the Triton POC")
    if x.ndim < 1 or x.shape[-1] != hidden_size:
        reasons.append("input last dimension does not equal hidden size")
    if not x.is_contiguous():
        reasons.append("input tensor is not contiguous")
    for name, parameter in (
        ("input_bias", input_bias),
        ("gamma", gamma),
        ("beta", beta),
    ):
        if parameter.shape != (hidden_size,):
            reasons.append(f"{name} shape is not ({hidden_size},)")
        if not parameter.is_contiguous():
            reasons.append(f"{name} is not contiguous")
        if parameter.dtype != x.dtype or parameter.device != x.device:
            reasons.append(f"{name} dtype/device does not match x")
    return reasons


def launch_triton(
    inputs: tuple[torch.Tensor, ...],
    hidden_size: int,
    eps: float,
    config: KernelConfig,
) -> torch.Tensor:
    if TRITON_KERNEL is None:
        raise RuntimeError("Triton kernel was not constructed")
    x, input_bias, gamma, beta = inputs
    row_count = x.numel() // hidden_size
    output = torch.empty_like(x)
    TRITON_KERNEL[(row_count,)](
        x,
        input_bias,
        gamma,
        beta,
        output,
        hidden_size,
        eps,
        BLOCK_SIZE=config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def tune_triton(
    suite: ValidationSuite,
    hidden_size: int,
    eps: float | None,
    warmup: int,
    iterations: int,
    rounds: int,
    skip_reasons: Sequence[str],
) -> tuple[list[CandidateResult], CandidateResult | None, dict[str, int]]:
    results: list[CandidateResult] = []
    counts = dict.fromkeys(("generated", "pruned", "shape_valid", "executed", "verified",
                            "correctness_rejected", "runtime_failed", "skipped"), 0)
    configs = all_configs()
    random.Random(TUNING_SEED).shuffle(configs)
    for config in configs:
        counts["generated"] += 1
        if config.block_size < hidden_size:
            counts["pruned"] += 1
            results.append(CandidateResult(config, "PRUNED", "BLOCK_SIZE cannot cover hidden_size"))
            continue
        counts["shape_valid"] += 1
        if skip_reasons:
            counts["skipped"] += 1
            results.append(CandidateResult(config, "SKIPPED", "; ".join(skip_reasons)))
            continue
        counts["executed"] += 1
        assert eps is not None  # extracted from the semantically matched FX node
        call = lambda *values, current=config: launch_triton(values, hidden_size, eps, current)
        result = CandidateResult(config, "FAILED", "")
        results.append(result)
        phase = "compilation/validation"
        try:
            result.verification = suite.validate(call)  # JIT and validation are untimed
            if not result.verification.passed:
                result.decision = "REJECTED"
                result.reason = result.verification.reason
                counts["correctness_rejected"] += 1
                continue
            phase = "initial sweep timing"
            result.initial_median_ms = benchmark(
                lambda: call(*suite.inputs), suite.device, warmup, iterations
            )
            result.decision = "VERIFIED"
            result.reason = "all validation cases passed"
            counts["verified"] += 1
        except ReferenceValidationError:
            raise
        except Exception as exc:
            result.reason = one_line_error(exc)
            result.failure = failure_record(exc, phase)
            counts["runtime_failed"] += 1

    verified = [result for result in results if result.decision == "VERIFIED"]
    # Retain three when available, or all if fewer than three survived.
    finalists = sorted(verified, key=lambda result: result.initial_median_ms)[:3]
    for result in finalists:
        result.is_finalist = True
    for round_index in range(rounds):
        order = list(finalists)
        random.Random(TUNING_SEED + 1 + round_index).shuffle(order)
        for result in order:
            if result.decision == "FAILED":
                continue
            try:
                latency = benchmark(
                    lambda current=result.config: launch_triton(
                        suite.inputs, hidden_size, eps, current
                    ), suite.device, warmup, iterations,
                )
                result.finalist_trials.append(latency)
            except Exception as exc:
                result.decision = "FAILED"
                result.reason = f"finalist round {round_index + 1}: {one_line_error(exc)}"
                result.failure = failure_record(exc, f"finalist round {round_index + 1}")
                counts["verified"] -= 1
                counts["runtime_failed"] += 1
    survivors = [result for result in finalists if result.decision == "VERIFIED"]
    best = min(survivors, key=lambda result: result.finalist_median_ms, default=None)
    return results, best, counts


def compare_paths(
    paths: dict[str, Callable[[], torch.Tensor]], device: torch.device,
    warmup: int, iterations: int, rounds: int,
) -> tuple[dict[str, list[float]], dict[str, dict[str, str]]]:
    """Final deployment comparison: interleave distinct paths, never aliases."""
    trials: dict[str, list[float]] = {name: [] for name in paths}
    failures: dict[str, dict[str, str]] = {}
    for round_index in range(rounds):
        order = list(paths)
        random.Random(TUNING_SEED + 1000 + round_index).shuffle(order)
        for name in order:
            if name in failures:
                continue
            try:
                trials[name].append(benchmark(paths[name], device, warmup, iterations))
            except Exception as exc:
                failures[name] = failure_record(exc, f"deployment round {round_index + 1}")
    # Partial measurements of a failed path must never participate in selection.
    return {name: samples for name, samples in trials.items() if name not in failures}, failures


def triton_clears_margin(triton_ms: float, framework_ms: float, margin: float) -> bool:
    return triton_ms < framework_ms and triton_ms <= framework_ms * (1.0 - margin)


def format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def print_candidate_summary(
    results: Sequence[CandidateResult],
    counts: dict[str, int],
    skip_reasons: Sequence[str],
) -> None:
    section("Candidate decisions")
    print(
        ascii_table(
            ("Category", "Count"),
            ((name.replace("_", " ").title(), count) for name, count in counts.items()),
        )
    )
    if skip_reasons:
        print("Triton tuning skipped: " + "; ".join(skip_reasons) + ".")
        print("No Triton configuration was selected.")
    if counts["pruned"]:
        print(
            f"Pruned {counts['pruned']} configurations because BLOCK_SIZE was smaller than "
            "hidden_size; these are not failures."
        )
    visible = [result for result in results if result.decision not in ("PRUNED", "SKIPPED")]
    if visible:
        print(
            ascii_table(
                ("Config", "Decision", "Initial sweep median ms", "Max abs", "Max rel", "Max normalized", "Mismatches"),
                (
                    (
                        result.config.compact(),
                        result.decision,
                        format_metric(result.initial_median_ms),
                        format_metric(result.verification.max_abs if result.verification else None),
                        format_metric(result.verification.max_rel if result.verification else None),
                        format_metric(
                            result.verification.max_normalized if result.verification else None
                        ),
                        (f"{result.verification.mismatched} ({result.verification.mismatch_percent:.3g}%)"
                         if result.verification else "-"),
                    )
                    for result in visible
                ),
            )
        )
        for result in visible:
            if result.decision != "VERIFIED":
                print(f"{result.config.compact()} {result.decision}: {result.reason}")
    finalists = [result for result in results if result.is_finalist]
    if finalists:
        print(ascii_table(
            ("Finalist", "Status", "Rounds", "Finalist median ms", "Min ms", "Max ms"),
            ((result.config.compact(), result.decision, len(result.finalist_trials),
              format_metric(result.finalist_median_ms) if result.decision == "VERIFIED" else "excluded",
              format_metric(min(result.finalist_trials)) if result.finalist_trials else "-",
              format_metric(max(result.finalist_trials)) if result.finalist_trials else "-")
             for result in finalists),
        ))
    if not visible and counts["skipped"]:
        print(
            f"Per-configuration rows omitted because all {counts['skipped']} otherwise shape-valid "
            f"candidates shared the global skip reason; {counts['pruned']} candidates were pruned, "
            "not failed."
        )


def useful_bandwidth_gbps(useful_bytes: float, median_ms: float) -> float:
    if median_ms <= 0.0:
        return math.inf
    return useful_bytes / (median_ms / 1000.0) / 1.0e9


def timing_statistics(trials: Sequence[float]) -> dict[str, Any]:
    return {
        "trial_medians_ms": list(trials), "rounds": len(trials),
        "median_ms": statistics.median(trials) if trials else None,
        "min_ms": min(trials) if trials else None,
        "max_ms": max(trials) if trials else None,
    }


def correctness_metrics(result: VerificationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "passed": result.passed, "max_abs": result.max_abs, "max_rel": result.max_rel,
        "max_normalized_error": result.max_normalized, "mismatch_count": result.mismatched,
        "mismatch_percentage": result.mismatch_percent, "elements_checked": result.elements,
        "atol": result.atol, "rtol": result.rtol, "reason": result.reason,
    }


def source_provenance() -> dict[str, str | None]:
    script = Path(__file__).resolve()
    commit = digest = None
    try:
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
    except OSError:
        pass
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script.parent,
                                capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": commit, "script_sha256": digest}


def json_safe(value: Any) -> Any:
    """JSON has no NaN/Infinity: unavailable or unbounded diagnostics become null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    # Serialize before opening anything; same-directory replace is atomic.
    payload = json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_cli = vars(args).copy()
    if args.quick:
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 10)
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        print(f"ERROR: {exc}; no benchmark was run.", file=sys.stderr)
        return 2
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    triton_module, triton_language, triton_status = maybe_load_triton(device)
    triton_version = getattr(triton_module, "__version__", None)
    del triton_module
    rounds = 2 if args.quick else 5
    tuning_warmup, tuning_iters = min(args.warmup, 10), min(args.iters, 25)

    section("Target fingerprint and reproducibility")
    print(ascii_table(("Property", "Value"), target_fingerprint(device, triton_status)))
    print(f"CLI: device={args.device}, rows={args.rows}, hidden_size={args.hidden_size}, "
          f"dtype={args.dtype}, quick={args.quick}, warmup={args.warmup}, iters={args.iters}, "
          f"deployment_margin={args.deployment_margin:g}, peak_tflops={args.peak_tflops}, "
          f"peak_bandwidth_gbps={args.peak_bandwidth_gbps}, json_output={args.json_output}.")
    print(f"Random seed policy: CPU-generated FP32 inputs cast to target dtype; seed={INPUT_SEED}; "
          "additional random seeds are seed+1 and seed+2; stress cases are deterministic.")
    print("Validation cases: " + ", ".join(VALIDATION_CASES) + ".")
    print("Representative benchmark case: primary_random (same inputs for every path).")
    print(f"Tuning order: shuffle seed={TUNING_SEED}; finalist round seeds=seed+1+round_index; "
          "deployment round seeds=seed+1000+round_index (zero-based rounds).")
    print(f"Tuning selection: initial sweep median; retain top 3 verified (all if fewer); "
          f"{rounds} independently shuffled interleaved finalist rounds; "
          "rank by median of trial medians, with min/max dispersion.")
    print(f"Timing: tuning warmup={tuning_warmup}, samples/trial={tuning_iters}; "
          f"final deployment comparison={rounds} interleaved rounds, "
          f"warmup/trial={args.warmup}, samples/trial={args.iters}; median of trial medians.")
    print("CUDA uses events and device synchronization; CPU uses perf_counter_ns. "
          "JIT/torch.compile compilation, validation, and autotuning overhead are excluded "
          "from steady-state latency; each trial also has one untimed setup call.")
    print(f"Deployment rule: Triton must reduce best framework robust latency by at least "
          f"{100 * args.deployment_margin:g}%; ties retain the framework.")
    atol, rtol = dtype_tolerances(dtype)
    print(f"Correctness policy: atol={atol:g}, rtol={rtol:g}; "
          "abs_error <= atol + rtol * abs(reference), max normalized error <= 1, zero mismatches.")
    print("FP16 policy intentionally allows fused reduction plus transcendental fast math. "
          "Conventional max relative error can exceed rtol near zero while the elementwise gate passes.")
    print("num_stages is a requested experimental dimension, mostly relevant to software-pipelined "
          "loops and matrix-multiplication-style kernels. This kernel has no such loop; "
          "observed differences may be measurement variance unless generated-code differences are established.")

    try:
        inputs = make_inputs(args.rows, args.hidden_size, dtype, device)
    except Exception as exc:
        print(f"ERROR: could not create workload inputs: {one_line_error(exc)}", file=sys.stderr)
        return 1
    module = BiasAddLayerNormSiLU(args.hidden_size).to(device).eval()
    suite = ValidationSuite(module, inputs, device)

    print("\n[1/7] Observe: capture the FX graph and inspect the active target.")
    try:
        graph_module, shape_error = capture_and_propagate(module, inputs)
    except Exception as exc:
        graph_module, shape_error = None, f"FX capture failed: {one_line_error(exc)}"
    fusion = (find_fusion_region(graph_module, inputs, args.hidden_size, shape_error)
              if graph_module is not None else FusionMatch(rejections=(shape_error,)))
    section("Captured FX graph")
    if graph_module is not None:
        print(ascii_table(("Node", "FX op", "Target", "Shape", "Dtype", "Fused"),
                          graph_rows(graph_module, fusion.nodes)))
    if shape_error:
        print(f"Fusion disabled: {shape_error}")

    print("[2/7] Decide: validate graph semantics and fusion eligibility.")
    section("Detected fusion region and analytical cost model")
    print(f"Pattern decision: {'ELIGIBLE' if fusion.nodes else 'NOT ELIGIBLE'} - {fusion.reason}.")
    print("Matcher scope: four-input functional operator.add/torch.add, "
          "F.layer_norm/torch.layer_norm, F.silu; sole tensor output. "
          "Module, method, and ATen forms are unsupported.")
    if fusion.nodes and fusion.rejections:
        print("Earlier rejected chains: " + "; ".join(fusion.rejections))
    costs = analytical_costs(args.rows, args.hidden_size, inputs[0].element_size())
    print(ascii_table(("Analytical estimate", "Value"), (
        ("Scalar operations (approx.)", f"{int(costs['flops']):,}"),
        ("Eager logical tensor traffic", f"{costs['eager_bytes'] / 1e6:.3f} MB"),
        ("Fused logical tensor traffic", f"{costs['fused_bytes'] / 1e6:.3f} MB"),
        ("Avoided intermediate traffic", f"{costs['avoided_bytes'] / 1e6:.3f} MB"),
        ("Eager arithmetic intensity", f"{costs['eager_ai']:.3f} ops/byte"),
        ("Fused arithmetic intensity", f"{costs['fused_ai']:.3f} ops/byte"),
    )))
    print("Convention: one operation each for add/multiply/divide/exp/rsqrt; "
          "reductions, LayerNorm, and transcendental counts are approximate. "
          "Logical traffic counts each parameter vector once; no hardware counters.")
    if args.peak_tflops is None:
        print("low-arithmetic-intensity fusion candidate; physical bottleneck unclassified")
    else:
        ridge = 1000.0 * args.peak_tflops / args.peak_bandwidth_gbps
        prediction = "memory-bound" if costs["eager_ai"] < ridge else "compute-bound"
        print(f"User-supplied roofline ridge point: {ridge:.6g} ops/byte; "
              f"eager analytical intensity={costs['eager_ai']:.6g} ops/byte. "
              f"Roofline prediction: {prediction}; analytical prediction, not profiler evidence. "
              "Physical bottleneck remains unclassified without profiler counters.")

    print("[3/7] Propose: enumerate the bounded 4 x 3 x 3 Triton configuration space.")
    eligibility_reasons = triton_eligibility(
        fusion.nodes, device, triton_language, inputs, args.hidden_size
    )
    if not fusion.nodes:
        eligibility_reasons.append(fusion.reason)
    print("[4/7] Compile/execute: compile outside measured regions.")
    print("[5/7] Verify: every executed candidate must pass every validation case before timing.")
    try:
        reference_check = suite.validate(module)
        candidate_results, best_triton, candidate_counts = tune_triton(
            suite, args.hidden_size, fusion.eps, tuning_warmup, tuning_iters, rounds, eligibility_reasons
        )
        compiled_path = prepare_compiled_path(suite, reference_check)
    except Exception as exc:
        print(f"ERROR: eager reference/validation setup failed: {one_line_error(exc)}", file=sys.stderr)
        return 1
    if not reference_check.passed:
        print(f"ERROR: eager reference failed: {reference_check.reason}", file=sys.stderr)
        return 1

    print("[6/7] Benchmark: interleave distinct verified deployment paths.")
    paths = {"Eager PyTorch": lambda: module(*inputs)}
    if compiled_path.used_compile:
        paths["torch.compile / Inductor"] = compiled_path.function
    if best_triton is not None:
        paths["fused Triton"] = lambda: launch_triton(
            inputs, args.hidden_size, fusion.eps, best_triton.config
        )
    trials, failures = compare_paths(paths, device, args.warmup, args.iters, rounds)
    if "Eager PyTorch" in failures:
        print(f"ERROR: eager steady-state benchmark failed: {failures['Eager PyTorch']['summary']}", file=sys.stderr)
        return 1
    if "torch.compile / Inductor" in failures:
        compiled_path = CompiledPath("torch.compile fallback: eager", paths["Eager PyTorch"],
                                     reference_check, False, failures["torch.compile / Inductor"]["summary"],
                                     failures["torch.compile / Inductor"], compiled_path.verification)
    if "fused Triton" in failures:
        best_triton.decision = "FAILED"
        best_triton.reason = failures["fused Triton"]["summary"]
        best_triton.failure = failures["fused Triton"]
        candidate_counts["verified"] -= 1
        candidate_counts["runtime_failed"] += 1
    print_candidate_summary(candidate_results, candidate_counts, eligibility_reasons)
    print("Verified count excludes candidates later disqualified by benchmark failures; "
          "runtime-failed includes compilation/launch/timing failures. Pruned/skipped were not executed.")

    medians = {name: statistics.median(values) for name, values in trials.items()}
    framework_names = [name for name in ("Eager PyTorch", "torch.compile / Inductor") if name in medians]
    framework = min(framework_names, key=medians.get)  # eager wins framework ties
    winner_name = framework
    if "fused Triton" in medians and triton_clears_margin(
        medians["fused Triton"], medians[framework], args.deployment_margin
    ):
        winner_name = "fused Triton"
    agent_label = ("Agent fused Triton (alias)" if winner_name == "fused Triton"
                   else f"Agent fallback: {winner_name} (alias)")
    checks = {"Eager PyTorch": reference_check, "torch.compile / Inductor": compiled_path.verification}
    if best_triton is not None:
        checks["fused Triton"] = best_triton.verification

    section("Correctness results")
    correctness_rows = []
    correctness_paths = [
        ("Eager reference", "PASS", reference_check),
        (compiled_path.label, "PASS" if compiled_path.used_compile else "FALLBACK", compiled_path.verification),
    ]
    if best_triton is not None:
        correctness_paths.append((f"Triton {best_triton.config.compact()}",
                                  "PASS" if "fused Triton" in medians else "BENCH FAILED",
                                  best_triton.verification))
    correctness_paths.append((agent_label, "PASS (alias)", checks[winner_name]))
    for name, status, check in correctness_paths:
        correctness_rows.append((name, status, format_metric(check.max_abs), format_metric(check.max_rel),
                                 format_metric(check.max_normalized), check.mismatched,
                                 f"{check.mismatch_percent:.4g}%"))
    print(ascii_table(("Path", "Status", "Max abs", "Max rel", "Max normalized", "Mismatches", "Mismatch %"),
                      correctness_rows))
    print(f"All PASS metrics aggregate all {len(VALIDATION_CASES)} cases; atol={atol:g}, rtol={rtol:g}.")
    print("Max relative = abs_error / abs(reference), with 0/0 defined as 0 and nonzero/0 as inf. "
          "It is diagnostic, not the acceptance threshold. Max normalized = abs_error / "
          "(atol + rtol * abs(reference)); PASS requires <= 1 and zero mismatches.")
    print(f"torch.compile: {compiled_path.detail}")

    section("Performance summary: final deployment comparison")
    performance = [(name, name) for name in paths]
    if not compiled_path.used_compile and "torch.compile / Inductor" not in paths:
        performance.append(("torch.compile fallback: eager (alias)", "Eager PyTorch"))
    performance.append((agent_label, winner_name))
    performance_rows = []
    for label, source in performance:
        latency = medians.get(source)
        samples = trials.get(source)
        performance_rows.append((
            label, f"{latency:.6f}" if latency is not None else "excluded",
            f"{min(samples):.6f} / {max(samples):.6f}" if samples else "-",
            f"{useful_bandwidth_gbps(costs['useful_bytes'], latency):.3f}" if latency is not None else "-",
            f"{medians['Eager PyTorch'] / latency:.2f}x" if latency else "-",
        ))
    print(ascii_table(("Path", "Robust median ms", "Trial min / max ms", "Est. useful GB/s", "Speedup"),
                      performance_rows))
    print("Estimated Effective Bandwidth uses input + output + one logical copy each of "
          "input_bias, gamma, and beta for every path. This useful-byte numerator is analytical; "
          "the ratio is not measured HBM bandwidth. Alias rows reuse source measurements.")
    for name, failure in failures.items():
        print(f"{name} excluded from final comparison: {failure['classification']} "
              f"failure during {failure['phase']}: {failure['summary']}")

    print("[7/7] Select/fallback: apply the deployment margin to robust medians.")
    section("Selected configuration or fallback")
    fallback_reason = None
    if "fused Triton" in failures:
        fallback_reason = "Final Triton benchmark failed: " + failures["fused Triton"]["summary"]
    elif "fused Triton" in medians and winner_name != "fused Triton":
        fallback_reason = (f"Triton statistically insufficient under the {100 * args.deployment_margin:g}% "
                           "deployment margin heuristic; this is not a significance test")
    elif best_triton is None:
        fallback_reason = "; ".join(eligibility_reasons) or "no verified Triton finalist survived"
    selection_rationale = (
        f"Triton cleared the {100 * args.deployment_margin:g}% margin against {framework} "
        "using median of interleaved deployment trial medians"
        if winner_name == "fused Triton" else f"Retain {framework}: {fallback_reason}"
    )
    if best_triton is not None:
        print(f"Best observed verified configuration from tuning: {best_triton.config.compact()}; "
              f"initial sweep median={best_triton.initial_median_ms:.6f} ms; "
              f"finalist median={best_triton.finalist_median_ms:.6f} ms.")
    if "fused Triton" in failures:
        print("Final Triton benchmark failed; excluded from deployment.")
    elif "fused Triton" in medians and winner_name != "fused Triton":
        print(f"Triton result is statistically insufficient to justify deployment under the "
              f"{100 * args.deployment_margin:g}% margin rule; retain {framework}. "
              "This is a practical margin heuristic, not a significance test.")
    elif best_triton is None:
        reason = "; ".join(eligibility_reasons) or "no verified Triton finalist survived"
        print(f"Triton fallback reason: {reason}.")
    print("Selected/deployed Triton configuration: " + (
        best_triton.config.compact() if winner_name == "fused Triton" else "none"
    ) + ".")
    print(f"Selected/deployed implementation: {winner_name}; "
          f"robust median={medians[winner_name]:.6f} ms.")
    print("Scope: bounded launch-parameter autotuning of a handwritten fixed Triton kernel body; "
          "inference-only/no backward support; no arbitrary program synthesis; "
          "no multi-vendor backend; no persistent kernel database; no profiler-counter evidence; "
          "no production dynamic-shape support.")
    if args.json_output is not None:
        hardware = {
            "device": str(device), "platform": platform.platform(),
            "gpu_name": None, "compute_capability": None, "memory_bytes": None,
            "cuda_version": torch.version.cuda, "pytorch_version": str(torch.__version__),
            "triton_version": triton_version, "triton_status": triton_status,
            "python_version": platform.python_version(),
        }
        if device.type == "cuda":
            hardware.update(
                gpu_name=torch.cuda.get_device_name(device),
                compute_capability=list(torch.cuda.get_device_capability(device)),
                memory_bytes=torch.cuda.get_device_properties(device).total_memory,
            )
        graph_nodes = []
        if graph_module is not None:
            for node in graph_module.graph.nodes:
                metadata = node.meta.get("tensor_meta")
                graph_nodes.append({
                    "name": node.name, "op": node.op, "target": format_target(node.target),
                    "input_edges": [parent.name for parent in node.all_input_nodes],
                    "shape": list(metadata.shape) if metadata is not None else None,
                    "dtype": str(metadata.dtype) if metadata is not None else None,
                    "in_fusion_boundary": node in fusion.nodes,
                })
        path_results = {}
        for key, name in (("eager", "Eager PyTorch"), ("torch_compile", "torch.compile / Inductor"),
                          ("triton", "fused Triton")):
            latency = medians.get(name)
            check = checks.get(name)
            if key == "torch_compile" and not compiled_path.used_compile:
                check = compiled_path.attempt_verification
            path_results[key] = {
                "implementation": name, "available_and_verified": name in medians,
                "latency_ms": latency, "timing": timing_statistics(trials.get(name, [])),
                "correctness": correctness_metrics(check),
                "estimated_effective_bandwidth_gbps": (
                    useful_bandwidth_gbps(costs["useful_bytes"], latency) if latency is not None else None
                ),
                "speedup_over_eager": medians["Eager PyTorch"] / latency if latency else None,
                "failure": failures.get(name) or (compiled_path.failure if key == "torch_compile" else None),
                "fallback_path": "eager" if key == "torch_compile" and not compiled_path.used_compile else None,
                "fallback_reason": compiled_path.detail if key == "torch_compile" and not compiled_path.used_compile else None,
            }
        report = {
            "schema_version": "1.0", "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_completed": True,
            "cli_arguments": requested_cli, "effective_cli_arguments": vars(args),
            "provenance": source_provenance(), "hardware": hardware,
            "random_seed": INPUT_SEED,
            "reproducibility": {
                "input_seed": INPUT_SEED, "additional_input_seeds": [INPUT_SEED + 1, INPUT_SEED + 2],
                "input_generation": "CPU FP32, then cast to target dtype/device",
                "tuning_seed": TUNING_SEED, "finalist_round_seeds": [TUNING_SEED + 1 + i for i in range(rounds)],
                "deployment_round_seeds": [TUNING_SEED + 1000 + i for i in range(rounds)],
                "validation_cases": list(VALIDATION_CASES), "benchmark_case": "primary_random",
            },
            "graph": {"nodes": graph_nodes, "shape_propagation_error": shape_error,
                      "fusion_boundary": [node.name for node in fusion.nodes],
                      "fusion_eligible": bool(fusion.nodes), "eps": fusion.eps,
                      "decision": fusion.reason, "rejected_chains": fusion.rejections},
            "analytical_model": {
                **costs, "all_quantities_are_estimates": True,
                "operation_convention": "14*hidden_size+2 ops/row; exp/rsqrt count as one; approximate reductions",
                "useful_byte_definition": "input + output + one logical copy each of input_bias, gamma, beta",
                "roofline_ridge_ops_per_byte": ridge if args.peak_tflops is not None else None,
                "roofline_prediction": prediction if args.peak_tflops is not None else None,
                "physical_bottleneck": "unclassified; no profiler-counter evidence",
            },
            "candidate_counts": {**candidate_counts, "failed": candidate_counts["runtime_failed"],
                                 "incorrect": candidate_counts["correctness_rejected"],
                                 "compilation_failed": sum(result.failure is not None and
                                                           result.failure["classification"] == "compilation"
                                                           for result in candidate_results)},
            "candidate_count_semantics": "executed counts launch attempts; runtime_failed/failed include compilation failures; verified excludes later disqualifications",
            "candidates": [{
                "configuration": asdict(result.config),
                "launch_attempted": result.decision not in ("PRUNED", "SKIPPED"),
                "launched": (True if result.verification is not None else
                             False if result.decision in ("PRUNED", "SKIPPED") or
                             (result.failure and result.failure["classification"] == "compilation") else None),
                "decision": result.decision, "reason": result.reason, "failure": result.failure,
                "correctness": correctness_metrics(result.verification),
                "initial_sweep": timing_statistics(
                    [result.initial_median_ms] if result.initial_median_ms is not None else []
                ),
                "is_finalist": result.is_finalist, "finalist": timing_statistics(result.finalist_trials),
            } for result in candidate_results],
            "timing_protocol": {
                "statistic": "median of trial medians", "tuning_warmup": tuning_warmup,
                "tuning_iterations": tuning_iters, "finalist_rounds": rounds, "finalist_count_limit": 3,
                "deployment_rounds": rounds, "deployment_warmup": args.warmup, "deployment_iterations": args.iters,
                "clock": "CUDA events with synchronization" if device.type == "cuda" else "perf_counter_ns",
                "compilation_validation_autotuning_excluded": True,
            },
            "paths": path_results,
            "selection": {
                "implementation": winner_name,
                "configuration": asdict(best_triton.config) if winner_name == "fused Triton" else None,
                "best_observed_tuning_configuration": asdict(best_triton.config) if best_triton else None,
                "tuning_rule": "shuffled initial sweep; top three verified; median of interleaved finalist trial medians",
                "deployment_margin": args.deployment_margin, "rationale": selection_rationale,
                "fallback_path": winner_name if winner_name != "fused Triton" else None,
                "fallback_reason": fallback_reason,
            },
            "nonfinite_encoding": "non-finite diagnostics are null, never NaN or Infinity",
            "failure_classification": "known compiler exception chains => compilation; other execution exceptions => runtime",
        }
        try:
            write_json_atomic(args.json_output, report)
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR: JSON output was not written: {one_line_error(exc)}", file=sys.stderr)
            return 1
        print(f"Completed evaluation JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
