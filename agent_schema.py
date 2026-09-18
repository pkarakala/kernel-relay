"""Versioned, JSON-only agent/evaluator contracts; no accelerator imports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "2.0"
MAX_SOURCE_BYTES = 65536
CANDIDATE_TYPES = ("eager", "pytorch_source", "triton_source")


def json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def strict_loads(payload: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(payload, parse_constant=reject, object_pairs_hook=unique_pairs)


class JsonRecord:
    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str):
        return cls.from_dict(strict_loads(payload))


@dataclass(frozen=True)
class HardwareFingerprint(JsonRecord):
    device: str
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HardwareFingerprint:
        return cls(**value)


@dataclass(frozen=True)
class AgentProposal(JsonRecord):
    candidate_id: str
    candidate_type: str
    strategy: str
    source: str | None = None
    launch_parameters: dict[str, int] = field(default_factory=dict)
    expected_optimization: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposal schema_version")
        for name in ("candidate_id", "strategy", "expected_optimization"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"{name} must be a string of at most 4096 characters")
        if not self.candidate_id or not self.strategy:
            raise ValueError("candidate_id and strategy must not be empty")
        if self.candidate_type not in CANDIDATE_TYPES:
            raise ValueError(f"unsupported candidate_type: {self.candidate_type}")
        if self.candidate_type == "eager":
            if self.source is not None or self.launch_parameters:
                raise ValueError("eager proposals cannot supply source or launch parameters")
        elif not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source candidates require nonempty source")
        if self.source is not None and len(self.source.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("candidate source exceeds 64 KiB")
        if not isinstance(self.launch_parameters, dict):
            raise ValueError("launch_parameters must be an object")
        if set(self.launch_parameters) - {"block_size", "num_warps", "num_stages"}:
            raise ValueError("unknown launch parameter")
        for name, value in self.launch_parameters.items():
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError(f"invalid integer launch parameter: {name}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentProposal:
        if not isinstance(value, dict):
            raise ValueError("proposal must be a JSON object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"invalid proposal fields: {exc}") from exc


@dataclass(frozen=True)
class AgentRequest(JsonRecord):
    task: dict[str, Any]
    hardware: HardwareFingerprint
    history: list[dict[str, Any]]
    baseline_results: list[dict[str, Any]]
    remaining_budget: dict[str, Any]
    round_index: int
    candidate_contract: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentRequest:
        values = dict(value)
        if values.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("unsupported request schema_version")
        values["hardware"] = HardwareFingerprint.from_dict(values["hardware"])
        return cls(**values)


@dataclass
class EvaluationResult(JsonRecord):
    status: str
    hardware_fingerprint: HardwareFingerprint
    compiled: bool | None = None
    correct: bool | None = None
    compile_error: str | None = None
    runtime_error: str | None = None
    correctness_error: str | None = None
    reason: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    speedup: float | None = None
    verification: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    wall_time_seconds: float = 0.0
    source_sha256: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationResult:
        values = dict(value)
        values["hardware_fingerprint"] = HardwareFingerprint.from_dict(values["hardware_fingerprint"])
        return cls(**values)


@dataclass
class BaselineResult(JsonRecord):
    name: str
    evaluation: EvaluationResult

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BaselineResult:
        return cls(value["name"], EvaluationResult.from_dict(value["evaluation"]))


@dataclass
class OptimizationRound(JsonRecord):
    round_index: int
    request: AgentRequest
    proposal: AgentProposal | None
    evaluation: EvaluationResult
    reward: float | None
    agent_error: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptimizationRound:
        values = dict(value)
        values["request"] = AgentRequest.from_dict(values["request"])
        values["proposal"] = AgentProposal.from_dict(values["proposal"]) if values["proposal"] is not None else None
        values["evaluation"] = EvaluationResult.from_dict(values["evaluation"])
        return cls(**values)


@dataclass
class RunMetrics(JsonRecord):
    compile_success_rate: float | None
    correctness_rate: float | None
    best_verified_candidate_latency_ms: float | None
    best_baseline_latency_ms: float | None
    speedup_over_eager: float | None
    speedup_over_best_compiler_baseline: float | None
    speedup_over_best_baseline: float | None
    number_of_optimization_rounds: int
    number_of_correct_candidates: int
    number_of_faster_than_baseline_candidates: int
    total_optimization_wall_time_seconds: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetrics:
        return cls(**value)


@dataclass
class OptimizationTrace(JsonRecord):
    task_id: str
    task: dict[str, Any]
    target: HardwareFingerprint
    baseline_results: list[BaselineResult]
    rounds: list[OptimizationRound] = field(default_factory=list)
    selected_round: int | None = None
    final_result: dict[str, Any] = field(default_factory=dict)
    metrics: RunMetrics | None = None
    gpu_performance_evaluated: bool = False
    evaluation_completed: bool = False
    stop_reason: str = ""
    timestamp: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    isolation: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptimizationTrace:
        values = dict(value)
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported trace schema_version")
        values["target"] = HardwareFingerprint.from_dict(values["target"])
        values["baseline_results"] = [BaselineResult.from_dict(item) for item in values["baseline_results"]]
        values["rounds"] = [OptimizationRound.from_dict(item) for item in values["rounds"]]
        values["metrics"] = RunMetrics.from_dict(values["metrics"]) if values["metrics"] is not None else None
        return cls(**values)
