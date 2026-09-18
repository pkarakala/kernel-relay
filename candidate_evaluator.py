"""Trusted target boundary: proposals are data, measurements belong to workers."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from typing import Protocol

import agentic_kernel_compiler as compiler
from agent_schema import AgentProposal, BaselineResult, EvaluationResult, HardwareFingerprint, strict_loads
from candidate_policy import validate_source
from evaluation_isolation import STRICT, TRUSTED_FIXTURE_COLAB, isolation_report, worker_command, worker_environment
from agent_backends.mock import validate_trusted_mock_proposal
from optimization_tasks import OptimizationTask, TaskSettings


ROOT = Path(__file__).resolve().parent
WORKER_FILES = (
    "agentic_kernel_compiler.py", "agent_schema.py", "optimization_tasks.py",
    "candidate_policy.py", "evaluation_isolation.py", "candidate_worker.py",
    "agent_backends/__init__.py", "agent_backends/base.py", "agent_backends/mock.py",
)


class CandidateEvaluator(Protocol):
    target: HardwareFingerprint

    def baselines(self, timeout_seconds: float) -> list[BaselineResult]:
        ...

    def evaluate(self, proposal: AgentProposal, timeout_seconds: float) -> EvaluationResult:
        ...


def verified_latency(result: EvaluationResult) -> float | None:
    import math

    value = result.latency_ms
    if (result.status == "measured" and result.compiled is True and result.correct is True
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0):
        return value
    return None


class SubprocessEvaluator:
    def __init__(self, task: OptimizationTask, settings: TaskSettings, target: HardwareFingerprint,
                 *, warmup: int = 100, iterations: int = 100, timing_rounds: int = 5,
                 max_autotune: bool = True, isolation_mode: str = STRICT):
        self.task, self.settings, self.target = task, settings, target
        self.isolation = isolation_report(isolation_mode)
        if isolation_mode == TRUSTED_FIXTURE_COLAB and task.task_id != "bias_layernorm_silu":
            raise ValueError("trusted-fixture mode supports only the audited bias_layernorm_silu task")
        self.timing = {"warmup": warmup, "iterations": iterations, "timing_rounds": timing_rounds,
                       "max_autotune": max_autotune}
        self.baseline_results: list[BaselineResult] = []

    def _run(self, operation: str, timeout_seconds: float, proposal: AgentProposal | None = None,
             baseline_name: str | None = None) -> dict:
        payload = {"operation": operation, "task_id": self.task.task_id,
                   "task_settings": asdict(self.settings), "target": self.target.to_dict(),
                   "timing": self.timing, "proposal": proposal.to_dict() if proposal else None,
                   "baseline_name": baseline_name, "isolation_mode": self.isolation["mode"]}
        with tempfile.TemporaryDirectory(prefix=".agent-trial-", dir=ROOT) as temporary:
            directory = Path(temporary).resolve()
            for name in WORKER_FILES:
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, directory / name)
            compiler.write_json_atomic(directory / "request.json", payload)
            with (directory / "worker.log").open("w+") as log:
                command = worker_command(directory, self.isolation["mode"])
                process = subprocess.Popen(command, cwd=directory,
                                           env=worker_environment(directory, self.isolation["mode"]),
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                           start_new_session=True, close_fds=True)
                try:
                    process.wait(timeout=max(0.001, timeout_seconds))
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process.returncode != 0:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 4000))
                    raise RuntimeError(f"worker exited {process.returncode}: {log.read()}")
            response = directory / "response.json"
            if response.stat().st_size > 1024 * 1024:
                raise RuntimeError("oversized worker response")
            result = strict_loads(response.read_text())
            if "fatal_reference_error" in result:
                raise compiler.ReferenceValidationError(result["fatal_reference_error"])
            if "infrastructure_error" in result:
                raise RuntimeError(result["infrastructure_error"])
            return result

    def baselines(self, timeout_seconds: float) -> list[BaselineResult]:
        deadline = time.monotonic() + timeout_seconds
        self.baseline_results = []
        for name in ("eager", "inductor_default", "inductor_max_autotune"):
            remaining = deadline - time.monotonic()
            if name != "eager" and self.target.device == "cpu":
                result = EvaluationResult("unavailable", self.target, reason="CPU infrastructure mode measures eager only")
            elif name == "inductor_max_autotune" and not self.timing["max_autotune"]:
                result = EvaluationResult("unavailable", self.target, reason="max-autotune disabled explicitly")
            elif remaining <= 0:
                result = EvaluationResult("unavailable", self.target, reason="baseline time budget exhausted")
            else:
                try:
                    response = self._run("baselines", remaining / (2 if name == "inductor_default" else 1),
                                         baseline_name=name)
                    result = BaselineResult.from_dict(response["baselines"][0]).evaluation
                except compiler.ReferenceValidationError:
                    raise
                except subprocess.TimeoutExpired:
                    result = EvaluationResult("timeout", self.target, runtime_error="baseline deadline exceeded")
                except Exception as exc:
                    result = EvaluationResult("infrastructure_error", self.target,
                                              runtime_error=compiler.one_line_error(exc))
            result.diagnostics.setdefault("isolation", copy.deepcopy(self.isolation))
            self.baseline_results.append(BaselineResult(name, result))
        eager = self.baseline_results[0].evaluation
        if verified_latency(eager) is None:
            raise compiler.ReferenceValidationError(f"eager baseline failed: {eager.to_json()}")
        return copy.deepcopy(self.baseline_results)

    def evaluate(self, proposal: AgentProposal, timeout_seconds: float) -> EvaluationResult:
        started = time.monotonic()
        result = EvaluationResult("invalid_proposal", self.target)
        try:
            proposal = AgentProposal.from_dict(proposal.to_dict())
            if self.isolation["mode"] == TRUSTED_FIXTURE_COLAB:
                validate_trusted_mock_proposal(proposal, self.settings.hidden_size)
            if proposal.candidate_type == "eager":
                result = copy.deepcopy(next(item.evaluation for item in self.baseline_results if item.name == "eager"))
                result.diagnostics["baseline_alias"] = "eager; no duplicate timing or invented gain"
            else:
                validate_source(proposal.source, proposal.candidate_type)
                unsupported = None
                if proposal.candidate_type == "triton_source":
                    if self.target.device == "cpu":
                        unsupported = "CPU infrastructure mode: Triton execution skipped; no GPU performance measured"
                    elif self.settings.hidden_size > compiler.MAX_TRITON_HIDDEN_SIZE:
                        unsupported = "POC Triton hidden-size limit is 256"
                    elif self.target.properties.get("Triton", "").startswith(("not installed", "import failed", "kernel construction failed")):
                        unsupported = "Triton unavailable: " + self.target.properties["Triton"]
                if unsupported:
                    result = EvaluationResult("skipped", self.target, reason=unsupported)
                else:
                    response = self._run("candidate", timeout_seconds, proposal)
                    result = EvaluationResult.from_dict(response["evaluation"])
        except compiler.ReferenceValidationError:
            raise
        except SyntaxError as exc:
            result = EvaluationResult("compile_error", self.target, compiled=False,
                                      compile_error=compiler.one_line_error(exc))
        except (ValueError, TypeError) as exc:
            result = EvaluationResult("invalid_proposal", self.target, compiled=False,
                                      compile_error=compiler.one_line_error(exc))
        except subprocess.TimeoutExpired:
            result = EvaluationResult("timeout", self.target, runtime_error=f"trial exceeded {timeout_seconds:.3g}s")
        except Exception as exc:
            result = EvaluationResult("infrastructure_error", self.target, runtime_error=compiler.one_line_error(exc))
        result.wall_time_seconds = time.monotonic() - started
        result.diagnostics.setdefault("isolation", copy.deepcopy(self.isolation))
        source = getattr(proposal, "source", None)
        result.source_sha256 = hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None
        valid_baselines = [verified_latency(item.evaluation) for item in self.baseline_results]
        valid_baselines = [value for value in valid_baselines if value is not None]
        result.baseline_latency_ms = min(valid_baselines, default=None)
        latency = verified_latency(result)
        result.speedup = result.baseline_latency_ms / latency if latency and result.baseline_latency_ms else None
        return result
