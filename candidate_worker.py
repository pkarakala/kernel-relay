"""Private subprocess entry point; generated code never runs in the controller."""

from __future__ import annotations

import builtins
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import types
from typing import Callable

import torch

import agentic_kernel_compiler as compiler
from agent_schema import AgentProposal, BaselineResult, EvaluationResult, HardwareFingerprint
from candidate_policy import BUILTINS, IMPORTS, validate_source
from evaluation_isolation import STRICT, TRUSTED_FIXTURE_COLAB, isolation_report, restrict_linux_filesystem
from agent_backends.mock import validate_trusted_mock_proposal
from optimization_tasks import TaskSettings, load_task


class CandidateCorrectnessError(Exception):
    """Interface or input immutability violation, not a valid timing candidate."""


def load_candidate(proposal: AgentProposal, directory: Path) -> Callable:
    source_path = directory / "candidate.py"
    source_path.write_text(proposal.source)
    tree = validate_source(proposal.source, proposal.candidate_type)
    module = types.ModuleType("evaluated_candidate")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in IMPORTS:
            raise ImportError(f"candidate import denied: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {name: getattr(builtins, name) for name in BUILTINS}
    module.__dict__["__builtins__"]["__import__"] = restricted_import
    exec(compile(tree, str(source_path), "exec", dont_inherit=True), module.__dict__)
    return module.run


def evaluate_function(factory, suite, target, settings, guarded=False) -> EvaluationResult:
    result = EvaluationResult("compile_error", target, compiled=False)
    started = time.monotonic()
    phase = "compile"
    try:
        function = factory()
        result.compiled = True
        phase = "execute/verify"

        def verified_call(*inputs):
            values = tuple(tensor.clone() for tensor in inputs) if guarded else inputs
            output = function(*values)
            compiler.synchronize(suite.device)
            if not isinstance(output, torch.Tensor) or output.device != inputs[0].device:
                raise CandidateCorrectnessError("run must return a tensor on the input device")
            if guarded and any(not torch.equal(original, passed) for original, passed in zip(inputs, values)):
                raise CandidateCorrectnessError("candidate modified an input tensor")
            return output

        verification = suite.validate(verified_call)
        result.correct = verification.passed
        result.verification = compiler.correctness_metrics(verification)
        result.max_abs_error = verification.max_abs
        result.max_rel_error = verification.max_rel
        if not verification.passed:
            result.status = "incorrect"
            result.correctness_error = verification.reason
            return result
        phase = "benchmark"
        snapshots = tuple(tensor.clone() for tensor in suite.inputs) if guarded else ()
        trials, failures = compiler.compare_paths(
            {"path": lambda: function(*suite.inputs)}, suite.device,
            settings["warmup"], settings["iterations"], settings["timing_rounds"],
        )
        if failures:
            raise RuntimeError(failures["path"]["summary"])
        if guarded and any(not torch.equal(original, current) for original, current in zip(snapshots, suite.inputs)):
            raise CandidateCorrectnessError("candidate modified benchmark inputs")
        timing = compiler.timing_statistics(trials["path"])
        latency = timing["median_ms"]
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("timing did not produce a finite positive latency")
        phase = "post-timing verification"
        verification = suite.validate(verified_call)
        if not verification.passed:
            result.correct = False
            result.status = "incorrect"
            result.correctness_error = "post-timing: " + verification.reason
            result.verification = compiler.correctness_metrics(verification)
            result.max_abs_error = verification.max_abs
            result.max_rel_error = verification.max_rel
            return result
        result.status = "measured"
        result.latency_ms = latency
        result.timing = {**timing, **settings, "statistic": "median of trial medians",
                         "clock": "synchronized CUDA events" if suite.device.type == "cuda" else "perf_counter_ns",
                         "compilation_validation_autotuning_excluded": True}
    except compiler.ReferenceValidationError:
        raise
    except Exception as exc:
        failure = compiler.failure_record(exc, phase)
        chain = exc
        incorrect_interface = False
        while chain is not None:
            incorrect_interface |= isinstance(chain, CandidateCorrectnessError)
            chain = chain.__cause__
        if incorrect_interface:
            result.status, result.correct = "incorrect", False
            result.correctness_error = failure["summary"]
        elif phase == "compile" or failure["classification"] == "compilation":
            result.status, result.compiled = "compile_error", False
            result.compile_error = failure["summary"]
        else:
            result.status = "runtime_error"
            result.runtime_error = failure["summary"]
        result.diagnostics["failure"] = failure
    finally:
        result.wall_time_seconds = time.monotonic() - started
    return result


def evaluate_baselines(suite, target, settings, baseline_name) -> list[BaselineResult]:
    results = []
    for name, mode in (("eager", None), ("inductor_default", "default"),
                       ("inductor_max_autotune", "max-autotune")):
        if name != baseline_name:
            continue
        unavailable = None
        if mode is not None and suite.device.type != "cuda":
            unavailable = "CPU infrastructure mode measures eager only; CUDA compiler baseline skipped"
        elif mode is not None and not callable(getattr(torch, "compile", None)):
            unavailable = "torch.compile is unavailable"
        elif mode == "max-autotune" and not settings["max_autotune"]:
            unavailable = "max-autotune disabled explicitly"
        if unavailable:
            results.append(BaselineResult(name, EvaluationResult("unavailable", target, reason=unavailable)))
            continue
        if mode == "max-autotune":
            try:
                from torch._inductor import list_mode_options

                if mode not in list_mode_options():
                    results.append(BaselineResult(name, EvaluationResult("unavailable", target, reason="mode not supported")))
                    continue
            except (ImportError, AttributeError):
                pass
            except Exception as exc:
                results.append(BaselineResult(name, EvaluationResult(
                    "unavailable", target, reason="mode discovery failed: " + compiler.one_line_error(exc))))
                continue
        factory = (lambda: suite.module) if mode is None else (
            lambda current=mode: torch.compile(suite.module, fullgraph=True, dynamic=False, mode=current)
        )
        results.append(BaselineResult(name, evaluate_function(factory, suite, target, settings)))
    return results


@torch.no_grad()
def main() -> None:
    directory = Path.cwd().resolve()
    payload = json.loads((directory / "request.json").read_text())
    target = HardwareFingerprint.from_dict(payload["target"])
    response = {}
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        isolation = isolation_report(payload.get("isolation_mode", STRICT))
        if isolation["mode"] == TRUSTED_FIXTURE_COLAB:
            if payload["task_id"] != "bias_layernorm_silu":
                raise ValueError("trusted-fixture mode rejects unaudited task")
            if payload["operation"] == "candidate":
                validate_trusted_mock_proposal(AgentProposal.from_dict(payload["proposal"]),
                                               payload["task_settings"]["hidden_size"])
            elif payload["operation"] != "baselines" or payload["baseline_name"] not in (
                "eager", "inductor_default", "inductor_max_autotune"
            ) or payload.get("proposal") is not None:
                raise ValueError("trusted-fixture mode rejects unaudited operation")
            isolation["enforcement"] = "exact fixture match; sanitized subprocess with timeout; no OS sandbox"
        else:
            isolation["enforcement"] = restrict_linux_filesystem(directory)
        torch.set_num_threads(1)
        device = compiler.resolve_device(target.device)
        task = load_task(payload["task_id"])
        suite = task.prepare(TaskSettings(**payload["task_settings"]), device)
        reference = suite.validate(suite.module)
        if not reference.passed:
            raise compiler.ReferenceValidationError(reference.reason)
        if payload["operation"] == "baselines":
            baselines = evaluate_baselines(suite, target, payload["timing"], payload["baseline_name"])
            for baseline in baselines:
                baseline.evaluation.diagnostics["isolation"] = isolation
            response["baselines"] = [baseline.to_dict() for baseline in baselines]
        else:
            proposal = AgentProposal.from_dict(payload["proposal"])
            if proposal.candidate_type == "triton_source" and importlib.util.find_spec("triton") is None:
                result = EvaluationResult("skipped", target, reason="Triton is not installed")
            else:
                def factory():
                    entry = load_candidate(proposal, directory)
                    return lambda *values: entry(*values, eps=suite.module.eps,
                                                launch_parameters=proposal.launch_parameters)

                result = evaluate_function(factory, suite, target, payload["timing"], guarded=True)
            result.diagnostics["isolation"] = isolation
            response["evaluation"] = result.to_dict()
    except compiler.ReferenceValidationError as exc:
        response["fatal_reference_error"] = str(exc)
    except Exception as exc:
        response["infrastructure_error"] = compiler.one_line_error(exc)
    compiler.write_json_atomic(directory / "response.json", response)


if __name__ == "__main__":
    main()
