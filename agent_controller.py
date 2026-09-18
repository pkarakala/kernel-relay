"""Bounded feedback orchestration. Agents cannot mutate the retained history."""

from __future__ import annotations

import copy
import math
import time
from datetime import datetime, timezone

import agentic_kernel_compiler as compiler
from agent_backends.base import AgentBackend, AgentExhausted
from agent_backends.mock import MockAgentBackend
from evaluation_isolation import TRUSTED_FIXTURE_COLAB
from agent_metrics import best_baseline, calculate_metrics, reward_function
from agent_schema import AgentProposal, AgentRequest, EvaluationResult, OptimizationRound, OptimizationTrace
from candidate_evaluator import CandidateEvaluator, verified_latency
from candidate_policy import ATTRIBUTES, IMPORTS


def run_optimization(task: dict, agent: AgentBackend, evaluator: CandidateEvaluator, *,
                     max_rounds: int = 3, trial_timeout_seconds: float = 120,
                     max_seconds: float = 1800, baseline_timeout_seconds: float = 600) -> OptimizationTrace:
    if type(max_rounds) is not int or not 1 <= max_rounds <= 64:
        raise ValueError("max_rounds must be an integer in [1, 64]")
    if any(not math.isfinite(value) or value <= 0 for value in
           (trial_timeout_seconds, max_seconds, baseline_timeout_seconds)):
        raise ValueError("time budgets must be positive finite values")
    isolation = copy.deepcopy(getattr(evaluator, "isolation", {}))
    if isolation.get("mode") == TRUSTED_FIXTURE_COLAB and (
        type(agent) is not MockAgentBackend or agent.proposals is not None
    ):
        raise ValueError("trusted-fixture mode requires the built-in mock agent without custom proposals")
    started = time.monotonic()
    deadline = started + max_seconds
    baselines = evaluator.baselines(min(baseline_timeout_seconds, max_seconds))
    baseline = best_baseline(baselines)
    if baseline is None:
        raise compiler.ReferenceValidationError("no measured verified framework baseline")
    trace = OptimizationTrace(
        task["task_id"], copy.deepcopy(task), evaluator.target, baselines,
        isolation=isolation,
        timestamp=datetime.now(timezone.utc).isoformat(),
        budget={"max_rounds": max_rounds, "trial_timeout_seconds": trial_timeout_seconds,
                "max_seconds": max_seconds, "baseline_timeout_seconds": baseline_timeout_seconds},
    )
    trace.stop_reason = "max_rounds"
    for round_index in range(max_rounds):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            trace.stop_reason = "wall_time_budget"
            break
        history = [{key: value for key, value in record.to_dict().items() if key != "request"}
                   for record in trace.rounds]
        request = AgentRequest(
            copy.deepcopy(task), copy.deepcopy(evaluator.target), history,
            [item.to_dict() for item in baselines],
            {"rounds": max_rounds - round_index, "seconds": remaining,
             "trial_timeout_seconds": min(remaining, trial_timeout_seconds)},
            round_index,
            {"types": ["eager", "pytorch_source", "triton_source"],
             "entrypoint": "run(x, input_bias, gamma, beta, *, eps, launch_parameters)",
             "imports": IMPORTS, "allowed_attributes": sorted(ATTRIBUTES),
             "source_limit_bytes": 65536, "triton_hidden_size_limit": 256,
             "isolation": copy.deepcopy(isolation),
             "rules": "No IO, private attributes, input mutation, or evaluator metrics; source is AST-restricted"},
        )
        proposal, agent_error = None, None
        try:
            proposed = agent.propose(copy.deepcopy(request))
            proposal = AgentProposal.from_dict(proposed.to_dict() if isinstance(proposed, AgentProposal) else proposed)
        except AgentExhausted as exc:
            trace.stop_reason = str(exc)
            break
        except Exception as exc:
            agent_error = compiler.one_line_error(exc)
        if agent_error is not None:
            result = EvaluationResult("agent_error", evaluator.target, compiled=False, reason=agent_error)
        elif time.monotonic() >= deadline:
            result = EvaluationResult("timeout", evaluator.target, reason="budget exhausted during agent proposal")
        else:
            try:
                result = evaluator.evaluate(proposal, min(trial_timeout_seconds, deadline - time.monotonic()))
            except compiler.ReferenceValidationError:
                raise
            except Exception as exc:
                result = EvaluationResult("infrastructure_error", evaluator.target,
                                          runtime_error=compiler.one_line_error(exc))
        result.diagnostics.setdefault("isolation", copy.deepcopy(isolation))
        result.baseline_latency_ms = verified_latency(baseline.evaluation)
        latency = verified_latency(result)
        result.speedup = result.baseline_latency_ms / latency if latency else None
        trace.rounds.append(OptimizationRound(round_index, request, proposal, result,
                                               reward_function(result), agent_error))
    verified = [record for record in trace.rounds if verified_latency(record.evaluation) is not None]
    best = min(verified, key=lambda record: record.evaluation.latency_ms, default=None)
    if best is not None and best.evaluation.latency_ms < baseline.evaluation.latency_ms:
        trace.selected_round = best.round_index
    trace.final_result = {
        "implementation": best.proposal.candidate_id if trace.selected_round is not None else baseline.name,
        "fallback": trace.selected_round is None,
        "reason": "lowest measured verified latency; ties retain framework; not a significance test",
        "best_verified_round": best.round_index if best else None,
        "evaluation": (best.evaluation if trace.selected_round is not None else baseline.evaluation).to_dict(),
    }
    trace.gpu_performance_evaluated = evaluator.target.device.startswith("cuda") and any(
        verified_latency(record.evaluation) is not None for record in trace.rounds
        if record.proposal and record.proposal.candidate_type != "eager"
    )
    trace.metrics = calculate_metrics(trace.rounds, baselines, time.monotonic() - started)
    trace.evaluation_completed = True
    return trace
