"""Pure evaluator-owned rewards and metrics. No training or model claims."""

import math

from agent_schema import BaselineResult, EvaluationResult, OptimizationRound, RunMetrics
from candidate_evaluator import verified_latency


def reward_function(result: EvaluationResult) -> float | None:
    if result.status in ("skipped", "unavailable"):
        return None
    if result.compiled is False or result.correct is False or result.status in (
        "compile_error", "runtime_error", "incorrect", "invalid_proposal", "timeout",
        "infrastructure_error", "agent_error",
    ):
        return -1.0
    latency, baseline = verified_latency(result), result.baseline_latency_ms
    if latency is None or baseline is None or not math.isfinite(baseline) or baseline <= 0:
        return None
    return math.log(baseline) - math.log(latency)


def best_baseline(baselines: list[BaselineResult]) -> BaselineResult | None:
    valid = [item for item in baselines if verified_latency(item.evaluation) is not None]
    return min(valid, key=lambda item: item.evaluation.latency_ms, default=None)


def calculate_metrics(rounds: list[OptimizationRound], baselines: list[BaselineResult],
                      wall_seconds: float) -> RunMetrics:
    attempted = [item.evaluation for item in rounds if item.evaluation.status not in ("skipped", "unavailable")]
    correct = [item for item in attempted if item.correct is True]
    latencies = [verified_latency(item) for item in correct]
    latency = min((value for value in latencies if value is not None), default=None)
    baseline = best_baseline(baselines)
    baseline_ms = verified_latency(baseline.evaluation) if baseline else None
    eager = next((verified_latency(item.evaluation) for item in baselines if item.name == "eager"), None)
    compiler = best_baseline([item for item in baselines if item.name != "eager"])
    compiler_ms = verified_latency(compiler.evaluation) if compiler else None
    return RunMetrics(
        sum(item.compiled is True for item in attempted) / len(attempted) if attempted else None,
        len(correct) / len(attempted) if attempted else None,
        latency, baseline_ms, eager / latency if eager and latency else None,
        compiler_ms / latency if compiler_ms and latency else None,
        baseline_ms / latency if baseline_ms and latency else None,
        len(rounds), len(correct),
        sum(value is not None and baseline_ms is not None and value < baseline_ms for value in latencies),
        wall_seconds,
    )


def suite_speedup_metrics(speedups: list[float | None]) -> dict[str, float | None]:
    """Future multi-task reducer; missing/failed tasks count as failures, not omissions.

    Geometric mean is unavailable if any task is unmeasured; fast_p uses the whole
    supplied task set. The single-task CLI deliberately never calls this reducer.
    """
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in speedups):
        raise ValueError("speedups must be positive finite numbers or None")
    result = {f"fast_{threshold:.1f}": (sum(value is not None and value > threshold for value in speedups)
                                      / len(speedups) if speedups else None)
              for threshold in (1.0, 1.1, 1.5)}
    result["geometric_mean_speedup"] = (math.exp(sum(math.log(value) for value in speedups) / len(speedups))
                                        if speedups and all(value is not None for value in speedups) else None)
    return result
