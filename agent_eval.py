#!/usr/bin/env python3
"""CPU-safe V2 agent evaluation CLI. V1 remains independently executable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

import agentic_kernel_compiler as compiler
from agent_backends.mock import MockAgentBackend
from agent_backends.replay import ReplayAgentBackend
from agent_controller import run_optimization
from candidate_evaluator import SubprocessEvaluator
from evaluation_isolation import STRICT, TRUSTED_FIXTURE_COLAB
from optimization_tasks import TASKS, TaskSettings, fingerprint, load_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="bias_layernorm_silu", choices=sorted(TASKS))
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--agent", choices=("mock", "replay"), default="mock")
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--trial-timeout", type=float, default=120)
    parser.add_argument("--baseline-timeout", type=float, default=600)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--skip-max-autotune", action="store_true")
    parser.add_argument("--trusted-fixture-colab", action="store_true",
                        help="explicitly run only exact built-in mock fixtures without an OS sandbox; never replay")
    args = parser.parse_args(argv)
    if not 1 <= args.max_rounds <= 64:
        parser.error("--max-rounds must be in [1, 64]")
    if args.agent == "replay" and args.replay_file is None:
        parser.error("--agent replay requires --replay-file")
    if args.agent == "mock" and args.replay_file is not None:
        parser.error("--replay-file requires --agent replay")
    if args.trusted_fixture_colab and args.agent != "mock":
        parser.error("--trusted-fixture-colab requires --agent mock; replay/arbitrary source is forbidden")
    return args


@torch.no_grad()
def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_tasks:
        print("\n".join(f"{name}: {task.description}" for name, task in sorted(TASKS.items())))
        return 0
    try:
        settings = TaskSettings(args.rows, args.hidden_size, args.dtype)
        device = compiler.resolve_device(args.device)
        agent = ReplayAgentBackend(args.replay_file) if args.agent == "replay" else MockAgentBackend()
        task = load_task(args.task)
        target = fingerprint(device)
        description = task.describe(settings, device)
        evaluator = SubprocessEvaluator(task, settings, target, warmup=3 if args.quick else 100,
                                        iterations=10 if args.quick else 100, timing_rounds=2 if args.quick else 5,
                                        max_autotune=not args.skip_max_autotune,
                                        isolation_mode=TRUSTED_FIXTURE_COLAB if args.trusted_fixture_colab else STRICT)
        if args.trusted_fixture_colab:
            print("WARNING: explicit trusted-fixture mode; exact repository mock fixtures only; NO OS SANDBOX.",
                  file=sys.stderr)
        trace = run_optimization(description, agent, evaluator, max_rounds=args.max_rounds,
                                 trial_timeout_seconds=args.trial_timeout, max_seconds=args.max_seconds,
                                 baseline_timeout_seconds=args.baseline_timeout)
        root = Path(__file__).resolve().parent
        sources = sorted(root.glob("*.py")) + sorted((root / "agent_backends").glob("*.py"))
        trace.provenance = {"v1": compiler.source_provenance(),
                            "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in sources},
                            "cli_arguments": {key: str(value) if isinstance(value, Path) else value
                                              for key, value in vars(args).items()},
                            "agent": args.agent}
        if args.replay_file:
            trace.provenance["replay_sha256"] = hashlib.sha256(args.replay_file.read_bytes()).hexdigest()
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            compiler.write_json_atomic(args.json_output, trace.to_dict())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {compiler.one_line_error(exc)}; no completed trace written", file=sys.stderr)
        return 1
    print(f"V2 {args.task} | {target.device} | {target.properties.get('Device', 'CPU infrastructure only')}")
    for baseline in trace.baseline_results:
        result = baseline.evaluation
        print(f"baseline {baseline.name}: {result.status}; {result.reason or result.compile_error or result.runtime_error or result.correctness_error or result.latency_ms}")
    for record in trace.rounds:
        result = record.evaluation
        print(f"round {record.round_index}: {result.status}; reward={record.reward}; "
              f"{result.reason or result.compile_error or result.runtime_error or result.correctness_error or result.latency_ms}")
    print(json.dumps({"gpu_performance_evaluated": trace.gpu_performance_evaluated,
                      "isolation": trace.isolation,
                      "selected_round": trace.selected_round, "stop_reason": trace.stop_reason,
                      "metrics": trace.metrics.to_dict()}, indent=2))
    if args.json_output:
        print(f"Completed trace: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
