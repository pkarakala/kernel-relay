"""Interleaved T4 confirmation for exact, trusted V2 mock fixtures only.

This script is not an isolation boundary. Never point it at arbitrary/replay
agent source; run only in a trusted, secret-free evaluation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import agentic_kernel_compiler as compiler
from agent_backends.mock import validate_trusted_mock_proposal
from agent_schema import AgentProposal
from candidate_worker import load_candidate
from optimization_tasks import TaskSettings, load_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "results/agent-eval/v2-full-t4-fp16.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/agent-eval/v2-paired-t4-fp16-rerun.json")
    args = parser.parse_args()

    trace_bytes = args.trace.read_bytes()
    trace = json.loads(trace_bytes)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU result can be produced")
    if trace["task_id"] != "bias_layernorm_silu" or not trace["evaluation_completed"] or not trace["gpu_performance_evaluated"]:
        raise ValueError("trace is not a completed GPU bias_layernorm_silu evaluation")
    if trace["isolation"]["mode"] != "trusted_fixture_colab" or trace["provenance"]["agent"] != "mock":
        raise ValueError("only the exact trusted Colab mock-fixture trace is supported")
    if trace["provenance"]["cli_arguments"]["rows"] != 8192 or trace["provenance"]["cli_arguments"]["hidden_size"] != 128 or trace["provenance"]["cli_arguments"]["dtype"] != "float16":
        raise ValueError("this confirmation requires the 8192 x 128 FP16 workload")
    for relative_path, expected_hash in trace["provenance"]["source_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"source differs from the trace: {relative_path}")

    device = torch.device("cuda")
    suite = load_task("bias_layernorm_silu").prepare(
        TaskSettings(rows=8192, hidden_size=128, dtype="float16"), device
    )
    with torch.no_grad(), tempfile.TemporaryDirectory() as temporary:
        compiled = torch.compile(suite.module, fullgraph=True, dynamic=False)
        functions = {"eager": suite.module, "inductor_default": compiled}
        for round_index in (2, 3):
            proposal = AgentProposal.from_dict(trace["rounds"][round_index]["proposal"])
            validate_trusted_mock_proposal(proposal, 128)
            candidate_dir = Path(temporary) / proposal.candidate_id
            candidate_dir.mkdir()
            run = load_candidate(proposal, candidate_dir)

            def candidate(*inputs, run=run, proposal=proposal):
                return run(*inputs, eps=suite.module.eps,
                           launch_parameters=proposal.launch_parameters)

            functions[proposal.candidate_id] = candidate

        correctness = {}
        for name, function in functions.items():
            check = suite.validate(function)
            correctness[name] = compiler.correctness_metrics(check)
            if not check.passed:
                raise RuntimeError(f"{name} failed correctness: {check.reason}")

        paths = {
            name: (lambda function=function: function(*suite.inputs))
            for name, function in functions.items()
        }
        trials, failures = compiler.compare_paths(
            paths, device, warmup=100, iterations=100, rounds=5
        )
        if failures or any(len(samples) != 5 for samples in trials.values()):
            raise RuntimeError(f"incomplete timing: {failures}")
        medians = {name: statistics.median(samples) for name, samples in trials.items()}
        best_baseline = min(medians["eager"], medians["inductor_default"])
        best_fused = min(medians["fused-2"], medians["fused-3"])
        output = {
            "method": "same-process interleaved paths; 5 rounds; 100 warmups and 100 CUDA-event samples per path per round",
            "device": torch.cuda.get_device_name(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "source_trace": str(args.trace),
            "source_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "correctness": correctness,
            "trial_medians_ms": trials,
            "path_medians_ms": medians,
            "best_fused_over_best_framework_speedup": best_baseline / best_fused,
            "isolation_note": "Exact trusted repository mock fixtures only; this direct confirmation has no OS sandbox.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        for name, samples in trials.items():
            print(f"{name}: median={medians[name]:.6f} ms; trial medians={samples}")
        print(f"Fused / best framework speedup: {best_baseline / best_fused:.3f}x")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
