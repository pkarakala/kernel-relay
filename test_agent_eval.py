"""CPU-first V2 tests. Synthetic timing fixtures are unit data, not GPU evidence."""

import contextlib
import copy
import errno
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import agent_eval
import agentic_kernel_compiler as compiler
from agent_backends.base import AgentExhausted
from agent_backends.mock import MockAgentBackend, PYTORCH_SOURCE, triton_source, validate_trusted_mock_proposal
from agent_backends.replay import ReplayAgentBackend
from agent_controller import run_optimization
from agent_metrics import best_baseline, calculate_metrics, reward_function, suite_speedup_metrics
from agent_schema import AgentProposal, AgentRequest, BaselineResult, EvaluationResult, HardwareFingerprint, OptimizationTrace, strict_loads
from candidate_evaluator import ROOT, SubprocessEvaluator, verified_latency
from candidate_policy import validate_source
from candidate_worker import evaluate_function, evaluate_baselines, load_candidate
from evaluation_isolation import (
    STRICT, TRUSTED_FIXTURE_COLAB, isolation_report, landlock_status,
    strict_isolation_unavailable_reason, worker_command, worker_environment,
)
from optimization_tasks import TASKS, TaskSettings, fingerprint, load_task


CPU = HardwareFingerprint("cpu", {"test_fixture": True})
TASK = {"task_id": "bias_layernorm_silu", "inputs": [{"shape": [3, 8]}]}


def measured(latency=2.0, baseline=2.0):
    return EvaluationResult("measured", CPU, compiled=True, correct=True, latency_ms=latency,
                            baseline_latency_ms=baseline)


def request():
    return AgentRequest(TASK, CPU, [], [], {"rounds": 3, "seconds": 10}, 0, {})


def require_strict_isolation():
    reason = strict_isolation_unavailable_reason()
    if reason:
        raise unittest.SkipTest("strict-isolation environment limitation: " + reason)


class FixtureEvaluator:
    target = CPU

    def __init__(self, results=None):
        self.results = results or [measured()]
        self.calls = 0

    def baselines(self, timeout_seconds):
        return [BaselineResult("eager", measured(3.0)), BaselineResult("inductor_default", measured(2.0)),
                BaselineResult("inductor_max_autotune", EvaluationResult("unavailable", CPU))]

    def evaluate(self, proposal, timeout_seconds):
        result = copy.deepcopy(self.results[self.calls % len(self.results)])
        self.calls += 1
        return result


class SchemaAndBackendTests(unittest.TestCase):
    def test_task_registry_reference_graph_and_determinism(self):
        self.assertEqual(list(TASKS), ["bias_layernorm_silu"])
        task = load_task("bias_layernorm_silu")
        settings = TaskSettings(3, 8)
        first, second = (task.prepare(settings, torch.device("cpu")) for _ in range(2))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(first.inputs, second.inputs)))
        description = task.describe(settings, torch.device("cpu"))
        self.assertTrue(description["fusion_eligible"])
        self.assertEqual(len(description["graph_nodes"]), 8)
        self.assertIn("forward", description["pytorch_source"])
        self.assertEqual(len(description["validation_cases"]), 8)
        self.assertTrue(first.validate(first.module).passed)
        with self.assertRaises(ValueError):
            load_task("unknown")
        for settings in ((0, 8, "float32"), (1, 0, "float32"), (True, 8, "float32"), (3, 8, "int32")):
            with self.assertRaises(ValueError):
                TaskSettings(*settings)

    def test_request_proposal_and_result_serialization(self):
        self.assertEqual(AgentRequest.from_json(request().to_json()), request())
        proposal = AgentProposal("cpu-source", "pytorch_source", "reference", PYTORCH_SOURCE)
        self.assertEqual(AgentProposal.from_json(proposal.to_json()), proposal)
        result = measured()
        result.max_rel_error = math.inf
        serialized = strict_loads(result.to_json())
        self.assertIsNone(serialized["max_rel_error"])
        self.assertEqual(EvaluationResult.from_dict(serialized).latency_ms, 2)

    def test_strict_proposal_rejects_claimed_metrics_and_bad_fields(self):
        base = AgentProposal("fallback", "eager", "safe").to_dict()
        for changes in ({"reward": 100}, {"correct": True}, {"latency_ms": 0.001}, {"candidate_type": "astra"},
                        {"source": PYTORCH_SOURCE}, {"schema_version": "1.0"}, {"strategy": 3},
                        {"launch_parameters": {"num_warps": True}}, {"candidate_id": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AgentProposal.from_dict({**base, **changes})
        with self.assertRaises(ValueError):
            AgentProposal("large", "pytorch_source", "source", "x" * 65537)
        for payload in ('{"key": 1, "key": 2}', '{"key": NaN}', '{"key": Infinity}'):
            with self.assertRaises(ValueError):
                strict_loads(payload)

    def test_mock_is_deterministic_and_contains_real_source(self):
        first, second = MockAgentBackend(), MockAgentBackend()
        first_values = [first.propose(request()).to_dict() for _ in range(8)]
        self.assertEqual(first_values, [second.propose(request()).to_dict() for _ in range(8)])
        self.assertEqual([value["candidate_type"] for value in first_values[:3]],
                         ["pytorch_source", "eager", "triton_source"])
        self.assertNotEqual(first_values[2]["source"], first_values[3]["source"])
        for value in first_values[2:]:
            validate_source(value["source"], "triton_source")
        with self.assertRaises(AgentExhausted):
            first.propose(request())

    def test_replay_bundle_trace_exhaustion_and_bad_proposal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "replay.json"
            proposal = AgentProposal("safe", "eager", "reference").to_dict()
            for body in ({"proposals": [proposal]}, {"rounds": [{"proposal": None}, {"proposal": proposal,
                                                                             "reward": 999, "evaluation": {"correct": False}}]}):
                path.write_text(json.dumps({"schema_version": "2.0", "task_id": TASK["task_id"], **body}))
                backend = ReplayAgentBackend(path)
                self.assertEqual(backend.propose(request()).to_dict(), proposal)
                with self.assertRaises(AgentExhausted):
                    backend.propose(request())
            path.write_text(json.dumps({"schema_version": "2.0", "proposals": [{"bad": True}, proposal]}))
            backend = ReplayAgentBackend(path)
            with self.assertRaises(ValueError):
                backend.propose(request())
            self.assertEqual(backend.propose(request()).candidate_id, "safe")
            path.write_text(json.dumps({"schema_version": "2.0", "task_id": "another", "proposals": [proposal]}))
            with self.assertRaises(ValueError):
                ReplayAgentBackend(path).propose(request())
            path.write_text('{"schema_version": "1.0", "proposals": []}')
            with self.assertRaises(ValueError):
                ReplayAgentBackend(path)


class RewardAndControllerTests(unittest.TestCase):
    def test_rewards_hard_gate_and_missing_measurements(self):
        self.assertAlmostEqual(reward_function(measured(1, 2)), math.log(2))
        self.assertAlmostEqual(reward_function(measured(4, 2)), math.log(0.5))
        self.assertEqual(reward_function(measured(2, 2)), 0)
        for status in ("compile_error", "incorrect", "runtime_error", "timeout", "agent_error"):
            result = measured(0.00001, 2)
            result.status = status
            self.assertEqual(reward_function(result), -1)
        result = measured(0.001, 2)
        result.correct = False
        self.assertEqual(reward_function(result), -1)
        for value in (None, 0, -1, math.nan, math.inf):
            result = measured(value, 2)
            self.assertIsNone(reward_function(result))
        for status in ("skipped", "unavailable"):
            self.assertIsNone(reward_function(EvaluationResult(status, CPU)))

    def test_bounded_feedback_selection_and_metrics(self):
        results = [EvaluationResult("compile_error", CPU, compiled=False, compile_error="syntax"),
                   measured(3), measured(1)]
        evaluator = FixtureEvaluator(results)
        trace = run_optimization(TASK, MockAgentBackend(), evaluator, max_rounds=3)
        self.assertEqual(trace.selected_round, 2)
        self.assertEqual(trace.stop_reason, "max_rounds")
        self.assertEqual(trace.rounds[1].request.history[0]["evaluation"]["compile_error"], "syntax")
        self.assertEqual([item.request.remaining_budget["rounds"] for item in trace.rounds], [3, 2, 1])
        self.assertEqual(trace.rounds[2].request.history[1]["evaluation"]["latency_ms"], 3)
        self.assertEqual(evaluator.calls, 3)
        self.assertEqual(trace.metrics.number_of_correct_candidates, 2)
        self.assertEqual(trace.metrics.number_of_faster_than_baseline_candidates, 1)
        self.assertAlmostEqual(trace.metrics.compile_success_rate, 2 / 3)
        self.assertAlmostEqual(trace.metrics.correctness_rate, 2 / 3)
        self.assertEqual(trace.metrics.speedup_over_eager, 3)
        self.assertEqual(trace.metrics.speedup_over_best_compiler_baseline, 2)
        self.assertEqual(trace.metrics.best_baseline_latency_ms, 2)
        self.assertFalse(trace.gpu_performance_evaluated)
        payload = strict_loads(trace.to_json())
        self.assertEqual(OptimizationTrace.from_dict(payload).to_dict(), payload)
        self.assertTrue(payload["evaluation_completed"])
        self.assertEqual(payload["rounds"][0]["reward"], -1)
        self.assertNotIn("request", payload["rounds"][2]["request"]["history"][0])

    def test_agent_cannot_mutate_prior_history_or_hardware(self):
        class MutatingAgent:
            def propose(self, request):
                request.hardware.properties["changed"] = True
                request.task["task_id"] = "changed"
                if request.history:
                    request.history[0]["evaluation"]["latency_ms"] = 0.000001
                return AgentProposal("safe", "eager", "reference")

        trace = run_optimization(TASK, MutatingAgent(), FixtureEvaluator(), max_rounds=2)
        self.assertEqual(trace.rounds[0].evaluation.latency_ms, 2)
        self.assertNotIn("changed", trace.target.properties)
        self.assertEqual(trace.task_id, TASK["task_id"])
        self.assertEqual(trace.rounds[0].request.task["task_id"], TASK["task_id"])

    def test_failed_agent_consumes_round_and_evaluator_failure_is_recorded(self):
        class BrokenAgent:
            def propose(self, request):
                return {"reward": 100}

        evaluator = FixtureEvaluator()
        trace = run_optimization(TASK, BrokenAgent(), evaluator, max_rounds=2)
        self.assertEqual(evaluator.calls, 0)
        self.assertTrue(all(record.evaluation.status == "agent_error" for record in trace.rounds))
        self.assertTrue(trace.final_result["fallback"])
        with patch.object(evaluator, "evaluate", side_effect=RuntimeError("worker error")):
            trace = run_optimization(TASK, MockAgentBackend(), evaluator, max_rounds=1)
        self.assertEqual(trace.rounds[0].evaluation.status, "infrastructure_error")

    def test_exhaustion_zero_candidates_ties_and_incorrect_fast_path(self):
        trace = run_optimization(TASK, MockAgentBackend([]), FixtureEvaluator(), max_rounds=3)
        self.assertEqual(len(trace.rounds), 0)
        self.assertTrue(trace.final_result["fallback"])
        self.assertIsNone(trace.metrics.correctness_rate)
        incorrect = measured(0.00001)
        incorrect.correct, incorrect.status = False, "incorrect"
        trace = run_optimization(TASK, MockAgentBackend(), FixtureEvaluator([incorrect, measured(2)]), max_rounds=2)
        self.assertIsNone(trace.selected_round)
        self.assertEqual(trace.metrics.best_verified_candidate_latency_ms, 2)

    def test_wall_budget_and_invalid_budget(self):
        with patch("agent_controller.time.monotonic", side_effect=[0, 20, 20]):
            trace = run_optimization(TASK, MockAgentBackend(), FixtureEvaluator(), max_seconds=10)
        self.assertEqual(trace.stop_reason, "wall_time_budget")
        self.assertEqual(len(trace.rounds), 0)
        for budget in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                run_optimization(TASK, MockAgentBackend(), FixtureEvaluator(), max_seconds=budget)
        with self.assertRaises(ValueError):
            run_optimization(TASK, MockAgentBackend(), FixtureEvaluator(), max_rounds=0)

    def test_future_suite_reducer_does_not_drop_failures(self):
        metrics = suite_speedup_metrics([2, 1, None])
        self.assertEqual(metrics["fast_1.0"], 1 / 3)
        self.assertIsNone(metrics["geometric_mean_speedup"])
        self.assertAlmostEqual(suite_speedup_metrics([2, 0.5])["geometric_mean_speedup"], 1)
        with self.assertRaises(ValueError):
            suite_speedup_metrics([0])

    def test_only_verified_framework_baselines_define_success(self):
        incorrect = measured(0.00001)
        incorrect.correct = False
        baselines = [BaselineResult("eager", measured(3)), BaselineResult("inductor_default", incorrect),
                     BaselineResult("inductor_max_autotune", measured(2))]
        self.assertEqual(best_baseline(baselines).name, "inductor_max_autotune")
        self.assertIsNone(verified_latency(incorrect))


class VerificationAndIsolationTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_task(TASK["task_id"]).prepare(TaskSettings(3, 8), torch.device("cpu"))
        self.timing = {"warmup": 0, "iterations": 1, "timing_rounds": 1, "max_autotune": True}

    def test_compile_runtime_and_correctness_failures_are_distinct(self):
        def bad_factory():
            raise RuntimeError("compile failed")

        def bad_runtime(*values):
            raise RuntimeError("launch failed")

        with patch.object(compiler, "benchmark") as benchmark:
            failed = evaluate_function(bad_factory, self.suite, CPU, self.timing)
            runtime = evaluate_function(lambda: bad_runtime, self.suite, CPU, self.timing)
            incorrect = evaluate_function(lambda: lambda *values: torch.zeros_like(values[0]), self.suite, CPU, self.timing)
            invalid = evaluate_function(lambda: lambda *values: None, self.suite, CPU, self.timing)
        benchmark.assert_not_called()
        self.assertEqual(failed.status, "compile_error")
        self.assertEqual(runtime.status, "runtime_error")
        self.assertEqual(incorrect.status, "incorrect")
        self.assertEqual(invalid.status, "incorrect")
        self.assertIsNone(incorrect.latency_ms)
        self.assertGreater(incorrect.max_abs_error, 0)

    def test_input_mutation_rejected_before_timing(self):
        def mutate(*values):
            output = self.suite.module(*values)
            values[0].zero_()
            return output

        with patch.object(compiler, "benchmark") as benchmark:
            result = evaluate_function(lambda: mutate, self.suite, CPU, self.timing, guarded=True)
        benchmark.assert_not_called()
        self.assertEqual(result.status, "incorrect")
        self.assertIn("modified", result.correctness_error)

    def test_unavailable_compile_and_mode_compilation_failure(self):
        from types import SimpleNamespace

        suite = SimpleNamespace(module=self.suite.module, inputs=self.suite.inputs, device=torch.device("cuda"))
        with patch.object(torch, "compile", None):
            result = evaluate_baselines(suite, CPU, self.timing, "inductor_default")[0]
        self.assertEqual(result.evaluation.status, "unavailable")
        with patch.object(torch, "compile", side_effect=RuntimeError("unsupported mode")):
            result = evaluate_baselines(suite, CPU, self.timing, "inductor_default")[0]
        self.assertEqual(result.evaluation.status, "compile_error")

    def test_source_policy_and_real_implementation_variants(self):
        validate_source(PYTORCH_SOURCE, "pytorch_source")
        for variant in (False, True):
            validate_source(triton_source(variant), "triton_source")
        for source in ("import os\n" + PYTORCH_SOURCE,
                       PYTORCH_SOURCE.replace("return functional.silu", "return open"),
                       PYTORCH_SOURCE.replace("return functional.silu", "return torch.save"),
                       PYTORCH_SOURCE.replace("return functional.silu", "return torch.load"),
                       PYTORCH_SOURCE.replace("return functional.silu", "return torch.jit.load"),
                       PYTORCH_SOURCE.replace("return functional.silu", "alias = torch\n    return alias.load"),
                       PYTORCH_SOURCE.replace("x.shape", "x.__class__"),
                       PYTORCH_SOURCE + "\ntorch.save = run\n",
                       PYTORCH_SOURCE.replace("*, eps,", "*, eps=open('x'),")):
            with self.subTest(source=source), self.assertRaises((ValueError, SyntaxError)):
                validate_source(source, "pytorch_source")

    def test_candidate_annotations_do_not_inherit_worker_future_flags(self):
        source = PYTORCH_SOURCE.replace("def run(x,", "def run(x: int,")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            entry = load_candidate(AgentProposal("annotation-fixture", "pytorch_source", "test", source), Path(directory))
            self.assertIs(entry.__annotations__["x"], int)

    def test_clean_environment_and_filesystem_confinement(self):
        require_strict_isolation()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            trial = root / "trial"
            trial.mkdir()
            sentinel = root / "repository-file"
            sentinel.write_text("untouched")
            with patch.dict(os.environ, {"FAKE_API_KEY": "do-not-forward", "PYTHONPATH": "/bad"}):
                environment = worker_environment(trial)
            self.assertNotIn("FAKE_API_KEY", environment)
            self.assertNotIn("PYTHONPATH", environment)
            command = worker_command(trial)
            command[-1] = (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                "from evaluation_isolation import restrict_linux_filesystem; "
                f"restrict_linux_filesystem(Path({str(trial)!r})); "
                f"Path({str(sentinel)!r}).write_text('overwritten')"
            )
            if sys.platform == "darwin":
                command[-1] = f"from pathlib import Path; Path({str(sentinel)!r}).write_text('overwritten')"
            process = subprocess.run(command, cwd=trial, env=environment, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("PermissionError", process.stderr)
            self.assertEqual(sentinel.read_text(), "untouched")


class SubprocessAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = SubprocessEvaluator(load_task(TASK["task_id"]), TaskSettings(3, 8), fingerprint(torch.device("cpu")),
                                            warmup=0, iterations=1, timing_rounds=1)

    def prepare_strict_baselines(self):
        require_strict_isolation()
        if not self.evaluator.baseline_results:
            self.evaluator.baselines(30)

    def test_cpu_reference_source_is_executed_and_measured(self):
        self.prepare_strict_baselines()
        result = self.evaluator.evaluate(AgentProposal("cpu", "pytorch_source", "reference", PYTORCH_SOURCE), 30)
        self.assertEqual(result.status, "measured", result.to_json())
        self.assertTrue(result.correct)
        self.assertGreater(result.latency_ms, 0)
        self.assertEqual(result.verification["elements_checked"], 3 * 8 * 8)
        self.assertEqual(len(result.source_sha256), 64)
        self.assertTrue(result.diagnostics["isolation"])

    def test_source_wrong_output_runtime_exception_timeout_and_cleanup(self):
        self.prepare_strict_baselines()
        before = set(ROOT.glob(".agent-trial-*"))
        for body, status in (("return torch.zeros_like(x)", "incorrect"),
                             ("raise RuntimeError('intentional')", "runtime_error"),
                             ("while True:\n        pass", "timeout")):
            source = "import torch\n\ndef run(x, input_bias, gamma, beta, *, eps, launch_parameters):\n    " + body
            result = self.evaluator.evaluate(AgentProposal("failure", "pytorch_source", "test", source),
                                              2 if status == "timeout" else 30)
            self.assertEqual(result.status, status, result.to_json())
            self.assertIsNone(result.latency_ms)
            self.assertEqual(reward_function(result), -1)
        self.assertEqual(set(ROOT.glob(".agent-trial-*")), before)

    def test_cpu_skip_missing_triton_and_missing_cuda(self):
        proposal = AgentProposal("gpu", "triton_source", "fuse", triton_source())
        with patch.object(self.evaluator, "_run") as worker:
            self.assertEqual(self.evaluator.evaluate(proposal, 30).status, "skipped")
            worker.assert_not_called()
        gpu = SubprocessEvaluator(load_task(TASK["task_id"]), TaskSettings(3, 8),
                                  HardwareFingerprint("cuda", {"Triton": "not installed"}))
        with patch.object(gpu, "_run") as worker:
            self.assertEqual(gpu.evaluate(proposal, 30).status, "skipped")
            worker.assert_not_called()
        with patch.object(compiler.torch.cuda, "is_available", return_value=False):
            self.assertEqual(compiler.resolve_device("auto").type, "cpu")
            with self.assertRaises(ValueError):
                compiler.resolve_device("cuda")

    def test_baseline_timeout_preserves_verified_eager_and_continues(self):
        evaluator = SubprocessEvaluator(load_task(TASK["task_id"]), TaskSettings(3, 8),
                                       HardwareFingerprint("cuda", {"unit_test_only": True}))
        eager = {"baselines": [BaselineResult("eager", measured(3)).to_dict()]}
        stronger = {"baselines": [BaselineResult("inductor_max_autotune", measured(2)).to_dict()]}
        with patch.object(evaluator, "_run", side_effect=[eager, subprocess.TimeoutExpired("fixture", 1), stronger]):
            baselines = evaluator.baselines(30)
        self.assertEqual([item.evaluation.status for item in baselines], ["measured", "timeout", "measured"])
        self.assertEqual(best_baseline(baselines).name, "inductor_max_autotune")

    def test_no_optional_accelerator_import_at_startup(self):
        code = "import agent_eval,sys; assert 'triton' not in sys.modules; import torch; assert not torch.cuda.is_initialized()"
        process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_smoke_json_replay_and_error_preserves_artifact(self):
        require_strict_isolation()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "nested" / "trace.json"
            command = [sys.executable, str(ROOT / "agent_eval.py"), "--device", "cpu", "--quick",
                       "--rows", "3", "--hidden-size", "8", "--max-rounds", "3"]
            process = subprocess.run(command + ["--json-output", str(output)], cwd=ROOT,
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            trace = strict_loads(output.read_text())
            self.assertEqual(trace["schema_version"], "2.0")
            self.assertFalse(trace["gpu_performance_evaluated"])
            self.assertEqual([record["evaluation"]["status"] for record in trace["rounds"]],
                             ["compile_error", "measured", "skipped"])
            self.assertEqual(trace["target"]["device"], "cpu")
            self.assertNotIn("fast_1.0", trace["metrics"])
            replay = subprocess.run(command + ["--agent", "replay", "--replay-file", str(output)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            original = output.read_bytes()
            with patch.object(compiler.torch.cuda, "is_available", return_value=False), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(agent_eval.main(["--device", "cuda", "--json-output", str(output)]), 2)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_discovery_and_validation(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(agent_eval.main(["--list-tasks"]), 0)
        self.assertIn("bias_layernorm_silu", stream.getvalue())
        for args in (["--max-rounds", "0"], ["--agent", "replay"], ["--replay-file", "x.json"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                agent_eval.main(args)
            self.assertEqual(caught.exception.code, 2)


class TrustedFixtureTests(unittest.TestCase):
    def evaluator(self, mode=TRUSTED_FIXTURE_COLAB):
        return SubprocessEvaluator(load_task(TASK["task_id"]), TaskSettings(3, 8), CPU,
                                   warmup=0, iterations=1, timing_rounds=1, isolation_mode=mode)

    def fixtures(self):
        agent = MockAgentBackend()
        return [agent.propose(request()) for _ in range(8)]

    def test_worker_uses_only_fixed_colab_driver_library_path(self):
        with (patch.dict(os.environ, {"LD_LIBRARY_PATH": "/untrusted/library", "FAKE_API_KEY": "secret"}),
              patch("evaluation_isolation.sys.platform", "linux"),
              patch("evaluation_isolation.Path.is_dir", return_value=True)):
            environment = worker_environment(ROOT, TRUSTED_FIXTURE_COLAB)
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/usr/lib64-nvidia")
        self.assertNotIn("FAKE_API_KEY", environment)
        with self.assertRaises(ValueError):
            worker_environment(ROOT, "unknown-mode")

    def test_exact_fixture_gate_includes_source_and_parameters(self):
        fixtures = self.fixtures()
        for proposal in fixtures:
            validate_trusted_mock_proposal(proposal, 8)
        for changes in ({"source": fixtures[2].source + "\n"}, {"source": PYTORCH_SOURCE},
                        {"launch_parameters": {"block_size": 64, "num_warps": 2, "num_stages": 2}},
                        {"candidate_id": "user-supplied"}, {"strategy": "changed"}):
            proposal = AgentProposal.from_dict({**fixtures[2].to_dict(), **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_trusted_mock_proposal(proposal, 8)
            evaluator = self.evaluator()
            with patch.object(evaluator, "_run") as worker:
                self.assertEqual(evaluator.evaluate(proposal, 10).status, "invalid_proposal")
                worker.assert_not_called()

    def test_controller_and_cli_reject_replay_and_custom_mock(self):
        evaluator = self.evaluator()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps({"schema_version": "2.0", "proposals": [self.fixtures()[1].to_dict()]}))
            for agent in (ReplayAgentBackend(path), MockAgentBackend(self.fixtures())):
                with patch.object(evaluator, "baselines") as baseline, self.assertRaises(ValueError):
                    run_optimization(TASK, agent, evaluator)
                baseline.assert_not_called()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                agent_eval.main(["--trusted-fixture-colab", "--agent", "replay", "--replay-file", str(path)])
            self.assertEqual(caught.exception.code, 2)

    def test_trusted_baseline_runs_in_real_subprocess_and_timeout_remains(self):
        evaluator = self.evaluator()
        with patch.dict(os.environ, {"FAKE_API_KEY": "not-forwarded"}):
            self.assertNotIn("FAKE_API_KEY", worker_environment(ROOT))
            baselines = evaluator.baselines(30)
        self.assertEqual(baselines[0].evaluation.status, "measured")
        detail = baselines[0].evaluation.diagnostics["isolation"]
        self.assertEqual(detail["mode"], TRUSTED_FIXTURE_COLAB)
        self.assertEqual(detail["filesystem_confinement"], "none")
        self.assertIn("no OS sandbox", detail["enforcement"])
        self.assertFalse(detail["production_sandbox"])
        self.assertEqual(evaluator.evaluate(self.fixtures()[1], 10).status, "measured")
        before = set(ROOT.glob(".agent-trial-*"))
        with self.assertRaises(subprocess.TimeoutExpired):
            evaluator._run("baselines", 0.001, baseline_name="eager")
        self.assertEqual(set(ROOT.glob(".agent-trial-*")), before)

    def test_worker_independently_rejects_unapproved_source_and_operation(self):
        evaluator = self.evaluator()
        modified = AgentProposal.from_dict({**self.fixtures()[2].to_dict(), "source": PYTORCH_SOURCE})
        with self.assertRaisesRegex(RuntimeError, "not an exact repository mock fixture"):
            evaluator._run("candidate", 30, modified)
        with self.assertRaisesRegex(RuntimeError, "unaudited operation"):
            evaluator._run("baselines", 30, baseline_name="arbitrary")

    def test_landlock_enosys_is_reported_and_strict_tests_skip_explicitly(self):
        from types import SimpleNamespace

        with patch("evaluation_isolation.sys.platform", "linux"), patch(
            "evaluation_isolation.platform.machine", return_value="x86_64"
        ), patch("evaluation_isolation.ctypes.CDLL", return_value=SimpleNamespace(syscall=lambda *args: -1)), patch(
            "evaluation_isolation.ctypes.get_errno", return_value=errno.ENOSYS
        ):
            status = landlock_status()
            self.assertFalse(status["available"])
            self.assertEqual(status["errno"], errno.ENOSYS)
            self.assertIn("ENOSYS", isolation_report(TRUSTED_FIXTURE_COLAB)["landlock"]["reason"])
            with self.assertRaisesRegex(unittest.SkipTest, "environment limitation.*ENOSYS"):
                require_strict_isolation()
            from evaluation_isolation import restrict_linux_filesystem

            with self.assertRaisesRegex(RuntimeError, "refusing unconfined"):
                restrict_linux_filesystem(ROOT)

    def test_strict_default_does_not_fall_back_on_isolation_failure(self):
        evaluator = self.evaluator(STRICT)
        self.assertEqual(evaluator.isolation["mode"], STRICT)
        with patch.object(evaluator, "_run", side_effect=RuntimeError("Landlock unavailable: ENOSYS")):
            with self.assertRaises(compiler.ReferenceValidationError):
                evaluator.baselines(30)
        self.assertTrue(all(item.evaluation.status == "infrastructure_error" for item in
                            evaluator.baseline_results if item.name == "eager"))

    def test_trusted_trajectory_with_simulated_enosys_in_real_worker(self):
        unavailable = {"available": False, "errno": errno.ENOSYS, "reason": "ENOSYS (simulated test)"}

        def simulated_command(directory, isolation_mode):
            command = worker_command(directory, isolation_mode)
            command[-1] = command[-1].replace(
                "candidate_worker.main()",
                "import evaluation_isolation; "
                f"evaluation_isolation.landlock_status = lambda: {unavailable!r}; "
                "candidate_worker.restrict_linux_filesystem = lambda directory: "
                "(_ for _ in ()).throw(RuntimeError('must not attempt confinement in explicit fixture mode')); "
                "candidate_worker.main()",
            )
            return command

        with patch("evaluation_isolation.landlock_status", return_value=unavailable), patch(
            "candidate_evaluator.worker_command", side_effect=simulated_command
        ):
            trace = run_optimization(TASK, MockAgentBackend(), self.evaluator(), max_rounds=3)
        payload = strict_loads(trace.to_json())
        self.assertEqual(payload["isolation"]["landlock"], unavailable)
        eager = payload["baseline_results"][0]["evaluation"]
        self.assertEqual(eager["status"], "measured")
        self.assertEqual(eager["diagnostics"]["isolation"]["landlock"], unavailable)
        self.assertFalse(payload["gpu_performance_evaluated"])

    def test_trusted_cli_trace_metadata_and_strict_replay_flag_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "trace.json"
            command = [sys.executable, str(ROOT / "agent_eval.py"), "--device", "cpu", "--agent", "mock",
                       "--trusted-fixture-colab", "--quick", "--max-rounds", "3", "--rows", "3", "--hidden-size", "8",
                       "--json-output", str(output)]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("NO OS SANDBOX", process.stderr)
            trace = OptimizationTrace.from_json(output.read_text())
            self.assertEqual(trace.isolation["mode"], TRUSTED_FIXTURE_COLAB)
            self.assertIn("exact repository mock", trace.isolation["trust_scope"])
            self.assertIn("available", trace.isolation["landlock"])
            self.assertFalse(trace.gpu_performance_evaluated)
            self.assertEqual([item.evaluation.status for item in trace.rounds], ["compile_error", "measured", "skipped"])
            self.assertTrue(all(item.evaluation.diagnostics["isolation"]["mode"] == TRUSTED_FIXTURE_COLAB
                                for item in trace.rounds))


if __name__ == "__main__":
    unittest.main()
