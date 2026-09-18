"""CPU regression checks; no Triton import, GPU execution, or synthetic GPU timings.

Run with PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_agentic_kernel_compiler.py
to leave any existing __pycache__ untouched.
"""

import contextlib
import hashlib
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
import torch.nn.functional as F

import agentic_kernel_compiler as compiler


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "agentic_kernel_compiler.py"


class SemanticAndCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.inputs = compiler.make_inputs(3, 8, torch.float32, torch.device("cpu"))
        self.module = compiler.BiasAddLayerNormSiLU(8, eps=0.125)

    def graph(self):
        graph, error = compiler.capture_and_propagate(self.module, self.inputs)
        self.assertIsNone(error)
        return graph

    def match(self, graph, error=None):
        return compiler.find_fusion_region(graph, self.inputs, 8, error)

    def test_valid_match_extracts_epsilon(self):
        match = self.match(self.graph())
        self.assertEqual(len(match.nodes), 3)
        self.assertEqual(match.eps, 0.125)

    def test_shape_propagation_failure_is_ineligible(self):
        match = self.match(self.graph(), "intentional ShapeProp failure")
        self.assertFalse(match.nodes)
        self.assertIn("ShapeProp failed", match.reason)

    def test_semantic_rejections(self):
        # Each mutation isolates a different previously unchecked condition.
        mutations = [
            ("silu", "inplace", True, "non-inplace"),
            ("layer_norm", "normalized_shape", (4, 2), "normalized_shape"),
            ("layer_norm", "weight", None, "both exist"),
            ("layer_norm", "bias", None, "both exist"),
            ("layer_norm", "eps", float("nan"), "epsilon"),
            ("layer_norm", "eps", 0.0, "epsilon"),
            ("add", "alpha", 2, "alpha=1"),
        ]
        for node_name, key, value, reason in mutations:
            with self.subTest(node=node_name, key=key, value=value):
                graph = self.graph()
                node = next(node for node in graph.graph.nodes if node.name == node_name)
                if key == "normalized_shape":
                    node.args = (node.args[0], value)
                else:
                    node.kwargs = {**node.kwargs, key: value}
                match = self.match(graph)
                self.assertFalse(match.nodes)
                self.assertIn(reason, match.reason)

    def test_metadata_rejections(self):
        for name in ("x", "input_bias", "gamma", "beta", "add", "layer_norm", "silu"):
            for change in ("missing", "shape", "dtype"):
                with self.subTest(node=name, change=change):
                    graph = self.graph()
                    node = next(node for node in graph.graph.nodes if node.name == name)
                    if change == "missing":
                        del node.meta["tensor_meta"]
                    else:
                        values = {"shape": (99,)} if change == "shape" else {"dtype": torch.float16}
                        node.meta["tensor_meta"] = node.meta["tensor_meta"]._replace(**values)
                    self.assertFalse(self.match(graph).nodes)

    def test_external_users_and_wrong_operand_binding(self):
        for name in ("add", "layer_norm"):
            graph = self.graph()
            node = next(node for node in graph.graph.nodes if node.name == name)
            with graph.graph.inserting_after(node):
                graph.graph.call_function(torch.neg, (node,))
            self.assertIn("external users", self.match(graph).reason)
        graph = self.graph()
        nodes = {node.name: node for node in graph.graph.nodes}
        nodes["add"].args = (nodes["x"], nodes["gamma"])
        self.assertIn("input_bias", self.match(graph).reason)
        graph = self.graph()
        nodes = {node.name: node for node in graph.graph.nodes}
        nodes["layer_norm"].kwargs = {**nodes["layer_norm"].kwargs, "weight": nodes["beta"]}
        self.assertIn("bind", self.match(graph).reason)

    def test_invalid_earlier_chain_does_not_hide_valid_chain(self):
        class TwoChains(torch.nn.Module):
            def forward(self, x, input_bias, gamma, beta):
                bad = F.layer_norm(x + input_bias, (8,), gamma, beta)
                F.silu(bad)
                torch.neg(bad)  # external user invalidates only this first chain
                return F.silu(F.layer_norm(x + input_bias, (8,), gamma, beta, eps=0.25))

        graph, error = compiler.capture_and_propagate(TwoChains(), self.inputs)
        match = self.match(graph, error)
        self.assertTrue(match.nodes)
        self.assertEqual(match.eps, 0.25)
        self.assertEqual(len(match.rejections), 1)
        self.assertIn("external users", match.rejections[0])

    def test_keyword_and_reversed_add_form(self):
        class KeywordForm(torch.nn.Module):
            def forward(self, x, input_bias, gamma, beta):
                return F.silu(input=torch.layer_norm(
                    input=torch.add(input=input_bias, other=x), normalized_shape=(8,),
                    weight=gamma, bias=beta, eps=0.03125,
                ))

        graph, error = compiler.capture_and_propagate(KeywordForm(), self.inputs)
        match = self.match(graph, error)
        self.assertTrue(match.nodes, match.reason)
        self.assertEqual(match.eps, 0.03125)

    def test_correctness_metrics_near_zero_and_mismatch(self):
        reference = torch.zeros(2)
        passing = compiler.verify_output(torch.tensor([0.00005, 0.0]), reference)
        self.assertTrue(passing.passed)
        self.assertTrue(math.isinf(passing.max_rel))
        self.assertLessEqual(passing.max_normalized, 1)
        self.assertEqual(passing.mismatched, 0)
        failing = compiler.verify_output(torch.tensor([0.0002, 0.0]), reference)
        self.assertFalse(failing.passed)
        self.assertGreater(failing.max_normalized, 1)
        self.assertEqual(failing.mismatched, 1)
        self.assertEqual(failing.mismatch_percent, 50)
        for candidate in (torch.zeros(3), torch.zeros(2, dtype=torch.float16),
                          torch.tensor([float("nan"), 0]), torch.tensor([float("inf"), 0])):
            self.assertFalse(compiler.verify_output(candidate, reference).passed)
        self.assertFalse(compiler.verify_output(reference, torch.tensor([float("inf"), 0])).passed)

    def test_validation_suite_is_deterministic_and_stops_on_failing_case(self):
        for dtype in (torch.float16, torch.float32):
            for name in compiler.VALIDATION_CASES:
                first = compiler.make_inputs(3, 8, dtype, torch.device("cpu"), name)
                second = compiler.make_inputs(3, 8, dtype, torch.device("cpu"), name)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
                if name in ("all_zero", "constant_rows"):
                    self.assertTrue(((first[0] + first[1]).float().var(dim=-1, unbiased=False) == 0).all())
        calls = 0

        def wrong_on_third(*inputs):
            nonlocal calls
            calls += 1
            output = self.module(*inputs)
            return output + 1 if calls == 3 else output

        suite = compiler.ValidationSuite(self.module, self.inputs, torch.device("cpu"))
        result = suite.validate(wrong_on_third)
        self.assertFalse(result.passed)
        self.assertEqual(calls, 3)
        self.assertIn("random_seed_plus_2", result.reason)

    def test_deployment_margin_and_cpu_failure_exclusion(self):
        self.assertTrue(compiler.triton_clears_margin(0.98, 1.0, 0.02))
        self.assertFalse(compiler.triton_clears_margin(0.981, 1.0, 0.02))
        self.assertFalse(compiler.triton_clears_margin(1.0, 1.0, 0.0))

        def broken_cpu_path():
            raise RuntimeError("intentional CPU timing failure")

        trials, failures = compiler.compare_paths(
            {"working CPU": lambda: torch.ones(1), "broken CPU": broken_cpu_path},
            torch.device("cpu"), 0, 1, 2,
        )
        self.assertEqual(len(trials["working CPU"]), 2)
        self.assertNotIn("broken CPU", trials)
        self.assertEqual(failures["broken CPU"]["classification"], "runtime")


class JsonOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix=".hardening-tests-", dir=ROOT)
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = Path(cls.directory.name) / "results.json"
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cls.directory.name) / "inductor")
        cls.process = subprocess.run(
            [sys.executable, str(SCRIPT), "--device", "cpu", "--quick", "--rows", "3",
             "--hidden-size", "8", "--warmup", "0", "--iters", "1", "--json-output", str(cls.path)],
            env=environment, capture_output=True, text=True, timeout=180,
        )
        if cls.process.returncode:
            raise AssertionError(cls.process.stdout + cls.process.stderr)

        def reject_constant(value):
            raise AssertionError(f"nonstandard JSON number: {value}")

        cls.report = json.loads(cls.path.read_text(), parse_constant=reject_constant)

    def test_schema_and_provenance(self):
        report = self.report
        self.assertEqual(report["schema_version"], "1.0")
        self.assertTrue(report["evaluation_completed"])
        self.assertTrue(report["timestamp"].endswith("+00:00"))
        self.assertEqual(report["cli_arguments"]["device"], "cpu")
        self.assertEqual(report["cli_arguments"]["json_output"], str(self.path))
        self.assertEqual(report["random_seed"], compiler.INPUT_SEED)
        self.assertEqual(report["provenance"]["script_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(report["hardware"]["device"], "cpu")
        for name in ("gpu_name", "compute_capability", "memory_bytes", "triton_version"):
            self.assertIsNone(report["hardware"][name])
        self.assertIn("pytorch_version", report["hardware"])
        self.assertIn("cuda_version", report["hardware"])
        self.assertEqual(report["graph"]["fusion_boundary"], ["add", "layer_norm", "silu"])
        self.assertEqual(len(report["graph"]["nodes"]), 8)
        self.assertEqual(report["analytical_model"]["useful_bytes"], (2 * 3 * 8 + 3 * 8) * 4)

    def test_candidates_and_paths_are_honest(self):
        report = self.report
        counts = report["candidate_counts"]
        self.assertEqual(counts["generated"], 36)
        self.assertEqual(counts["executed"], 0)
        self.assertEqual(counts["verified"], 0)
        self.assertEqual(counts["failed"], 0)
        self.assertEqual(counts["incorrect"], 0)
        self.assertEqual(counts["generated"], counts["pruned"] + counts["skipped"] + counts["executed"])
        self.assertEqual(len(report["candidates"]), 36)
        self.assertTrue(all(not candidate["launched"] for candidate in report["candidates"]))
        self.assertTrue(all(candidate["initial_sweep"]["median_ms"] is None for candidate in report["candidates"]))
        self.assertEqual(set(report["paths"]), {"eager", "torch_compile", "triton"})
        self.assertIsNone(report["paths"]["triton"]["latency_ms"])
        self.assertIsNone(report["selection"]["configuration"])
        self.assertIn("CPU", report["selection"]["fallback_reason"])
        self.assertIsNotNone(report["selection"]["fallback_path"])
        eager = report["paths"]["eager"]
        self.assertGreater(eager["latency_ms"], 0)
        self.assertEqual(eager["timing"]["rounds"], 2)
        for key in ("max_abs", "max_rel", "max_normalized_error", "mismatch_count"):
            self.assertEqual(eager["correctness"][key], 0)
        self.assertEqual(eager["speedup_over_eager"], 1)
        self.assertGreater(eager["estimated_effective_bandwidth_gbps"], 0)
        self.assertIn("Performance summary", self.process.stdout)

    def test_atomic_write_and_nonfinite_encoding(self):
        path = Path(self.directory.name) / "atomic.json"
        path.write_text("previous artifact")
        with patch.object(compiler.os, "replace", side_effect=OSError("intentional replace failure")):
            with self.assertRaises(OSError):
                compiler.write_json_atomic(path, {"number": 1})
        self.assertEqual(path.read_text(), "previous artifact")
        self.assertFalse(list(path.parent.glob(".atomic.json.*.tmp")))
        compiler.write_json_atomic(path, {"inf": math.inf, "nan": math.nan, "value": 1})
        self.assertEqual(json.loads(path.read_text()), {"inf": None, "nan": None, "value": 1})

    def test_incomplete_evaluation_preserves_existing_artifact(self):
        path = Path(self.directory.name) / "incomplete.json"
        path.write_text("previous artifact")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                compiler.main(["--rows", "0", "--json-output", str(path)])
            self.assertEqual(caught.exception.code, 2)
            with patch.object(compiler.ValidationSuite, "validate",
                              side_effect=compiler.ReferenceValidationError("intentional oracle failure")):
                code = compiler.main(["--device", "cpu", "--quick", "--rows", "2",
                                      "--hidden-size", "8", "--json-output", str(path)])
            self.assertEqual(code, 1)
        self.assertEqual(path.read_text(), "previous artifact")

    def test_compile_failure_is_reported_with_eager_fallback(self):
        path = Path(self.directory.name) / "compile-fallback.json"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            compiler.torch, "compile", side_effect=RuntimeError("intentional unavailable compiler")
        ):
            code = compiler.main(["--device", "cpu", "--quick", "--rows", "2", "--hidden-size", "8",
                                  "--warmup", "0", "--iters", "1", "--json-output", str(path)])
        self.assertEqual(code, 0)
        report = json.loads(path.read_text())
        path_result = report["paths"]["torch_compile"]
        self.assertIsNone(path_result["latency_ms"])
        self.assertIsNone(path_result["correctness"])
        self.assertEqual(path_result["fallback_path"], "eager")
        self.assertIn("intentional unavailable compiler", path_result["failure"]["summary"])
        self.assertEqual(report["selection"]["implementation"], "Eager PyTorch")
        self.assertEqual(report["paths"]["eager"]["timing"]["rounds"], 2)

    @unittest.skipIf(torch.cuda.is_available(), "requires a CPU-only host")
    def test_explicit_cuda_returns_two_without_artifact(self):
        path = Path(self.directory.name) / "cuda.json"
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = compiler.main(["--device", "cuda", "--quick", "--json-output", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("CUDA is unavailable", errors.getvalue())
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
