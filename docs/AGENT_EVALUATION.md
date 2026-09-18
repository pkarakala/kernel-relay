# KernelRelay agent evaluation (V2)

## Experiment and version boundary

Given one PyTorch workload and one NVIDIA target, can an agent propose and refine an implementation that beats verified framework baselines within a fixed budget?

V1 searches launch configurations for a handwritten kernel. V2 admits **actual implementation source** and feeds compile errors, numerical errors, latency, and remaining budget back to an `AgentBackend`. Mock/replay agents exercise this interface without an external API. Mock proposals are predefined fixtures, not evidence of reasoning or learning; replay re-executes proposals, never imports another machine's correctness claims, latencies, or rewards.

This repository does not yet train a frontier model.
It generates evaluation trajectories that can later be used for
supervised fine-tuning, preference optimization, or reinforcement learning.

Harbor and Prime Intellect Verifiers are not dependencies for this POC.
The trace/evaluator interfaces are intentionally structured so integration
with those systems could be added later.

The V1 executable, tests, and original `results/t4` artifacts remain unchanged. Those artifacts are not V2 GPU evidence. No Astra API is implemented or assumed.

## Architecture

```text
PyTorch task (optimization_tasks.py)
     |
     v
FX graph + shapes + dtype + target fingerprint
     |
     v
AgentRequest --> AgentBackend (mock / replay / future API adapter)
                     |
                     v
              AgentProposal: implementation source + strategy + launch parameters
                     |
                     v
              CandidateEvaluator (trusted controller)
                     |
                     +--> schema / source policy / target eligibility
                     +--> temporary evaluation subprocess (strict confinement by default)
                              compile / first execution
                                      |
                              eager numerical oracle
                                      |
                              correctness hard gate
                                      |
                              synchronized steady-state benchmark
                     |
                     v
              EvaluationResult --> pure reward + metrics
                     |
                     v
              OptimizationTrace -- structured history --> next bounded round
                     |
                     v
              best verified candidate vs best verified framework baseline
                     |
                     v
              candidate or verified framework fallback

Training: not implemented; downstream consumer of future evaluated trajectories
```

The controller owns history, selection, and reward. Requests are deep-copied so a backend cannot mutate retained feedback. Generated code does not run in the controller. Reusable V1 functions provide eight deterministic correctness cases, FP16/FP32 tolerances, semantic graph analysis, synchronization, robust timing, and atomic output. The new task carries module/input/validation factories; adding a different input signature also requires a corresponding candidate contract and verifier, not just a dataset import.

## Run locally

```bash
python3 agent_eval.py --list-tasks
python3 agent_eval.py --task bias_layernorm_silu --device cpu --agent mock --max-rounds 3 --quick
python3 agent_eval.py --device cpu --agent replay --replay-file examples/replay_cpu.json --max-rounds 3 --quick --json-output results/agent-eval/cpu-replay.json
```

CPU is the canonical local fallback; MPS optimization is out of scope. CPU runs execute the eager task, FX capture, validation, orchestration, PyTorch-source candidates, real CPU timing, schema/reward/metric calculations, and trace writing. Inductor GPU baselines and Triton execution are explicitly skipped on CPU. `gpu_performance_evaluated` is false; CPU latency and speedup fields are **CPU measurements only**, not CUDA predictions.

`auto` selects CUDA if available, otherwise CPU. Explicit unavailable `--device cuda` returns exit code 2 without replacing an output artifact. Invalid reference behavior or an unmeasurable eager baseline fails the run; candidate failures instead become round feedback. Successful evaluation returns 0 even when every optimization fails. CLI/data errors return 2; fatal evaluation errors return 1. Use a new output path to retain earlier completed runs.

`--quick` preserves shape and all correctness cases, but reduces timing from 100 warmups, 100 samples, 5 trials to 3 warmups, 10 samples, 2 trials. Use `--rows 257` for a smaller GPU smoke workload. Quick timing is infrastructure validation, not strong performance evidence.

## Agent protocol and source contract

`AgentBackend.propose(request: AgentRequest) -> AgentProposal` is the only backend method. Requests contain the task description and PyTorch source, FX graph/code/metadata, tensor shapes/strides/dtype, tolerance policy, target hardware/software fingerprint, previous proposals and evaluator feedback, baseline records, and rounds/seconds remaining. `request.to_json()` is a ready structured prompt payload; no natural-language result parsing is necessary.

Proposal fields (schema `2.0`):

| Field | Meaning |
|---|---|
| `candidate_id` | Human-readable ID, never used as a filesystem path |
| `candidate_type` | `eager`, `pytorch_source`, or `triton_source` |
| `strategy` | Short strategy summary, not private chain-of-thought |
| `source` | Python implementation, at most 64 KiB; null for eager |
| `launch_parameters` | Optional positive integer `block_size`, `num_warps`, `num_stages` |
| `expected_optimization` | Hypothesis, not an evaluator assertion |

Unknown fields, including `correct`, `latency_ms`, and `reward`, are rejected. Duplicate JSON keys and NaN/Infinity input are rejected. Non-finite diagnostic output is encoded as JSON null.

Source candidates must define:

```python
def run(x, input_bias, gamma, beta, *, eps, launch_parameters):
    ...
```

Return one tensor with reference shape/dtype/device, and leave all inputs unchanged. Supported imports are canonical `import torch`, `import torch.nn.functional as functional`, and, for Triton only, `import triton` / `import triton.language as tl`. Only public top-level functions, constant defaults, a small expression/control-flow subset, whitelisted tensor/module operations, and local `@triton.jit` kernels are accepted. No filesystem/network APIs, arbitrary imports, private attributes, module aliasing, attribute writes, subscript writes, dynamic evaluation, or unrestricted Python. See `candidate_policy.py` for the exact allowlists. The first task's Triton execution limit remains hidden size 256; unsupported shapes fall back honestly. The architecture accepts source changes, not just launch-knob choices.

The mock emits invalid syntax, an eager fallback, then six predefined fused-source candidates: V1's kernel body reused without rewriting it, plus a SiLU implementation variant and warp variations. None has a preassigned winning configuration or latency. The supplied replay example executes actual CPU-compatible source. Any completed V2 trace can be a replay input:

```bash
python3 agent_eval.py --device cuda --agent mock --max-rounds 8 --json-output results/agent-eval/candidate_trace.json
python3 agent_eval.py --device cuda --agent replay --replay-file results/agent-eval/candidate_trace.json --max-rounds 8 --json-output results/agent-eval/replayed.json
```

A replay bundle is `{"schema_version":"2.0","task_id":"bias_layernorm_silu","proposals":[...]}`. A trace uses `rounds[].proposal`; null proposals from failed agent calls are skipped. Exhaustion stops normally. Malformed individual proposals consume a failed round but later proposals remain usable. Task IDs must match. Replay ignores recorded device settings; the new CLI controls shape/dtype/device and all proposals are reverified there.

### Future frontier-model adapter

Implement the same protocol in `agent_backends/`, serialize the request, ask the provider for the proposal schema, and validate with `AgentProposal.from_dict`. Keep credentials solely in the trusted adapter process; never place them in requests, sources, traces, or worker environments. Honor the request deadline with provider/network timeouts. Backend calls are trusted Python: the controller checks the wall deadline after a call but cannot forcibly interrupt a hung in-process adapter. An untrusted/remote backend needs its own timeout-capable transport. There is no fabricated Astra endpoint or SDK.

## Evaluator and baselines

The evaluator performs compile/load, execute, verify, and **only then** benchmark. It reuses the eager oracle and V1's eight deterministic cases with FP16 `(atol, rtol) = (5e-3, 5e-3)` and FP32 `(1e-4, 1e-4)`. Correctness checks shape, dtype, device, finite values, elementwise errors, and input immutability. It also verifies again after timing; failed/partial measurements never participate in selection.

CUDA baselines are eager, `torch.compile(..., mode="default")`, and `torch.compile(..., mode="max-autotune")`. Every available baseline is verified independently before timing. `--skip-max-autotune` explicitly disables the expensive optional mode; it is enabled by default even in quick mode. Baselines run in separate bounded workers, so one compiler failure or timeout does not discard eager or another successful mode.

Statuses distinguish `unavailable`, `compile_error`, `incorrect`, and `measured`, plus runtime errors, timeouts, infrastructure errors, and candidate skips. The authoritative baseline is the minimum finite positive latency across **measured, compiled, correct** baselines only. Eager proposals reuse the verified eager record as a labeled alias; they cannot acquire a noise-induced gain from duplicate timing.

CUDA uses synchronized events; CPU uses `perf_counter_ns`. V1's untimed setup call, warmups, deterministic seeds, median-of-trial-medians statistic, and dispersion records are retained. Compilation, validation, process startup, and autotuning are outside steady-state measurements but inside optimization wall time. Each process has fresh temporary compilation caches. CPU workers use one thread consistently; Inductor compilation uses one worker to avoid shared-memory worker pools outside the trial. The initial framework measurements and candidate trials are **not interleaved across subprocesses**; thermal/clock drift and noise remain limitations. Selection uses the lowest verified observed latency, retaining the framework on ties; this is not a significance test or a production deployment decision. Repeated paired confirmation on the GPU is the recommended next measurement improvement.

## Budget, traces, rewards, metrics

Budgets are bounded by `--max-rounds` (1–64), `--trial-timeout` (default 120 seconds per source trial), `--baseline-timeout` (600 seconds total), and `--max-seconds` (1800 seconds including baseline setup, proposal calls, and trials). Task construction/graph capture precede that optimization clock. Compiler-default receives half the remaining baseline budget to leave room for max-autotune. Child process groups are killed on deadline, and trial directories are removed. Local mock/replay calls are immediate; future adapters must enforce their own call deadlines.

Each `OptimizationRound` records the exact structured request, proposal, evaluator result, reward, and any agent error. History excludes nested request copies to avoid exponential growth. `OptimizationTrace` records schema version, timestamp, task, hardware, baselines, rounds, budgets, stop reason, selected round, fallback/final result, metrics, and source/replay hashes. `selected_round` is null when the framework wins; `final_result.best_verified_round` still identifies the fastest verified proposal, even if slower. `gpu_performance_evaluated` becomes true only for an actually measured non-alias agent candidate on CUDA. A completed trajectory may contain failed rounds; it is not proof of optimization success.

The pure reward is `-1` for failed compilation, execution, correctness, invalid proposal, agent failure, or timeout; null for unavailable/skipped/unmeasured cases; otherwise `log(best_verified_baseline_ms / candidate_ms)`. The implementation subtracts logarithms to avoid ratio overflow. No cost penalty is enabled. Incorrect candidates never receive positive performance reward. Reward is not evidence that a model improved.

Run metrics include compile-success and correctness rates, best verified candidate/baseline latency, speedup over eager, best compiler-only baseline and best overall baseline, optimization-round count, correct-candidate count, faster-than-baseline count, and total optimization wall time. Both rates use non-skipped attempted rounds as denominator, including failed proposals; an empty denominator yields null. Correctness-passing candidates that subsequently fail timing count as correct but cannot win or receive a positive reward. Compiler-only speedup is null when no compiler baseline was verified.

`suite_speedup_metrics` is a pure future multi-task reducer for strict `speedup > p` fractions (`fast_1.0`, `fast_1.1`, `fast_1.5`) and geometric mean. Missing/failed tasks remain in fraction denominators; geometric mean is unavailable if any task is unmeasured. The single-task CLI does not emit or claim meaningful suite statistics. Held-out re-evaluation, SFT/preference construction, RL rollouts, policy optimization, and training are future work.

## Security boundary and limitations

### Explicit Colab trusted-fixture mode

Some Colab environments block Landlock syscall 444 with `ENOSYS` through host seccomp. Default V2 evaluation still fails closed in that environment. For this narrow demonstration, explicitly opt into the following **mock-only** mode:

```bash
python3 agent_eval.py --task bias_layernorm_silu --device cuda --agent mock --trusted-fixture-colab --max-rounds 3 --quick --rows 257 --dtype float16 --json-output results/agent-eval/trusted-fixture-smoke.json
python3 agent_eval.py --task bias_layernorm_silu --device cuda --agent mock --trusted-fixture-colab --max-rounds 8 --rows 8192 --hidden-size 128 --dtype float16 --trial-timeout 180 --baseline-timeout 900 --max-seconds 2400 --json-output results/agent-eval/trusted-fixture-full.json
```

The same flag supports `--device cpu` for local infrastructure testing. It deliberately uses **no OS filesystem/network sandbox**, even if strict confinement would be available; no automatic downgrade is performed. It can run only the fixed `bias_layernorm_silu` task, trusted eager/Inductor baselines, and the eight built-in mock proposals. Both evaluator and worker require canonical exact equality of the full proposal, including source bytes, parameters, and metadata, to fixtures reconstructed from the trusted repository. Agent-supplied hashes, alternate source text (even added whitespace), other tasks/operations, and arbitrary proposals cannot authorize execution. The trust root is the audited repository and installed dependencies, not an immutable vendor signature; modifying trusted repository code changes that trust root.

`--agent replay --trusted-fixture-colab` is rejected before evaluation. The controller also rejects replay objects, custom mock proposal streams, and mock subclasses in this mode. A recorded mock trajectory is not itself a trusted fixture bundle: replay it only under normal strict confinement. The syntactically invalid first mock fixture remains a recorded compile failure, eager remains a verified alias, and Triton fixtures are still skipped on CPU. Correctness checks, sanitized environments, temporary files, size limits, and subprocess timeouts are retained.

`OptimizationTrace.isolation`, request contracts, and each evaluator result's `diagnostics.isolation` record `mode`, `explicit_opt_in`, `trust_scope`, `filesystem_confinement`, `landlock` availability/error, and `production_sandbox: false`. Successful workers also report the actual enforcement used. An unavailable Linux Landlock probe is recorded with its error code/reason; on macOS the probe is not applicable. Trust metadata is independent of measured correctness and performance: a GPU result never proves sandbox security or model intelligence. The CLI prints a no-sandbox warning for every opted-in run.

Strict-confinement-dependent tests explicitly skip with the capability/environment reason when Landlock is absent. Other tests, including real trusted-fixture subprocess/CLI tests, still run. A skip is **not** a passed isolation test; rule-installation errors on a host advertising Landlock still fail rather than being hidden by a broad exception skip. Production sandboxing and a live frontier-model backend remain out of scope.

### Default strict mode

- Only evaluator-controlled filenames are written in a fresh temporary `.agent-trial-*` directory inside this repository. Trusted worker code is copied there; the worker does not import repository modules in place.
- Worker environments are built from a small allowlist, not copied from `os.environ`. API keys, tokens, `PYTHONPATH`, and user compiler hooks are not inherited. HOME, caches, and temporary paths point to the trial. Only `CUDA_VISIBLE_DEVICES` is optionally propagated.
- In default strict mode, macOS `sandbox-exec` denies writes outside the trial, denies network, and denies reads under the real user home except required interpreter prefixes and the trial. Linux Landlock restricts writes to the trial and restricts reads/execution to runtime/system/device paths; GPU device handles are necessarily allowed. Unsupported confinement fails closed. Linux policy requires Landlock-enabled x86-64/AArch64; a disabled host policy remains a blocker for replay/arbitrary-source evaluation. Only the explicit trusted-fixture mode above avoids that requirement for canonical mock fixtures.
- The AST policy and restricted builtins are defense in depth, not a claim that Python or native Triton is a complete sandbox. Linux Landlock is a filesystem policy, not a network or GPU-memory isolation system. Public source APIs do not expose network access. Candidates and verifier share a worker address space; native library/compiler exploits, GPU memory attacks, resource exhaustion, and malicious benchmark-specific behavior are not solved.
- The evaluator caps source size, response size, output-file size, and execution time, but does not enforce a complete memory/device quota. No credentials should exist on a machine evaluating genuinely hostile code. Production requires disposable containers/VMs, read-only mounts, network restrictions, device/resource quotas, and an independent verifier.

This is not full KernelBench, Harbor, Prime Verifiers, RL training, distributed/multi-GPU execution, AMD support, arbitrary CUDA C++ synthesis, a persistent learned kernel database, or production sandbox security. Actual CUDA/Triton/Inductor V2 validation must be performed in [Colab](../colab/README.md); Mac tests cannot establish it.
