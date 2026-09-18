# Colab NVIDIA workflow (T4 when assigned)

Use a GPU runtime and run these cells in order. This workflow preserves Colab's existing PyTorch/CUDA stack, detects the real GPU, runs the configuration-search and proposal-evaluation paths separately, and saves JSON with the actual hardware fingerprint. A non-T4 NVIDIA GPU is accepted and labeled honestly; its measurements must not be presented as T4 results. Clone a known repository revision for a reproducible run. This notebook workflow does not commit or push changes.

## 1. Clone

```python
import os
import subprocess
from pathlib import Path

root = Path('/content/kernel-relay')
if not root.exists():
    subprocess.run(['git', 'clone', 'https://github.com/pkarakala/kernel-relay.git', str(root)], check=True)
os.chdir(root)
assert Path('agent_eval.py').exists(), 'Upload the V2 working-tree files or check out a published V2 revision first'
```

## 2. Detect hardware and dependencies

```python
import importlib.metadata
import importlib.util
import platform
import sys

assert importlib.util.find_spec('torch') is not None, 'Use a Colab PyTorch GPU runtime; do not replace its CUDA stack blindly'
import torch

subprocess.run(['nvidia-smi'], check=True)
assert torch.cuda.is_available(), 'Enable a GPU runtime; no CUDA results can be produced on CPU'
if importlib.util.find_spec('triton') is None:
    requirements = importlib.metadata.requires('torch') or []
    triton_requirement = next((item.split(';')[0].strip() for item in requirements if item.lower().startswith('triton==')), None)
    assert triton_requirement is not None, 'No matching Triton pin found in PyTorch metadata; select a supported Colab runtime rather than guessing a version'
    subprocess.run([sys.executable, '-m', 'pip', 'install', triton_requirement], check=True)
import triton

print({'python': platform.python_version(), 'pytorch': torch.__version__, 'cuda': torch.version.cuda,
       'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
       'compute_capability': torch.cuda.get_device_capability(0)})
gpu = torch.cuda.get_device_name(0)
print('T4 run' if 'T4' in gpu else f'Not a T4: all results apply to {gpu}')
```

No model SDK, dataset, notebook package, Harbor, or Verifiers installation is needed. Only a missing PyTorch-matched Triton package is installed. Some Colab hosts return `ENOSYS` for the Landlock syscall under their seccomp policy, even with a recent kernel. Inspect capability without assuming confinement:

```python
from evaluation_isolation import landlock_status
print(landlock_status())
```

**Explicit trusted-fixture option:** the commands below opt into `--trusted-fixture-colab`, which runs only the exact repository-audited built-in mock proposals and trusted framework baselines, **without an OS sandbox**. This is not arbitrary-source isolation, production security, or a live frontier-model run. Source, launch parameters, and proposal metadata must match the canonical repository fixtures exactly; the parent evaluator and worker both check them. Replay, custom mock streams, and modified sources are forbidden. Trust the local repository and its dependencies before using this flag; do not mount secrets in the runtime. Sanitized worker environments, temporary directories, subprocess deadlines, and correctness gates remain active.

Colab exposes its NVIDIA driver library at `/usr/lib64-nvidia`. The sanitized worker includes only that fixed system path in `LD_LIBRARY_PATH` when present; it never copies an arbitrary notebook `LD_LIBRARY_PATH` into candidate processes. If an eager CUDA baseline still reports unavailable, inspect the worker's CUDA visibility before claiming a GPU result.

Without this explicit flag, strict Landlock isolation still fails closed when unavailable. On a host with working Landlock, omit the flag to use strict mode. There is no automatic downgrade.

## 3. Tests and existing compiler smoke

```python
results = Path('results/agent-eval')
results.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'py_compile', 'agentic_kernel_compiler.py', 'agent_eval.py'], check=True)
subprocess.run([sys.executable, '-m', 'unittest', '-v'], check=True)
subprocess.run([sys.executable, 'agentic_kernel_compiler.py', '--device', 'cuda', '--quick',
                '--rows', '257', '--hidden-size', '128', '--dtype', 'float16',
                '--json-output', str(results / 'v1-smoke.json')], check=True)
```

The unit suite is CPU-first even on CUDA hosts; successful unit tests alone do not validate Triton execution. If Landlock is unavailable, strict-confinement-dependent tests report `skipped` with the environment reason, not a passing security check. The trusted-fixture subprocess/CLI tests and other CPU tests still run. Inspect the skip summary; it is not evidence that strict confinement works.

## 4. Agent smoke and full evaluation

```python
subprocess.run([sys.executable, 'agent_eval.py', '--task', 'bias_layernorm_silu', '--device', 'cuda',
                '--agent', 'mock', '--trusted-fixture-colab', '--max-rounds', '3', '--quick', '--rows', '257', '--dtype', 'float16',
                '--json-output', str(results / 'v2-smoke.json')], check=True)
subprocess.run([sys.executable, 'agent_eval.py', '--task', 'bias_layernorm_silu', '--device', 'cuda',
                '--agent', 'mock', '--trusted-fixture-colab', '--max-rounds', '8', '--rows', '8192', '--hidden-size', '128',
                '--dtype', 'float16', '--trial-timeout', '180', '--baseline-timeout', '900',
                '--max-seconds', '2400', '--json-output', str(results / 'candidate_trace.json')], check=True)
```

Equivalent full shell command from the clone:

```bash
python3 agent_eval.py --task bias_layernorm_silu --device cuda --agent mock --trusted-fixture-colab --max-rounds 8 --rows 8192 --hidden-size 128 --dtype float16 --trial-timeout 180 --baseline-timeout 900 --max-seconds 2400 --json-output results/agent-eval/candidate_trace.json
```

This runs eager and both Inductor modes where supported, then deterministic mock source proposals with real target feedback. Max-autotune can take minutes; failures/timeouts remain in the trace. An incorrect kernel or a slower result is a valid experimental outcome, not permission to loosen tolerances or invent results.

## 5. Inspect and save

```python
import json
import zipfile

trace = json.loads((results / 'candidate_trace.json').read_text())
print(trace['target'])
print(trace['isolation'])
assert trace['isolation']['mode'] == 'trusted_fixture_colab'
assert trace['isolation']['filesystem_confinement'] == 'none'
print(trace['metrics'])
print('Agent GPU performance actually evaluated:', trace['gpu_performance_evaluated'])
for record in trace['baseline_results']:
    print(record['name'], record['evaluation']['status'], record['evaluation']['latency_ms'])
for record in trace['rounds']:
    print(record['round_index'], record['evaluation']['status'], record['evaluation']['compile_error'],
          record['evaluation']['correctness_error'], record['evaluation']['runtime_error'])

with zipfile.ZipFile('/content/agent-eval-artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(results.glob('*.json')):
        archive.write(path, path.name)
from google.colab import files
files.download('/content/agent-eval-artifacts.zip')
```

The trace and every baseline/candidate evaluation record identify the isolation mode and trust scope; `isolation.landlock` records availability and the error (for example `ENOSYS`) on the actual host. On macOS it reports not applicable, not a fictional Landlock failure.

Replay remains **strict-mode only**. Move the trace to an appropriately confined host before running:

```bash
python3 agent_eval.py --task bias_layernorm_silu --device cuda --agent replay --replay-file results/agent-eval/candidate_trace.json --max-rounds 8 --dtype float16 --json-output results/agent-eval/replayed.json
```

Do not add `--trusted-fixture-colab` to that replay command; it is rejected even if the trace originally came from mock. Replay remeasures on the current GPU; historical rewards/timings are never trusted. Check actual candidate/baseline statuses even when the CLI succeeds: safe fallback is a successful infrastructure run, not a GPU optimization result. Record isolation, source hashes, and hardware with any reported performance claim, and confirm apparent wins with repeated paired measurements before drawing conclusions. Local Mac tests do not establish that CUDA/Triton/Inductor work in this Colab runtime; the commands above still need actual GPU execution.

## 6. Confirm finalists on the same GPU

After a full trusted-mock run, use the bounded confirmation script to
revalidate rounds 2 and 3, then interleave eager, Inductor default, and
both fused paths in one process. It checks that repository source hashes
match the trace and that both proposals exactly match built-in mock
fixtures. It is **not sandboxed**; never use arbitrary or replay source.

```bash
python3 colab/paired_confirmation.py \
  --trace results/agent-eval/candidate_trace.json \
  --output results/agent-eval/paired_confirmation.json
```

The timing uses five shuffled path rounds, 100 warmups and 100 CUDA-event
samples per path per round, excluding first-use compilation. Inspect the
per-round values for drift, not only the ratio of medians. The
[recorded T4 results](../results/agent-eval/README.md) show the actual
full-run and paired-confirmation evidence and its limitations.
