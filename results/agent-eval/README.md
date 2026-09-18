# V2 Tesla T4 evaluation records

These are measured Google Colab results for the 8192 × 128 FP16
`bias_layernorm_silu` task on a Tesla T4 with PyTorch 2.11.0+cu128,
CUDA runtime 12.8, and Triton 3.6.0. They validate the V2 evaluator loop
with the **deterministic repository mock agent**, not a frontier model,
training run, Harbor task, or general kernel-synthesis benchmark.

The published JSON and log files are **path-normalized copies** of the
original Colab outputs. No measurements or verification outcomes were
changed. [The publication manifest](PUBLICATION_MANIFEST.md) records
original and public hashes and the exact normalization rule.

The Colab host returned `ENOSYS` for Landlock. Each full trace explicitly
used `trusted_fixture_colab`: exact repository mock fixtures only, **no OS
sandbox**. The normal evaluator mode still fails closed without supported
confinement. Do not use this exception for replay or arbitrary candidate code.

## Three full evaluations

Each run used eight proposal rounds, eight correctness cases per executed
candidate, 100 warmups, 100 timed iterations, and five trial medians.
The first proposal intentionally has invalid syntax and is recorded as a
compile error; seven of eight proposals compiled and passed correctness.
Four candidates beat the best measured framework baseline in each run.

| Run | Best framework baseline | Best verified fused candidate | Selected round | Ratio |
| --- | ---: | ---: | ---: | ---: |
| [1](v2-full-t4-fp16.json) | Inductor default, 0.127168 ms | 0.057248 ms | 2 | 2.221× |
| [2](v2-full-t4-fp16-repeat2.json) | Inductor default, 0.124912 ms | 0.057120 ms | 3 | 2.187× |
| [3](v2-full-t4-fp16-repeat3.json) | Eager, 0.157040 ms | 0.057344 ms | 2 | 2.739× |

The three per-run ratios have a median of 2.221×; this is a descriptive
statistic, not a confidence interval or a guaranteed deployment speedup.
Full-run baselines and candidates were timed in separate worker processes.
The best configuration changed between runs, and `fused-2` was slower in
run 2. This motivated a same-process, interleaved confirmation.

## Paired finalist confirmation

[Public confirmation record](v2-paired-t4-fp16.json) reloaded rounds 2 and 3 from
run 1, checked the recorded repository source hashes, enforced exact mock
fixture matching, and passed all eight correctness cases for eager,
Inductor default, `fused-2`, and `fused-3` (zero mismatches at FP16
`atol=rtol=0.005`). It excluded compilation from timing and shuffled the
four paths over five rounds, each with 100 warmups and 100 CUDA-event
measurements. The median of the five per-round medians was:

| Path | Median latency |
| --- | ---: |
| Eager | 0.119824 ms |
| Inductor default | 0.140480 ms |
| `fused-2` | 0.055312 ms |
| `fused-3` | 0.053088 ms |

The fastest fused median divided into the fastest framework median gives
**2.257×** for this confirmation. Inductor max-autotune was measured in
the three full evaluations but **not** in this paired confirmation; it
was slower than the best framework path in every full run. This comparison
uses path medians, not one-to-one matched invocation ratios. Eager's first
two confirmation rounds were about 0.201 ms and its last three about
0.117–0.120 ms, so timing drift is material. The result supports a
repeatable T4 win for these trusted fixtures, but not a precise universal
speedup claim. No HBM counter, energy, cross-GPU, or statistical confidence
interval is reported.

The direct confirmation code had **no OS sandbox**, even though it accepted
only exact repository mock candidates. Reproduce it only in a trusted,
secret-free Colab runtime:

```bash
python3 colab/paired_confirmation.py \
  --trace results/agent-eval/v2-full-t4-fp16.json \
  --output results/agent-eval/v2-paired-t4-fp16-rerun.json
```

The `.log` files preserve terminal summaries. The JSON traces are the
primary records: they contain target fingerprints, source hashes,
candidate source, verification results, timing, rewards, and isolation
metadata. The confirmation JSON preserves all five per-path trial medians.
