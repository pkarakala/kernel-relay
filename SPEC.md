# KernelRelay specification

## Version boundary

The specification below describes **V1**, retained as the self-contained `agentic_kernel_compiler.py` deliverable. **V2** extends rather than replaces that substrate with `agent_eval.py`, a task registry, structured agent backends, isolated candidate evaluation, bounded feedback, and versioned trajectories. Its implemented contract, security limits, baseline/reward rules, and extension points are specified in [Agent evaluation V2](docs/AGENT_EVALUATION.md). The original `results/t4` artifacts validate V1; [separate V2 T4 records](results/agent-eval/README.md) validate the trusted mock-fixture evaluation path on a Colab GPU without an OS sandbox.

## Goal

Build a credible, end-to-end demonstration of a hardware-aware compiler loop:

```text
PyTorch module
  -> FX graph capture
  -> cost and fusion analysis
  -> target inspection
  -> candidate Triton configurations
  -> compile + numerical verification
  -> benchmark + select
  -> safe fallback
```

The prototype should demonstrate the engineering ideas behind an agentic compiler without claiming to reproduce a production compiler or a multi-vendor deployment system.

## Deliverable

One executable Python file:

```text
agentic_kernel_compiler.py
```

PyTorch is required. Triton is optional and must be imported only when a supported CUDA environment is detected. The script must run to completion on a CPU-only machine with PyTorch installed.

## Workload

Define a `torch.nn.Module` that computes:

```python
y = silu(layer_norm(x + input_bias, weight=gamma, bias=beta))
```

Use a realistic default such as `rows=8192`, `hidden_size=128`, with CLI overrides. Keep the last dimension contiguous. Treat all leading dimensions as rows.

The distinction between `input_bias` and LayerNorm's affine `beta` must be clear in names and output.

## Stage 1: graph capture and analysis

Use `torch.fx.symbolic_trace` and shape propagation to capture the module. Print an ASCII node table with at least:

- node name
- FX operation kind
- target
- inferred shape
- inferred dtype
- whether the node is inside the proposed fusion region

Pattern-match the `add -> layer_norm -> silu` chain. Do not merely assume that the pattern exists.

Estimate:

- scalar FLOPs, with the counting convention disclosed
- logical tensor bytes for the eager operator sequence
- logical tensor bytes for the fused sequence
- arithmetic intensity in FLOPs/byte
- avoided intermediate bytes

These are analytical estimates, not profiler counters. LayerNorm and transcendental operation counts are approximate and must be labeled as such.

## Stage 2: target-aware candidate generation

Probe the active target and print a compact hardware fingerprint. On CUDA include the device name, compute capability, total memory, and relevant PyTorch/Triton versions. On CPU include the processor/platform and PyTorch version.

The optimization controller should make its phases visible:

1. observe the FX graph and target
2. decide whether the cluster is eligible
3. propose configurations
4. compile and execute candidates
5. verify against the reference
6. benchmark valid candidates
7. select the fastest verified candidate or fall back

Explore this requested Cartesian search space:

- `BLOCK_SIZE`: 32, 64, 128, 256
- `num_warps`: 2, 4, 8
- `num_stages`: 2, 3, 4

Prune configurations that cannot cover `hidden_size` or violate target/kernel constraints. A pruned candidate is not a failed candidate. Catch compilation/runtime failures and continue the search.

`num_stages` may have little effect for this reduction/pointwise kernel; report that honestly rather than claiming all knobs are equally meaningful.

## Stage 3: fused Triton kernel

For eligible CUDA inputs, implement a Triton JIT kernel that uses one program per row:

1. masked-load `x`, `input_bias`, `gamma`, and `beta`
2. accumulate LayerNorm statistics in FP32
3. normalize and apply affine parameters
4. apply numerically stable SiLU
5. write the final output once

Do not materialize the post-add or post-LayerNorm intermediates in global memory. Support at least FP16 and FP32 inputs. Document or guard the supported hidden-size limit and shape assumptions.

The kernel can be template-generated/parameterized inside the script; do not describe a fixed handwritten kernel plus grid search as unrestricted program synthesis. Call it a bounded candidate generator or synthesis loop.

## Stage 4: correctness and fallback

Use eager PyTorch as the numerical oracle. For every candidate:

- compare shape and dtype
- check finite values
- calculate maximum absolute and relative error
- apply dtype-appropriate tolerances
- reject candidates that fail

If CUDA/Triton is unavailable, the agent-selected path must fall back to `torch.compile` when available and working, otherwise eager PyTorch. The script must not crash merely because Triton, CUDA, or `torch.compile` is unavailable.

On a CPU-only run, the final table must explicitly show that Triton tuning was skipped and that the third path is a fallback; it must not invent a block configuration.

## Stage 5: benchmark

Benchmark these steady-state paths:

1. eager PyTorch reference
2. `torch.compile`/Inductor, with a labeled eager fallback if compilation fails
3. selected verified Triton kernel, or the labeled CPU fallback

Use 100 warm-up iterations and 100 measured iterations by default. Provide `--quick` for a much shorter smoke test. Exclude first-call compilation and autotuning from steady-state timing. Synchronize CUDA before and after measurements. Report median latency, or another robust statistic if clearly stated.

For comparable effective bandwidth, use the same useful-byte definition for all paths:

```text
input tensor + output tensor + one copy of each parameter vector
```

Report it as `estimated effective bandwidth`, not measured HBM bandwidth. Separately print the analytical eager/fused traffic model so the fusion benefit is visible without mixing incompatible byte definitions.

## Output

Print concise ASCII sections for:

- target fingerprint
- captured FX graph
- detected fusion region and cost model
- candidate decisions and rejection counts
- selected configuration, or explicit fallback reason
- correctness results
- performance summary with latency, estimated effective GB/s, and speedup over eager

Example column layout:

```text
+------------------------------+------------+------------------+---------+
| Path                         | Median ms  | Est. useful GB/s | Speedup |
+------------------------------+------------+------------------+---------+
| Eager PyTorch                | ...        | ...              | 1.00x   |
| torch.compile                | ...        | ...              | ...     |
| Agent fused Triton/fallback  | ...        | ...              | ...     |
+------------------------------+------------+------------------+---------+
```

## CLI and exit behavior

Support at least:

```text
--device auto|cpu|cuda
--rows INT
--hidden-size INT
--dtype float16|float32
--warmup INT
--iters INT
--quick
```

Return a nonzero exit code for invalid CLI input or reference-correctness failure. Unsupported optimization cases should normally use the safe fallback and finish successfully.

## Acceptance criteria

- A CPU-only run completes and prints all summary sections with an honest fallback.
- A supported CUDA run explores valid members of the requested search space, validates each executed candidate, and selects the fastest verified configuration from measured results.
- The FX fusion boundary is discovered from the graph.
- No intermediate tensors are written by the Triton fused path.
- Timing excludes compile/tune overhead.
- Metrics clearly distinguish analytical estimates from measurements.
- The code is readable enough to explain in a 5-10 minute technical walkthrough.

## Demo narrative

Frame the prototype as a miniature closed-loop compiler:

- **Observe:** capture the graph and quantify a memory-bound region.
- **Reason:** choose a safe fusion boundary and target-specific constraints.
- **Act:** generate bounded kernel configurations.
- **Verify:** compare every candidate with the framework reference.
- **Learn/select:** retain the fastest verified configuration for that run and hardware fingerprint.
- **Fail safely:** preserve correctness through `torch.compile` or eager fallback.

Be explicit about what is not demonstrated: multi-vendor code generation, production-grade dynamic-shape support, persistent fleet-scale kernel memory, or an LLM generating arbitrary kernel programs.
