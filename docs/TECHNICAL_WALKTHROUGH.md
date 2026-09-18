# KernelRelay technical walkthrough

This walkthrough describes the preserved **V1** compiler and its recorded measurements. [V2](AGENT_EVALUATION.md) adds a separate agent-evaluation layer: structured requests and source proposals, evaluator-owned verification/timing, iterative feedback, and JSON trajectories. V2 reuses V1's workload, graph matcher, deterministic inputs, numerical oracle, tolerances, synchronized timing, and atomic JSON writer. It does not reinterpret the V1 T4 measurements as agent-generated results.

## Problem

The workload computes:

```python
y = silu(layer_norm(x + input_bias, weight=gamma, bias=beta))
```

An eager implementation may materialize the output of the addition and LayerNorm in device memory. A fused kernel can keep those intermediate values inside one kernel and write only the final result.

The prototype tests that optimization through a controlled compiler loop.

## Compiler loop

### 1. Observe

PyTorch FX captures the module as a graph. Shape propagation records the tensor shape and dtype for each node.

### 2. Match

The matcher follows graph edges and validates the operator semantics. It accepts the supported bias addition, functional LayerNorm, and non inplace SiLU sequence. It also verifies tensor roles, metadata, epsilon, and external users.

The matcher does not accept a region based only on node names or order.

### 3. Estimate

The analytical model estimates scalar operations and logical tensor traffic. For the full FP32 case, the eager traffic estimate is 25.167 MB and the fused estimate is 8.390 MB. The model predicts an opportunity to reduce intermediate traffic.

These values are analytical estimates. They are not hardware counter measurements.

### 4. Propose

The controller enumerates 36 configurations from four block sizes, three warp counts, and three stage counts. It prunes configurations that cannot cover the hidden dimension or violate the bounded kernel assumptions.

This is configuration search over one handwritten kernel body. It is not arbitrary program synthesis.

### 5. Verify

Eager PyTorch is the numerical oracle. Every executed candidate must pass eight deterministic cases before timing:

1. Three random inputs
2. All zero input
3. Constant rows
4. Values near zero
5. Alternating large values
6. Nontrivial affine parameters

The check covers shape, dtype, finite values, elementwise error, normalized error, and mismatch count. FP16 and FP32 use different tolerances.

### 6. Measure

Compilation and tuning are excluded from steady state latency. CUDA timing uses events and explicit synchronization. Candidate order is shuffled. Finalists run in repeated interleaved rounds, and selection uses the median of trial medians.

The same representative input is used for eager PyTorch, `torch.compile`, and Triton.

### 7. Select

Triton must be correct and at least 2 percent faster than the best verified framework path. Otherwise the controller keeps the framework implementation.

Unsupported devices, shapes, compiler failures, and kernel failures use a verified fallback.

## Measured outcome

The recorded environment was an NVIDIA Tesla T4 with compute capability 7.5, CUDA 12.8, PyTorch 2.11.0, Triton 3.6.0, and Python 3.13.15.

<table>
  <thead>
    <tr><th>Dtype</th><th>Triton median</th><th>Speedup over eager</th><th>Maximum normalized error</th><th>Mismatches</th></tr>
  </thead>
  <tbody>
    <tr><td>FP16</td><td>0.112576 ms</td><td>1.65x</td><td>0.135593</td><td>0</td></tr>
    <tr><td>FP32</td><td>0.106512 ms</td><td>1.96x</td><td>0.003513</td><td>0</td></tr>
  </tbody>
</table>

These measurements describe one Colab T4 session. They do not establish performance on another GPU or software version.

## Claims that are supported

1. The FX matcher finds the supported fusion region from graph semantics.
2. Every executed Triton candidate is checked before timing.
3. The selected kernel was faster than eager PyTorch in the recorded T4 runs.
4. The bounded shape limit triggers a verified fallback.
5. The JSON output records the environment, correctness, timing, candidate outcomes, and selection rationale.

## Claims that are not supported

1. The system does not generate arbitrary kernel programs.
2. The analytical roofline result does not prove the physical bottleneck.
3. The benchmark does not generalize beyond the recorded environment.
4. The kernel does not implement backward propagation.
5. The prototype does not support multiple accelerator vendors.
6. The prototype does not maintain a persistent tuning database.

## Short explanation

This project demonstrates a small closed compiler loop. It observes a PyTorch graph, validates a fusion opportunity, searches a bounded Triton configuration space, rejects incorrect candidates, compares verified implementations under one timing protocol, and falls back when the optimized path is unsupported or not clearly faster.

The word agentic refers to that feedback loop. It does not imply that a language model writes arbitrary kernels.

## Common questions

### Why use eager PyTorch as the oracle?

It provides a clear reference implementation with the expected framework semantics. The optimized paths must match it within dtype appropriate tolerances.

### Why can `torch.compile` be slower?

Compiler output and launch behavior depend on the graph, shape, hardware, and software versions. The result is specific to this run and is not a general claim about Inductor.

### Why require a 2 percent margin?

Small timing differences can result from measurement noise. The margin avoids replacing a verified framework path for a negligible gain.

### Why include `num_stages` in the search?

It is part of the requested configuration space. This kernel has no software pipelined loop, so stage count may have limited effect. The program states that limitation rather than treating each parameter as equally important.

### Is the workload proven to be memory bound?

No. The logical traffic model and optional roofline calculation support that hypothesis. Profiler counters would be required to classify the physical bottleneck.
