# Reproducing the Results

The historical commands below reproduce **V1**. For V2, follow [the Colab workflow](../colab/README.md), which detects the actual assigned GPU without assuming a T4. The [V2 T4 records](../results/agent-eval/README.md) include full traces and an interleaved finalist confirmation. CPU-only V2 verification:

```bash
python3 -m py_compile agentic_kernel_compiler.py
python3 -m py_compile agent_eval.py
python3 -m unittest -v
python3 agentic_kernel_compiler.py --quick
python3 agent_eval.py --task bias_layernorm_silu --device cpu --agent mock --max-rounds 3 --quick
python3 agent_eval.py --device cpu --agent replay --replay-file examples/replay_cpu.json --max-rounds 3 --quick
```

Use `python` instead of `python3` if that is the interpreter containing PyTorch. By default, V2 workers require macOS `sandbox-exec` or Linux Landlock (x86-64/AArch64); unavailable confinement is a reported failure, never an implicit unrestricted-code fallback. The [Colab workflow](../colab/README.md) documents the explicit mock-only trusted-fixture exception, which is not an OS sandbox and rejects replay/arbitrary source. CUDA/Triton measurements require a separate GPU run.

## CPU check

PyTorch is required. Triton is not imported on the CPU path.

```bash
python3 -m py_compile agentic_kernel_compiler.py
python3 -m unittest -v test_agentic_kernel_compiler.py
python3 agentic_kernel_compiler.py --quick
```

## NVIDIA T4 check

The recorded run used Google Colab with a Tesla T4. Upload `agentic_kernel_compiler.py`, select a T4 runtime, and verify the environment:

```python
import importlib.util
import torch

assert torch.cuda.is_available()
assert importlib.util.find_spec("triton") is not None
assert "T4" in torch.cuda.get_device_name(0)
```

Run the full FP16 evaluation:

```bash
python3 agentic_kernel_compiler.py \
  --device cuda \
  --rows 8192 \
  --hidden-size 128 \
  --dtype float16 \
  --json-output full-fp16.json
```

Run the full FP32 evaluation:

```bash
python3 agentic_kernel_compiler.py \
  --device cuda \
  --rows 8192 \
  --hidden-size 128 \
  --dtype float32 \
  --json-output full-fp32.json
```

## Expected checks

1. The FX region is eligible.
2. Eighteen configurations execute for hidden size 128.
3. Every executed candidate passes all eight correctness cases.
4. The final deployment decision applies the 2 percent margin.
5. The JSON ends with `evaluation_completed` set to `true`.

Exact latency and the selected configuration can vary between sessions. Correctness and fallback behavior should remain stable.

## Recorded evidence

The original JSON files and compressed boundary logs are in [`results/t4`](../results/t4/README.md). Each record contains the SHA256 value of the compiler source used for that run.
