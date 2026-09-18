# Contributing

This is a focused research prototype. Please open an issue before proposing a new workload or backend so the correctness oracle, failure handling, and measurement contract stay clear.

For a code change, run:

```bash
python3 -m py_compile agentic_kernel_compiler.py agent_eval.py
python3 -m unittest -q
python3 agentic_kernel_compiler.py --quick
```

GPU claims require an actual CUDA/Triton run, a recorded hardware/software fingerprint, per-candidate correctness results, and raw timing samples. Do not fabricate results, silently loosen tolerances, or describe analytical byte estimates as hardware-counter measurements. Never run untrusted candidate source in the Colab trusted-fixture mode.

Small, auditable changes with tests and explicit limitations are preferred.
