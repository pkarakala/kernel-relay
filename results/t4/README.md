# NVIDIA T4 validation artifacts

These files were produced on a Google Colab Tesla T4 on September 16, 2026 local time. Each completed JSON record identifies the hardware and software environment, effective CLI arguments, correctness results, candidate outcomes, timing protocol, and deployment rationale.

The compiler source used for every recorded run has SHA256:

```text
4683c4726c41128181b28a877de00ffab4c78ec2d8bb3d207c65581845a2f0cd
```

This matches `agentic_kernel_compiler.py` in the working tree at the time these artifacts were added.

## Files

| File | Contents |
|---|---|
| `t4-smoke.json` | Quick FP16 CUDA compilation, correctness, tuning, and deployment check |
| `full-fp16.json` | Full FP16 evaluation at 8192 by 128 |
| `full-fp32.json` | Full FP32 evaluation at 8192 by 128 |
| `boundary-results.zip` | JSON and terminal logs for 16 dtype and hidden size boundary cases |
| `roofline-t4-fp32.zip` | Roofline JSON and terminal log using user supplied nominal T4 inputs |
| `hidden-257-fallback.json` | Verified unsupported shape fallback above the bounded kernel limit |

## Download checksums

```text
0e2db012e429d1b66ba5a5e17c182aa512d79da89e20e08a3d2b379e988f39b3  t4-smoke.json
e462904b604e718b990681b55af4f7d0d576d1c0d5a7791dc56af46b98e048f7  full-fp16.json
0ad7044cc4f50c9c4255c66d98a40f8d5082fae1fa0f66e14905a354aa99aa98  full-fp32.json
0ab450e4181a6b1db95ead7ec2d881b9bd497562ca2a5e27274e48796b55fabc  boundary-results.zip
303d36cb277dc8539d9a52f2e4b01c59fa2bf74160a5aff77176a8cf927bf782  roofline-t4-fp32.zip
22ad2bbaaa557a8a89a80b4cee6660e42ccaba31925234aa92ea8ebc9e47fc81  hidden-257-fallback.json
```

Latency values are measurements from one Colab T4 session and should not be generalized across hardware or software versions. Effective bandwidth and roofline quantities are analytical estimates rather than hardware counter measurements.
