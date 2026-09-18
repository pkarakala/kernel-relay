# Acknowledgments and provenance

KernelRelay is an independent prototype. Its fused-kernel LayerNorm reduction structure was informed by the [official Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html). PyTorch FX, `torch.compile`, and Triton are upstream projects and runtime dependencies; none is vendored here.

The repository's MIT license covers this project's original code and documentation. Upstream projects retain their own copyright and license terms. See the [Triton repository](https://github.com/triton-lang/triton) and the [PyTorch repository](https://github.com/pytorch/pytorch) for those terms.

The public V2 result files normalize an old Colab extraction path. They are derived copies, not byte-identical raw outputs. [The result manifest](../results/agent-eval/PUBLICATION_MANIFEST.md) records the pre- and post-normalization hashes; the original files remain in the private research archive. No timing, correctness, or hardware fields were changed.
