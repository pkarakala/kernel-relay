# Security policy

KernelRelay is a research prototype, not a service for executing untrusted code. Candidate source can reach native GPU tooling. The AST restrictions and subprocess controls are defense-in-depth, **not** a complete security boundary. Default evaluation requires supported OS confinement; the explicit Colab trusted-fixture mode has no OS sandbox and accepts only exact built-in mock fixtures.

Do not run arbitrary proposals, replay files, or external model output with the trusted-fixture option. Use disposable environments without credentials for GPU experiments.

For a security-sensitive report, use **Security → Report a vulnerability** on this repository; private vulnerability reporting is enabled. Please do not publish exploit details in an issue before coordination. Ordinary bugs and documentation problems can be filed as public issues.
