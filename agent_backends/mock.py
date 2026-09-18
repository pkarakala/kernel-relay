"""Deterministic fixtures, not a learned optimizer or frontier-model adapter."""

import ast
import inspect
from collections.abc import Sequence

import agentic_kernel_compiler as compiler
from agent_backends.base import AgentExhausted
from agent_schema import AgentProposal, AgentRequest


PYTORCH_SOURCE = '''import torch.nn.functional as functional

def run(x, input_bias, gamma, beta, *, eps, launch_parameters):
    return functional.silu(functional.layer_norm(x + input_bias, (x.shape[-1],), gamma, beta, eps))
'''


def triton_source(use_sigmoid: bool = False) -> str:
    tree = ast.parse(inspect.getsource(compiler._triton_fused_impl))
    kernel = tree.body[0]
    kernel.name = "fused"
    kernel.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    kernel.args.args[-1].annotation = ast.parse("tl.constexpr", mode="eval").body
    if use_sigmoid:
        kernel.body[-4:-1] = ast.parse("silu = affine * tl.sigmoid(affine)").body
    source = "import torch\nimport triton\nimport triton.language as tl\n\n"
    source += ast.unparse(ast.fix_missing_locations(tree))
    return source + '''

def run(x, input_bias, gamma, beta, *, eps, launch_parameters):
    hidden_size = x.shape[-1]
    output = torch.empty_like(x)
    fused[(x.numel() // hidden_size,)](
        x, input_bias, gamma, beta, output, hidden_size, eps,
        BLOCK_SIZE=launch_parameters["block_size"],
        num_warps=launch_parameters["num_warps"],
        num_stages=launch_parameters["num_stages"])
    return output
'''


class MockAgentBackend:
    def __init__(self, proposals: Sequence[AgentProposal] | None = None):
        self.proposals = list(proposals) if proposals is not None else None
        self.index = 0

    def propose(self, request: AgentRequest) -> AgentProposal:
        index = self.index
        self.index += 1
        if self.proposals is not None:
            if index >= len(self.proposals):
                raise AgentExhausted("mock proposal stream exhausted")
            return self.proposals[index]
        if index == 0:
            return AgentProposal("invalid-syntax", "pytorch_source", "Exercise compile-error feedback", "def run(:")
        if index == 1:
            return AgentProposal("reference-fallback", "eager", "Recover using the trusted eager reference")
        if index >= 8:
            raise AgentExhausted("eight predefined mock proposals exhausted")
        hidden_size = request.task["inputs"][0]["shape"][-1]
        block_size = min(4096, max(32, 1 << (hidden_size - 1).bit_length()))
        return AgentProposal(
            f"fused-{index}", "triton_source",
            "Predefined fused implementation; next request includes previous evaluator feedback",
            triton_source(use_sigmoid=index % 2 == 1),
            {"block_size": block_size, "num_warps": (2, 4, 8)[(index - 2) // 2], "num_stages": 2},
            "Avoid global-memory add/LayerNorm intermediates; speedup is only a hypothesis",
        )


def validate_trusted_mock_proposal(proposal: AgentProposal, hidden_size: int) -> None:
    """Exact canonical match, including source bytes, parameters, and metadata.

    Trust is rooted in this audited working tree, never in agent-supplied hashes
    or a replay file. Both the controller and copied worker perform this check.
    """
    from agent_schema import HardwareFingerprint

    request = AgentRequest(
        {"inputs": [{"shape": [1, hidden_size]}]}, HardwareFingerprint("cpu"),
        [], [], {}, 0, {},
    )
    backend = MockAgentBackend()
    expected = [backend.propose(request).to_dict() for _ in range(8)]
    if proposal.to_dict() not in expected:
        raise ValueError("trusted-fixture mode rejects proposal: not an exact repository mock fixture")
