"""The agent proposes; it cannot assign correctness, latency, or reward."""

from typing import Protocol

from agent_schema import AgentProposal, AgentRequest


class AgentExhausted(Exception):
    """A finite proposal stream has ended normally."""


class AgentBackend(Protocol):
    def propose(self, request: AgentRequest) -> AgentProposal:
        ...
