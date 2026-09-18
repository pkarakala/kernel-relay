"""Replay proposal JSON, never recorded rewards or hardware measurements."""

from pathlib import Path

from agent_backends.base import AgentExhausted
from agent_schema import AgentProposal, AgentRequest, SCHEMA_VERSION, strict_loads


class ReplayAgentBackend:
    def __init__(self, path: Path):
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("replay file exceeds 256 MiB")
        payload = strict_loads(path.read_text())
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("replay requires a version 2.0 proposal bundle or optimization trace")
        self.task_id = payload.get("task_id")
        if "proposals" in payload:
            self.proposals = payload["proposals"]
        elif "rounds" in payload and isinstance(payload["rounds"], list):
            self.proposals = [record["proposal"] for record in payload["rounds"] if record.get("proposal") is not None]
        else:
            raise ValueError("replay requires proposals or rounds")
        if not isinstance(self.proposals, list):
            raise ValueError("replay proposals must be a list")
        self.index = 0

    def propose(self, request: AgentRequest) -> AgentProposal:
        if self.task_id is not None and self.task_id != request.task["task_id"]:
            raise ValueError("replay task_id does not match requested task")
        if self.index >= len(self.proposals):
            raise AgentExhausted("replay exhausted")
        payload = self.proposals[self.index]
        self.index += 1
        return AgentProposal.from_dict(payload)
