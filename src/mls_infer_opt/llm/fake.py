"""Test doubles for code that depends on the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from .openai_client import AgentResult
from .tooling import ToolSpec

__all__ = ["FakeAgentClient"]


@dataclass
class FakeAgentClient:
    """Small fake that supports both generate() and run_agent()."""

    responses: list[str | AgentResult | Exception]
    available: bool = True
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str | None:
        result = self.run_agent(prompt)
        return result.text if result.ok else None

    def run_agent(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        *,
        instructions: str = "",
        max_tool_rounds: int | None = None,
    ) -> AgentResult:
        _ = (tools, instructions, max_tool_rounds)
        self.prompts.append(prompt)
        if not self.available:
            return AgentResult(ok=False, error={"kind": "unavailable", "message": "fake disabled"})
        if not self.responses:
            return AgentResult(ok=False, error={"kind": "empty", "message": "no scripted response"})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, AgentResult):
            return value
        return AgentResult(ok=True, text=value)
