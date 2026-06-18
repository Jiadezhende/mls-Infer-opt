"""Small exception hierarchy for the optional LLM layer."""

from __future__ import annotations

__all__ = ["LLMError", "LLMUnavailableError", "LLMCallError", "ToolExecutionError"]


class LLMError(Exception):
    """Base class for local LLM-layer failures."""


class LLMUnavailableError(LLMError):
    """Raised internally when the configured LLM client cannot be used."""


class LLMCallError(LLMError):
    """传输/基建层调用失败（C2）：网络 / API / 超时等。

    run_agent 在真实调用抛错时抛出本异常，**不**静默降级成 ok=False——C2 必须穿透到总控的循环
    边界，由总控记 C2 + 仍发布 best-so-far（见 PIPELINE_SPEC §3）。内容层失败（模型没给最终答复 /
    工具循环没收敛）仍是 ok=False，属 C1 邻域，调用方可回退 rule-based，不抛。
    """


class ToolExecutionError(LLMError):
    """Raised internally when a tool invocation cannot be completed."""
