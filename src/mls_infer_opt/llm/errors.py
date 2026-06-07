"""Small exception hierarchy for the optional LLM layer."""

from __future__ import annotations

__all__ = ["LLMError", "LLMUnavailableError", "ToolExecutionError"]


class LLMError(Exception):
    """Base class for local LLM-layer failures."""


class LLMUnavailableError(LLMError):
    """Raised internally when the configured LLM client cannot be used."""


class ToolExecutionError(LLMError):
    """Raised internally when a tool invocation cannot be completed."""
