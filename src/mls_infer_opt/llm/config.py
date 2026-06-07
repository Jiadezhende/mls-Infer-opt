"""Configuration for optional LLM access.

The LLM layer is an optional accelerator. Missing credentials, missing SDKs, or
explicit disable flags must never make the outer optimization loop fail.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LLMConfig"]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_dotenv(path: str | os.PathLike[str] | None) -> dict[str, str]:
    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_env_value(value.strip())
    return values


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _merged_env(
    env: Mapping[str, str] | None,
    env_file: str | os.PathLike[str] | None,
) -> dict[str, str]:
    shell_values = dict(env if env is not None else os.environ)
    dotenv_path = env_file or shell_values.get("MLS_LLM_ENV_FILE") or ".env"
    dotenv_values = _read_dotenv(dotenv_path)
    # Exported shell values win over .env so CI/local overrides remain predictable.
    return {**dotenv_values, **shell_values}


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for the OpenAI-backed agent client."""

    provider: str = "openai"
    model: str = "gpt-5.5"
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = 120.0  # 产一份 engine.py 实测 ~80s（reasoning 模型），留足余量
    max_tool_rounds: int = 4
    disabled: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        env_file: str | os.PathLike[str] | None = None,
    ) -> LLMConfig:
        values = _merged_env(env, env_file)
        return cls(
            provider=values.get("MLS_LLM_PROVIDER", "openai"),
            model=values.get("MLS_LLM_MODEL") or values.get("OPENAI_MODEL", "gpt-5.5"),
            api_key=values.get("OPENAI_API_KEY") or values.get("MLS_LLM_API_KEY"),
            base_url=(
                values.get("MLS_LLM_BASE_URL")
                or values.get("OPENAI_BASE_URL")
                or values.get("OPENAI_API_BASE")
            ),
            timeout_s=_float_env(values, "MLS_LLM_TIMEOUT_S", 120.0),
            max_tool_rounds=_int_env(values, "MLS_LLM_MAX_TOOL_ROUNDS", 4),
            disabled=_truthy(values.get("MLS_LLM_DISABLED")),
        )

    @property
    def can_attempt_request(self) -> bool:
        """Whether a real network-backed client may be constructed."""

        return not self.disabled and bool(self.api_key)
