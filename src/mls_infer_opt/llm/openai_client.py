"""OpenAI Responses API client with local function-tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import present
from .config import LLMConfig
from .tooling import ToolExecutor, ToolRegistry, ToolResult, ToolSpec

__all__ = ["AgentResult", "ToolCallRecord", "OpenAIAgentClient"]


@dataclass(frozen=True)
class ToolCallRecord:
    """Audit record for one model-requested local tool call."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult
    call_id: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Result of a model run, with optional tool-call audit data."""

    ok: bool
    text: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    raw: Any = None


@dataclass(frozen=True)
class _PendingToolCall:
    name: str
    arguments: str | dict[str, Any] | None
    call_id: str | None


class OpenAIAgentClient:
    """Small, defensive wrapper around OpenAI Responses API.

    Exposes the single run_agent(prompt, tools=...) entry that generate/analyze drive.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = client
        self._unavailable_reason: str | None = None
        # 最近一次 run_agent 失败的结构化 error（kind/message）——供上层（loop）记录「为什么这次
        # 调用没成功」，而不是把网络/API 错误静默丢弃。
        self._last_error: dict[str, Any] | None = None
        if self.config.disabled:
            self._unavailable_reason = "LLM disabled by MLS_LLM_DISABLED"
        elif self._client is None:
            self._client = self._build_client()

    @property
    def available(self) -> bool:
        return self._client is not None and self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def last_error(self) -> dict[str, Any] | None:
        """最近一次 run_agent 失败的 error（kind/message），成功则保持上一次值。"""
        return self._last_error

    def run_agent(
        self,
        prompt: str,
        tools: list[ToolSpec] | None = None,
        *,
        instructions: str = "",
        max_tool_rounds: int | None = None,
    ) -> AgentResult:
        self._last_error = None
        if not self.available:
            self._last_error = {
                "kind": "unavailable",
                "message": self._unavailable_reason or "LLM client is unavailable",
            }
            return AgentResult(ok=False, error=self._last_error)

        registry = ToolRegistry(tools or [])
        executor = ToolExecutor(registry, default_timeout_s=self.config.timeout_s)
        input_items: list[Any] = [{"role": "user", "content": prompt}]
        tool_records: list[ToolCallRecord] = []
        rounds = max_tool_rounds if max_tool_rounds is not None else self.config.max_tool_rounds

        try:
            for round_index in range(rounds + 1):
                response = self._create_response(
                    input_items,
                    instructions=instructions,
                    tools=registry.openai_schemas(),
                )
                output_items = _response_output(response)
                pending_calls = _extract_tool_calls(output_items)
                text = _extract_text(response)
                if not pending_calls:
                    # 模型给出最终答复、无更多工具调用——这一轮 agent 收束。
                    present.emit(f"  agent r{round_index}: 完成（{len(text or '')} 字）")
                    return AgentResult(
                        ok=True,
                        text=text,
                        tool_calls=tool_records,
                        usage=_extract_usage(response),
                        raw=response,
                    )
                # 让终端看到「这一轮模型想调哪些工具」——agent 分析过程的实时投影。
                present.emit(
                    f"  agent r{round_index}: 调用工具 {[c.name for c in pending_calls]}"
                )
                if round_index >= rounds:
                    self._last_error = {
                        "kind": "max_tool_rounds",
                        "message": "tool loop limit reached",
                    }
                    return AgentResult(
                        ok=False,
                        text=text,
                        tool_calls=tool_records,
                        usage=_extract_usage(response),
                        error=self._last_error,
                        raw=response,
                    )

                input_items.extend(_item_to_input(item) for item in output_items)
                for call in pending_calls:
                    result = executor.execute(call.name, call.arguments)
                    present.emit(
                        f"    ↳ {call.name} "
                        + ("ok" if result.ok else f"err:{(result.error or {}).get('kind', '?')}")
                    )
                    parsed_arguments = _arguments_to_dict(call.arguments)
                    tool_records.append(
                        ToolCallRecord(
                            name=call.name,
                            arguments=parsed_arguments,
                            result=result,
                            call_id=call.call_id,
                        )
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": result.to_json(),
                        }
                    )
        except Exception as exc:
            self._last_error = {"kind": type(exc).__name__, "message": str(exc)}
            return AgentResult(
                ok=False,
                tool_calls=tool_records,
                error=self._last_error,
            )

        self._last_error = {"kind": "empty", "message": "model produced no final answer"}
        return AgentResult(ok=False, tool_calls=tool_records, error=self._last_error)

    def _build_client(self) -> Any | None:
        if not self.config.can_attempt_request:
            self._unavailable_reason = "missing OPENAI_API_KEY"
            return None
        try:
            from openai import OpenAI
        except Exception as exc:
            self._unavailable_reason = f"openai SDK unavailable: {exc}"
            return None

        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_s,
            # SDK 默认 max_retries=2 会在超时后静默再试（最坏 ≈ timeout×3），
            # codegen 长输出下会放大成数分钟「假死」；收敛为不重试，重试交上层循环。
            "max_retries": 0,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        try:
            client = OpenAI(**kwargs)
        except Exception as exc:
            self._unavailable_reason = f"failed to create OpenAI client: {exc}"
            return None
        if not hasattr(client, "responses"):
            self._unavailable_reason = "openai SDK does not expose Responses API"
            return None
        return client

    def _create_response(
        self,
        input_items: list[Any],
        *,
        instructions: str,
        tools: list[dict[str, Any]],
    ) -> Any:
        assert self._client is not None
        params: dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
        }
        if instructions:
            params["instructions"] = instructions
        if tools:
            params["tools"] = tools
        return self._client.responses.create(**params)


def _response_output(response: Any) -> list[Any]:
    output = _get(response, "output")
    if isinstance(output, list):
        return output
    return []


def _extract_tool_calls(output_items: list[Any]) -> list[_PendingToolCall]:
    calls: list[_PendingToolCall] = []
    for item in output_items:
        if _get(item, "type") != "function_call":
            continue
        name = _get(item, "name")
        if not isinstance(name, str):
            continue
        calls.append(
            _PendingToolCall(
                name=name,
                arguments=_get(item, "arguments"),
                call_id=_get(item, "call_id"),
            )
        )
    return calls


def _extract_text(response: Any) -> str | None:
    direct = _get(response, "output_text")
    if isinstance(direct, str) and direct:
        return direct

    chunks: list[str] = []
    for item in _response_output(response):
        item_type = _get(item, "type")
        if item_type == "message":
            content = _get(item, "content")
            if isinstance(content, list):
                chunks.extend(_text_from_content(part) for part in content)
        elif item_type in {"output_text", "text"}:
            text = _get(item, "text")
            if isinstance(text, str):
                chunks.append(text)
    text = "".join(chunks).strip()
    return text or None


def _text_from_content(part: Any) -> str:
    text = _get(part, "text")
    return text if isinstance(text, str) else ""


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = _get(response, "usage")
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else None
    return None


def _item_to_input(item: Any) -> Any:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return item


def _arguments_to_dict(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    try:
        value = __import__("json").loads(arguments or "{}")
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
