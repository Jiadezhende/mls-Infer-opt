"""Tool registration and execution for LLM agents.

This module intentionally contains no optimization business logic. It is the
adapter that turns deterministic project functions into model-callable tools.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolHandler",
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    "to_openai_tools",
]

ToolHandler = Callable[[Mapping[str, Any]], "ToolResult | Mapping[str, Any] | str | None"]


@dataclass(frozen=True)
class ToolResult:
    """Structured output returned to the model after a tool call."""

    ok: bool
    data: Any = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any = None, **metadata: Any) -> ToolResult:
        return cls(ok=True, data=data, metadata=dict(metadata))

    @classmethod
    def failure(cls, kind: str, message: str, **metadata: Any) -> ToolResult:
        return cls(ok=False, error={"kind": kind, "message": message}, metadata=dict(metadata))

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.data is not None:
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ToolSpec:
    """A single model-callable tool and its local handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    timeout_s: float | None = None
    strict: bool = True

    def openai_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.strict:
            schema["strict"] = True
        return schema


class ToolRegistry:
    """Name-indexed collection of tools available in one agent phase."""

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        tool = self.get(name)
        if tool is None:
            raise KeyError(name)
        return tool

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]


class ToolExecutor:
    """Validates and executes tool calls, converting all failures to ToolResult."""

    def __init__(self, registry: ToolRegistry, *, default_timeout_s: float | None = None) -> None:
        self.registry = registry
        self.default_timeout_s = default_timeout_s

    def execute(self, name: str, arguments: Mapping[str, Any] | str | None) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult.failure("unknown_tool", f"tool {name!r} is not registered")

        parsed = _parse_arguments(arguments)
        if not parsed.ok:
            return parsed
        assert isinstance(parsed.data, dict)

        validation_error = _validate_schema(parsed.data, tool.parameters)
        if validation_error is not None:
            return ToolResult.failure("invalid_arguments", validation_error)

        timeout_s = tool.timeout_s if tool.timeout_s is not None else self.default_timeout_s
        return _run_handler(tool.handler, parsed.data, timeout_s=timeout_s)


def to_openai_tools(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Convert local ToolSpec objects into Responses API tool schemas."""

    return ToolRegistry(tools).openai_schemas()


def _parse_arguments(arguments: Mapping[str, Any] | str | None) -> ToolResult:
    if arguments is None:
        return ToolResult.success({})
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult.failure("invalid_json", f"arguments are not valid JSON: {exc}")
        if not isinstance(value, dict):
            return ToolResult.failure("invalid_arguments", "tool arguments must be a JSON object")
        return ToolResult.success(value)
    return ToolResult.success(dict(arguments))


def _run_handler(
    handler: ToolHandler,
    arguments: Mapping[str, Any],
    *,
    timeout_s: float | None,
) -> ToolResult:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(handler, arguments)
    try:
        result = future.result(timeout=timeout_s)
    except TimeoutError:
        future.cancel()
        return ToolResult.failure("timeout", "tool call timed out")
    except Exception as exc:
        return ToolResult.failure(type(exc).__name__, str(exc))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if isinstance(result, ToolResult):
        return result
    if isinstance(result, Mapping):
        return ToolResult.success(dict(result))
    return ToolResult.success(result)


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> str | None:
    """Small JSON-schema subset validator for tool arguments.

    The project does not depend on jsonschema; this covers the strict schemas we
    use for function tools: object/properties/required/additionalProperties,
    scalar types, arrays, and enum.
    """

    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']!r}"

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        return f"{path} must be {_format_type(expected_type)}"

    if _allows_type(expected_type, "object") and isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return f"{path}.properties must be an object in tool schema"
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                return f"{path} has unexpected properties: {extra}"
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                error = _validate_schema(value[key], child_schema, f"{path}.{key}")
                if error is not None:
                    return error

    if _allows_type(expected_type, "array") and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                error = _validate_schema(item, item_schema, f"{path}[{index}]")
                if error is not None:
                    return error

    return None


def _allows_type(expected_type: Any, type_name: str) -> bool:
    if expected_type == type_name:
        return True
    if isinstance(expected_type, list):
        return type_name in expected_type
    return False


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "null":
        return value is None
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    return True


def _format_type(expected_type: Any) -> str:
    if isinstance(expected_type, list):
        return " or ".join(str(item) for item in expected_type)
    return str(expected_type)
