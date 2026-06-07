from __future__ import annotations

import pytest

from mls_infer_opt.llm import ToolExecutor, ToolRegistry, ToolResult, ToolSpec, to_openai_tools


def make_echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo a value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=lambda args: {"echo": args["value"]},
    )


def test_registry_rejects_duplicate_names():
    tool = make_echo_tool()
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])


def test_to_openai_tools_emits_strict_function_schema():
    [schema] = to_openai_tools([make_echo_tool()])
    assert schema["type"] == "function"
    assert schema["name"] == "echo"
    assert schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False


def test_executor_runs_handler_with_valid_arguments():
    executor = ToolExecutor(ToolRegistry([make_echo_tool()]))
    result = executor.execute("echo", '{"value": "ok"}')
    assert result.ok
    assert result.data == {"echo": "ok"}


def test_executor_rejects_unknown_tool():
    executor = ToolExecutor(ToolRegistry())
    result = executor.execute("missing", {})
    assert not result.ok
    assert result.error and result.error["kind"] == "unknown_tool"


def test_executor_rejects_bad_json():
    executor = ToolExecutor(ToolRegistry([make_echo_tool()]))
    result = executor.execute("echo", "{")
    assert not result.ok
    assert result.error and result.error["kind"] == "invalid_json"


def test_executor_validates_required_and_extra_arguments():
    executor = ToolExecutor(ToolRegistry([make_echo_tool()]))
    missing = executor.execute("echo", {})
    assert not missing.ok
    assert missing.error and missing.error["kind"] == "invalid_arguments"

    extra = executor.execute("echo", {"value": "ok", "surprise": True})
    assert not extra.ok
    assert extra.error and extra.error["kind"] == "invalid_arguments"


def test_executor_converts_handler_exceptions_to_result():
    def boom(_args):
        raise RuntimeError("boom")

    tool = ToolSpec(
        name="boom",
        description="Raise.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=boom,
    )
    result = ToolExecutor(ToolRegistry([tool])).execute("boom", {})
    assert not result.ok
    assert result.error and result.error["kind"] == "RuntimeError"


def test_tool_result_serializes_structured_payload():
    payload = ToolResult.failure("runtime", "nope", candidate_id="c1").to_payload()
    assert payload == {
        "ok": False,
        "error": {"kind": "runtime", "message": "nope"},
        "metadata": {"candidate_id": "c1"},
    }
