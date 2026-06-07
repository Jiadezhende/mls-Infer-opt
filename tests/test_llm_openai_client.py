from __future__ import annotations

from typing import Any

from mls_infer_opt.llm import LLMConfig, OpenAIAgentClient, ToolSpec


class FakeResponses:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class FakeOpenAI:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.responses = FakeResponses(outputs)


def test_client_is_unavailable_without_key_and_sdk():
    client = OpenAIAgentClient(LLMConfig(api_key=None))
    assert not client.available
    assert client.generate("hello") is None
    assert client.unavailable_reason == "missing OPENAI_API_KEY"


def test_client_extracts_plain_text_from_response():
    fake = FakeOpenAI(
        [
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    client = OpenAIAgentClient(LLMConfig(api_key="x"), client=fake)
    result = client.run_agent("prompt")
    assert result.ok
    assert result.text == "done"
    assert result.usage == {"input_tokens": 1, "output_tokens": 1}
    assert fake.responses.calls[0]["model"] == "gpt-5.5"


def test_client_executes_function_tool_and_sends_output_back():
    fake = FakeOpenAI(
        [
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "echo",
                        "arguments": '{"value": "hello"}',
                        "call_id": "call-1",
                    }
                ]
            },
            {"output_text": "final", "output": []},
        ]
    )
    tool = ToolSpec(
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
    client = OpenAIAgentClient(LLMConfig(api_key="x"), client=fake)
    result = client.run_agent("prompt", tools=[tool])

    assert result.ok
    assert result.text == "final"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "echo"
    assert result.tool_calls[0].result.data == {"echo": "hello"}

    second_input = fake.responses.calls[1]["input"]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call-1"
    assert '"echo": "hello"' in second_input[-1]["output"]


def test_client_stops_after_max_tool_rounds():
    fake = FakeOpenAI(
        [
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "unknown",
                        "arguments": "{}",
                        "call_id": "call-1",
                    }
                ]
            }
        ]
    )
    client = OpenAIAgentClient(LLMConfig(api_key="x", max_tool_rounds=0), client=fake)
    result = client.run_agent("prompt")
    assert not result.ok
    assert result.error and result.error["kind"] == "max_tool_rounds"
