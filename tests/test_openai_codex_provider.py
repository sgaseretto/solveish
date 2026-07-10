import asyncio
import base64

import pytest

from dialeng.services.llm.providers.openai_codex_provider import (
    OpenAICodexProvider,
    ResponseState,
    build_codex_input,
    build_codex_tools,
)
from dialeng.services.skill_service import set_skill_root


def test_codex_input_is_full_stateless_transcript_with_images():
    image = base64.b64encode(b"image").decode()
    items = build_codex_input([
        {"role": "user", "content": [{"type": "text", "text": "Look"}, {
            "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image},
        }]},
        {"role": "assistant", "content": "I can see it."},
    ], "What changed?")
    assert items[0]["content"][1]["type"] == "input_image"
    assert items[1]["role"] == "assistant"
    assert items[-1]["content"][0]["text"] == "What changed?"


def test_codex_request_is_ephemeral_and_sequential(tmp_path):
    set_skill_root(tmp_path)
    body = OpenAICodexProvider._request_body(
        model="gpt-5.4", system_prompt="system", input_items=[], tools=[], use_thinking=True,
    )
    assert body["store"] is False
    assert "previous_response_id" not in body
    assert body["parallel_tool_calls"] is False
    assert body["reasoning"]["summary"] == "auto"


def test_tool_schema_and_event_normalization():
    tools = build_codex_tools([{
        "name": "view", "description": "View a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }])
    assert tools[0]["type"] == "function"
    assert tools[0]["parameters"]["properties"]["path"]["type"] == "string"

    provider = OpenAICodexProvider()
    state = ResponseState()

    async def collect(event):
        return [item async for item in provider._normalize_event(event, state)]

    assert asyncio.run(collect({"type": "response.output_text.delta", "delta": "hello"})) == [
        {"type": "chunk", "content": "hello"}
    ]
    asyncio.run(collect({
        "type": "response.output_item.done",
        "item": {"type": "function_call", "call_id": "call_1", "name": "view", "arguments": '{"path":"x"}'},
    }))
    assert state.tool_calls == [{"call_id": "call_1", "name": "view", "arguments": {"path": "x"}}]
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}
    asyncio.run(collect({"type": "response.output_item.done", "item": reasoning}))
    assert reasoning in state.output_items
    usage_events = asyncio.run(collect({
        "type": "response.completed",
        "response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
    }))
    assert usage_events == [{
        "type": "usage",
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }]


def test_auto_transport_falls_back_only_before_meaningful_output(monkeypatch):
    provider = OpenAICodexProvider()

    async def websocket_before_output(_body, _cancel):
        yield {"type": "response.created"}
        raise RuntimeError("socket failed")

    async def sse(_body, _cancel, *, allow_retry):
        assert allow_retry is True
        yield {"type": "response.output_text.delta", "delta": "fallback"}

    monkeypatch.setattr(provider, "_websocket_events", websocket_before_output)
    monkeypatch.setattr(provider, "_sse_events", sse)

    async def collect():
        config = type("Config", (), {"openai_codex_transport": "auto"})()
        return [event async for event in provider._request_events({}, config, None, allow_retry=True)]

    events = asyncio.run(collect())
    assert events[-1]["delta"] == "fallback"

    async def websocket_after_output(_body, _cancel):
        yield {"type": "response.output_text.delta", "delta": "started"}
        raise RuntimeError("socket failed")

    monkeypatch.setattr(provider, "_websocket_events", websocket_after_output)
    with pytest.raises(RuntimeError, match="socket failed"):
        asyncio.run(collect())
