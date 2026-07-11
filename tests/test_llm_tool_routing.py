"""Tests for provider-independent tool routing in LLMService."""

import asyncio
from types import SimpleNamespace

from dialeng.services.llm.llm_service import LLMService


class _FakeProvider:
    def __init__(self):
        self.last_tools = None

    async def stream(self, **kwargs):
        yield {"type": "stream_fallback"}

    async def stream_with_tools(self, **kwargs):
        self.last_tools = kwargs["tools"]
        yield {"type": "provider_tool_loop", "tool_names": [tool["name"] for tool in kwargs["tools"]]}


class _FakeRegistry:
    def __init__(self):
        self.calls = []

    async def get_tools_for_prompt(self, func_names, kernel, notebook_id, include_builtins=True):
        self.calls.append(
            {
                "func_names": list(func_names),
                "notebook_id": notebook_id,
                "include_builtins": include_builtins,
            }
        )
        if include_builtins:
            return [{"name": "view", "description": "View files", "input_schema": {"type": "object"}}]
        return []

    def resolve_tool_display_name(self, _notebook_id, name):
        return name

    def push_pyrun_context(self, **_kwargs):
        return None

    def reset_pyrun_context(self, _token):
        return None


async def _collect(async_iterable):
    return [item async for item in async_iterable]


def test_builtin_tools_are_routed_consistently_for_all_providers(monkeypatch):
    for provider_name in ("claudette:bedrock", "openai_codex"):
        service = LLMService()
        provider = _FakeProvider()
        registry = _FakeRegistry()

        async def get_provider(_provider_id, provider=provider, provider_name=provider_name):
            return provider_name, provider
        monkeypatch.setattr(service, "_get_provider", get_provider)

        monkeypatch.setattr(
            "dialeng.services.tool_registry.get_tool_registry",
            lambda registry=registry: registry,
        )
        monkeypatch.setattr(
            service,
            "_resolve_model_and_prompt",
            lambda mode, model, provider_id: ("system prompt", model, SimpleNamespace()),
        )

        items = asyncio.run(
            _collect(
                service.stream_response_with_tools(
                    prompt="Summarize this file tree",
                    context_messages=[],
                    mode="standard",
                    model="claude-sonnet-4-5",
                    include_builtins=True,
                        notebook_id="demo",
                        provider_id=provider_name,
                )
            )
        )

        assert registry.calls == [
            {
                "func_names": [],
                "notebook_id": "demo",
                "include_builtins": True,
            }
        ]
        assert any(item["type"] == "tool_available" and item["name"] == "view" for item in items)
        assert any(item["type"] == "provider_tool_loop" for item in items)
        assert provider.last_tools == [{"name": "view", "description": "View files", "input_schema": {"type": "object"}}]
