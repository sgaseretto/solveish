"""Deterministic, network-free LLM provider for browser E2E tests."""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator, Dict, List

from dialeng.core.registry import ProviderRegistration, registry

from ..base_provider import BaseLLMProvider, LLMResult, ProviderInfo


class DeterministicTestProvider(BaseLLMProvider):
    """Exercise the production streaming path without external credentials."""

    async def initialize(self) -> None:
        return None

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="e2e_fake",
            display_name="Deterministic E2E",
            supports_native_tools=True,
            supports_mcp_tools=False,
            capabilities=["streaming", "thinking", "usage", "cancellation"],
        )

    def check_thinking_support(self, model: str) -> bool:
        return model in {"e2e-fast", "e2e-slow"}

    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        del context_messages, system_prompt, config
        if model == "e2e-error":
            yield {"type": "error", "content": "Deterministic E2E provider error"}
            return

        if use_thinking:
            yield {"type": "thinking_start"}
            yield {"type": "thinking", "content": "Deterministic reasoning summary"}
            yield {"type": "thinking_end"}

        response = f"E2E response: {prompt}"
        delay = 0.08 if model == "e2e-slow" else 0
        emitted = 0
        for offset in range(0, len(response), 4):
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "error", "content": "Request cancelled"}
                return
            if delay:
                await asyncio.sleep(delay)
            chunk = response[offset : offset + 4]
            emitted += len(chunk)
            yield {"type": "chunk", "content": chunk}

        usage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": emitted,
            "total_tokens": len(prompt.split()) + emitted,
        }
        self._last_result = LLMResult(usage=usage, cost=0.0)
        yield {"type": "usage", "usage": usage}

    async def stream_with_tools(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
        tools: List[Dict],
        kernel: Any,
        notebook_id: str,
        registry: Any,
        max_steps: int,
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        del tools, kernel, notebook_id, registry, max_steps
        async for event in self.stream(
            prompt,
            context_messages,
            system_prompt,
            model,
            use_thinking,
            config,
            cancel_event,
        ):
            yield event


if os.environ.get("DIALENG_E2E") == "1":
    registry.register_provider(
        ProviderRegistration(
            name="e2e_fake",
            label="Deterministic E2E",
            factory=DeterministicTestProvider,
            priority=-100,
        )
    )
