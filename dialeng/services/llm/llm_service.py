"""Provider-neutral LLM coordinator for stateless prompt-cell execution."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base_provider import BaseLLMProvider
from .constants import SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


class LLMService:
    """Resolve and cache an authenticated provider for each prompt request."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_locks: Dict[str, asyncio.Lock] = {}
        self._last_provider_key: Optional[str] = None

    def invalidate_providers(self, provider_id: Optional[str] = None) -> None:
        """Discard initialized providers after auth or configuration changes."""
        if provider_id:
            self._providers.pop(provider_id, None)
            self._provider_locks.pop(provider_id, None)
        else:
            self._providers.clear()
            self._provider_locks.clear()
        if provider_id is None or self._last_provider_key == provider_id:
            self._last_provider_key = None

    @staticmethod
    def _inventory():
        from dialeng.services.credential_service import detect_credential_inventory
        return detect_credential_inventory()

    def resolve_provider_id(self, provider_id: Optional[str]) -> str:
        return provider_id or self._inventory().default_provider()

    async def _get_provider(self, provider_id: Optional[str]) -> tuple[str, BaseLLMProvider]:
        provider_id = self.resolve_provider_id(provider_id)
        status = self._inventory().get(provider_id)
        if not status or provider_id == "mock_only":
            raise RuntimeError("No authenticated LLM provider is selected")
        if not status.authenticated:
            raise RuntimeError(f"{status.label} is not authenticated")
        if provider_id in self._providers:
            self._last_provider_key = provider_id
            return provider_id, self._providers[provider_id]

        lock = self._provider_locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            if provider_id in self._providers:
                self._last_provider_key = provider_id
                return provider_id, self._providers[provider_id]

            from dialeng.core.registry import registry as extension_registry

            registration = extension_registry.providers.get(status.provider)
            if not registration:
                raise RuntimeError(f"LLM provider '{status.provider}' is not registered")
            if status.provider == "claudette":
                provider = registration.factory(backend=status.backend)
            else:
                provider = registration.factory()
            await provider.initialize()
            self._providers[provider_id] = provider
            self._last_provider_key = provider_id
            logger.info(
                "Initialized LLM provider (selection=%s, provider=%s, backend=%s)",
                provider_id,
                status.provider,
                status.backend,
            )
            return provider_id, provider

    @staticmethod
    def _resolve_model_and_prompt(mode: str, model: str, provider_id: str):
        from dialeng.services.dialeng_config import get_config

        config = get_config()
        valid_model = config.validate_model(provider_id, model)
        if model != valid_model:
            raise ValueError(f"Model '{model}' is not available for provider '{provider_id}'")
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])
        api_model = config.get_api_model_name(valid_model, provider_id)

        if provider_id == "openai_codex" and config.skills_enabled:
            from dialeng.services.skill_service import get_skill_service
            skill_service = get_skill_service()
            system_prompt = skill_service.augment_system_prompt(system_prompt)

        return system_prompt, api_model, config

    async def stream_response(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str = "claude-sonnet-4-5",
        use_thinking: bool = False,
        *,
        provider_id: Optional[str] = None,
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        try:
            provider_id, provider = await self._get_provider(provider_id)
            system_prompt, api_model, config = self._resolve_model_and_prompt(mode, model, provider_id)
            async for item in provider.stream(
                prompt=prompt,
                context_messages=context_messages,
                system_prompt=system_prompt,
                model=api_model,
                use_thinking=use_thinking,
                config=config,
                cancel_event=cancel_event,
            ):
                yield item
        except Exception as exc:
            logger.exception("LLM request failed (provider=%s): %s", provider_id, exc)
            yield {"type": "error", "content": f"LLM Error: {exc}"}

    async def stream_response_with_tools(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str = "claude-sonnet-4-5",
        use_thinking: bool = False,
        kernel=None,
        notebook_id: str = "",
        max_steps: int = 5,
        include_builtins: bool = True,
        *,
        provider_id: Optional[str] = None,
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        from dialeng.services.prompt_parser import parse_prompt, substitute_variables
        from dialeng.services.tool_registry import get_tool_registry

        try:
            provider_id, provider = await self._get_provider(provider_id)
            var_names, func_names = parse_prompt(prompt)
            for message in context_messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                if not content:
                    continue
                context_vars, context_funcs = parse_prompt(content)
                var_names.extend(name for name in context_vars if name not in var_names)
                func_names.extend(name for name in context_funcs if name not in func_names)

            processed_prompt = prompt
            if var_names and kernel:
                processed_prompt, variable_info = await substitute_variables(
                    prompt, kernel, notebook_id, var_names
                )
                for name, info in variable_info.items():
                    if info.get("exists"):
                        yield {
                            "type": "var_substituted",
                            "name": name,
                            "var_type": info.get("var_type"),
                            "value": info.get("repr", "")[:100],
                        }

            registry = get_tool_registry()
            tools = await registry.get_tools_for_prompt(
                func_names,
                kernel,
                notebook_id,
                include_builtins=include_builtins,
            )

            # Skills have a narrow reader even if general built-in tools are off.
            if provider_id == "openai_codex":
                from dialeng.services.dialeng_config import get_config
                from dialeng.services.skill_service import get_skill_service

                if get_config().skills_enabled and get_skill_service().model_skills:
                    schema = registry.get_builtin_schema("read_skill")
                    if schema and all(tool.get("name") != "read_skill" for tool in tools):
                        tools.append(schema)

            for tool in tools:
                yield {
                    "type": "tool_available",
                    "name": registry.resolve_tool_display_name(notebook_id, tool["name"]),
                    "description": tool.get("description", "")[:100],
                }

            if not tools:
                async for item in self.stream_response(
                    processed_prompt,
                    context_messages,
                    mode,
                    model,
                    use_thinking,
                    provider_id=provider_id,
                    cancel_event=cancel_event,
                ):
                    yield item
                return

            pyrun_token = registry.push_pyrun_context(
                notebook_id=notebook_id,
                kernel=kernel,
                func_names=func_names,
                include_builtins=include_builtins,
            )
            try:
                system_prompt, api_model, config = self._resolve_model_and_prompt(
                    mode, model, provider_id
                )
                async for item in provider.stream_with_tools(
                    prompt=processed_prompt,
                    context_messages=context_messages,
                    system_prompt=system_prompt,
                    model=api_model,
                    use_thinking=use_thinking,
                    config=config,
                    tools=tools,
                    kernel=kernel,
                    notebook_id=notebook_id,
                    registry=registry,
                    max_steps=max_steps,
                    cancel_event=cancel_event,
                ):
                    yield item
            finally:
                registry.reset_pyrun_context(pyrun_token)
        except Exception as exc:
            logger.exception("Tool-enabled LLM request failed (provider=%s): %s", provider_id, exc)
            yield {"type": "error", "content": f"Tool LLM Error: {exc}"}

    def get_provider(self) -> str:
        return self._last_provider_key or self._inventory().default_provider()

    @property
    def last_usage(self) -> Optional[Any]:
        provider = self._providers.get(self._last_provider_key or "")
        return provider.last_result.usage if provider else None

    @property
    def last_cost(self) -> Optional[float]:
        provider = self._providers.get(self._last_provider_key or "")
        return provider.last_result.cost if provider else None


llm_service = LLMService()
