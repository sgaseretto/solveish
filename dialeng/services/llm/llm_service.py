"""LLM Service coordinator - Routes requests to the appropriate provider.

This is the slim coordinator that keeps the same public API as the original
monolithic LLMService. It owns:
- Provider selection (credential detection + use_sdk_directly flag)
- Mode -> system prompt mapping
- Model name mapping via config.get_api_model_name()
- Prompt parsing (parse_prompt, substitute_variables) and tool registry interaction
- Error wrapping around provider streaming
- last_usage / last_cost properties delegating to provider.last_result
"""
from typing import AsyncIterator, List, Dict, Any, Optional
import logging

from .constants import SYSTEM_PROMPTS
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMService:
    """Service for streaming LLM responses.

    Supports three providers:
    - claudette: Direct Anthropic API or AWS Bedrock (requires credentials)
    - claudette_agent: Claude Code subscription wrapper (claudette-agent library)
    - claude_agent_sdk: Claude Code subscription via SDK directly (most isolated)

    Usage:
        service = LLMService()
        async for item in service.stream_response(prompt, context, "standard"):
            if item["type"] == "chunk":
                print(item["content"], end="")
            elif item["type"] == "error":
                print(f"Error: {item['content']}")
    """

    def __init__(self):
        self._initialized = False
        self._provider_name: Optional[str] = None
        self._backend: Optional[str] = None  # "anthropic_api" or "bedrock"
        self._provider: Optional[BaseLLMProvider] = None

    def _detect_credentials(self):
        """Detect credentials and store provider/backend info."""
        from dialeng.services.credential_service import detect_credentials
        cred_status = detect_credentials()
        self._provider_name = cred_status.provider
        self._backend = cred_status.backend
        return cred_status

    async def _ensure_initialized(self):
        """Lazy initialization - detect provider and create appropriate instance."""
        if self._initialized:
            return

        cred_status = self._detect_credentials()
        logger.info(f"LLM Service initializing with provider: {self._provider_name}, backend: {self._backend}")

        from dialeng.core.registry import registry as ext_registry

        # Resolve effective provider name (claudette_agent may map to SDK)
        effective_provider = self._provider_name
        if self._provider_name == "claudette_agent":
            from dialeng.services.dialeng_config import get_config
            config = get_config()
            if getattr(config, 'use_sdk_directly', False):
                effective_provider = "claude_agent_sdk"

        # Try registry-based lookup
        reg = ext_registry.providers.get(effective_provider)
        if reg:
            # Construct provider — pass backend kwarg for claudette
            if effective_provider == "claudette":
                self._provider = reg.factory(backend=self._backend or "anthropic_api")
            else:
                self._provider = reg.factory()
            await self._provider.initialize()
            self._initialized = True
            logger.info(f"Initialized with {type(self._provider).__name__} via registry")
        elif self._provider_name and self._provider_name != "mock_only":
            logger.warning(f"Provider '{self._provider_name}' not found in registry, falling back to mock")
            self._initialized = True
        else:
            self._initialized = True
            logger.warning("LLM Service initialized in mock-only mode (no credentials)")

    def _resolve_model_and_prompt(self, mode: str, model: str):
        """Resolve system prompt from mode and API model name from config.

        Returns:
            (system_prompt, api_model) tuple
        """
        from dialeng.services.dialeng_config import get_config
        config = get_config()

        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])
        api_model = config.get_api_model_name(model, self._backend)

        return system_prompt, api_model, config

    async def stream_response(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str = "claude-sonnet-4-5",
        use_thinking: bool = False
    ) -> AsyncIterator[Dict]:
        """Stream LLM response via the appropriate provider.

        Args:
            prompt: The user's prompt/question
            context_messages: Previous conversation context as list of
                             {"role": "user"/"assistant", "content": "..."} dicts
            mode: One of "learning", "concise", "standard"
            model: The Claude model to use (e.g., "claude-sonnet-4-5", "claude-haiku-4-5")
            use_thinking: Whether to enable thinking mode (extended thinking)

        Yields:
            Dict with "type" key:
            - {"type": "thinking_start"} - Start of thinking phase
            - {"type": "thinking", "content": "..."} - Thinking content
            - {"type": "thinking_end"} - End of thinking phase
            - {"type": "chunk", "content": "..."} - Response chunk
            - {"type": "error", "content": "..."} - Error occurred
        """
        try:
            await self._ensure_initialized()

            if not self._provider:
                yield {"type": "error", "content": "No LLM credentials available. Please use Mock mode."}
                return

            system_prompt, api_model, config = self._resolve_model_and_prompt(mode, model)

            async for item in self._provider.stream(
                prompt=prompt,
                context_messages=context_messages,
                system_prompt=system_prompt,
                model=api_model,
                use_thinking=use_thinking,
                config=config,
            ):
                yield item

        except ImportError as e:
            yield {"type": "error", "content": str(e)}
        except Exception as e:
            logger.exception(f"LLM error: {e}")
            yield {"type": "error", "content": f"LLM Error: {str(e)}"}

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
        include_builtins: bool = True
    ) -> AsyncIterator[Dict]:
        """Stream LLM response with tool calling support.

        This method handles:
        1. Parsing $`var` and &`func` syntax in prompt
        2. Substituting variable values from kernel
        3. Building tool schemas from functions
        4. Running the tool loop (up to max_steps iterations)

        Args:
            prompt: User's prompt (may contain $`var` and &`func` syntax)
            context_messages: Previous conversation context
            mode: Dialog mode (learning, concise, standard)
            model: Claude model to use
            use_thinking: Enable extended thinking
            kernel: SubprocessKernel instance for tool execution
            notebook_id: Notebook identifier
            max_steps: Maximum tool loop iterations (default 5)
            include_builtins: Include built-in file tools

        Yields:
            Dicts with 'type' key:
            - {"type": "var_substituted", "name": ..., "value": ...}
            - {"type": "tool_available", "name": ..., "schema": ...}
            - {"type": "thinking_start/thinking/thinking_end", ...}
            - {"type": "chunk", "content": ...}
            - {"type": "tool_call", "id": ..., "name": ..., "input": ...}
            - {"type": "tool_result", "id": ..., "name": ..., "result": ...}
            - {"type": "error", "content": ...}
        """
        from dialeng.services.prompt_parser import parse_prompt, substitute_variables
        from dialeng.services.tool_registry import get_tool_registry

        try:
            await self._ensure_initialized()

            # Parse prompt for special syntax
            var_names, func_names = parse_prompt(prompt)

            # Also parse context_messages for $`var` and &`func` syntax
            for msg in context_messages:
                content = msg.get('content', '')
                # content may be a list of blocks (multimodal) — extract text only
                if isinstance(content, list):
                    content = ' '.join(
                        b.get('text', '') for b in content
                        if isinstance(b, dict) and b.get('type') == 'text'
                    )
                if content:
                    ctx_vars, ctx_funcs = parse_prompt(content)
                    for v in ctx_vars:
                        if v not in var_names:
                            var_names.append(v)
                    for f in ctx_funcs:
                        if f not in func_names:
                            func_names.append(f)

            # If no special syntax and no builtins, fall back to regular streaming
            if not var_names and not func_names and not include_builtins:
                async for item in self.stream_response(prompt, context_messages, mode, model, use_thinking):
                    yield item
                return

            # Substitute variables if kernel is available
            processed_prompt = prompt
            if var_names and kernel:
                processed_prompt, var_info = await substitute_variables(prompt, kernel, notebook_id, var_names)
                for name, info in var_info.items():
                    if info.get('exists'):
                        yield {
                            "type": "var_substituted",
                            "name": name,
                            "var_type": info.get('var_type'),
                            "value": info.get('repr', '')[:100]
                        }

            # Determine if we actually need tool calling
            needs_tool_loop = len(func_names) > 0

            # Get tool registry and build tool list
            registry = get_tool_registry()

            # For claudette provider, include builtins; for others, only if we have func refs
            effective_builtins = include_builtins if self._provider_name == "claudette" else (include_builtins and needs_tool_loop)

            tools = await registry.get_tools_for_prompt(
                func_names,
                kernel,
                notebook_id,
                include_builtins=effective_builtins
            )

            # Notify about available tools
            for tool in tools:
                yield {
                    "type": "tool_available",
                    "name": tool['name'],
                    "description": tool.get('description', '')[:100]
                }

            # If no tools or provider doesn't need tool loop, fall back to regular streaming
            if not tools or (self._provider_name == "claudette_agent" and not needs_tool_loop):
                async for item in self.stream_response(
                    processed_prompt, context_messages, mode, model, use_thinking
                ):
                    yield item
                return

            # Resolve model/prompt for tool calling
            system_prompt, api_model, config = self._resolve_model_and_prompt(mode, model)

            # Delegate to provider's stream_with_tools
            async for item in self._provider.stream_with_tools(
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
            ):
                yield item

        except Exception as e:
            logger.exception(f"Tool-enabled LLM error: {e}")
            yield {"type": "error", "content": f"Tool LLM Error: {str(e)}"}

    def get_provider(self) -> str:
        """Get the current provider name (for debugging/logging)."""
        if self._provider_name is None:
            self._detect_credentials()
        return self._provider_name

    @property
    def last_usage(self) -> Optional[Any]:
        """Get usage stats from the last API call."""
        if self._provider:
            return self._provider.last_result.usage
        return None

    @property
    def last_cost(self) -> Optional[float]:
        """Get cost from the last API call (in USD)."""
        if self._provider:
            return self._provider.last_result.cost
        return None


# Global instance for convenience
llm_service = LLMService()
