"""Base LLM provider protocol for multi-provider support.

Defines the abstract interface that all LLM provider backends must implement.
Follows the same pattern as services/kernel/base_kernel.py.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ProviderInfo:
    """Static metadata about an LLM provider backend."""
    provider_name: str           # "claudette" | "claudette_agent" | "claude_agent_sdk"
    display_name: str            # "Claudette (API/Bedrock)" | "Claude Agent SDK"
    supports_native_tools: bool  # True for claudette (Anthropic tool calling)
    supports_mcp_tools: bool     # True for claude_agent_sdk (MCP-based tools)
    supports_streaming: bool = True


@dataclass
class LLMResult:
    """Usage and cost from a completed LLM call."""
    usage: Optional[Any] = None
    cost: Optional[float] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider implementations.

    Each provider handles the specifics of streaming responses from a particular
    LLM library or SDK. The coordinator (LLMService) resolves mode→system_prompt
    and model→api_model before calling provider methods, so providers receive
    ready-to-use values.

    Yielded event dicts must use these types:
    - {"type": "chunk", "content": "..."} - Text response fragment
    - {"type": "thinking_start"} - Extended thinking begins
    - {"type": "thinking", "content": "..."} - Thinking content
    - {"type": "thinking_end"} - Extended thinking complete
    - {"type": "error", "content": "..."} - Error occurred
    - {"type": "tool_call", "id": ..., "name": ..., "input": ...} - Tool invoked
    - {"type": "tool_result", "id": ..., "name": ..., "result": ...} - Tool result
    """

    def __init__(self):
        self._last_result = LLMResult()

    @abstractmethod
    async def initialize(self) -> None:
        """Import libraries and validate the provider is ready.

        Called once lazily by the coordinator on first use.
        Should raise ImportError if required libraries are missing.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
    ) -> AsyncIterator[Dict]:
        """Stream a plain response (no tools).

        Args:
            prompt: The user's prompt/question
            context_messages: Previous conversation as [{"role": ..., "content": ...}]
            system_prompt: Already resolved from mode by the coordinator
            model: API model name, already mapped by the coordinator
            use_thinking: Whether to enable extended thinking
            config: DialengConfig instance for provider-specific settings

        Yields:
            Event dicts (chunk, thinking_start, thinking, thinking_end, error)
        """
        ...

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
    ) -> AsyncIterator[Dict]:
        """Stream a response with tool calling support.

        Default implementation raises NotImplementedError. Override in providers
        that support tool calling.

        Yields:
            Event dicts (chunk, tool_call, tool_result, thinking_*, error)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling"
        )
        # Make this an async generator
        yield  # pragma: no cover

    @abstractmethod
    def check_thinking_support(self, model: str) -> bool:
        """Whether the given model supports extended thinking."""
        ...

    @abstractmethod
    def get_info(self) -> ProviderInfo:
        """Return static provider metadata."""
        ...

    @property
    def last_result(self) -> LLMResult:
        """Usage/cost from the most recent call."""
        return self._last_result
