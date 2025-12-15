"""
LLM Service - Streaming LLM responses via claudette or claudette-agent.

This module provides the interface between Indianapolis notebooks and
Claude models. It supports two providers:

1. claudette: Direct Anthropic API or AWS Bedrock (requires credentials)
2. claudette-agent: Uses Claude Code subscription credentials

The provider is selected based on available credentials via credential_service.

Modes:
- learning: Guide user to discover answers, ask leading questions
- concise: Brief answers, code-focused, minimal explanation
- standard: Balanced, default helpful assistant behavior
"""
from typing import AsyncIterator, List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Mode-specific system prompts
SYSTEM_PROMPTS = {
    "learning": """You are a coding tutor. Guide the user to discover answers themselves.
Ask leading questions, provide hints, and explain concepts step-by-step.
Don't give direct solutions - help them learn by doing.
When they ask a question, first check their understanding, then guide them with hints.
Celebrate their progress and encourage exploration.""",

    "concise": """Be brief and code-focused. Provide minimal explanation.
Answer with code examples when possible. Skip pleasantries.
If asked a question, give the direct answer or code solution.
Only explain if explicitly asked or if the code is complex.""",

    "standard": """You are a helpful coding assistant. Provide clear, accurate answers
with appropriate code examples and explanations.
Balance being thorough with being concise.
Explain your reasoning when helpful, but don't over-explain simple things.""",
}

# Model mappings are now loaded from dialeng_config.json
# See services/dialeng_config.py for configuration management


class LLMService:
    """
    Service for streaming LLM responses.

    Supports two providers:
    - claudette: For direct API/Bedrock access
    - claudette_agent: For Claude Code subscription

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
        self._provider: Optional[str] = None
        self._backend: Optional[str] = None  # "anthropic_api" or "bedrock"
        self._AsyncChat = None  # claudette-agent
        self._Chat = None       # claudette
        self._Client = None     # claudette Client wrapper

    def _detect_credentials(self):
        """Detect credentials and store provider/backend info."""
        from .credential_service import detect_credentials
        cred_status = detect_credentials()
        self._provider = cred_status.provider
        self._backend = cred_status.backend
        return cred_status

    async def _ensure_initialized(self):
        """Lazy initialization - detect provider and import appropriate library."""
        if self._initialized:
            return

        cred_status = self._detect_credentials()
        logger.info(f"LLM Service initializing with provider: {self._provider}, backend: {self._backend}")

        if self._provider == "claudette":
            try:
                from claudette import Chat, Client
                self._Chat = Chat
                self._Client = Client
                self._initialized = True
                logger.info(f"Initialized with claudette ({self._backend})")
            except ImportError as e:
                logger.error(f"Failed to import claudette: {e}")
                raise ImportError(
                    "claudette is not installed. Install with: pip install claudette"
                ) from e

        elif self._provider == "claudette_agent":
            try:
                from claudette_agent import AsyncChat
                self._AsyncChat = AsyncChat
                self._initialized = True
                logger.info("Initialized with claudette-agent (Claude Code subscription)")
            except ImportError as e:
                logger.error(f"Failed to import claudette_agent: {e}")
                raise ImportError(
                    "claudette-agent is not installed. "
                    "Install with: pip install git+https://github.com/sgaseretto/claudette-agent.git"
                ) from e

        else:
            # mock_only - shouldn't reach stream_response, but be safe
            self._initialized = True
            logger.warning("LLM Service initialized in mock-only mode (no credentials)")

    async def stream_response(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str = "claude-sonnet-4-5",
        use_thinking: bool = False
    ) -> AsyncIterator[Dict]:
        """
        Stream LLM response via the appropriate provider.

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

            if self._provider == "claudette":
                async for item in self._stream_claudette(prompt, context_messages, mode, model, use_thinking):
                    yield item
            elif self._provider == "claudette_agent":
                async for item in self._stream_claudette_agent(prompt, context_messages, mode, model, use_thinking):
                    yield item
            else:
                yield {"type": "error", "content": "No LLM credentials available. Please use Mock mode."}

        except ImportError as e:
            yield {"type": "error", "content": str(e)}
        except Exception as e:
            logger.exception(f"LLM error: {e}")
            yield {"type": "error", "content": f"LLM Error: {str(e)}"}

    def _create_claudette_client(self, api_model: str):
        """
        Create the appropriate claudette Client based on backend type.

        For Bedrock: Create AnthropicBedrock client and wrap in Client
        For Anthropic API: Create regular Client (uses ANTHROPIC_API_KEY env var)
        """
        if self._backend == "bedrock":
            # For Bedrock, create AnthropicBedrock client
            from anthropic import AnthropicBedrock

            # AnthropicBedrock auto-detects credentials from environment
            # (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION)
            ab = AnthropicBedrock()
            return self._Client(api_model, ab)
        else:
            # For direct Anthropic API, create simple Client
            # (uses ANTHROPIC_API_KEY from environment)
            return self._Client(api_model)

    async def _stream_claudette(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str,
        use_thinking: bool
    ) -> AsyncIterator[Dict]:
        """
        Stream response using claudette (direct API/Bedrock).

        Claudette API:
        - Sync streaming: `for chunk in chat(prompt, stream=True): ...`
        - For Bedrock: Need to create AnthropicBedrock client and pass via `cli` param

        Note: claudette's Chat is synchronous, so we wrap sync iteration in async context.
        """
        import asyncio

        # Get system prompt for mode
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        # Map model name based on backend (Bedrock vs direct API) using config
        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        # Create appropriate client based on backend (API vs Bedrock)
        client = self._create_claudette_client(api_model)

        # Create claudette Chat instance with the configured client
        chat = self._Chat(cli=client, sp=system_prompt)

        # Add context messages to history
        for msg in context_messages:
            chat.h.append(msg)

        # Handle thinking mode if enabled
        if use_thinking:
            yield {"type": "thinking_start"}
            yield {"type": "thinking_end"}

        # Stream the response using claudette's API: chat(prompt, stream=True)
        try:
            # claudette's sync streaming: for chunk in chat(prompt, stream=True)
            # Each chunk is typically a string
            for chunk in chat(prompt, stream=True):
                if chunk:
                    # chunk is usually a string directly
                    content = str(chunk) if not isinstance(chunk, str) else chunk
                    if content:
                        yield {"type": "chunk", "content": content}
                # Small yield to keep async cooperative
                await asyncio.sleep(0)

        except Exception as e:
            logger.exception(f"claudette streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

    async def _stream_claudette_agent(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str,
        use_thinking: bool
    ) -> AsyncIterator[Dict]:
        """Stream response using claudette-agent (Claude Code subscription)."""
        # Get system prompt for mode
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        # Map model name using config (claudette-agent uses simple names like "claude-sonnet-4-5")
        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        logger.info(f"claudette-agent: Using model {api_model} (from {model})")

        # Create chat instance with model and system prompt
        chat = self._AsyncChat(model=api_model, sp=system_prompt)

        # Add context messages to history
        for msg in context_messages:
            chat.h.append(msg)

        # Handle thinking mode if enabled
        if use_thinking:
            yield {"type": "thinking_start"}
            # Note: claudette-agent may support extended thinking in the future
            yield {"type": "thinking_end"}

        # Stream the response with error handling
        try:
            async for block in chat.stream(prompt):
                # claudette-agent stream yields complete message blocks
                if hasattr(block, 'text'):
                    yield {"type": "chunk", "content": block.text}
                elif isinstance(block, str):
                    yield {"type": "chunk", "content": block}
                else:
                    content = str(block) if block else ""
                    if content:
                        yield {"type": "chunk", "content": content}
        except Exception as e:
            logger.exception(f"claudette-agent streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

    def get_provider(self) -> str:
        """Get the current provider (for debugging/logging)."""
        if self._provider is None:
            self._detect_credentials()
        return self._provider


# Global instance for convenience
llm_service = LLMService()
