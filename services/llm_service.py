"""
LLM Service - Streaming LLM responses via claudette, claudette-agent, or direct SDK.

This module provides the interface between Dialeng notebooks and
Claude models. It supports three providers:

1. claudette: Direct Anthropic API or AWS Bedrock (requires credentials)
2. claudette-agent: Uses Claude Code subscription credentials (wrapper library)
3. claude-agent-sdk (direct): Uses Claude Code subscription via SDK directly (most isolated)

The provider is selected based on available credentials via credential_service.
When using Claude Code subscription, the `use_sdk_direct` config option controls
whether to use claudette-agent wrapper or the SDK directly.

Modes:
- learning: Guide user to discover answers, ask leading questions
- concise: Brief answers, code-focused, minimal explanation
- standard: Balanced, default helpful assistant behavior
"""
from typing import AsyncIterator, List, Dict, Any, Optional
import logging
import os
import tempfile
import shutil
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Context explanation added to all prompts
# The conversation history may include code cells (shown as python code blocks) and
# notes from the user's notebook. Focus on responding to the user's latest message.
_CONTEXT_PREAMBLE = """You are in an interactive notebook environment. The conversation history may include:
- Code cells (shown as python code blocks with optional output)
- Notes (markdown text)
- Previous prompts and your responses

Focus on responding to the user's LATEST message. The code cells and notes are context from their notebook - don't analyze or list them unless specifically asked."""

# Mode-specific system prompts
SYSTEM_PROMPTS = {
    "learning": f"""{_CONTEXT_PREAMBLE}

You are a coding tutor. Guide the user to discover answers themselves.
Ask leading questions, provide hints, and explain concepts step-by-step.
Don't give direct solutions - help them learn by doing.
When they ask a question, first check their understanding, then guide them with hints.
Celebrate their progress and encourage exploration.""",

    "concise": f"""{_CONTEXT_PREAMBLE}

Be brief and code-focused. Provide minimal explanation.
Answer with code examples when possible. Skip pleasantries.
If asked a question, give the direct answer or code solution.
Only explain if explicitly asked or if the code is complex.""",

    "standard": f"""{_CONTEXT_PREAMBLE}

You are a helpful coding assistant. Provide clear, accurate answers
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
        # Usage and cost tracking from last API call
        self._last_usage: Optional[Any] = None
        self._last_cost: Optional[float] = None

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
                logger.info("Initialized with claudette-agent (Claude Code subscription) - stateless mode")
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
                # Check if we should use SDK directly (more isolated) or claudette-agent wrapper
                from .dialeng_config import get_config
                config = get_config()
                use_sdk_direct = getattr(config, 'use_sdk_direct', True)  # Default to True for isolation

                if use_sdk_direct:
                    logger.info("Using claude-agent-sdk directly (use_sdk_direct=True)")
                    async for item in self._stream_claude_sdk_direct(prompt, context_messages, mode, model, use_thinking):
                        yield item
                else:
                    logger.info("Using claudette-agent wrapper (use_sdk_direct=False)")
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

        logger.info(f"claudette: Using model {api_model} (from {model})")

        # Debug: Log the prompt and context being sent
        logger.info(f"claudette: PROMPT = {prompt[:100]}..." if len(prompt) > 100 else f"claudette: PROMPT = {prompt}")
        logger.info(f"claudette: Context has {len(context_messages)} messages")
        for i, msg in enumerate(context_messages):
            role = msg.get('role', '?')
            content = msg.get('content', '')
            content_preview = content[:80] + "..." if len(content) > 80 else content
            logger.info(f"claudette: Context[{i}] {role}: {content_preview}")

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
        """
        Stream response using claudette-agent (Claude Code subscription).

        Uses AsyncChat with stateless configuration to ensure the notebook
        cells are the sole source of truth for conversation history, not
        Claude Code's internal session tracking.

        Key stateless mechanisms:
        1. setting_sources=[] - Prevents loading settings files
        2. cwd=None - No working directory specified, SDK creates new session each time
        3. extra_args={'no-session-persistence': None} - Prevents saving new sessions

        Note: claudette-agent's _build_options() also sets continue_conversation=False
        and resume=None to ensure no session continuation or resumption.

        Supports real extended thinking via maxthinktok parameter when
        the model supports it (Claude Sonnet 3.7+, Sonnet 4+, Opus 4+).
        """
        # Get system prompt for mode
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        # Map model name using config (claudette-agent uses simple names like "claude-sonnet-4-5")
        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        logger.info(f"claudette-agent: Using model {api_model} (from {model})")

        # Debug: Log the prompt and context being sent
        logger.info(f"claudette-agent: PROMPT = {prompt[:100]}..." if len(prompt) > 100 else f"claudette-agent: PROMPT = {prompt}")
        logger.info(f"claudette-agent: Context has {len(context_messages)} messages")
        for i, msg in enumerate(context_messages[-5:]):  # Log last 5 context messages
            role = msg.get('role', '?')
            content = msg.get('content', '')
            content_preview = content[:80] + "..." if len(content) > 80 else content
            logger.info(f"claudette-agent: Context[{i}] {role}: {content_preview}")

        # Check if model supports extended thinking
        thinking_enabled = use_thinking and self._check_thinking_support(api_model)
        if use_thinking and not thinking_enabled:
            logger.warning(f"Model {api_model} does not support extended thinking, disabling")

        # Build a single prompt that includes context
        # This avoids the issue where multiple "User:" messages in history
        # confuse the SDK about which message to respond to.
        full_prompt = self._build_prompt_with_context(prompt, context_messages)
        logger.info(f"claudette-agent: Built full prompt with {len(context_messages)} context messages")
        # Debug: Log the FULL prompt being sent to see exactly what Claude receives
        logger.info(f"claudette-agent: ===== FULL PROMPT START =====")
        for line in full_prompt.split('\n')[:30]:  # First 30 lines
            logger.info(f"claudette-agent: {line}")
        if full_prompt.count('\n') > 30:
            logger.info(f"claudette-agent: ... ({full_prompt.count(chr(10)) - 30} more lines)")
        logger.info(f"claudette-agent: ===== FULL PROMPT END =====")

        # Create AsyncChat with fully stateless configuration:
        # - setting_sources=[] prevents loading settings files
        # - cwd=None allows SDK to create fresh session each time (no per-project sessions)
        # - extra_args={'no-session-persistence': None} prevents saving new sessions
        # Note: claudette-agent's _build_options() also sets continue_conversation=False
        # and resume=None to ensure no session continuation or resumption.
        chat = self._AsyncChat(
            model=api_model,
            sp=system_prompt,
            setting_sources=[],  # Don't load settings files
            cwd=None,  # No cwd - SDK creates fresh session each time
            extra_args={"no-session-persistence": None}  # Don't save new sessions
        )

        # Debug: Verify the stateless configuration was properly set on the client
        actual_setting_sources = getattr(chat.c, 'setting_sources', None)
        actual_extra_args = getattr(chat.c, 'extra_args', None)
        actual_cwd = getattr(chat.c, 'cwd', None)
        logger.info(f"claudette-agent: VERIFY - chat.c.setting_sources = {actual_setting_sources}")
        logger.info(f"claudette-agent: VERIFY - chat.c.extra_args = {actual_extra_args}")
        logger.info(f"claudette-agent: VERIFY - chat.c.cwd = {actual_cwd}")

        # IMPORTANT: Manually append to chat.h before calling stream()
        # In AsyncChat, _append_pr is async but stream() calls it without await,
        # so the prompt never gets added to history. We must add it manually.
        chat.h.append({"role": "user", "content": full_prompt})

        # Determine maxthinktok value (0 = disabled, >0 = enabled with token budget)
        maxthinktok = config.thinking_max_tokens if thinking_enabled else 0

        if thinking_enabled:
            yield {"type": "thinking_start"}
            logger.info(f"Extended thinking enabled with maxthinktok={maxthinktok}")

        # Stream the response with error handling
        thinking_phase_ended = False
        try:
            # Pass None since we already added the prompt to chat.h manually.
            # This avoids the RuntimeWarning about _append_pr not being awaited.
            async for block in chat.stream(None, maxthinktok=maxthinktok):
                # claudette-agent stream yields complete message blocks
                # Check for thinking blocks (have type='thinking' attribute)
                if hasattr(block, 'type') and block.type == 'thinking':
                    # Extended thinking content
                    thinking_content = getattr(block, 'thinking', str(block))
                    yield {"type": "thinking", "content": thinking_content}
                elif hasattr(block, 'text'):
                    # Regular text block - end thinking phase first if needed
                    if thinking_enabled and not thinking_phase_ended:
                        yield {"type": "thinking_end"}
                        thinking_phase_ended = True
                    yield {"type": "chunk", "content": block.text}
                elif isinstance(block, str):
                    if thinking_enabled and not thinking_phase_ended:
                        yield {"type": "thinking_end"}
                        thinking_phase_ended = True
                    yield {"type": "chunk", "content": block}
                else:
                    content = str(block) if block else ""
                    if content:
                        if thinking_enabled and not thinking_phase_ended:
                            yield {"type": "thinking_end"}
                            thinking_phase_ended = True
                        yield {"type": "chunk", "content": content}

            # Ensure thinking_end is yielded if thinking was enabled but no text blocks came
            if thinking_enabled and not thinking_phase_ended:
                yield {"type": "thinking_end"}

            # Capture usage/cost after streaming completes
            if hasattr(chat, 'use'):
                self._last_usage = chat.use
            if hasattr(chat, 'cost'):
                self._last_cost = chat.cost
                logger.info(f"claudette-agent: Usage={chat.use}, Cost=${chat.cost:.6f}")

        except Exception as e:
            logger.exception(f"claudette-agent streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

    async def _stream_claude_sdk_direct(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str,
        use_thinking: bool
    ) -> AsyncIterator[Dict]:
        """
        Stream response using claude-agent-sdk directly (bypassing claudette-agent wrapper).

        This is the most isolated approach for stateless queries. Each query:
        1. Creates a completely fresh subprocess
        2. Uses a unique temporary directory as cwd
        3. Sets all stateless options explicitly
        4. Cleans up the temp directory after completion

        This bypasses any potential state management in the claudette-agent wrapper.
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
        from claude_agent_sdk.types import AssistantMessage, ResultMessage

        # Get system prompt for mode
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        # Map model name using config
        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        logger.info(f"SDK-direct: Using model {api_model}")

        # Build full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, context_messages)

        # Create unique temporary directory for complete session isolation
        temp_cwd = tempfile.mkdtemp(prefix=f"dialeng_sdk_{uuid.uuid4().hex[:8]}_")
        logger.info(f"SDK-direct: Created temp cwd: {temp_cwd}")

        # Debug logging - save prompt to file if debug mode enabled
        debug_mode = getattr(config, 'debug_mode', False)
        debug_log_dir = getattr(config, 'debug_log_dir', './debug_logs')
        if debug_mode:
            self._save_debug_log(debug_log_dir, "prompt", {
                "timestamp": datetime.now().isoformat(),
                "model": api_model,
                "mode": mode,
                "temp_cwd": temp_cwd,
                "prompt": full_prompt,
                "system_prompt": system_prompt,
            })

        # Log the full prompt
        logger.info(f"SDK-direct: ===== FULL PROMPT START =====")
        for line in full_prompt.split('\n')[:30]:
            logger.info(f"SDK-direct: {line}")
        if full_prompt.count('\n') > 30:
            logger.info(f"SDK-direct: ... ({full_prompt.count(chr(10)) - 30} more lines)")
        logger.info(f"SDK-direct: ===== FULL PROMPT END =====")

        # Build ClaudeAgentOptions with maximum isolation
        options = ClaudeAgentOptions(
            # Core stateless settings
            continue_conversation=False,  # Don't continue any conversation
            resume=None,  # Don't resume any session
            # Session isolation
            setting_sources=[],  # Don't load any settings files
            cwd=temp_cwd,  # Use unique temp cwd per query
            # Model and system prompt
            model=api_model,
            system_prompt=system_prompt,
        )

        logger.info(f"SDK-direct: Options - continue_conversation={options.continue_conversation}, "
                    f"resume={options.resume}, setting_sources={options.setting_sources}, cwd={options.cwd}")

        # Handle thinking mode
        thinking_enabled = use_thinking and self._check_thinking_support(api_model)
        if use_thinking and not thinking_enabled:
            logger.warning(f"Model {api_model} does not support extended thinking, disabling")

        if thinking_enabled:
            yield {"type": "thinking_start"}
            logger.info("SDK-direct: Extended thinking enabled")

        thinking_phase_ended = False
        collected_response = []

        try:
            # Use SDK query() directly - this is fully stateless
            async for message in sdk_query(prompt=full_prompt, options=options):
                # Process different message types
                if isinstance(message, ResultMessage):
                    # ResultMessage contains usage and cost info
                    if hasattr(message, 'usage') and message.usage:
                        self._last_usage = message.usage
                        logger.info(f"SDK-direct: Usage = {message.usage}")
                    if hasattr(message, 'total_cost_usd'):
                        self._last_cost = message.total_cost_usd
                        logger.info(f"SDK-direct: Cost = ${message.total_cost_usd:.6f}")
                    continue

                if isinstance(message, AssistantMessage):
                    # AssistantMessage contains the response content
                    if hasattr(message, 'content') and message.content:
                        for block in message.content:
                            # Check for thinking blocks
                            if hasattr(block, 'type') and block.type == 'thinking':
                                thinking_content = getattr(block, 'thinking', str(block))
                                yield {"type": "thinking", "content": thinking_content}
                            # Check for text blocks
                            elif hasattr(block, 'text'):
                                if thinking_enabled and not thinking_phase_ended:
                                    yield {"type": "thinking_end"}
                                    thinking_phase_ended = True
                                yield {"type": "chunk", "content": block.text}
                                collected_response.append(block.text)

            # Ensure thinking_end is yielded if needed
            if thinking_enabled and not thinking_phase_ended:
                yield {"type": "thinking_end"}

            # Debug logging - save response
            if debug_mode:
                self._save_debug_log(debug_log_dir, "response", {
                    "timestamp": datetime.now().isoformat(),
                    "model": api_model,
                    "response": "".join(collected_response),
                    "usage": str(self._last_usage),
                    "cost": self._last_cost,
                })

        except Exception as e:
            logger.exception(f"SDK-direct streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

        finally:
            # CRITICAL: Clean up temp directory to prevent any future session loading
            if temp_cwd and os.path.exists(temp_cwd):
                try:
                    shutil.rmtree(temp_cwd, ignore_errors=True)
                    logger.info(f"SDK-direct: Cleaned up temp cwd: {temp_cwd}")
                except Exception as cleanup_err:
                    logger.warning(f"SDK-direct: Failed to clean up {temp_cwd}: {cleanup_err}")

    def _save_debug_log(self, debug_log_dir: str, log_type: str, data: dict):
        """Save debug data to a timestamped JSON file."""
        try:
            os.makedirs(debug_log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{log_type}.json"
            filepath = os.path.join(debug_log_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Debug log saved to: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug log: {e}")

    def _build_prompt_with_context(self, prompt: str, context_messages: List[Dict]) -> str:
        """
        Build a single prompt string that includes conversation context.

        Instead of appending context messages to chat.h (which creates multiple
        "User:" messages that can confuse the Claude Agent SDK), we build a
        single prompt that clearly presents the context and the current question.

        This ensures the SDK sees ONE clear user message to respond to.

        Args:
            prompt: The current user prompt
            context_messages: Previous conversation context

        Returns:
            A formatted prompt string including context
        """
        if not context_messages:
            return prompt

        # Build context section
        context_parts = []
        for msg in context_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")

        context_text = "\n\n".join(context_parts)

        # Build the full prompt with clear structure
        full_prompt = f"""Here is the conversation history:

{context_text}

---

Now respond to my latest message:

{prompt}"""

        return full_prompt

    def get_provider(self) -> str:
        """Get the current provider (for debugging/logging)."""
        if self._provider is None:
            self._detect_credentials()
        return self._provider

    def _check_thinking_support(self, model: str) -> bool:
        """
        Check if the given model supports extended thinking.

        Uses claudette-agent's can_use_extended_thinking() for capability detection.
        """
        if self._provider == "claudette_agent":
            try:
                from claudette_agent import can_use_extended_thinking
                return can_use_extended_thinking(model)
            except (ImportError, AttributeError):
                # can_use_extended_thinking not available
                return False
        elif self._provider == "claudette":
            # For claudette, check model name patterns
            # Extended thinking supported on Sonnet 3.7+, Sonnet 4+, Opus 4+
            model_lower = model.lower()
            return ("sonnet-4" in model_lower or
                    "opus-4" in model_lower or
                    "3-7" in model_lower or
                    "3.7" in model_lower)
        return False

    @property
    def last_usage(self) -> Optional[Any]:
        """Get usage stats from the last API call."""
        return self._last_usage

    @property
    def last_cost(self) -> Optional[float]:
        """Get cost from the last API call (in USD)."""
        return self._last_cost


# Global instance for convenience
llm_service = LLMService()
