"""
LLM Service - Streaming LLM responses via claudette, claudette-agent, or direct SDK.

This module provides the interface between Dialeng notebooks and
Claude models. It supports three providers:

1. claudette: Direct Anthropic API or AWS Bedrock (requires credentials)
2. claudette-agent: Uses Claude Code subscription credentials (wrapper library)
3. claude-agent-sdk (direct): Uses Claude Code subscription via SDK directly (most isolated)

The provider is selected based on available credentials via credential_service.
When using Claude Code subscription, the `use_sdk_directly` config option controls
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
                use_sdk_directly = getattr(config, 'use_sdk_directly', False)  # Default to False

                if use_sdk_directly:
                    logger.info("Using claude-agent-sdk directly (use_sdk_directly=True)")
                    async for item in self._stream_claude_sdk_direct(prompt, context_messages, mode, model, use_thinking):
                        yield item
                else:
                    logger.info("Using claudette-agent wrapper (use_sdk_directly=False)")
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
            from .dialeng_config import get_config

            # Get region from config
            config = get_config()
            aws_region = config.aws_region

            # AnthropicBedrock with explicit region
            print(f"[BEDROCK DEBUG] Creating AnthropicBedrock with region={aws_region}")
            ab = AnthropicBedrock(aws_region=aws_region)
            print(f"[BEDROCK DEBUG] Creating claudette Client with model={api_model}")
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
        """
        Stream LLM response with tool calling support.

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
        from .prompt_parser import parse_prompt, substitute_variables, has_special_syntax
        from .tool_registry import get_tool_registry, is_file_modifying_tool

        try:
            await self._ensure_initialized()

            # Parse prompt for special syntax
            var_names, func_names = parse_prompt(prompt)

            # Also parse context_messages (includes note cells) for $`var` and &`func` syntax
            # This allows declaring variables/functions in note cells that become available
            for msg in context_messages:
                content = msg.get('content', '')
                if content:
                    ctx_vars, ctx_funcs = parse_prompt(content)
                    # Add unique vars/funcs from context
                    for v in ctx_vars:
                        if v not in var_names:
                            var_names.append(v)
                    for f in ctx_funcs:
                        if f not in func_names:
                            func_names.append(f)

            # If no special syntax and no kernel, fall back to regular streaming
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
            # For claudette_agent, only enable tool loop when there are actual &`func` references
            # (the SDK doesn't support custom tool definitions, so built-ins alone don't help)
            needs_tool_loop = len(func_names) > 0

            # Get tool registry and build tool list
            registry = get_tool_registry()

            # For claudette provider, include builtins; for claudette_agent, only if we have func refs
            effective_builtins = include_builtins if self._provider == "claudette" else (include_builtins and needs_tool_loop)

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

            # If no tools or claudette_agent without function references, fall back to regular streaming
            # This handles variable-only substitution for claudette_agent
            if not tools or (self._provider == "claudette_agent" and not needs_tool_loop):
                async for item in self.stream_response(
                    processed_prompt, context_messages, mode, model, use_thinking
                ):
                    yield item
                return

            # Run tool loop based on provider
            if self._provider == "claudette":
                async for item in self._stream_claudette_with_tools(
                    processed_prompt, context_messages, mode, model, use_thinking,
                    tools, kernel, notebook_id, registry, max_steps
                ):
                    yield item
            elif self._provider == "claudette_agent":
                # claudette_agent uses text-based tool calling (tools embedded in prompt)
                async for item in self._stream_sdk_with_text_tools(
                    processed_prompt, context_messages, mode, model, use_thinking,
                    tools, kernel, notebook_id, registry, max_steps
                ):
                    yield item
            else:
                yield {"type": "error", "content": "No LLM credentials available for tool calling."}

        except Exception as e:
            logger.exception(f"Tool-enabled LLM error: {e}")
            yield {"type": "error", "content": f"Tool LLM Error: {str(e)}"}

    async def _stream_claudette_with_tools(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str,
        use_thinking: bool,
        tools: List[Dict],
        kernel,
        notebook_id: str,
        registry,
        max_steps: int
    ) -> AsyncIterator[Dict]:
        """Stream with tool loop using claudette (direct API/Bedrock).

        IMPORTANT: Claudette streaming only yields text chunks. Tool calls are
        stored in chat.h (conversation history) after streaming completes.
        We need to check chat.h[-1] for ToolUseBlock content after streaming.
        """
        import asyncio
        from anthropic.types import ToolUseBlock

        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        # Debug: Print model info to console
        print(f"[BEDROCK DEBUG] UI model: {model}, backend: {self._backend}, API model: {api_model}")

        # Create client
        client = self._create_claudette_client(api_model)

        # Debug: Print tools info
        print(f"[CLAUDETTE TOOLS] Tools passed: {len(tools) if tools else 0}")
        if tools:
            for t in tools:
                t_name = t.get('name') if isinstance(t, dict) else getattr(t, 'name', 'unknown')
                print(f"[CLAUDETTE TOOLS]   Tool: {t_name}")

        # Pass tools to Chat constructor - claudette expects tools set on the instance
        chat = self._Chat(cli=client, sp=system_prompt, tools=tools)

        # Debug: Verify tools are set on chat instance
        chat_tools = getattr(chat, 'tools', None)
        print(f"[CLAUDETTE TOOLS] Chat.tools after init: {len(chat_tools) if chat_tools else 'None'}")

        # Add context messages to history
        for msg in context_messages:
            chat.h.append(msg)

        # Handle thinking
        if use_thinking:
            yield {"type": "thinking_start"}
            yield {"type": "thinking_end"}

        steps = 0
        current_prompt = prompt

        while steps < max_steps:
            steps += 1
            print(f"[CLAUDETTE TOOLS] Step {steps}/{max_steps}, prompt: {current_prompt[:100]}...")

            try:
                response_text = ""

                # Stream text chunks - tool calls will be in chat.h after streaming
                print(f"[CLAUDETTE TOOLS] Starting stream...")
                print(f"[CLAUDETTE TOOLS] History length BEFORE streaming: {len(chat.h)}")

                chunk_count = 0
                stream_result = chat(current_prompt, stream=True)
                print(f"[CLAUDETTE TOOLS] Stream result type: {type(stream_result)}")

                for chunk in stream_result:
                    chunk_count += 1
                    if chunk:
                        if hasattr(chunk, 'text'):
                            response_text += chunk.text
                            yield {"type": "chunk", "content": chunk.text}
                        elif isinstance(chunk, str):
                            response_text += chunk
                            yield {"type": "chunk", "content": chunk}
                    await asyncio.sleep(0)

                print(f"[CLAUDETTE TOOLS] Stream done. Chunks: {chunk_count}, Response length: {len(response_text)}")
                print(f"[CLAUDETTE TOOLS] History length AFTER streaming: {len(chat.h)}")

                # IMPORTANT: Per claudette docs, streaming result has a .value attribute
                # that contains the full response including tool calls
                stream_value = None
                if hasattr(stream_result, 'value'):
                    stream_value = stream_result.value
                    print(f"[CLAUDETTE TOOLS] stream_result.value type: {type(stream_value)}")
                    if stream_value and hasattr(stream_value, 'content'):
                        print(f"[CLAUDETTE TOOLS] stream_result.value.content type: {type(stream_value.content)}")
                        if isinstance(stream_value.content, list):
                            for idx, item in enumerate(stream_value.content):
                                item_type = type(item).__name__
                                if hasattr(item, 'type'):
                                    print(f"[CLAUDETTE TOOLS] stream_result.value.content[{idx}]: type={item.type}, class={item_type}")
                else:
                    print(f"[CLAUDETTE TOOLS] stream_result has no .value attribute")

                # Also check if chat.res holds the response (claudette may store it there)
                if hasattr(chat, 'res') and chat.res:
                    print(f"[CLAUDETTE TOOLS] chat.res type: {type(chat.res)}")
                    if hasattr(chat.res, 'content'):
                        res_content = chat.res.content
                        print(f"[CLAUDETTE TOOLS] chat.res.content type: {type(res_content)}")
                        if isinstance(res_content, list):
                            for idx, item in enumerate(res_content):
                                if hasattr(item, 'type'):
                                    print(f"[CLAUDETTE TOOLS] chat.res.content[{idx}]: type={item.type}")
                else:
                    print(f"[CLAUDETTE TOOLS] chat.res is None or not present")

                # After streaming, check chat.h[-1] for tool calls
                # The last message in history is the assistant's response
                tool_calls = []

                # Debug: print all messages in history
                for i, msg in enumerate(chat.h):
                    role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
                    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                    content_type = type(content).__name__
                    print(f"[CLAUDETTE TOOLS] History[{i}]: role={role}, content_type={content_type}")
                    if isinstance(content, list):
                        for j, item in enumerate(content):
                            item_type = type(item).__name__
                            if hasattr(item, 'type'):
                                print(f"[CLAUDETTE TOOLS]   content[{j}]: type={item.type}, item_type={item_type}")
                            elif isinstance(item, dict):
                                print(f"[CLAUDETTE TOOLS]   content[{j}]: dict_type={item.get('type')}, keys={list(item.keys())}")

                # Helper function to extract tool calls from content
                def extract_tool_calls_from_content(content, source=""):
                    extracted = []
                    if isinstance(content, list):
                        for item in content:
                            # Check for ToolUseBlock type (object with type attribute)
                            if hasattr(item, 'type') and item.type == 'tool_use':
                                print(f"[CLAUDETTE TOOLS] Found tool call from {source}: {item.name}")
                                extracted.append({
                                    'id': item.id,
                                    'name': item.name,
                                    'input': item.input if hasattr(item, 'input') else {}
                                })
                            # Check for dict with type key
                            elif isinstance(item, dict) and item.get('type') == 'tool_use':
                                print(f"[CLAUDETTE TOOLS] Found tool call (dict) from {source}: {item.get('name')}")
                                extracted.append({
                                    'id': item.get('id', f'tool_{len(extracted)}'),
                                    'name': item.get('name'),
                                    'input': item.get('input', {})
                                })
                    return extracted

                # First try to get tool calls from history
                if chat.h and len(chat.h) > 0:
                    last_msg = chat.h[-1]
                    msg_role = last_msg.get('role', 'unknown') if isinstance(last_msg, dict) else getattr(last_msg, 'role', 'unknown')
                    print(f"[CLAUDETTE TOOLS] Last message role: {msg_role}")

                    # Check if the last message has content with tool_use blocks
                    content = last_msg.get('content', []) if isinstance(last_msg, dict) else getattr(last_msg, 'content', [])
                    tool_calls = extract_tool_calls_from_content(content, "history")

                # If no tool calls from history, try stream_result.value (per claudette docs)
                if not tool_calls and stream_value and hasattr(stream_value, 'content'):
                    print(f"[CLAUDETTE TOOLS] Trying to extract from stream_result.value...")
                    tool_calls = extract_tool_calls_from_content(stream_value.content, "stream_value")

                # If still no tool calls, try chat.res as fallback
                if not tool_calls and hasattr(chat, 'res') and chat.res:
                    print(f"[CLAUDETTE TOOLS] Trying to extract from chat.res...")
                    if hasattr(chat.res, 'content'):
                        tool_calls = extract_tool_calls_from_content(chat.res.content, "chat.res")

                print(f"[CLAUDETTE TOOLS] Total tool calls found: {len(tool_calls)}")

                # If no tool calls, we're done
                if not tool_calls:
                    print(f"[CLAUDETTE TOOLS] No tool calls, done")
                    break

                # Execute tool calls
                tool_results = []
                for tc in tool_calls:
                    yield {
                        "type": "tool_call",
                        "id": tc['id'],
                        "name": tc['name'],
                        "input": tc['input']
                    }

                    # Execute tool
                    result = await self._execute_tool(
                        tc['name'], tc['input'], kernel, notebook_id, registry
                    )

                    yield {
                        "type": "tool_result",
                        "id": tc['id'],
                        "name": tc['name'],
                        "result": result
                    }

                    tool_results.append({
                        "tool_use_id": tc['id'],
                        "content": self._format_tool_result_for_llm(result)
                    })

                # For claudette, we need to send tool results back to continue the conversation
                # Build the tool result content
                tool_result_content = []
                for tr in tool_results:
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tr["tool_use_id"],
                        "content": tr["content"]
                    })

                # Check if claudette already added a tool_result placeholder to history
                # This happens because claudette's streaming may auto-add entries
                last_msg = chat.h[-1] if chat.h else None
                last_role = None
                last_content = None
                if last_msg:
                    last_role = last_msg.get('role') if isinstance(last_msg, dict) else getattr(last_msg, 'role', None)
                    last_content = last_msg.get('content', []) if isinstance(last_msg, dict) else getattr(last_msg, 'content', [])

                print(f"[CLAUDETTE TOOLS] Last message in history: role={last_role}")

                # Check if last message already has tool_result (claudette auto-added)
                has_existing_tool_result = False
                if last_role == 'user' and isinstance(last_content, list):
                    for item in last_content:
                        item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                        if item_type == 'tool_result':
                            has_existing_tool_result = True
                            break

                if has_existing_tool_result:
                    # Claudette already added tool_result entries with placeholders
                    # Update the content IN-PLACE to preserve claudette's AttrDict format
                    print(f"[CLAUDETTE TOOLS] Updating existing tool_result entries in-place...")

                    # Get the actual content list from the last message
                    existing_content = last_content

                    # Build a map of our results by tool_use_id
                    results_by_id = {tr["tool_use_id"]: tr["content"] for tr in tool_result_content}

                    # Update each existing tool_result item's content
                    for item in existing_content:
                        item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                        if item_type == 'tool_result':
                            tool_use_id = item.get('tool_use_id') if isinstance(item, dict) else getattr(item, 'tool_use_id', None)
                            if tool_use_id in results_by_id:
                                # Update content in-place (works for both dict and AttrDict)
                                if isinstance(item, dict):
                                    item['content'] = results_by_id[tool_use_id]
                                else:
                                    # AttrDict - update via attribute or dict-like access
                                    try:
                                        item['content'] = results_by_id[tool_use_id]
                                    except:
                                        item.content = results_by_id[tool_use_id]
                                print(f"[CLAUDETTE TOOLS] Updated content for tool_use_id={tool_use_id}")
                else:
                    # No existing tool_result, add as new user message
                    print(f"[CLAUDETTE TOOLS] Adding new tool_result message to history...")
                    chat.h.append({"role": "user", "content": tool_result_content})

                # Continue without prompt - claudette will use the tool results from history
                current_prompt = ""

            except Exception as e:
                import traceback
                print(f"[CLAUDETTE TOOLS] ERROR: {e}")
                traceback.print_exc()
                yield {"type": "error", "content": f"Tool loop error: {str(e)}"}
                break

    async def _stream_sdk_with_text_tools(
        self,
        prompt: str,
        context_messages: List[Dict],
        mode: str,
        model: str,
        use_thinking: bool,
        tools: List[Dict],
        kernel,
        notebook_id: str,
        registry,
        max_steps: int
    ) -> AsyncIterator[Dict]:
        """
        Stream with MCP-based tool calling for claude-agent-sdk.

        The Claude Agent SDK supports custom tools via MCP servers. We create an
        in-process MCP server that wraps our kernel functions, allowing Claude to
        call them natively.

        Key requirements from SDK docs:
        1. Use @tool decorator and create_sdk_mcp_server to define tools
        2. Pass MCP server to ClaudeAgentOptions.mcp_servers
        3. Use streaming input mode (async generator) for prompt
        4. Allow tools with mcp__{server_name}__{tool_name} pattern
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions, tool as sdk_tool, create_sdk_mcp_server
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, ToolUseBlock

        # Get system prompt and model
        base_system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["standard"])

        from .dialeng_config import get_config
        config = get_config()
        api_model = config.get_api_model_name(model, self._backend)

        # Debug: Print model and provider info
        print(f"[SDK-MCP DEBUG] UI model: {model}, backend: {self._backend}, API model: {api_model}")
        print(f"[SDK-MCP DEBUG] Provider: {self._provider}, mode: {mode}")

        # Debug: Print tools info
        print(f"[SDK-MCP DEBUG] Tools passed: {len(tools) if tools else 0}")
        if tools:
            for t in tools:
                t_name = t.get('name') if isinstance(t, dict) else getattr(t, 'name', 'unknown')
                print(f"[SDK-MCP DEBUG]   Tool: {t_name}")

        logger.info(f"sdk-mcp-tools: Using model {api_model} with {len(tools)} tools")

        # Create temp directory for SDK isolation
        temp_cwd = tempfile.mkdtemp(prefix=f"dialeng_tools_{uuid.uuid4().hex[:8]}_")
        logger.info(f"sdk-mcp-tools: Created temp cwd: {temp_cwd}")

        # Store tool execution results for yielding to the UI
        tool_execution_events = []

        # Create MCP tools that wrap our kernel functions
        # Each tool will execute in the kernel when called by Claude
        sdk_tools = []
        allowed_tool_names = []

        for tool_def in tools:
            tool_name = tool_def.get('name', 'unknown')
            tool_desc = tool_def.get('description', f'Tool: {tool_name}')
            tool_schema = tool_def.get('input_schema', {})

            # Build parameter type mapping for @tool decorator
            # SDK expects: {"param_name": type} where type is Python type or JSON schema
            params = {}
            if 'properties' in tool_schema:
                for param_name, param_info in tool_schema['properties'].items():
                    param_type = param_info.get('type', 'string')
                    # Map JSON schema types to Python types
                    type_mapping = {
                        'string': str,
                        'integer': int,
                        'number': float,
                        'boolean': bool,
                        'array': list,
                        'object': dict,
                    }
                    params[param_name] = type_mapping.get(param_type, str)

            print(f"[SDK-MCP DEBUG] Creating MCP tool: {tool_name}, params: {params}")

            # Create a closure that captures the tool name for execution
            # We need to use a factory function to properly capture the tool_name
            def make_tool_handler(captured_tool_name, captured_kernel, captured_notebook_id, captured_registry, captured_events, captured_tool_schema):
                async def tool_handler(args: dict) -> dict:
                    """Execute the tool in the kernel and return results."""
                    print(f"[SDK-MCP DEBUG] MCP tool called: {captured_tool_name}, args: {args}")

                    # MCP passes args as JSON-serialized values, need to convert back to proper types
                    # For example, lists may come as strings like '[1, 2, 3]'
                    converted_args = {}
                    for key, value in args.items():
                        if isinstance(value, str):
                            # Try to parse JSON strings back to proper types
                            import json
                            try:
                                parsed = json.loads(value)
                                # Check if the schema expects this type
                                if captured_tool_schema.get('properties', {}).get(key, {}).get('type') == 'array':
                                    converted_args[key] = parsed if isinstance(parsed, list) else value
                                elif captured_tool_schema.get('properties', {}).get(key, {}).get('type') == 'object':
                                    converted_args[key] = parsed if isinstance(parsed, dict) else value
                                else:
                                    converted_args[key] = value
                            except (json.JSONDecodeError, TypeError):
                                converted_args[key] = value
                        else:
                            converted_args[key] = value

                    print(f"[SDK-MCP DEBUG] Converted args: {converted_args}")

                    # Record the tool call event
                    tool_id = f"mcp_tool_{captured_tool_name}_{len(captured_events)}"
                    captured_events.append({
                        "type": "tool_call",
                        "id": tool_id,
                        "name": captured_tool_name,
                        "input": converted_args
                    })

                    try:
                        # Execute via our existing tool execution mechanism
                        result = await self._execute_tool(
                            captured_tool_name, converted_args, captured_kernel, captured_notebook_id, captured_registry
                        )

                        result_text = self._format_tool_result_for_llm(result)
                        print(f"[SDK-MCP DEBUG] Tool result: {result_text[:200]}...")

                        # Record the result event
                        captured_events.append({
                            "type": "tool_result",
                            "id": tool_id,
                            "name": captured_tool_name,
                            "result": result
                        })

                        return {
                            "content": [{"type": "text", "text": result_text}]
                        }
                    except Exception as e:
                        error_msg = f"Error executing {captured_tool_name}: {str(e)}"
                        print(f"[SDK-MCP DEBUG] Tool error: {error_msg}")

                        captured_events.append({
                            "type": "tool_result",
                            "id": tool_id,
                            "name": captured_tool_name,
                            "result": {"status": "error", "error": error_msg}
                        })

                        return {
                            "content": [{"type": "text", "text": error_msg}],
                            "is_error": True
                        }

                return tool_handler

            # Create the decorated tool function
            handler = make_tool_handler(tool_name, kernel, notebook_id, registry, tool_execution_events, tool_schema)

            # Apply the @sdk_tool decorator
            decorated_tool = sdk_tool(tool_name, tool_desc, params)(handler)
            sdk_tools.append(decorated_tool)

            # Track allowed tool name (mcp__{server}__{tool} format)
            allowed_tool_names.append(f"mcp__notebook_tools__{tool_name}")

        print(f"[SDK-MCP DEBUG] Created {len(sdk_tools)} SDK tools")
        print(f"[SDK-MCP DEBUG] Allowed tools: {allowed_tool_names}")

        # Create the MCP server with all tools
        mcp_server = None
        if sdk_tools:
            mcp_server = create_sdk_mcp_server(
                name="notebook_tools",
                version="1.0.0",
                tools=sdk_tools
            )
            print(f"[SDK-MCP DEBUG] Created MCP server: notebook_tools")

        # Build full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, context_messages)

        print(f"[SDK-MCP DEBUG] Full prompt length: {len(full_prompt)}")

        try:
            # Create async generator for streaming input mode (required for MCP)
            async def message_generator():
                yield {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": full_prompt
                    }
                }

            # Build options with MCP server
            options = ClaudeAgentOptions(
                continue_conversation=False,
                resume=None,
                setting_sources=[],
                cwd=temp_cwd,
                model=api_model,
                system_prompt=base_system_prompt,
                mcp_servers={"notebook_tools": mcp_server} if mcp_server else {},
                allowed_tools=allowed_tool_names if allowed_tool_names else [],
                max_turns=max_steps,
            )

            print(f"[SDK-MCP DEBUG] Starting SDK query with MCP tools...")

            # Track which tool events we've yielded to maintain proper ordering
            # The goal is to interleave text and tool events in the order they occur
            yielded_event_count = 0

            async for message in sdk_query(prompt=message_generator(), options=options):
                print(f"[SDK-MCP DEBUG] Message type: {type(message).__name__}")

                if isinstance(message, ResultMessage):
                    # Usage info
                    if hasattr(message, 'usage') and message.usage:
                        self._last_usage = message.usage
                    print(f"[SDK-MCP DEBUG] ResultMessage: subtype={getattr(message, 'subtype', 'unknown')}")
                    continue

                if isinstance(message, AssistantMessage):
                    if hasattr(message, 'content') and message.content:
                        # CRITICAL: Before processing this message, yield any pending tool events
                        # that were recorded during the PREVIOUS message's tool execution.
                        # This ensures that text in THIS message (which comes AFTER tools ran)
                        # is yielded AFTER the tool events, so it appears outside LLM Steps.
                        while yielded_event_count < len(tool_execution_events):
                            event = tool_execution_events[yielded_event_count]
                            print(f"[SDK-MCP DEBUG] Yielding pending tool event BEFORE text: {event.get('type')} - {event.get('name')}")
                            yield event
                            yielded_event_count += 1

                        # Now collect text from this message
                        message_text = ""
                        has_tool_use = False

                        for block in message.content:
                            if hasattr(block, 'text') and block.text:
                                message_text += block.text
                            elif hasattr(block, 'type') and block.type == 'tool_use':
                                has_tool_use = True
                                print(f"[SDK-MCP DEBUG] Tool use block: {block.name}")

                        # Yield text AFTER any pending tool events have been yielded
                        # Text in a message AFTER tool execution is the response to the tool result
                        if message_text.strip():
                            print(f"[SDK-MCP DEBUG] Yielding text chunk: {len(message_text)} chars")
                            for chunk in self._chunk_text(message_text.strip(), 50):
                                yield {"type": "chunk", "content": chunk}

            # Yield any remaining tool events that weren't yielded during the loop
            while yielded_event_count < len(tool_execution_events):
                event = tool_execution_events[yielded_event_count]
                print(f"[SDK-MCP DEBUG] Yielding remaining tool event: {event.get('type')}")
                yield event
                yielded_event_count += 1

            print(f"[SDK-MCP DEBUG] Completed - yielded {yielded_event_count} tool events")

        except Exception as e:
            import traceback
            print(f"[SDK-MCP DEBUG] Error: {e}")
            traceback.print_exc()
            logger.exception(f"sdk-mcp-tools error: {e}")
            yield {"type": "error", "content": f"Tool loop error: {str(e)}"}
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_cwd, ignore_errors=True)
            except Exception:
                pass

    def _build_text_tool_definitions(self, tools: List[Dict]) -> str:
        """Build a text description of available tools for the system prompt."""
        lines = ["### Tool Definitions\n"]

        for tool in tools:
            name = tool.get('name', 'unknown')
            desc = tool.get('description', 'No description')
            schema = tool.get('input_schema', {})
            props = schema.get('properties', {})
            required = schema.get('required', [])

            lines.append(f"**{name}**")
            lines.append(f"  {desc}")

            if props:
                lines.append("  Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

            lines.append("")

        return "\n".join(lines)

    def _parse_text_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text using ```tool_call markers."""
        import re

        pattern = r'```tool_call\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if 'tool' in parsed:
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
                continue

        return tool_calls

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks for streaming."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        kernel,
        notebook_id: str,
        registry
    ) -> dict:
        """Execute a tool (builtin or kernel function)."""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        # Check if it's a builtin tool
        if registry.is_builtin(tool_name):
            return await registry.execute_builtin(tool_name, tool_input)

        # Execute in kernel
        if kernel:
            return await kernel.execute_tool(tool_name, tool_input)

        return {
            "status": "error",
            "error": f"Tool '{tool_name}' not found and no kernel available"
        }

    def _format_tool_result_for_llm(self, result: dict) -> str:
        """Format a tool result for sending back to the LLM."""
        if result.get('status') == 'error':
            return f"Error: {result.get('error', 'Unknown error')}"

        result_data = result.get('result', {})
        if isinstance(result_data, dict):
            result_type = result_data.get('type', 'text')
            content = result_data.get('content', '')

            if result_type == 'text':
                return content
            elif result_type == 'html':
                return f"[HTML output: {len(content)} chars]\n{content[:500]}..."
            elif result_type == 'image':
                return "[Image output]"
            else:
                return str(content)
        else:
            return str(result_data)

    def _build_tool_results_prompt(self, tool_results: List[dict]) -> str:
        """Build a prompt string containing tool results."""
        parts = ["Here are the results from the tool calls:"]
        for tr in tool_results:
            content = tr.get('content', '')
            parts.append(f"\n{content}")
        parts.append("\nPlease continue based on these results.")
        return "\n".join(parts)

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
