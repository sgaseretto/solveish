"""Claudette-agent provider - Claude Code subscription wrapper.

Uses the claudette-agent library which wraps the Claude Agent SDK with
a higher-level AsyncChat interface. Supports stateless configuration
to ensure notebook cells are the sole source of truth for conversation history.
"""
from typing import AsyncIterator, Dict, List, Any
import logging

from ..base_provider import BaseLLMProvider, ProviderInfo, LLMResult
from .. import utils

logger = logging.getLogger(__name__)


class ClaudetteAgentProvider(BaseLLMProvider):
    """LLM provider using claudette-agent (Claude Code subscription wrapper)."""

    def __init__(self):
        super().__init__()
        self._AsyncChat = None

    async def initialize(self) -> None:
        try:
            from claudette_agent import AsyncChat
            self._AsyncChat = AsyncChat
            logger.info("ClaudetteAgentProvider initialized (Claude Code subscription) - stateless mode")
        except ImportError as e:
            logger.error(f"Failed to import claudette_agent: {e}")
            raise ImportError(
                "claudette-agent is not installed. "
                "Install with: pip install git+https://github.com/sgaseretto/claudette-agent.git"
            ) from e

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="claudette_agent",
            display_name="Claudette Agent (Claude Code subscription)",
            supports_native_tools=False,
            supports_mcp_tools=False,
        )

    def check_thinking_support(self, model: str) -> bool:
        try:
            from claudette_agent import can_use_extended_thinking
            return can_use_extended_thinking(model)
        except (ImportError, AttributeError):
            return False

    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
    ) -> AsyncIterator[Dict]:
        """Stream response using claudette-agent (Claude Code subscription).

        Uses AsyncChat with stateless configuration to ensure the notebook
        cells are the sole source of truth for conversation history.

        Key stateless mechanisms:
        1. setting_sources=[] - Prevents loading settings files
        2. cwd=None - No working directory, SDK creates new session each time
        3. extra_args={'no-session-persistence': None} - Prevents saving sessions
        """
        logger.info(f"claudette-agent: Using model {model}")
        logger.info(f"claudette-agent: PROMPT = {prompt[:100]}..." if len(prompt) > 100 else f"claudette-agent: PROMPT = {prompt}")
        logger.info(f"claudette-agent: Context has {len(context_messages)} messages")
        for i, msg in enumerate(context_messages[-5:]):
            role = msg.get('role', '?')
            content = msg.get('content', '')
            content_preview = content[:80] + "..." if len(content) > 80 else content
            logger.info(f"claudette-agent: Context[{i}] {role}: {content_preview}")

        thinking_enabled = use_thinking and self.check_thinking_support(model)
        if use_thinking and not thinking_enabled:
            logger.warning(f"Model {model} does not support extended thinking, disabling")

        full_prompt = utils.build_prompt_with_context(prompt, context_messages)
        logger.info(f"claudette-agent: Built full prompt with {len(context_messages)} context messages")
        logger.info(f"claudette-agent: ===== FULL PROMPT START =====")
        for line in full_prompt.split('\n')[:30]:
            logger.info(f"claudette-agent: {line}")
        if full_prompt.count('\n') > 30:
            logger.info(f"claudette-agent: ... ({full_prompt.count(chr(10)) - 30} more lines)")
        logger.info(f"claudette-agent: ===== FULL PROMPT END =====")

        # Create AsyncChat with fully stateless configuration
        chat = self._AsyncChat(
            model=model,
            sp=system_prompt,
            setting_sources=[],
            cwd=None,
            extra_args={"no-session-persistence": None}
        )

        # Debug: Verify stateless configuration
        actual_setting_sources = getattr(chat.c, 'setting_sources', None)
        actual_extra_args = getattr(chat.c, 'extra_args', None)
        actual_cwd = getattr(chat.c, 'cwd', None)
        logger.info(f"claudette-agent: VERIFY - chat.c.setting_sources = {actual_setting_sources}")
        logger.info(f"claudette-agent: VERIFY - chat.c.extra_args = {actual_extra_args}")
        logger.info(f"claudette-agent: VERIFY - chat.c.cwd = {actual_cwd}")

        # Manually append to chat.h before calling stream()
        # In AsyncChat, _append_pr is async but stream() calls it without await
        chat.h.append({"role": "user", "content": full_prompt})

        maxthinktok = config.thinking_max_tokens if thinking_enabled else 0

        if thinking_enabled:
            yield {"type": "thinking_start"}
            logger.info(f"Extended thinking enabled with maxthinktok={maxthinktok}")

        thinking_phase_ended = False
        try:
            async for block in chat.stream(None, maxthinktok=maxthinktok):
                if hasattr(block, 'type') and block.type == 'thinking':
                    thinking_content = getattr(block, 'thinking', str(block))
                    yield {"type": "thinking", "content": thinking_content}
                elif hasattr(block, 'text'):
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

            if thinking_enabled and not thinking_phase_ended:
                yield {"type": "thinking_end"}

            # Capture usage/cost
            if hasattr(chat, 'use'):
                self._last_result.usage = chat.use
            if hasattr(chat, 'cost'):
                self._last_result.cost = chat.cost
                logger.info(f"claudette-agent: Usage={chat.use}, Cost=${chat.cost:.6f}")

        except Exception as e:
            logger.exception(f"claudette-agent streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}
