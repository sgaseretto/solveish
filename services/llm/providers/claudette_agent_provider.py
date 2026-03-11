"""Claudette-agent provider - Claude Code subscription wrapper.

Uses the claudette-agent library which wraps the Claude Agent SDK with
a higher-level Chat/AsyncChat interface. Now supports:
- Character-level streaming via StreamEvent (chat.stream())
- Native tool calling via MCP servers (Chat(tools=[...]))
- Tool loop with automatic follow-up (chat.toolloop())
- Extended thinking (non-streaming, incompatible with streaming per SDK)
- Stateless by default (setting_sources=[])
"""
from typing import AsyncIterator, Dict, List, Any
import logging

from ..base_provider import BaseLLMProvider, ProviderInfo, LLMResult
from .. import utils

logger = logging.getLogger(__name__)


class ClaudetteAgentProvider(BaseLLMProvider):
    """LLM provider using claudette-agent (Claude Code subscription wrapper).

    Uses Chat (not AsyncChat) for streaming because AsyncChat's async
    _append_pr override is not properly awaited by the inherited stream()
    method. Chat's sync _append_pr works correctly with the async generator.
    """

    def __init__(self):
        super().__init__()
        self._Chat = None
        self._contents = None

    async def initialize(self) -> None:
        try:
            from claudette_agent import Chat, contents
            self._Chat = Chat
            self._contents = contents
            logger.info("ClaudetteAgentProvider initialized (Claude Code subscription)")
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
            supports_native_tools=True,
            supports_mcp_tools=True,
        )

    def check_thinking_support(self, model: str) -> bool:
        try:
            from claudette_agent import can_use_extended_thinking
            return can_use_extended_thinking(model)
        except (ImportError, AttributeError):
            return False

    def _create_chat(self, model: str, system_prompt: str, tools: List = None,
                     extra_args: Dict = None) -> Any:
        """Create a Chat instance with stateless configuration.

        Uses Chat (not AsyncChat) because Chat.stream() calls sync
        _append_pr which works correctly. AsyncChat's async _append_pr
        is not awaited by the inherited stream() method.
        """
        kwargs = dict(
            model=model,
            sp=system_prompt,
            setting_sources=[],  # Stateless by default
        )
        if extra_args:
            kwargs['extra_args'] = extra_args
        if tools:
            kwargs['tools'] = tools

        return self._Chat(**kwargs)

    @staticmethod
    def _split_context_images(context_messages: List[Dict]):
        """Separate image blocks from context messages.

        Why this is needed (current SDK limitations):
        - claudette-agent's chat.stream() uses _build_conversation_prompt() which
          flattens ALL messages (including images) into a plain text string, losing
          image data entirely.
        - chat() (non-streaming) checks _has_images() but only preserves images in
          the LAST message via _call_with_images(), which uses ClaudeSDKClient to
          send structured content via stdin transport.
        - claude-agent-sdk's query() passes prompts as CLI arguments — base64 images
          cause "Argument list too long" (OS limit ~256KB on macOS).

        Future: When claude-agent-sdk's query() supports images natively (stdin
        transport or chunked args), this workaround can be simplified. The images
        could stay in context_messages and chat.stream() could handle them directly.

        Returns:
            (text_only_messages, image_blocks): Context with images stripped,
            and list of image content blocks to attach to the prompt.
        """
        text_messages = []
        image_blocks = []

        for msg in context_messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'image':
                        image_blocks.append(block)
                    else:
                        text_parts.append(block)
                if text_parts:
                    text_messages.append({"role": msg["role"], "content": text_parts})
            else:
                text_messages.append(msg)

        return text_messages, image_blocks

    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
    ) -> AsyncIterator[Dict]:
        """Stream response using claudette-agent.

        For streaming (no thinking, no images): Uses chat.stream(prompt) which
        yields text strings directly (handles StreamEvent internally).

        For thinking or images: Uses chat(prompt) non-streaming since:
        - Streaming and thinking are incompatible per SDK docs
        - chat.stream() doesn't support image content blocks; chat() routes
          to _call_with_images only when images are in the LAST message
        """
        logger.info(f"claudette-agent: Using model {model}")
        logger.info(f"claudette-agent: Context has {len(context_messages)} messages")

        thinking_enabled = use_thinking and self.check_thinking_support(model)
        if use_thinking and not thinking_enabled:
            logger.warning(f"Model {model} does not support extended thinking, disabling")

        # Separate images from context — they'll be attached to the prompt
        text_messages, image_blocks = self._split_context_images(context_messages)
        has_images = len(image_blocks) > 0
        if has_images:
            logger.info(f"claudette-agent: {len(image_blocks)} image(s) extracted from context, attaching to prompt")

        chat = self._create_chat(model, system_prompt)

        # Add text-only context to history
        for msg in text_messages:
            chat.h.append(msg)

        # Build the prompt: multimodal (text + images) or plain text
        if has_images:
            # Multimodal prompt — mk_msg will create content blocks,
            # _has_images() will detect them, _call_with_images sends via stdin
            multimodal_prompt = [{"type": "text", "text": prompt}] + image_blocks
        else:
            multimodal_prompt = prompt

        # Non-streaming path: thinking mode OR image content
        if thinking_enabled or has_images:
            if thinking_enabled:
                yield {"type": "thinking_start"}

            maxthinktok = getattr(config, 'thinking_max_tokens', 10000) if thinking_enabled else 0
            if thinking_enabled:
                logger.info(f"claudette-agent: Extended thinking with maxthinktok={maxthinktok}")

            try:
                response = await chat(multimodal_prompt, maxthinktok=maxthinktok)

                thinking_ended = False
                if hasattr(response, 'content'):
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == 'thinking':
                            thinking_content = getattr(block, 'thinking', str(block))
                            yield {"type": "thinking", "content": thinking_content}
                        elif hasattr(block, 'text'):
                            if thinking_enabled and not thinking_ended:
                                yield {"type": "thinking_end"}
                                thinking_ended = True
                            yield {"type": "chunk", "content": block.text}

                if thinking_enabled and not thinking_ended:
                    yield {"type": "thinking_end"}

                # Capture usage/cost
                self._last_result.usage = chat.use
                self._last_result.cost = chat.cost
                logger.info(f"claudette-agent: Usage={chat.use}, Cost=${chat.cost:.6f}")

            except Exception as e:
                logger.exception(f"claudette-agent error: {e}")
                yield {"type": "error", "content": f"Error: {str(e)}"}
        else:
            # Streaming mode: chat.stream() yields text strings directly
            try:
                async for text_chunk in chat.stream(prompt):
                    if text_chunk:
                        yield {"type": "chunk", "content": text_chunk}

                # Capture usage/cost
                self._last_result.usage = chat.use
                self._last_result.cost = chat.cost
                logger.info(f"claudette-agent: Usage={chat.use}, Cost=${chat.cost:.6f}")

            except Exception as e:
                logger.exception(f"claudette-agent streaming error: {e}")
                yield {"type": "error", "content": f"Streaming error: {str(e)}"}

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
        """Stream with tool calling using claudette-agent's native tool support.

        claudette-agent now supports tools natively via Chat(tools=[...]).
        Tools are automatically converted to SDK MCP tools. For streaming
        with tools, uses chat.stream(prompt). For non-streaming tool loops,
        uses chat.toolloop(prompt, max_steps=N).
        """
        from claudette_agent import tool as ca_tool

        logger.info(f"claudette-agent-tools: Using model {model} with {len(tools)} tools")

        # Convert our tool definitions to claudette-agent @tool decorated functions
        sdk_tools = []
        for tool_def in tools:
            tool_name = tool_def.get('name', 'unknown')
            tool_desc = tool_def.get('description', f'Tool: {tool_name}')
            tool_schema = tool_def.get('input_schema', {})

            # Build parameter type mapping
            params = {}
            if 'properties' in tool_schema:
                for param_name, param_info in tool_schema['properties'].items():
                    param_type = param_info.get('type', 'string')
                    type_mapping = {
                        'string': str, 'integer': int, 'number': float,
                        'boolean': bool, 'array': list, 'object': dict,
                    }
                    params[param_name] = type_mapping.get(param_type, str)

            # Create tool handler closure
            def make_handler(captured_name, captured_schema):
                async def handler(args: dict) -> dict:
                    # Convert JSON string args back to proper types
                    converted = {}
                    for key, value in args.items():
                        if isinstance(value, str):
                            import json
                            try:
                                parsed = json.loads(value)
                                expected = captured_schema.get('properties', {}).get(key, {}).get('type')
                                if expected == 'array' and isinstance(parsed, list):
                                    converted[key] = parsed
                                elif expected == 'object' and isinstance(parsed, dict):
                                    converted[key] = parsed
                                else:
                                    converted[key] = value
                            except (json.JSONDecodeError, TypeError):
                                converted[key] = value
                        else:
                            converted[key] = value

                    result = await utils.execute_tool(
                        captured_name, converted, kernel, notebook_id, registry
                    )
                    result_text = utils.format_tool_result_for_llm(result)
                    return {"content": [{"type": "text", "text": result_text}]}
                return handler

            handler = make_handler(tool_name, tool_schema)

            # Use claude_agent_sdk's @tool decorator directly
            from claude_agent_sdk import tool as sdk_tool
            decorated = sdk_tool(tool_name, tool_desc, params)(handler)
            sdk_tools.append(decorated)

        # Create chat with tools
        chat = self._create_chat(model, system_prompt, tools=sdk_tools)

        # Add context messages to history
        for msg in context_messages:
            chat.h.append(msg)

        try:
            # Use streaming with tools - chat.stream() handles MCP tools
            async for text_chunk in chat.stream(prompt):
                if text_chunk:
                    yield {"type": "chunk", "content": text_chunk}

            # Capture usage/cost
            self._last_result.usage = chat.use
            self._last_result.cost = chat.cost
            logger.info(f"claudette-agent-tools: Usage={chat.use}, Cost=${chat.cost:.6f}")

        except Exception as e:
            logger.exception(f"claudette-agent-tools error: {e}")
            yield {"type": "error", "content": f"Tool loop error: {str(e)}"}



# Register as an LLM provider
def _register_claudette_agent_provider():
    from core.registry import registry, ProviderRegistration
    registry.register_provider(ProviderRegistration(
        name="claudette_agent", label="Claude Code (claudette-agent)",
        factory=ClaudetteAgentProvider,
        priority=5
    ))

_register_claudette_agent_provider()
