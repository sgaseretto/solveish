"""Claudette provider - Direct Anthropic API or AWS Bedrock.

Uses the claudette library for streaming responses. Supports both
direct Anthropic API (via ANTHROPIC_API_KEY) and AWS Bedrock.
"""
from typing import AsyncIterator, Dict, List, Any, Optional
import asyncio
import logging

from ..base_provider import BaseLLMProvider, ProviderInfo, LLMResult
from .. import utils

logger = logging.getLogger(__name__)


class ClaudetteProvider(BaseLLMProvider):
    """LLM provider using claudette (direct API/Bedrock)."""

    def __init__(self, backend: str = "anthropic_api"):
        super().__init__()
        self._backend = backend
        self._Chat = None
        self._Client = None

    @staticmethod
    def _split_context_images(context_messages: List[Dict]):
        """Separate image blocks from context messages.

        Why this is needed (Anthropic API constraints):
        - Images can only appear in user turns, not assistant turns.
        - Context messages include prior prompt cell outputs (assistant role) which
          may contain image blocks after cell_to_llm_messages processing.
        - claudette's _append_pr auto-resolves consecutive user messages by calling
          self(), which can reorder messages and place images in assistant turns.

        By stripping images from context and attaching them to the current prompt
        (always a user turn), we guarantee correct API message structure.

        Future: If finalize_cell_execution preserves structured outputs alongside
        HTML, the two-source extraction in _extract_image_blocks could be simplified.
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

    async def initialize(self) -> None:
        try:
            from claudette import Chat, Client
            self._Chat = Chat
            self._Client = Client
            logger.info(f"ClaudetteProvider initialized ({self._backend})")
        except ImportError as e:
            logger.error(f"Failed to import claudette: {e}")
            raise ImportError(
                "claudette is not installed. Install with: pip install claudette"
            ) from e

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="claudette",
            display_name=f"Claudette ({self._backend})",
            supports_native_tools=True,
            supports_mcp_tools=False,
        )

    def check_thinking_support(self, model: str) -> bool:
        model_lower = model.lower()
        return ("sonnet-4" in model_lower or
                "opus-4" in model_lower or
                "3-7" in model_lower or
                "3.7" in model_lower)

    def _create_client(self, api_model: str):
        """Create the appropriate claudette Client based on backend type."""
        if self._backend == "bedrock":
            from anthropic import AnthropicBedrock
            from dialeng.services.dialeng_config import get_config
            config = get_config()
            aws_region = config.aws_region
            logger.debug(f"Creating AnthropicBedrock with region={aws_region}")
            ab = AnthropicBedrock(aws_region=aws_region)
            return self._Client(api_model, ab)
        else:
            return self._Client(api_model)

    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
    ) -> AsyncIterator[Dict]:
        logger.info(f"claudette: Using model {model}")
        logger.info(f"claudette: Context has {len(context_messages)} messages")

        # Separate images from context — attach to prompt (user turn only)
        text_messages, image_blocks = self._split_context_images(context_messages)
        if image_blocks:
            logger.info(f"claudette: {len(image_blocks)} image(s) extracted, attaching to prompt")

        client = self._create_client(model)
        chat = self._Chat(cli=client, sp=system_prompt)

        for msg in text_messages:
            chat.h.append(msg)

        # Build prompt: multimodal if images, plain text otherwise
        if image_blocks:
            actual_prompt = [{"type": "text", "text": prompt}] + image_blocks
        else:
            actual_prompt = prompt

        if use_thinking:
            yield {"type": "thinking_start"}
            yield {"type": "thinking_end"}

        try:
            for chunk in chat(actual_prompt, stream=True):
                if chunk:
                    content = str(chunk) if not isinstance(chunk, str) else chunk
                    if content:
                        yield {"type": "chunk", "content": content}
                await asyncio.sleep(0)
        except Exception as e:
            logger.exception(f"claudette streaming error: {e}")
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
        """Stream with tool loop using claudette (direct API/Bedrock).

        Claudette streaming only yields text chunks. Tool calls are
        stored in chat.h (conversation history) after streaming completes.
        We check chat.h[-1] for ToolUseBlock content after streaming.
        """
        from anthropic.types import ToolUseBlock

        logger.debug(f"claudette-tools: UI model={model}, backend={self._backend}")

        # Separate images from context — attach to prompt (user turn only)
        text_messages, image_blocks = self._split_context_images(context_messages)
        if image_blocks:
            logger.info(f"claudette-tools: {len(image_blocks)} image(s) extracted, attaching to prompt")

        client = self._create_client(model)

        logger.debug(f"claudette-tools: {len(tools)} tools passed")
        for t in tools:
            t_name = t.get('name') if isinstance(t, dict) else getattr(t, 'name', 'unknown')
            logger.debug(f"claudette-tools: Tool: {t_name}")

        chat = self._Chat(cli=client, sp=system_prompt, tools=tools)

        for msg in text_messages:
            chat.h.append(msg)

        if use_thinking:
            yield {"type": "thinking_start"}
            yield {"type": "thinking_end"}

        steps = 0
        # First iteration: attach images to prompt if present
        if image_blocks:
            current_prompt = [{"type": "text", "text": prompt}] + image_blocks
        else:
            current_prompt = prompt

        while steps < max_steps:
            steps += 1
            logger.debug(f"claudette-tools: Step {steps}/{max_steps}")

            try:
                response_text = ""
                stream_result = chat(current_prompt, stream=True)

                for chunk in stream_result:
                    if chunk:
                        if hasattr(chunk, 'text'):
                            response_text += chunk.text
                            yield {"type": "chunk", "content": chunk.text}
                        elif isinstance(chunk, str):
                            response_text += chunk
                            yield {"type": "chunk", "content": chunk}
                    await asyncio.sleep(0)

                # Extract tool calls from various sources
                stream_value = getattr(stream_result, 'value', None)
                tool_calls = []

                # Try history first
                if chat.h:
                    last_msg = chat.h[-1]
                    content = last_msg.get('content', []) if isinstance(last_msg, dict) else getattr(last_msg, 'content', [])
                    tool_calls = self._extract_tool_calls(content)

                # Fallback to stream_result.value
                if not tool_calls and stream_value and hasattr(stream_value, 'content'):
                    tool_calls = self._extract_tool_calls(stream_value.content)

                # Fallback to chat.res
                if not tool_calls and hasattr(chat, 'res') and chat.res and hasattr(chat.res, 'content'):
                    tool_calls = self._extract_tool_calls(chat.res.content)

                if not tool_calls:
                    break

                # Execute tool calls
                tool_results = []
                for tc in tool_calls:
                    if hasattr(registry, "resolve_tool_display_name"):
                        display_name = registry.resolve_tool_display_name(notebook_id, tc['name'])
                    else:
                        display_name = tc['name']
                    yield {"type": "tool_call", "id": tc['id'], "name": display_name, "input": tc['input']}

                    result = await utils.execute_tool(tc['name'], tc['input'], kernel, notebook_id, registry)

                    yield {"type": "tool_result", "id": tc['id'], "name": display_name, "result": result}

                    tool_results.append({
                        "tool_use_id": tc['id'],
                        "content": utils.format_tool_result_for_llm(result)
                    })

                # Send tool results back to continue conversation
                tool_result_content = [
                    {"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}
                    for tr in tool_results
                ]

                # Check if claudette already added a tool_result placeholder
                last_msg = chat.h[-1] if chat.h else None
                if last_msg:
                    last_role = last_msg.get('role') if isinstance(last_msg, dict) else getattr(last_msg, 'role', None)
                    last_content = last_msg.get('content', []) if isinstance(last_msg, dict) else getattr(last_msg, 'content', [])

                    has_existing = False
                    if last_role == 'user' and isinstance(last_content, list):
                        for item in last_content:
                            item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                            if item_type == 'tool_result':
                                has_existing = True
                                break

                    if has_existing:
                        # Update existing tool_result entries in-place
                        results_by_id = {tr["tool_use_id"]: tr["content"] for tr in tool_result_content}
                        for item in last_content:
                            item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                            if item_type == 'tool_result':
                                tool_use_id = item.get('tool_use_id') if isinstance(item, dict) else getattr(item, 'tool_use_id', None)
                                if tool_use_id in results_by_id:
                                    if isinstance(item, dict):
                                        item['content'] = results_by_id[tool_use_id]
                                    else:
                                        try:
                                            item['content'] = results_by_id[tool_use_id]
                                        except Exception:
                                            item.content = results_by_id[tool_use_id]
                    else:
                        chat.h.append({"role": "user", "content": tool_result_content})

                current_prompt = ""

            except Exception as e:
                logger.exception(f"claudette-tools error: {e}")
                yield {"type": "error", "content": f"Tool loop error: {str(e)}"}
                break

    @staticmethod
    def _extract_tool_calls(content) -> List[Dict]:
        """Extract tool calls from content (list of blocks)."""
        extracted = []
        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'type') and item.type == 'tool_use':
                    extracted.append({
                        'id': item.id,
                        'name': item.name,
                        'input': item.input if hasattr(item, 'input') else {}
                    })
                elif isinstance(item, dict) and item.get('type') == 'tool_use':
                    extracted.append({
                        'id': item.get('id', f'tool_{len(extracted)}'),
                        'name': item.get('name'),
                        'input': item.get('input', {})
                    })
        return extracted



# Register as an LLM provider
def _register_claudette_provider():
    from dialeng.core.registry import registry, ProviderRegistration
    registry.register_provider(ProviderRegistration(
        name="claudette", label="Anthropic API / Bedrock",
        factory=ClaudetteProvider,
        priority=10  # Highest priority - direct API access
    ))

_register_claudette_provider()
