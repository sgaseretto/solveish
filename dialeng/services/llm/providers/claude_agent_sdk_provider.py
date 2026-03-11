"""Claude Agent SDK provider - Direct SDK usage for maximum isolation.

Uses claude-agent-sdk's query() function directly. Each query creates a fresh
subprocess with a unique temporary directory for complete session isolation.
Supports real-time token streaming via StreamEvent when thinking is disabled.
"""
from typing import AsyncIterator, Dict, List, Any
import logging
import os
import tempfile
import shutil
import uuid
import json
from datetime import datetime

from ..base_provider import BaseLLMProvider, ProviderInfo, LLMResult
from .. import utils

logger = logging.getLogger(__name__)


class ClaudeAgentSdkProvider(BaseLLMProvider):
    """LLM provider using claude-agent-sdk directly (most isolated)."""

    async def initialize(self) -> None:
        # Validate imports are available (actual imports are lazy per-call)
        try:
            from claude_agent_sdk import query as _q, ClaudeAgentOptions as _o
            logger.info("ClaudeAgentSdkProvider initialized")
        except ImportError as e:
            logger.error(f"Failed to import claude_agent_sdk: {e}")
            raise ImportError(
                "claude-agent-sdk is not installed."
            ) from e

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="claude_agent_sdk",
            display_name="Claude Agent SDK (direct)",
            supports_native_tools=False,
            supports_mcp_tools=True,
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
        """Stream response using claude-agent-sdk directly.

        Each query creates a fresh subprocess with a unique temp directory
        for complete session isolation. Supports real-time token streaming
        via StreamEvent (incompatible with extended thinking per SDK docs).
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, StreamEvent

        logger.info(f"SDK-direct: Using model {model}")

        full_prompt = utils.build_prompt_with_context(prompt, context_messages)

        temp_cwd = tempfile.mkdtemp(prefix=f"dialeng_sdk_{uuid.uuid4().hex[:8]}_")
        logger.info(f"SDK-direct: Created temp cwd: {temp_cwd}")

        # Debug logging
        debug_mode = getattr(config, 'debug_mode', False)
        debug_log_dir = getattr(config, 'debug_log_dir', './debug_logs')
        if debug_mode:
            utils.save_debug_log(debug_log_dir, "prompt", {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "temp_cwd": temp_cwd,
                "prompt": full_prompt,
                "system_prompt": system_prompt,
            })

        logger.info(f"SDK-direct: ===== FULL PROMPT START =====")
        for line in full_prompt.split('\n')[:30]:
            logger.info(f"SDK-direct: {line}")
        if full_prompt.count('\n') > 30:
            logger.info(f"SDK-direct: ... ({full_prompt.count(chr(10)) - 30} more lines)")
        logger.info(f"SDK-direct: ===== FULL PROMPT END =====")

        # Determine thinking mode before building options
        # Streaming (include_partial_messages) is incompatible with extended thinking
        thinking_enabled = use_thinking and self.check_thinking_support(model)
        if use_thinking and not thinking_enabled:
            logger.warning(f"Model {model} does not support extended thinking, disabling")

        use_streaming = not thinking_enabled

        options = ClaudeAgentOptions(
            continue_conversation=False,
            resume=None,
            setting_sources=[],
            cwd=temp_cwd,
            model=model,
            system_prompt=system_prompt,
            include_partial_messages=use_streaming,
        )

        logger.info(f"SDK-direct: Options - continue_conversation={options.continue_conversation}, "
                    f"resume={options.resume}, setting_sources={options.setting_sources}, cwd={options.cwd}, "
                    f"include_partial_messages={use_streaming}")

        if thinking_enabled:
            yield {"type": "thinking_start"}
            logger.info("SDK-direct: Extended thinking enabled (streaming disabled)")

        thinking_phase_ended = False
        collected_response = []

        try:
            async for message in sdk_query(prompt=full_prompt, options=options):
                if isinstance(message, ResultMessage):
                    if hasattr(message, 'usage') and message.usage:
                        self._last_result.usage = message.usage
                        logger.info(f"SDK-direct: Usage = {message.usage}")
                    if hasattr(message, 'total_cost_usd'):
                        self._last_result.cost = message.total_cost_usd
                        logger.info(f"SDK-direct: Cost = ${message.total_cost_usd:.6f}")
                    continue

                # StreamEvent: real-time token streaming
                if use_streaming and isinstance(message, StreamEvent):
                    event = message.event
                    event_type = event.get("type")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield {"type": "chunk", "content": text}
                                collected_response.append(text)
                    continue

                # AssistantMessage: complete message (fallback, and for thinking mode)
                if isinstance(message, AssistantMessage):
                    if hasattr(message, 'content') and message.content:
                        for block in message.content:
                            if hasattr(block, 'type') and block.type == 'thinking':
                                thinking_content = getattr(block, 'thinking', str(block))
                                yield {"type": "thinking", "content": thinking_content}
                            elif hasattr(block, 'text') and not use_streaming:
                                if thinking_enabled and not thinking_phase_ended:
                                    yield {"type": "thinking_end"}
                                    thinking_phase_ended = True
                                yield {"type": "chunk", "content": block.text}
                                collected_response.append(block.text)

            if thinking_enabled and not thinking_phase_ended:
                yield {"type": "thinking_end"}

            if debug_mode:
                utils.save_debug_log(debug_log_dir, "response", {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "response": "".join(collected_response),
                    "usage": str(self._last_result.usage),
                    "cost": self._last_result.cost,
                })

        except Exception as e:
            logger.exception(f"SDK-direct streaming error: {e}")
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

        finally:
            if temp_cwd and os.path.exists(temp_cwd):
                try:
                    shutil.rmtree(temp_cwd, ignore_errors=True)
                    logger.info(f"SDK-direct: Cleaned up temp cwd: {temp_cwd}")
                except Exception as cleanup_err:
                    logger.warning(f"SDK-direct: Failed to clean up {temp_cwd}: {cleanup_err}")

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
        """Stream with MCP-based tool calling for claude-agent-sdk.

        Creates an in-process MCP server that wraps kernel functions, allowing
        Claude to call them natively via the SDK's tool support.
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions, tool as sdk_tool, create_sdk_mcp_server
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, ToolUseBlock, StreamEvent

        logger.debug(f"sdk-mcp-tools: model={model}, {len(tools)} tools")

        temp_cwd = tempfile.mkdtemp(prefix=f"dialeng_tools_{uuid.uuid4().hex[:8]}_")
        logger.info(f"sdk-mcp-tools: Created temp cwd: {temp_cwd}")

        # Store tool execution results for yielding to the UI
        tool_execution_events = []

        # Create MCP tools wrapping kernel functions
        sdk_tools = []
        allowed_tool_names = []

        for tool_def in tools:
            tool_name = tool_def.get('name', 'unknown')
            tool_desc = tool_def.get('description', f'Tool: {tool_name}')
            tool_schema = tool_def.get('input_schema', {})

            params = {}
            if 'properties' in tool_schema:
                type_mapping = {
                    'string': str, 'integer': int, 'number': float,
                    'boolean': bool, 'array': list, 'object': dict,
                }
                for param_name, param_info in tool_schema['properties'].items():
                    param_type = param_info.get('type', 'string')
                    params[param_name] = type_mapping.get(param_type, str)

            logger.debug(f"sdk-mcp-tools: Creating MCP tool: {tool_name}")

            def make_tool_handler(captured_tool_name, captured_kernel, captured_notebook_id, captured_registry, captured_events, captured_tool_schema):
                async def tool_handler(args: dict) -> dict:
                    logger.debug(f"MCP tool called: {captured_tool_name}, args: {args}")

                    # Convert JSON-serialized values back to proper types
                    converted_args = {}
                    for key, value in args.items():
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
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

                    tool_id = f"mcp_tool_{captured_tool_name}_{len(captured_events)}"
                    captured_events.append({
                        "type": "tool_call", "id": tool_id,
                        "name": captured_tool_name, "input": converted_args
                    })

                    try:
                        result = await utils.execute_tool(
                            captured_tool_name, converted_args, captured_kernel, captured_notebook_id, captured_registry
                        )
                        result_text = utils.format_tool_result_for_llm(result)

                        captured_events.append({
                            "type": "tool_result", "id": tool_id,
                            "name": captured_tool_name, "result": result
                        })

                        return {"content": [{"type": "text", "text": result_text}]}
                    except Exception as e:
                        error_msg = f"Error executing {captured_tool_name}: {str(e)}"
                        captured_events.append({
                            "type": "tool_result", "id": tool_id,
                            "name": captured_tool_name,
                            "result": {"status": "error", "error": error_msg}
                        })
                        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

                return tool_handler

            handler = make_tool_handler(tool_name, kernel, notebook_id, registry, tool_execution_events, tool_schema)
            decorated_tool = sdk_tool(tool_name, tool_desc, params)(handler)
            sdk_tools.append(decorated_tool)
            allowed_tool_names.append(f"mcp__notebook_tools__{tool_name}")

        # Create MCP server
        mcp_server = None
        if sdk_tools:
            mcp_server = create_sdk_mcp_server(name="notebook_tools", version="1.0.0", tools=sdk_tools)

        full_prompt = utils.build_prompt_with_context(prompt, context_messages)

        try:
            async def message_generator():
                yield {"type": "user", "message": {"role": "user", "content": full_prompt}}

            options = ClaudeAgentOptions(
                continue_conversation=False,
                resume=None,
                setting_sources=[],
                cwd=temp_cwd,
                model=model,
                system_prompt=system_prompt,
                mcp_servers={"notebook_tools": mcp_server} if mcp_server else {},
                allowed_tools=allowed_tool_names if allowed_tool_names else [],
                max_turns=max_steps,
                include_partial_messages=True,
            )

            yielded_event_count = 0
            in_tool_block = False

            async for message in sdk_query(prompt=message_generator(), options=options):
                if isinstance(message, ResultMessage):
                    if hasattr(message, 'usage') and message.usage:
                        self._last_result.usage = message.usage
                    continue

                # StreamEvent: real-time token streaming
                if isinstance(message, StreamEvent):
                    event = message.event
                    event_type = event.get("type")

                    if event_type == "content_block_start":
                        content_block = event.get("content_block", {})
                        if content_block.get("type") == "tool_use":
                            in_tool_block = True
                        else:
                            in_tool_block = False
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and not in_tool_block:
                            text = delta.get("text", "")
                            if text:
                                yield {"type": "chunk", "content": text}
                    elif event_type == "content_block_stop":
                        if in_tool_block:
                            in_tool_block = False
                    continue

                if isinstance(message, AssistantMessage):
                    if hasattr(message, 'content') and message.content:
                        # Yield pending tool execution events
                        while yielded_event_count < len(tool_execution_events):
                            event = tool_execution_events[yielded_event_count]
                            yield event
                            yielded_event_count += 1

                        # Text already yielded via StreamEvent; detect tool_use blocks
                        for block in message.content:
                            if hasattr(block, 'type') and block.type == 'tool_use':
                                logger.debug(f"sdk-mcp-tools: Tool use block: {block.name}")

            # Yield remaining tool events
            while yielded_event_count < len(tool_execution_events):
                yield tool_execution_events[yielded_event_count]
                yielded_event_count += 1

        except Exception as e:
            logger.exception(f"sdk-mcp-tools error: {e}")
            yield {"type": "error", "content": f"Tool loop error: {str(e)}"}
        finally:
            try:
                shutil.rmtree(temp_cwd, ignore_errors=True)
            except Exception:
                pass



# Register as an LLM provider
def _register_claude_agent_sdk_provider():
    from dialeng.core.registry import registry, ProviderRegistration
    registry.register_provider(ProviderRegistration(
        name="claude_agent_sdk", label="Claude Code (SDK)",
        factory=ClaudeAgentSdkProvider,
        priority=5
    ))

_register_claude_agent_sdk_provider()
