"""Shared utilities for LLM providers.

These are stateless functions extracted from the monolithic LLMService class.
Any provider can import and use them without needing a reference to the service.
"""
from typing import List, Dict, Any, Tuple
import logging
import os
import json
import re
from copy import deepcopy
from datetime import datetime

logger = logging.getLogger(__name__)


def _extract_text_from_content(content) -> str:
    """Extract text from message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif block.get('type') == 'image':
                    parts.append('[Image]')
        return ' '.join(parts)
    return str(content)


def build_prompt_with_context(prompt: str, context_messages: List[Dict]) -> str:
    """Build a single prompt string that includes conversation context.

    Instead of appending context messages to chat.h (which creates multiple
    "User:" messages that can confuse the Claude Agent SDK), we build a
    single prompt that clearly presents the context and the current question.

    This ensures the SDK sees ONE clear user message to respond to.
    Image content blocks are replaced with [Image] placeholders since
    the SDK text-only path cannot handle them.
    """
    if not context_messages:
        return prompt

    context_parts = []
    for msg in context_messages:
        role = msg.get('role', 'user')
        content = _extract_text_from_content(msg.get('content', ''))
        if role == 'user':
            context_parts.append(f"User: {content}")
        elif role == 'assistant':
            context_parts.append(f"Assistant: {content}")

    context_text = "\n\n".join(context_parts)

    full_prompt = f"""Here is the conversation history:

{context_text}

---

Now respond to my latest message:

{prompt}"""

    return full_prompt


def _extract_sdk_text_and_images(content, starting_image_index: int = 1) -> Tuple[str, List[Dict[str, Any]], int]:
    """Extract text plus numbered image placeholders for SDK prompt building."""
    if isinstance(content, str):
        return content, [], starting_image_index

    if not isinstance(content, list):
        return str(content), [], starting_image_index

    parts: List[str] = []
    image_blocks: List[Dict[str, Any]] = []
    image_index = starting_image_index

    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue

        block_type = block.get('type')
        if block_type == 'text':
            text = block.get('text', '')
            if text:
                parts.append(text)
        elif block_type == 'image':
            parts.append(f"[Notebook image {image_index}]")
            image_blocks.append(deepcopy(block))
            image_index += 1

    return ' '.join(p for p in parts if p).strip(), image_blocks, image_index


def build_sdk_query_payload(
    prompt: str,
    context_messages: List[Dict],
    system_prompt: str,
) -> Tuple[str, str | List[Dict[str, Any]]]:
    """Build a stateless Claude Agent SDK payload preserving notebook images.

    The SDK `query()` path is kept stateless by encoding the authoritative
    notebook transcript into the system prompt while passing notebook images as
    real multimodal blocks in the current user turn. This mirrors the working
    pattern from `test_claude_agent_query.ipynb`.
    """
    transcript_parts: List[str] = []
    image_blocks: List[Dict[str, Any]] = []
    next_image_index = 1

    for msg in context_messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        content_text, content_images, next_image_index = _extract_sdk_text_and_images(
            content, next_image_index
        )
        image_blocks.extend(content_images)

        if not content_text:
            continue

        if role == 'assistant':
            transcript_parts.append(f"Assistant: {content_text}")
        else:
            transcript_parts.append(f"User: {content_text}")

    effective_system_prompt = system_prompt
    if transcript_parts:
        transcript = "\n\n".join(transcript_parts)
        effective_system_prompt = (
            f"{system_prompt}\n\n"
            "Authoritative current notebook context:\n"
            f"{transcript}\n\n"
            "Treat the notebook context above as the sole conversation history. "
            "Use the notebook's current state only, and ignore any prior session memory."
        )

    if not image_blocks:
        return effective_system_prompt, prompt

    image_refs = ', '.join(f"[Notebook image {i}]" for i in range(1, len(image_blocks) + 1))
    prompt_block = {
        "type": "text",
        "text": (
            "The attached notebook images correspond to these placeholders from the "
            f"authoritative notebook context: {image_refs}.\n\n{prompt}"
        ),
    }
    return effective_system_prompt, image_blocks + [prompt_block]


async def execute_tool(tool_name: str, tool_input: dict, kernel, notebook_id: str, registry) -> dict:
    """Execute a tool (builtin or kernel function)."""
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

    if registry.is_builtin(tool_name):
        return await registry.execute_builtin(tool_name, tool_input)

    if kernel:
        return await kernel.execute_tool(tool_name, tool_input)

    return {
        "status": "error",
        "error": f"Tool '{tool_name}' not found and no kernel available"
    }


def format_tool_result_for_llm(result: dict) -> str:
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


def save_debug_log(debug_log_dir: str, log_type: str, data: dict):
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


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks for streaming."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def build_text_tool_definitions(tools: List[Dict]) -> str:
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


def parse_text_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from text using ```tool_call markers."""
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


def build_tool_results_prompt(tool_results: List[dict]) -> str:
    """Build a prompt string containing tool results."""
    parts = ["Here are the results from the tool calls:"]
    for tr in tool_results:
        content = tr.get('content', '')
        parts.append(f"\n{content}")
    parts.append("\nPlease continue based on these results.")
    return "\n".join(parts)
