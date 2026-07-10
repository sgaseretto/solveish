"""OpenAI Codex subscription provider using Dialeng-owned transports."""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .. import utils
from ..base_provider import BaseLLMProvider, LLMResult, ProviderInfo
from dialeng.services.openai_codex_auth import get_openai_codex_auth_service

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_WEBSOCKET_URL = "wss://chatgpt.com/backend-api/codex/responses"
WEBSOCKET_BETA = "responses_websockets=2026-02-06"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_meaningful_event(event: Dict[str, Any]) -> bool:
    return event.get("type") not in {
        "response.created", "response.queued", "response.in_progress"
    }


@dataclass
class ResponseState:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    output_items: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    thinking_active: bool = False
    emitted: bool = False


def _text_content(content: Any, *, assistant: bool = False) -> Any:
    if isinstance(content, str):
        if assistant:
            return [{"type": "output_text", "text": content, "annotations": []}]
        return [{"type": "input_text", "text": content}]
    blocks = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            blocks.append({
                "type": "output_text" if assistant else "input_text",
                "text": block.get("text", ""),
                **({"annotations": []} if assistant else {}),
            })
        elif block_type == "image" and not assistant:
            source = block.get("source", {})
            if source.get("type") == "base64" and source.get("data"):
                media_type = source.get("media_type", "image/png")
                blocks.append({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{media_type};base64,{source['data']}",
                })
            elif block.get("image_url"):
                blocks.append({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": block["image_url"],
                })
    return blocks


def build_codex_input(context_messages: List[Dict], prompt: str) -> List[Dict[str, Any]]:
    """Convert Dialeng's authoritative notebook transcript to Responses input."""
    items: List[Dict[str, Any]] = []
    for index, message in enumerate(context_messages):
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant":
            blocks = _text_content(content, assistant=True)
            if blocks:
                items.append({
                    "type": "message",
                    "id": f"msg_history_{index}",
                    "role": "assistant",
                    "status": "completed",
                    "content": blocks,
                })
        else:
            blocks = _text_content(content)
            if blocks:
                items.append({"role": "user", "content": blocks})
    items.append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})
    return items


def build_codex_tools(tools: List[Dict]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            "strict": False,
        }
        for tool in tools
    ]


class OpenAICodexProvider(BaseLLMProvider):
    """Stateless Codex Responses transport with a Dialeng-controlled tool loop."""

    def __init__(self) -> None:
        super().__init__()
        self.auth = get_openai_codex_auth_service()

    async def initialize(self) -> None:
        credentials = await self.auth.get_valid_credentials()
        if not credentials:
            raise RuntimeError("OpenAI Codex is not authenticated")
        logger.info("OpenAI Codex provider initialized (source=%s)", credentials.source)

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="openai_codex",
            display_name="OpenAI Codex subscription",
            supports_native_tools=True,
            supports_mcp_tools=False,
            capabilities=["tools", "images", "thinking", "skills", "usage"],
        )

    def check_thinking_support(self, model: str) -> bool:
        return model.startswith("gpt-5")

    async def refresh_authentication(self) -> bool:
        return bool(await self.auth.get_valid_credentials(force_refresh=True))

    async def stream(
        self,
        prompt: str,
        context_messages: List[Dict],
        system_prompt: str,
        model: str,
        use_thinking: bool,
        config: Any,
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        body = self._request_body(
            model=model,
            system_prompt=system_prompt,
            input_items=build_codex_input(context_messages, prompt),
            tools=[],
            use_thinking=use_thinking,
        )
        state = ResponseState()
        async for event in self._request_events(body, config, cancel_event, allow_retry=True):
            async for normalized in self._normalize_event(event, state):
                yield normalized
        self._set_usage(state.usage)

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
        cancel_event: Any = None,
    ) -> AsyncIterator[Dict]:
        input_items = build_codex_input(context_messages, prompt)
        codex_tools = build_codex_tools(tools)

        for _step in range(max_steps):
            if cancel_event and cancel_event.is_set():
                return
            state = ResponseState()
            body = self._request_body(
                model=model,
                system_prompt=system_prompt,
                input_items=input_items,
                tools=codex_tools,
                use_thinking=use_thinking,
            )
            async for event in self._request_events(
                body, config, cancel_event, allow_retry=(_step == 0)
            ):
                async for normalized in self._normalize_event(event, state):
                    yield normalized
            self._set_usage(state.usage)
            if not state.tool_calls:
                return

            input_items.extend(state.output_items)
            for tool_call in state.tool_calls:
                if cancel_event and cancel_event.is_set():
                    return
                display_name = registry.resolve_tool_display_name(
                    notebook_id, tool_call["name"]
                )
                yield {
                    "type": "tool_call",
                    "id": tool_call["call_id"],
                    "name": display_name,
                    "input": tool_call["arguments"],
                }
                result = await utils.execute_tool(
                    tool_call["name"],
                    tool_call["arguments"],
                    kernel,
                    notebook_id,
                    registry,
                )
                yield {
                    "type": "tool_result",
                    "id": tool_call["call_id"],
                    "name": display_name,
                    "result": result,
                }
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": utils.format_tool_result_for_llm(result),
                })

        yield {"type": "error", "content": f"Tool loop reached the {max_steps}-step limit"}

    @staticmethod
    def _request_body(
        *,
        model: str,
        system_prompt: str,
        input_items: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        use_thinking: bool,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        from dialeng.services.dialeng_config import get_config
        if get_config().skills_enabled:
            from dialeng.services.skill_service import get_skill_service
            skill_metadata = get_skill_service().execution_metadata()
            if skill_metadata:
                body["metadata"] = {
                    "dialeng_skills": ",".join(
                        f"{item['name']}:{item['sha256'][:12]}" for item in skill_metadata
                    )[:500]
                }
                logger.info(
                    "Codex prompt skills (skills=%s)",
                    ", ".join(item["name"] for item in skill_metadata),
                )
        if tools:
            body["tools"] = tools
        if use_thinking:
            body["reasoning"] = {"effort": "medium", "summary": "auto"}
        return body

    async def _request_events(
        self,
        body: Dict[str, Any],
        config: Any,
        cancel_event: Any,
        *,
        allow_retry: bool,
    ) -> AsyncIterator[Dict[str, Any]]:
        transport = getattr(config, "openai_codex_transport", "auto")
        if transport in {"auto", "websocket"}:
            emitted = False
            try:
                async for event in self._websocket_events(body, cancel_event):
                    emitted = emitted or _is_meaningful_event(event)
                    yield event
                return
            except Exception as exc:
                if transport == "websocket" or emitted or not allow_retry:
                    raise
                logger.warning("Codex WebSocket did not start; falling back to SSE: %s", exc)
        async for event in self._sse_events(body, cancel_event, allow_retry=allow_retry):
            yield event

    async def _headers(self, websocket: bool = False) -> Dict[str, str]:
        credentials = await self.auth.get_valid_credentials()
        if not credentials:
            raise RuntimeError("OpenAI Codex authentication expired; reconnect in Settings")
        request_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {credentials.access}",
            "chatgpt-account-id": credentials.account_id,
            "originator": "dialeng",
            "User-Agent": f"dialeng ({platform.system()} {platform.release()}; {platform.machine()})",
            "session_id": request_id,
        }
        if websocket:
            headers["OpenAI-Beta"] = WEBSOCKET_BETA
            headers["x-client-request-id"] = request_id
        else:
            headers.update({
                "OpenAI-Beta": "responses=experimental",
                "accept": "text/event-stream",
                "content-type": "application/json",
            })
        return headers

    async def _websocket_events(
        self, body: Dict[str, Any], cancel_event: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        from websockets.asyncio.client import connect

        headers = await self._headers(websocket=True)
        async with connect(
            CODEX_WEBSOCKET_URL,
            additional_headers=headers,
            user_agent_header=None,
            open_timeout=20,
        ) as socket:
            await socket.send(json.dumps({"type": "response.create", **body}))
            while True:
                if cancel_event and cancel_event.is_set():
                    await socket.close(code=1000, reason="cancelled")
                    raise asyncio.CancelledError
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                event = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                self._raise_event_error(event)
                yield event
                if event.get("type") in {
                    "response.completed", "response.done", "response.incomplete"
                }:
                    return

    async def _sse_events(
        self, body: Dict[str, Any], cancel_event: Any, *, allow_retry: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        refreshed = False
        for attempt in range(3):
            emitted = False
            try:
                headers = await self._headers()
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=20.0)) as client:
                    async with client.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as response:
                        if response.status_code == 401 and not refreshed and allow_retry:
                            refreshed = True
                            if await self.auth.get_valid_credentials(force_refresh=True):
                                continue
                        if response.status_code >= 400:
                            error_body = (await response.aread()).decode("utf-8", errors="replace")
                            if response.status_code in RETRYABLE_STATUS and attempt < 2 and allow_retry:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            raise RuntimeError(self._format_http_error(response.status_code, error_body))

                        cancel_watcher = None
                        if cancel_event:
                            async def close_on_cancel() -> None:
                                await cancel_event.wait()
                                await response.aclose()
                            cancel_watcher = asyncio.create_task(close_on_cancel())
                        try:
                            data_lines: List[str] = []
                            async for line in response.aiter_lines():
                                if cancel_event and cancel_event.is_set():
                                    raise asyncio.CancelledError
                                if not line:
                                    if data_lines:
                                        payload = "\n".join(data_lines)
                                        data_lines.clear()
                                        if payload != "[DONE]":
                                            event = json.loads(payload)
                                            self._raise_event_error(event)
                                            emitted = emitted or _is_meaningful_event(event)
                                            yield event
                                    continue
                                if line.startswith("data:"):
                                    data_lines.append(line[5:].strip())
                            if data_lines:
                                payload = "\n".join(data_lines)
                                if payload != "[DONE]":
                                    event = json.loads(payload)
                                    self._raise_event_error(event)
                                    yield event
                            return
                        finally:
                            if cancel_watcher:
                                cancel_watcher.cancel()
            except (httpx.TransportError, httpx.TimeoutException):
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError
                if emitted or not allow_retry or attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    @staticmethod
    def _raise_event_error(event: Dict[str, Any]) -> None:
        if event.get("type") == "error":
            raise RuntimeError(event.get("message") or event.get("code") or "Codex error")
        if event.get("type") == "response.failed":
            error = (event.get("response") or {}).get("error") or {}
            raise RuntimeError(error.get("message") or "Codex response failed")

    @staticmethod
    def _format_http_error(status: int, body: str) -> str:
        try:
            error = (json.loads(body).get("error") or {})
            message = error.get("message") or error.get("code")
            if message:
                return f"OpenAI Codex request failed ({status}): {message}"
        except (json.JSONDecodeError, AttributeError):
            pass
        return f"OpenAI Codex request failed ({status}): {body[:500]}"

    async def _normalize_event(
        self, event: Dict[str, Any], state: ResponseState
    ) -> AsyncIterator[Dict[str, Any]]:
        event_type = event.get("type")
        if event_type == "response.reasoning_summary_text.delta":
            delta = event.get("delta", "")
            if delta:
                if not state.thinking_active:
                    state.thinking_active = True
                    yield {"type": "thinking_start"}
                state.emitted = True
                yield {"type": "thinking", "content": delta}
        elif event_type in {"response.reasoning_summary_part.done", "response.reasoning_summary_text.done"}:
            if state.thinking_active:
                state.thinking_active = False
                yield {"type": "thinking_end"}
        elif event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                state.emitted = True
                yield {"type": "chunk", "content": delta}
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                raw_arguments = item.get("arguments") or "{}"
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    try:
                        arguments = json.loads(raw_arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                call = {
                    "call_id": item.get("call_id") or item.get("id") or str(uuid.uuid4()),
                    "name": item.get("name", ""),
                    "arguments": arguments,
                }
                state.tool_calls.append(call)
                output_item = {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": json.dumps(arguments),
                }
                if item.get("id"):
                    output_item["id"] = item["id"]
                state.output_items.append(output_item)
            elif item.get("type") in {"message", "reasoning"}:
                # Codex tool continuation needs the completed output items from
                # this response, including encrypted reasoning when present.
                state.output_items.append(item)
        elif event_type in {"response.completed", "response.done", "response.incomplete"}:
            if state.thinking_active:
                state.thinking_active = False
                yield {"type": "thinking_end"}
            response = event.get("response") or {}
            state.usage = response.get("usage")
            if state.usage:
                yield {"type": "usage", "usage": state.usage}

    def _set_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        details = usage.get("input_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        self._last_result = LLMResult(usage={
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cached_tokens": cached,
            "total_tokens": int(usage.get("total_tokens") or 0),
        }, cost=None)


def _register_openai_codex_provider() -> None:
    from dialeng.core.registry import ProviderRegistration, registry

    registry.register_provider(ProviderRegistration(
        name="openai_codex",
        label="OpenAI Codex subscription",
        factory=OpenAICodexProvider,
        priority=4,
    ))


_register_openai_codex_provider()
