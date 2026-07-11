"""Explicitly opt-in network smoke tests for real external services."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os

import pytest


def _enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def _assert_stream(events: list[dict]) -> None:
    errors = [event["content"] for event in events if event.get("type") == "error"]
    assert not errors, errors
    assert any(event.get("type") == "chunk" and event.get("content") for event in events)


def _stream_text(events: list[dict]) -> str:
    return "".join(
        str(event.get("content", ""))
        for event in events
        if event.get("type") == "chunk"
    )


@pytest.mark.external_openai
@pytest.mark.skipif(
    not _enabled("DIALENG_RUN_EXTERNAL_OPENAI"),
    reason="set DIALENG_RUN_EXTERNAL_OPENAI=1 to use stored Codex authentication",
)
def test_real_openai_codex_minimal_stream():
    from dialeng.services.dialeng_config import get_config
    from dialeng.services.llm.providers.openai_codex_provider import OpenAICodexProvider

    async def run():
        provider = OpenAICodexProvider()
        await provider.initialize()
        return [
            event
            async for event in provider.stream(
                "Reply with exactly DIALENG_OK.",
                [],
                "Follow the user's instruction exactly.",
                os.environ.get("DIALENG_EXTERNAL_OPENAI_MODEL", "gpt-5.4"),
                False,
                get_config(),
            )
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        events = pool.submit(asyncio.run, run()).result(timeout=120)
    _assert_stream(events)
    assert "DIALENG_OK" in _stream_text(events)


@pytest.mark.external_colab
@pytest.mark.skipif(
    not _enabled("DIALENG_RUN_EXTERNAL_COLAB"),
    reason="set DIALENG_RUN_EXTERNAL_COLAB=1 to validate stored Colab authentication",
)
def test_real_colab_auth_and_assignment_listing():
    from dialeng.services.colab import ColabAPIClient, ColabAuthService

    async def run():
        auth = ColabAuthService(auth_provider=os.environ.get("DIALENG_COLAB_AUTH_PROVIDER", "oauth_local"))
        assert await auth.validate_session(), auth.session_error
        return await ColabAPIClient(auth).list_tunnel_assignments()

    assignments = asyncio.run(run())
    assert isinstance(assignments, list)
