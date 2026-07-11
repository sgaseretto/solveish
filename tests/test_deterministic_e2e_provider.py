"""Contract tests for the browser-suite LLM provider."""

from __future__ import annotations

import asyncio


def test_e2e_inventory_isolated_from_external_credentials(monkeypatch):
    monkeypatch.setenv("DIALENG_E2E", "1")

    from dialeng.services.credential_service import build_initial_credential_inventory

    inventory = build_initial_credential_inventory()

    assert inventory.default_provider() == "e2e_fake"
    assert set(inventory.providers) == {"e2e_fake", "mock_only"}
    assert [model["id"] for model in inventory.providers["e2e_fake"].models] == [
        "e2e-fast",
        "e2e-slow",
        "e2e-error",
    ]


def test_deterministic_provider_stream_and_usage():
    from dialeng.services.llm.providers.deterministic_test_provider import (
        DeterministicTestProvider,
    )

    async def collect():
        provider = DeterministicTestProvider()
        events = [
            event
            async for event in provider.stream(
                "hello world", [], "", "e2e-fast", True, None
            )
        ]
        return provider, events

    provider, events = asyncio.run(collect())

    assert events[0]["type"] == "thinking_start"
    assert "".join(event.get("content", "") for event in events if event["type"] == "chunk") == (
        "E2E response: hello world"
    )
    assert events[-1]["type"] == "usage"
    assert provider.last_result.cost == 0.0


def test_deterministic_provider_honors_cancellation():
    from dialeng.services.llm.providers.deterministic_test_provider import (
        DeterministicTestProvider,
    )

    async def collect():
        cancel_event = asyncio.Event()
        cancel_event.set()
        return [
            event
            async for event in DeterministicTestProvider().stream(
                "cancel me", [], "", "e2e-slow", False, None, cancel_event
            )
        ]

    assert asyncio.run(collect()) == [{"type": "error", "content": "Request cancelled"}]


def test_e2e_fault_routes_are_absent_in_normal_mode(monkeypatch):
    monkeypatch.delenv("DIALENG_E2E", raising=False)

    from starlette.testclient import TestClient
    from dialeng.app import app

    client = TestClient(app)
    response = client.post("/dialeng/default/__e2e__/websocket/disconnect")

    assert response.status_code == 404
