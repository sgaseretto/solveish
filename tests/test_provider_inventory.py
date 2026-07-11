import asyncio
import time

from dialeng.core.registry import ProviderRegistration, registry
from dialeng.services.credential_service import (
    CredentialInventory,
    ProviderStatus,
    build_initial_credential_inventory,
    detect_credential_inventory,
    refresh_credential_inventory_async,
    reset_credential_cache,
)
from dialeng.services.llm.llm_service import LLMService


def _status(provider_id, provider, authenticated=True):
    return ProviderStatus(
        id=provider_id,
        provider=provider,
        backend=provider_id,
        label=provider_id,
        authenticated=authenticated,
        source="test",
        details="test",
        models=[],
        capabilities=[],
    )


def test_inventory_keeps_codex_when_bedrock_probe_fails(monkeypatch, tmp_path):
    import dialeng.services.credential_service as credentials

    monkeypatch.setattr(
        credentials,
        "_probe_openai_codex_provider",
        lambda _path: _status("openai_codex", "openai_codex"),
    )
    monkeypatch.setattr(
        credentials,
        "_probe_bedrock_provider",
        lambda _path: _status("claudette:bedrock", "claudette", False),
    )
    monkeypatch.setattr(credentials, "_provider_models", lambda _provider: [])

    reset_credential_cache()
    inventory = detect_credential_inventory(tmp_path / ".env", force=True)
    assert inventory.get("openai_codex").authenticated is True
    assert inventory.get("claudette:bedrock").authenticated is False
    assert inventory.default_provider() == "openai_codex"
    assert set(inventory.providers) == {
        "openai_codex", "claudette:bedrock", "mock_only",
    }
    reset_credential_cache()


def test_provider_initialization_is_locked_per_provider(monkeypatch):
    initialized = 0

    class Provider:
        async def initialize(self):
            nonlocal initialized
            initialized += 1
            await asyncio.sleep(0.01)

    provider = Provider()
    inventory = CredentialInventory({"test": _status("test", "test_provider")})
    service = LLMService()
    monkeypatch.setattr(service, "_inventory", lambda: inventory)
    monkeypatch.setitem(registry.providers, "test_provider", ProviderRegistration(
        name="test_provider", label="Test", factory=lambda: provider,
    ))

    async def initialize_twice():
        return await asyncio.gather(service._get_provider("test"), service._get_provider("test"))

    results = asyncio.run(initialize_twice())
    assert initialized == 1
    assert results[0][1] is results[1][1] is provider


def test_initial_inventory_uses_local_hints_without_remote_probes(monkeypatch, tmp_path):
    import dialeng.services.credential_service as credentials
    import dialeng.services.openai_codex_auth as codex_auth

    monkeypatch.setattr(credentials, "_check_claudette_available", lambda: True)
    monkeypatch.setenv("AWS_PROFILE", "dialeng-test")
    monkeypatch.setattr(credentials, "_provider_models", lambda _provider: [])
    monkeypatch.setattr(codex_auth.get_openai_codex_auth_service(), "status", lambda: {
        "authenticated": True, "source": "stored", "error": None,
    })

    inventory = build_initial_credential_inventory(tmp_path / ".env")

    assert inventory.get("openai_codex").authenticated is True
    assert inventory.get("claudette:bedrock").validation_state == "validating"
    assert inventory.default_provider() == "openai_codex"


def test_async_inventory_probes_concurrently_and_isolates_failures(monkeypatch, tmp_path):
    import dialeng.services.credential_service as credentials

    monkeypatch.setattr(credentials, "_load_dotenv_if_present", lambda _path: (False, "none"))
    monkeypatch.setattr(credentials, "build_initial_credential_inventory", lambda _path: CredentialInventory({
        "openai_codex": _status("openai_codex", "openai_codex", False),
        "claudette:bedrock": _status("claudette:bedrock", "claudette", False),
        "mock_only": _status("mock_only", "mock_only", True),
    }))

    def delayed(provider_id):
        def probe(_path):
            time.sleep(0.05)
            if provider_id == "claudette:bedrock":
                raise RuntimeError("isolated failure")
            return _status(provider_id, provider_id, True)
        return probe

    monkeypatch.setattr(credentials, "_probe_bedrock_provider", delayed("claudette:bedrock"))
    monkeypatch.setattr(credentials, "_probe_openai_codex_provider", delayed("openai_codex"))

    started_at = time.perf_counter()
    inventory = asyncio.run(refresh_credential_inventory_async(tmp_path / ".env"))
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.12
    assert inventory.get("claudette:bedrock").authenticated is False
    assert inventory.get("claudette:bedrock").validation_state == "unavailable"
    assert inventory.get("openai_codex").authenticated is True
    reset_credential_cache()
