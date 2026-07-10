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


def test_inventory_keeps_valid_providers_when_another_probe_fails(monkeypatch, tmp_path):
    import dialeng.services.credential_service as credentials
    import dialeng.services.openai_codex_auth as codex_auth

    monkeypatch.setattr(credentials, "_check_claudette_available", lambda: True)
    monkeypatch.setattr(credentials, "_find_anthropic_api_key", lambda _path: ("key", "test"))
    monkeypatch.setattr(credentials, "_check_anthropic_sdk_with_api_key", lambda _key: (False, "bad key"))
    monkeypatch.setattr(credentials, "_resolve_aws_credentials", lambda: None)
    monkeypatch.setattr(credentials, "_check_claude_agent_sdk_credentials", lambda: (True, "ok"))
    monkeypatch.setattr(credentials, "_provider_models", lambda _provider: [])
    monkeypatch.setattr(codex_auth.get_openai_codex_auth_service(), "status", lambda: {
        "authenticated": True, "source": "stored", "error": None,
    })

    reset_credential_cache()
    inventory = detect_credential_inventory(tmp_path / ".env", force=True)
    assert inventory.get("claudette:anthropic_api").authenticated is False
    assert inventory.get("claude_agent_sdk").authenticated is True
    assert inventory.get("openai_codex").authenticated is True
    assert inventory.default_provider() == "claude_agent_sdk"
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
    monkeypatch.setattr(credentials, "_find_anthropic_api_key", lambda _path: ("key", "env"))
    monkeypatch.setattr(credentials, "_resolve_aws_credentials", lambda: (_ for _ in ()).throw(
        AssertionError("AWS resolution must not run during local inventory")
    ))
    monkeypatch.setattr(credentials, "_check_claude_agent_sdk_credentials", lambda: (_ for _ in ()).throw(
        AssertionError("Claude SDK probe must not run during local inventory")
    ))
    monkeypatch.setattr(credentials, "_check_anthropic_sdk_with_api_key", lambda _key: (_ for _ in ()).throw(
        AssertionError("Anthropic probe must not run during local inventory")
    ))
    monkeypatch.setattr(credentials, "_find_claude_cli_path", lambda: ("/tmp/claude", "path"))
    monkeypatch.setattr(credentials, "_provider_models", lambda _provider: [])
    monkeypatch.setattr(codex_auth.get_openai_codex_auth_service(), "status", lambda: {
        "authenticated": False, "source": None, "error": None,
    })

    inventory = build_initial_credential_inventory(tmp_path / ".env")

    assert inventory.get("claudette:anthropic_api").validation_state == "validating"
    assert inventory.get("claude_agent_sdk").validation_state == "validating"
    assert inventory.default_provider() == "claudette:anthropic_api"
    assert inventory.available is False


def test_async_inventory_probes_concurrently_and_isolates_failures(monkeypatch, tmp_path):
    import dialeng.services.credential_service as credentials

    monkeypatch.setattr(credentials, "_load_dotenv_if_present", lambda _path: (False, "none"))
    monkeypatch.setattr(credentials, "build_initial_credential_inventory", lambda _path: CredentialInventory({
        "claudette:anthropic_api": _status("claudette:anthropic_api", "claudette", False),
        "claudette:bedrock": _status("claudette:bedrock", "claudette", False),
        "claude_agent_sdk": _status("claude_agent_sdk", "claude_agent_sdk", False),
        "openai_codex": _status("openai_codex", "openai_codex", False),
        "mock_only": _status("mock_only", "mock_only", True),
    }))

    def delayed(provider_id):
        def probe(_path):
            time.sleep(0.05)
            if provider_id == "claudette:bedrock":
                raise RuntimeError("isolated failure")
            return _status(provider_id, provider_id, True)
        return probe

    monkeypatch.setattr(credentials, "_probe_anthropic_provider", delayed("claudette:anthropic_api"))
    monkeypatch.setattr(credentials, "_probe_bedrock_provider", delayed("claudette:bedrock"))
    monkeypatch.setattr(credentials, "_probe_claude_agent_provider", delayed("claude_agent_sdk"))
    monkeypatch.setattr(credentials, "_probe_openai_codex_provider", delayed("openai_codex"))

    started_at = time.perf_counter()
    inventory = asyncio.run(refresh_credential_inventory_async(tmp_path / ".env"))
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.15
    assert inventory.get("claudette:anthropic_api").authenticated is True
    assert inventory.get("claudette:bedrock").authenticated is False
    assert inventory.get("claudette:bedrock").validation_state == "unavailable"
    assert inventory.get("openai_codex").authenticated is True
    reset_credential_cache()
