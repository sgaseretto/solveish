"""
Credential Service - Detects available LLM credentials and selects provider.

This module probes OpenAI Codex and AWS Bedrock independently. Mock mode
remains available when neither production provider is authenticated.

Based on cred_probe.py logic but integrated as a service module.
"""
from __future__ import annotations

import asyncio
import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from dialeng.services.performance_service import performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class CredentialStatus:
    """Result of credential detection."""
    available: bool
    provider: str  # "openai_codex" | "claudette" | "mock_only"
    backend: str   # "openai_codex" | "bedrock" | "none"
    source: str    # Where credentials were found
    details: str   # Human-readable details


@dataclass
class ProviderStatus:
    """Availability and capabilities for one selectable LLM backend."""
    id: str
    provider: str
    backend: str
    label: str
    authenticated: bool
    source: str
    details: str
    models: List[Dict[str, Any]]
    capabilities: List[str]
    validation_state: str = "ready"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "backend": self.backend,
            "label": self.label,
            "authenticated": self.authenticated,
            "source": self.source,
            "details": self.details,
            "models": self.models,
            "capabilities": self.capabilities,
            "validation_state": self.validation_state,
        }


@dataclass
class CredentialInventory:
    """All independently detected LLM providers for the running server."""
    providers: Dict[str, ProviderStatus]

    @property
    def available(self) -> bool:
        return any(p.authenticated and p.id != "mock_only" for p in self.providers.values())

    def authenticated(self) -> List[ProviderStatus]:
        return [p for p in self.providers.values() if p.authenticated]

    def get(self, provider_id: Optional[str]) -> Optional[ProviderStatus]:
        return self.providers.get(provider_id or "")

    def default_provider(self) -> str:
        provider_order = (
            "e2e_fake",
            "openai_codex",
            "claudette:bedrock",
        )
        for provider_id in provider_order:
            status = self.providers.get(provider_id)
            if status and status.authenticated:
                return provider_id
        for provider_id in provider_order:
            status = self.providers.get(provider_id)
            if status and status.validation_state == "validating":
                return provider_id
        return "mock_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "default_provider": self.default_provider(),
            "providers": [p.to_dict() for p in self.providers.values()],
        }


@dataclass
class AWSCreds:
    """AWS credentials for Bedrock."""
    access_key: str
    secret_key: str
    session_token: Optional[str]
    region: str
    source: str
    details: str


# Module-level cached credential status
_credential_status: Optional[CredentialStatus] = None
_credential_inventory: Optional[CredentialInventory] = None


def _e2e_enabled() -> bool:
    return os.environ.get("DIALENG_E2E") == "1"


def _mock_provider_status() -> ProviderStatus:
    return ProviderStatus(
        id="mock_only",
        provider="mock_only",
        backend="none",
        label="Mock",
        authenticated=True,
        source="built_in",
        details="No external credentials required",
        models=[],
        capabilities=[],
        validation_state="ready",
    )


def _e2e_inventory() -> CredentialInventory:
    """Return a network-free inventory that cannot inherit developer credentials."""
    models = [
        {"id": model, "name": model, "capabilities": ["streaming", "thinking"]}
        for model in ("e2e-fast", "e2e-slow", "e2e-error")
    ]
    return CredentialInventory({
        "e2e_fake": ProviderStatus(
            id="e2e_fake",
            provider="e2e_fake",
            backend="e2e_fake",
            label="Deterministic E2E",
            authenticated=True,
            source="DIALENG_E2E",
            details="Deterministic network-free test provider",
            models=models,
            capabilities=["streaming", "thinking", "usage"],
            validation_state="ready",
        ),
        "mock_only": _mock_provider_status(),
    })


def credential_status_from_inventory(inventory: CredentialInventory) -> CredentialStatus:
    """Adapt the provider inventory to the legacy single-provider status."""
    selected = inventory.get(inventory.default_provider())
    if selected and selected.id != "mock_only" and selected.authenticated:
        return CredentialStatus(
            available=True,
            provider=selected.provider,
            backend=selected.backend,
            source=selected.source,
            details=selected.details,
        )
    return CredentialStatus(
        available=False,
        provider="mock_only",
        backend="none",
        source="none",
        details="No authenticated OpenAI Codex or AWS Bedrock provider found.",
    )


def _load_dotenv_if_present(dotenv_path: Path) -> Tuple[bool, str]:
    """Load .env file if present."""
    if not dotenv_path.exists():
        return False, f"no .env at {dotenv_path}"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path, override=False)
        return True, f"loaded .env via python-dotenv: {dotenv_path}"
    except ImportError:
        # Minimal parser fallback
        try:
            count = 0
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    count += 1
            return True, f"loaded .env via minimal parser: {dotenv_path} (set {count} vars)"
        except Exception as e:
            return False, f"failed to load .env: {e}"


def _resolve_aws_credentials() -> Optional[AWSCreds]:
    """Resolve AWS credentials for Bedrock."""
    # Claudette-style env vars
    ak = os.environ.get("AWS_ACCESS_KEY")
    sk = os.environ.get("AWS_SECRET_KEY")
    if ak and sk:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        return AWSCreds(
            access_key=ak, secret_key=sk,
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region=region, source="aws:env (claudette-style)",
            details="AWS_ACCESS_KEY/AWS_SECRET_KEY detected"
        )

    # Standard AWS env vars
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        return AWSCreds(
            access_key=ak, secret_key=sk,
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region=region, source="aws:env (standard)",
            details="AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY detected"
        )

    # Botocore chain
    try:
        import botocore.session
        sess = botocore.session.get_session()
        creds = sess.get_credentials()
        if creds is None:
            return None

        frozen = creds.get_frozen_credentials()
        method = getattr(creds, "method", "unknown")
        region = (
            sess.get_config_variable("region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        if not frozen.access_key or not frozen.secret_key:
            return None

        ak_suffix = frozen.access_key[-4:] if frozen.access_key else "????"
        return AWSCreds(
            access_key=frozen.access_key, secret_key=frozen.secret_key,
            session_token=frozen.token, region=region,
            source=f"aws:botocore ({method})",
            details=f"botocore_method={method}, access_key_suffix=...{ak_suffix}, region={region}"
        )
    except Exception:
        return None


def _check_anthropic_sdk_with_bedrock(creds: AWSCreds) -> Tuple[bool, str]:
    """Check if anthropic SDK can be instantiated with Bedrock credentials."""
    try:
        from anthropic import AnthropicBedrock
        kwargs = dict(
            aws_access_key=creds.access_key,
            aws_secret_key=creds.secret_key,
            aws_region=creds.region,
        )
        if creds.session_token:
            kwargs["aws_session_token"] = creds.session_token

        _client = AnthropicBedrock(**kwargs)
        return True, f"anthropic-sdk: AnthropicBedrock(...) instantiated ({creds.source})"
    except Exception as e:
        return False, f"anthropic-sdk: failed to instantiate AnthropicBedrock: {e}"


def _check_claudette_available() -> bool:
    """Check if claudette library is installed."""
    try:
        import claudette
        return True
    except ImportError:
        return False

def _provider_models(provider_id: str) -> List[Dict[str, Any]]:
    from dialeng.services.dialeng_config import load_config

    return [
        {
            "id": model.id,
            "name": model.name,
            "capabilities": list(model.capabilities),
        }
        for model in load_config(create_if_missing=False).get_models_for_provider(provider_id)
    ]


def build_initial_credential_inventory(
    dotenv_path: Optional[Path] = None,
) -> CredentialInventory:
    """Build a local-only inventory without network or subprocess probes."""
    if _e2e_enabled():
        return _e2e_inventory()
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    has_claudette = _check_claudette_available()
    aws_pending = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or (Path.home() / ".aws" / "credentials").exists()
    ) and has_claudette

    from dialeng.services.dialeng_config import get_config
    from dialeng.services.openai_codex_auth import get_openai_codex_auth_service

    codex_status = get_openai_codex_auth_service().status()
    codex_enabled = get_config().openai_codex_enabled
    codex_authenticated = bool(codex_enabled and codex_status["authenticated"])

    providers = {
        "openai_codex": ProviderStatus(
            id="openai_codex",
            provider="openai_codex",
            backend="openai_codex",
            label="OpenAI Codex",
            authenticated=codex_authenticated,
            source=str(codex_status.get("source") or "none"),
            details=("OpenAI Codex subscription authenticated"
                     if codex_authenticated else str(codex_status.get("error") or "Not connected")),
            models=_provider_models("openai_codex"),
            capabilities=["tools", "images", "thinking", "skills", "usage"],
            validation_state="ready" if codex_authenticated else "unavailable",
        ),
        "claudette:bedrock": ProviderStatus(
            id="claudette:bedrock",
            provider="claudette",
            backend="bedrock",
            label="AWS Bedrock",
            authenticated=False,
            source="local_configuration" if aws_pending else "none",
            details=("Validating configured AWS credentials"
                     if aws_pending else "AWS credentials not found"),
            models=_provider_models("claudette:bedrock"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="validating" if aws_pending else "unavailable",
        ),
        "mock_only": _mock_provider_status(),
    }
    performance_monitor.increment("provider_inventory.initial_builds")
    return CredentialInventory(providers)


def _probe_bedrock_provider(_dotenv_path: Path) -> ProviderStatus:
    started_at = time.perf_counter()
    try:
        has_claudette = _check_claudette_available()
        aws = _resolve_aws_credentials()
        authenticated, details = (False, "AWS credentials not found")
        if aws:
            authenticated, details = _check_anthropic_sdk_with_bedrock(aws)
        authenticated = bool(authenticated and has_claudette)
        status = ProviderStatus(
            id="claudette:bedrock",
            provider="claudette",
            backend="bedrock",
            label="AWS Bedrock",
            authenticated=authenticated,
            source=aws.source if aws else "none",
            details=details if has_claudette else "claudette is not installed",
            models=_provider_models("claudette:bedrock"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="ready" if authenticated else "unavailable",
        )
        return status
    finally:
        performance_monitor.observe_duration(
            "provider.probe",
            (time.perf_counter() - started_at) * 1000,
            labels={"provider": "claudette:bedrock"},
            authenticated=bool(locals().get("status") and status.authenticated),
        )


def _probe_openai_codex_provider(_dotenv_path: Path) -> ProviderStatus:
    from dialeng.services.dialeng_config import get_config
    from dialeng.services.openai_codex_auth import get_openai_codex_auth_service

    started_at = time.perf_counter()
    try:
        config = get_config()
        codex_status = get_openai_codex_auth_service().status()
        authenticated = bool(config.openai_codex_enabled and codex_status["authenticated"])
        status = ProviderStatus(
            id="openai_codex",
            provider="openai_codex",
            backend="openai_codex",
            label="OpenAI Codex",
            authenticated=authenticated,
            source=str(codex_status.get("source") or "none"),
            details=("OpenAI Codex subscription authenticated" if authenticated
                     else str(codex_status.get("error") or "Not connected")),
            models=_provider_models("openai_codex"),
            capabilities=["tools", "images", "thinking", "skills", "usage"],
            validation_state="ready" if authenticated else "unavailable",
        )
        return status
    finally:
        performance_monitor.observe_duration(
            "provider.probe",
            (time.perf_counter() - started_at) * 1000,
            labels={"provider": "openai_codex"},
            authenticated=bool(locals().get("status") and status.authenticated),
        )


def detect_credential_inventory(
    dotenv_path: Optional[Path] = None,
    *,
    force: bool = False,
) -> CredentialInventory:
    """Detect every usable provider without allowing one failure to hide another."""
    global _credential_inventory

    if _e2e_enabled():
        _credential_inventory = _e2e_inventory()
        return _credential_inventory

    if _credential_inventory is not None and not force:
        performance_monitor.increment("provider_inventory.cache_hits")
        return _credential_inventory
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    providers = {
        status.id: status
        for status in (
            _probe_openai_codex_provider(dotenv_path),
            _probe_bedrock_provider(dotenv_path),
        )
    }
    providers["mock_only"] = build_initial_credential_inventory(dotenv_path).providers["mock_only"]

    _credential_inventory = CredentialInventory(providers)
    return _credential_inventory


async def refresh_credential_inventory_async(
    dotenv_path: Optional[Path] = None,
) -> CredentialInventory:
    """Refresh provider probes concurrently without blocking the event loop."""
    global _credential_inventory

    if _e2e_enabled():
        _credential_inventory = _e2e_inventory()
        return _credential_inventory

    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"
    _load_dotenv_if_present(dotenv_path)

    probe_functions = (
        _probe_openai_codex_provider,
        _probe_bedrock_provider,
    )
    results = await asyncio.gather(
        *(asyncio.to_thread(probe, dotenv_path) for probe in probe_functions),
        return_exceptions=True,
    )

    initial = build_initial_credential_inventory(dotenv_path)
    providers: Dict[str, ProviderStatus] = {}
    for provider_id, result in zip(
        (
            "openai_codex",
            "claudette:bedrock",
        ),
        results,
    ):
        if isinstance(result, BaseException):
            logger.exception(
                "Provider probe failed (%s)",
                provider_id,
                exc_info=(type(result), result, result.__traceback__),
            )
            fallback = initial.providers[provider_id]
            fallback.validation_state = "unavailable"
            fallback.details = f"Provider validation failed: {type(result).__name__}"
            providers[provider_id] = fallback
        else:
            providers[provider_id] = result
    providers["mock_only"] = initial.providers["mock_only"]
    _credential_inventory = CredentialInventory(providers)
    return _credential_inventory


def detect_credentials(dotenv_path: Optional[Path] = None) -> CredentialStatus:
    """
    Detect available LLM credentials.

    Returns the preferred available Codex or Bedrock credential status, or
    mock-only status when neither production provider is authenticated.
    """
    global _credential_status

    if _credential_status is not None:
        return _credential_status

    # Preserve the legacy single-provider API on top of the new inventory.
    inventory = detect_credential_inventory(dotenv_path)
    _credential_status = credential_status_from_inventory(inventory)
    return _credential_status


def reset_credential_cache() -> None:
    """Invalidate both the compatibility result and the provider inventory."""
    global _credential_status, _credential_inventory
    _credential_status = None
    _credential_inventory = None


def refresh_credential_inventory(dotenv_path: Optional[Path] = None) -> CredentialInventory:
    """Re-run all provider checks after an authentication/configuration change."""
    reset_credential_cache()
    return detect_credential_inventory(dotenv_path, force=True)


def get_available_modes(cred_status: CredentialStatus) -> List[Tuple[str, str]]:
    """
    Get available dialog modes based on credential status.

    Returns list of (value, label) tuples for UI.
    """
    modes = [("mock", "Mock")]

    if cred_status.available:
        modes.extend([
            ("learning", "Learning"),
            ("concise", "Concise"),
            ("standard", "Standard"),
        ])

    return modes


def print_credential_status(cred_status: CredentialStatus) -> None:
    """Print credential status in a formatted way for startup logging."""
    if cred_status.available:
        print(f"   ✅ LLM Credentials available")
        print(f"      Provider: {cred_status.provider}")
        print(f"      Backend:  {cred_status.backend}")
        print(f"      Source:   {cred_status.source}")
        print(f"      Details:  {cred_status.details}")
    else:
        print(f"   ⚠️  No LLM credentials found - Mock mode only")
        print(f"      Details: {cred_status.details}")


def print_credential_inventory(inventory: CredentialInventory) -> None:
    """Print all selectable providers rather than only the legacy default."""
    available = [p for p in inventory.authenticated() if p.id != "mock_only"]
    if not available:
        print("   No authenticated LLM providers found; Mock mode is available.")
        return
    print(f"   LLM providers available ({len(available)}):")
    for provider in available:
        marker = " (default)" if provider.id == inventory.default_provider() else ""
        print(f"      {provider.label}: authenticated via {provider.source}{marker}")
