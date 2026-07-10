"""
Credential Service - Detects available LLM credentials and selects provider.

This module probes for credentials in the following order:
1. Anthropic API key (ANTHROPIC_API_KEY) - uses claudette
2. AWS Bedrock credentials - uses claudette
3. Claude Code subscription (CLI) - uses claude-agent-sdk
4. No credentials - Mock mode only

Based on cred_probe.py logic but integrated as a service module.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
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
    provider: str  # "claudette" | "claude_agent_sdk" | "mock_only"
    backend: str   # "anthropic_api" | "bedrock" | "claude_code_subscription" | "none"
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
            "claudette:anthropic_api",
            "claudette:bedrock",
            "claude_agent_sdk",
            "openai_codex",
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
        details="No authenticated Anthropic, Bedrock, Claude Code, or OpenAI Codex provider found.",
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


def _find_anthropic_api_key(dotenv_path: Path) -> Tuple[Optional[str], str]:
    """Check for ANTHROPIC_API_KEY in env or .env."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key, "env:ANTHROPIC_API_KEY"

    _load_dotenv_if_present(dotenv_path)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key, f".env:{dotenv_path}"

    return None, "none"


def _check_anthropic_sdk_with_api_key(api_key: str) -> Tuple[bool, str]:
    """Check if anthropic SDK can be instantiated with API key."""
    try:
        from anthropic import Anthropic
        _client = Anthropic(api_key=api_key)
        return True, "anthropic-sdk: Anthropic(api_key=...) instantiated"
    except Exception as e:
        return False, f"anthropic-sdk: failed to instantiate: {e}"


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


def _search_for_claude_executable(root: Path, max_depth: int = 6) -> Optional[str]:
    """Search for claude executable in directory tree."""
    if not root.exists():
        return None

    names = {"claude", "claude.exe"}
    queue: List[Tuple[Path, int]] = [(root, 0)]

    while queue:
        p, d = queue.pop(0)
        if d > max_depth:
            continue
        try:
            for child in p.iterdir():
                if child.is_file() and child.name in names:
                    if os.name == "nt" or os.access(str(child), os.X_OK):
                        return str(child)
                if child.is_dir() and child.name not in {".git", "__pycache__", "node_modules"}:
                    queue.append((child, d + 1))
        except Exception:
            continue
    return None


def _find_claude_cli_path() -> Tuple[Optional[str], str]:
    """Find claude CLI path."""
    # Check PATH
    p = shutil.which("claude")
    if p:
        return p, "path:claude"

    # Check claude-agent-sdk bundle
    try:
        import claude_agent_sdk
        pkg_dir = Path(claude_agent_sdk.__file__).resolve().parent
        candidate = _search_for_claude_executable(pkg_dir)
        if candidate:
            return candidate, "bundled:claude-agent-sdk (package dir)"

        candidate = _search_for_claude_executable(pkg_dir.parent)
        if candidate:
            return candidate, "bundled:claude-agent-sdk (site-packages)"
    except Exception:
        pass

    return None, "none"


def _check_claude_code_credentials(cli_path: str) -> Tuple[bool, str]:
    """Check Claude Code subscription credentials via CLI."""
    # Sanity check
    try:
        r_ver = subprocess.run([cli_path, "-v"], capture_output=True, text=True, timeout=3)
        if r_ver.returncode != 0:
            return False, f"claude CLI present but -v failed: {r_ver.stderr.strip() or r_ver.stdout.strip()}"
    except Exception as e:
        return False, f"claude CLI not runnable: {e}"

    # Zero-turn auth probe
    try:
        r = subprocess.run(
            [cli_path, "-p", "--max-turns", "0", "ping"],
            capture_output=True, text=True, timeout=5,
            env=os.environ.copy(),
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        out_l = out.lower()

        if "please run /login" in out_l:
            return False, "claude CLI says not logged in (needs /login)"
        if "invalid api key" in out_l or "authentication_error" in out_l:
            return False, "claude CLI reports authentication error"
        if r.returncode == 0:
            return True, "claude CLI accepted 0-turn run; Claude Code stored credentials available"
        return False, f"claude CLI returned non-zero (code={r.returncode})"
    except subprocess.TimeoutExpired:
        return False, "claude CLI probe timed out"
    except Exception as e:
        return False, f"claude CLI probe failed: {e}"


def _check_claudette_available() -> bool:
    """Check if claudette library is installed."""
    try:
        import claudette
        return True
    except ImportError:
        return False



def _check_claude_agent_sdk_credentials() -> Tuple[bool, str]:
    """
    Check Claude Code subscription credentials via claude_agent_sdk directly.

    This is more reliable than searching for the CLI binary because:
    1. The SDK handles finding/bundling the CLI internally
    2. The SDK may have credentials cached from previous runs
    3. This matches how the SDK is actually used at runtime
    """
    try:
        import asyncio
        from claude_agent_sdk import query, ClaudeAgentOptions

        async def _probe():
            """Run a minimal probe to check if SDK auth works."""
            options = ClaudeAgentOptions(
                model="sonnet",
                max_turns=0,  # Don't actually run a turn
            )
            # Just initialize - if auth fails, this will raise
            # We use a simple prompt that will be rejected due to max_turns=0
            # but the auth check happens before that
            try:
                async for _ in query(prompt="test", options=options):
                    pass
            except Exception as e:
                err_str = str(e).lower()
                # These errors indicate auth is working but something else failed
                if "max_turns" in err_str or "turn" in err_str:
                    return True, "claude_agent_sdk: auth verified (max_turns reached)"
                # Auth-specific failures
                if "login" in err_str or "auth" in err_str or "credential" in err_str:
                    return False, f"claude_agent_sdk: auth failed: {e}"
                # Unknown error - might still work, be optimistic
                return True, f"claude_agent_sdk: probe completed with: {e}"
            return True, "claude_agent_sdk: probe completed successfully"

        # Run the async probe — check for running loop first to avoid
        # creating a coroutine that can never be awaited
        try:
            asyncio.get_running_loop()
            # Already inside an event loop (e.g. during uvicorn startup) —
            # run in a separate thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _probe()).result(timeout=10)
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(_probe())

    except ImportError:
        return False, "claude_agent_sdk: not installed"
    except Exception as e:
        err_str = str(e).lower()
        if "login" in err_str or "auth" in err_str:
            return False, f"claude_agent_sdk: auth failed: {e}"
        return False, f"claude_agent_sdk: probe failed: {e}"


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
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    has_claudette = _check_claudette_available()
    api_key, key_source = _find_anthropic_api_key(dotenv_path)
    anthropic_pending = bool(api_key and has_claudette)

    aws_pending = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or (Path.home() / ".aws" / "credentials").exists()
    ) and has_claudette

    claude_cli_path, claude_cli_source = _find_claude_cli_path()
    claude_pending = bool(claude_cli_path)

    from dialeng.services.dialeng_config import get_config
    from dialeng.services.openai_codex_auth import get_openai_codex_auth_service

    codex_status = get_openai_codex_auth_service().status()
    codex_enabled = get_config().openai_codex_enabled
    codex_authenticated = bool(codex_enabled and codex_status["authenticated"])

    providers = {
        "claudette:anthropic_api": ProviderStatus(
            id="claudette:anthropic_api",
            provider="claudette",
            backend="anthropic_api",
            label="Anthropic API",
            authenticated=False,
            source=key_source,
            details=("Validating configured Anthropic credentials"
                     if anthropic_pending else "ANTHROPIC_API_KEY not found"),
            models=_provider_models("claudette:anthropic_api"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="validating" if anthropic_pending else "unavailable",
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
        "claude_agent_sdk": ProviderStatus(
            id="claude_agent_sdk",
            provider="claude_agent_sdk",
            backend="claude_code_subscription",
            label="Claude Code",
            authenticated=False,
            source=claude_cli_source if claude_pending else "none",
            details=("Validating Claude Code credentials"
                     if claude_pending else "Claude Code CLI not found"),
            models=_provider_models("claude_agent_sdk"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="validating" if claude_pending else "unavailable",
        ),
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
        "mock_only": ProviderStatus(
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
        ),
    }
    performance_monitor.increment("provider_inventory.initial_builds")
    return CredentialInventory(providers)


def _probe_anthropic_provider(dotenv_path: Path) -> ProviderStatus:
    def probe() -> ProviderStatus:
        has_claudette = _check_claudette_available()
        api_key, key_source = _find_anthropic_api_key(dotenv_path)
        authenticated, details = (False, "ANTHROPIC_API_KEY not found")
        if api_key:
            authenticated, details = _check_anthropic_sdk_with_api_key(api_key)
        authenticated = bool(authenticated and has_claudette)
        return ProviderStatus(
            id="claudette:anthropic_api",
            provider="claudette",
            backend="anthropic_api",
            label="Anthropic API",
            authenticated=authenticated,
            source=key_source,
            details=details if has_claudette else "claudette is not installed",
            models=_provider_models("claudette:anthropic_api"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="ready" if authenticated else "unavailable",
        )

    started_at = time.perf_counter()
    try:
        status = probe()
        return status
    finally:
        performance_monitor.observe_duration(
            "provider.probe",
            (time.perf_counter() - started_at) * 1000,
            labels={"provider": "claudette:anthropic_api"},
            authenticated=bool(locals().get("status") and status.authenticated),
        )


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


def _probe_claude_agent_provider(_dotenv_path: Path) -> ProviderStatus:
    started_at = time.perf_counter()
    try:
        authenticated, details = _check_claude_agent_sdk_credentials()
        source = "claude_agent_sdk"
        if not authenticated:
            cli_path, cli_source = _find_claude_cli_path()
            if cli_path:
                authenticated, details = _check_claude_code_credentials(cli_path)
                source = cli_source
        status = ProviderStatus(
            id="claude_agent_sdk",
            provider="claude_agent_sdk",
            backend="claude_code_subscription",
            label="Claude Code",
            authenticated=authenticated,
            source=source if authenticated else "none",
            details=details,
            models=_provider_models("claude_agent_sdk"),
            capabilities=["tools", "images", "thinking", "usage"],
            validation_state="ready" if authenticated else "unavailable",
        )
        return status
    finally:
        performance_monitor.observe_duration(
            "provider.probe",
            (time.perf_counter() - started_at) * 1000,
            labels={"provider": "claude_agent_sdk"},
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

    if _credential_inventory is not None and not force:
        performance_monitor.increment("provider_inventory.cache_hits")
        return _credential_inventory
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    providers = {
        status.id: status
        for status in (
            _probe_anthropic_provider(dotenv_path),
            _probe_bedrock_provider(dotenv_path),
            _probe_claude_agent_provider(dotenv_path),
            _probe_openai_codex_provider(dotenv_path),
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

    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"
    _load_dotenv_if_present(dotenv_path)

    probe_functions = (
        _probe_anthropic_provider,
        _probe_bedrock_provider,
        _probe_claude_agent_provider,
        _probe_openai_codex_provider,
    )
    results = await asyncio.gather(
        *(asyncio.to_thread(probe, dotenv_path) for probe in probe_functions),
        return_exceptions=True,
    )

    initial = build_initial_credential_inventory(dotenv_path)
    providers: Dict[str, ProviderStatus] = {}
    for provider_id, result in zip(
        (
            "claudette:anthropic_api",
            "claudette:bedrock",
            "claude_agent_sdk",
            "openai_codex",
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

    Returns CredentialStatus with:
    - provider: "claudette" if Anthropic API/Bedrock available
    - provider: "claude_agent_sdk" if Claude Code subscription available
    - provider: "mock_only" if no credentials found
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
