"""
Credential Service - Detects available LLM credentials and selects provider.

This module probes for credentials in the following order:
1. Anthropic API key (ANTHROPIC_API_KEY) - uses claudette
2. AWS Bedrock credentials - uses claudette
3. Claude Code subscription (CLI) - uses claudette-agent
4. No credentials - Mock mode only

Based on cred_probe.py logic but integrated as a service module.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class CredentialStatus:
    """Result of credential detection."""
    available: bool
    provider: str  # "claudette" | "claudette_agent" | "mock_only"
    backend: str   # "anthropic_api" | "bedrock" | "claude_code_subscription" | "none"
    source: str    # Where credentials were found
    details: str   # Human-readable details


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


def _check_claudette_agent_available() -> bool:
    """Check if claudette-agent library is installed."""
    try:
        from claudette_agent import AsyncChat
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

        # Run the async probe
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_probe())
        finally:
            loop.close()

    except ImportError:
        return False, "claude_agent_sdk: not installed"
    except Exception as e:
        err_str = str(e).lower()
        if "login" in err_str or "auth" in err_str:
            return False, f"claude_agent_sdk: auth failed: {e}"
        return False, f"claude_agent_sdk: probe failed: {e}"


def detect_credentials(dotenv_path: Optional[Path] = None) -> CredentialStatus:
    """
    Detect available LLM credentials.

    Returns CredentialStatus with:
    - provider: "claudette" if Anthropic API/Bedrock available
    - provider: "claudette_agent" if Claude Code subscription available
    - provider: "mock_only" if no credentials found
    """
    global _credential_status

    if _credential_status is not None:
        return _credential_status

    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    has_claudette = _check_claudette_available()
    has_claudette_agent = _check_claudette_agent_available()

    # Step 1: Check Anthropic API key
    api_key, key_src = _find_anthropic_api_key(dotenv_path)
    if api_key:
        ok, msg = _check_anthropic_sdk_with_api_key(api_key)
        if ok and has_claudette:
            _credential_status = CredentialStatus(
                available=True, provider="claudette", backend="anthropic_api",
                source=key_src, details=msg
            )
            return _credential_status
        elif ok:
            # Has API key but no claudette - fall through to check if claudette-agent works
            logger.warning("Anthropic API key found but claudette not installed, checking Claude Code...")

    # Step 2: Check AWS Bedrock
    aws = _resolve_aws_credentials()
    if aws:
        ok, msg = _check_anthropic_sdk_with_bedrock(aws)
        if ok and has_claudette:
            _credential_status = CredentialStatus(
                available=True, provider="claudette", backend="bedrock",
                source=aws.source, details=f"{aws.details}; {msg}"
            )
            return _credential_status

    # Step 3: Check Claude Code subscription
    # First try direct SDK probe (more reliable as SDK handles CLI internally)
    if has_claudette_agent:
        ok, msg = _check_claude_agent_sdk_credentials()
        if ok:
            _credential_status = CredentialStatus(
                available=True, provider="claudette_agent", backend="claude_code_subscription",
                source="claude_agent_sdk", details=msg
            )
            return _credential_status
        else:
            logger.info(f"claude_agent_sdk probe: {msg}")

    # Fallback: try CLI-based detection
    cli_path, cli_src = _find_claude_cli_path()
    if cli_path:
        ok, msg = _check_claude_code_credentials(cli_path)
        if ok and has_claudette_agent:
            _credential_status = CredentialStatus(
                available=True, provider="claudette_agent", backend="claude_code_subscription",
                source=cli_src, details=msg
            )
            return _credential_status

    # No credentials found
    _credential_status = CredentialStatus(
        available=False, provider="mock_only", backend="none",
        source="none",
        details="No ANTHROPIC_API_KEY, no AWS Bedrock creds, and no Claude Code CLI found."
    )
    return _credential_status


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


def reset_credential_cache() -> None:
    """Reset cached credential status (useful for testing)."""
    global _credential_status
    _credential_status = None
