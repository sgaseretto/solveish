import base64
import json
import os
import time

import pytest

from dialeng.services.openai_codex_auth import (
    OpenAICodexAuthService,
    OpenAICodexCredentials,
    PendingOAuthFlow,
    inspect_access_token,
    parse_authorization_input,
)


def _jwt(account_id="acct_test", expires_at=None):
    payload = {
        "exp": expires_at or time.time() + 3600,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def test_codex_jwt_requires_chatgpt_account():
    account_id, expires_at = inspect_access_token(_jwt())
    assert account_id == "acct_test"
    assert expires_at > time.time()
    with pytest.raises(ValueError, match="subscription"):
        inspect_access_token("e30.e30.signature")


def test_auth_file_is_private_and_environment_wins(tmp_path, monkeypatch):
    service = OpenAICodexAuthService(tmp_path / "auth.json")
    service.save_credentials(OpenAICodexCredentials(
        access=_jwt("stored"), refresh="refresh", expires_at=time.time() + 3600,
        account_id="stored",
    ))
    assert os.stat(service.auth_path).st_mode & 0o777 == 0o600

    monkeypatch.setenv("DIALENG_OPENAI_CODEX_ACCESS_TOKEN", _jwt("environment"))
    credentials = service.load_credentials()
    assert credentials.account_id == "environment"
    assert credentials.source == "env:DIALENG_OPENAI_CODEX_ACCESS_TOKEN"


def test_manual_completion_requires_matching_state(tmp_path, monkeypatch):
    service = OpenAICodexAuthService(tmp_path / "auth.json")
    service.pending = PendingOAuthFlow("expected", "verifier", "https://example.test", time.time())

    async def exchange(code, verifier):
        assert code == "code"
        assert verifier == "verifier"
        return OpenAICodexCredentials(_jwt(), "refresh", time.time() + 3600, "acct_test")

    monkeypatch.setattr(service, "_exchange_code", exchange)
    with pytest.raises(ValueError, match="state"):
        import asyncio
        asyncio.run(service.complete_authorization("code#wrong"))

    import asyncio
    result = asyncio.run(service.complete_authorization("code#expected"))
    assert result.account_id == "acct_test"
    assert service.status()["authenticated"] is True


def test_authorization_input_formats():
    assert parse_authorization_input("code#state") == ("code", "state")
    assert parse_authorization_input("http://localhost:1455/auth/callback?code=x&state=y") == ("x", "y")


def test_refresh_response_can_reuse_or_rotate_refresh_token():
    access = _jwt()
    reused = OpenAICodexAuthService._credentials_from_token_response(
        {"access_token": access, "expires_in": 3600}, fallback_refresh="old"
    )
    rotated = OpenAICodexAuthService._credentials_from_token_response(
        {"access_token": access, "refresh_token": "new", "expires_in": 3600},
        fallback_refresh="old",
    )
    assert reused.refresh == "old"
    assert rotated.refresh == "new"
