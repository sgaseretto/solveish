# OpenAI Codex OAuth

Dialeng can use an OpenAI Codex subscription for prompt cells. This is separate
from the public OpenAI Platform API and does not use `OPENAI_API_KEY`.

## Connect

1. Start Dialeng and open Settings.
2. Expand **OpenAI Codex** and select **Connect OpenAI**.
3. Complete authorization in the opened browser tab.
4. Return to Dialeng. The provider selector updates without a server restart.

Dialeng normally receives the redirect on `http://localhost:1455/auth/callback`.
If that port is unavailable, paste the final redirect URL or `code#state` into
the prompt shown by Dialeng.

Credentials are stored in `~/.dialeng/auth.json` with private file permissions.
Select **Disconnect** to remove stored OpenAI credentials.

## Explicit Environment Token

For a managed local environment, set:

```bash
export DIALENG_OPENAI_CODEX_ACCESS_TOKEN="..."
uv run dialeng .
```

The token must be a Codex-compatible JWT containing a ChatGPT account ID.
Environment credentials take precedence, are not written to disk, and cannot be
refreshed by Dialeng.

## Notebook Selection

Choose **OpenAI Codex** in the provider selector, then choose a Codex model in
the adjacent model selector. Provider and model are saved in notebook metadata
and synchronized across tabs.

If authentication expires, Dialeng preserves the notebook selection and prior
prompt output. Reconnect or choose another authenticated provider before rerun.
