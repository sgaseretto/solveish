# Google Colab Kernel Integration

This document explains how Dialeng connects to Google Colab runtimes to execute code remotely, using the same APIs as the official Colab VS Code extension.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Structure](#module-structure)
3. [Authentication Flow](#authentication-flow)
4. [Credential Resolution & Validation](#credential-resolution--validation)
5. [Connection Lifecycle](#connection-lifecycle)
6. [Jupyter Wire Protocol over WebSocket](#jupyter-wire-protocol-over-websocket)
7. [Code Execution & Streaming Output](#code-execution--streaming-output)
8. [Rich Output Handling](#rich-output-handling)
9. [Multiplexed WebSocket Subtlety](#multiplexed-websocket-subtlety)
10. [Background Tasks](#background-tasks)
11. [Integration with Multi-Kernel System](#integration-with-multi-kernel-system)
12. [How to Extend](#how-to-extend)

## Architecture Overview

The Colab kernel allows Dialeng to execute code on Google's cloud infrastructure — the same machines you get when opening a notebook on colab.research.google.com. This gives users access to free GPUs/TPUs without installing anything locally.

The integration is built by reverse-engineering the same API endpoints used by the [colab-vscode](https://github.com/googlecolab/colab-vscode) extension (Apache 2.0, Google LLC).

```mermaid
graph TB
    subgraph "Dialeng (Local)"
        Auth[ColabAuthService]
        API[ColabAPIClient]
        SM[ColabSessionManager]
        CK[ColabKernel]
        KS[KernelService]
        EQ[ExecutionQueue]
    end

    subgraph "Google Cloud"
        GA[Google OAuth2]
        CA[Colab API<br/>colab.research.google.com]
        GP[Colab GAPI<br/>colab.pa.googleapis.com]
        RT[Colab Runtime<br/>xxx-colab.googleusercontent.com]
    end

    Auth -->|"OAuth2 tokens"| GA
    API -->|"REST: assign, keep-alive"| CA
    API -->|"REST: user info, tokens"| GP
    API -->|"REST: Jupyter sessions"| RT
    CK -->|"WebSocket: Jupyter wire protocol"| RT

    KS -->|"manages"| SM
    SM -->|"creates"| CK
    SM -->|"uses"| API
    API -->|"auth"| Auth
    EQ -->|"uses"| KS
```

## Module Structure

All Colab-related code lives in `services/colab/`:

| File | Class | Purpose |
|------|-------|---------|
| `colab_auth.py` | `ColabAuthService` | Google OAuth2 flow, token storage/refresh |
| `colab_api.py` | `ColabAPIClient` | REST API calls to Colab backends |
| `colab_kernel.py` | `ColabKernel` | WebSocket-based Jupyter wire protocol execution |
| `colab_session.py` | `ColabSessionManager` | Manages kernel instances per notebook |
| `__init__.py` | — | Re-exports public classes |

### Relationship to Base Kernel

`ColabKernel` extends `BaseKernel` (defined in `services/kernel/base_kernel.py`), the same abstract interface that `SubprocessKernel` implements. This means `KernelService` and `ExecutionQueue` work identically regardless of whether code runs locally or on Colab.

```mermaid
classDiagram
    class BaseKernel {
        <<abstract>>
        +execute_streaming(code) AsyncIterator~CellOutput~
        +interrupt() bool
        +restart() bool
        +shutdown()
        +is_alive bool
        +get_status() KernelStatus
        +get_info() KernelInfo
        +get_namespace_info() dict
    }

    class SubprocessKernel {
        Local Python subprocess
        SIGINT interrupt
    }

    class ColabKernel {
        Remote Colab runtime
        WebSocket Jupyter protocol
        Keep-alive + token refresh
    }

    BaseKernel <|-- SubprocessKernel
    BaseKernel <|-- ColabKernel
```

## Authentication Flow

Dialeng uses a Google OAuth2 desktop/native client to authenticate with the Colaboratory API. Built-in credentials from the [Colab VS Code extension](https://github.com/googlecolab/colab-vscode) are used by default — no configuration required. You can optionally override them via environment variables:

- `COLAB_CLIENT_ID` — OAuth2 client ID (optional override)
- `COLAB_CLIENT_SECRET` — OAuth2 client secret (optional override)

```mermaid
sequenceDiagram
    participant User
    participant Dialeng
    participant Google as Google OAuth2

    User->>Dialeng: Click "Connect Colab"
    Dialeng->>Google: Redirect to auth URL<br/>(scope: colaboratory)
    Google->>User: Login & consent screen
    User->>Google: Approve
    Google->>Dialeng: Redirect to localhost callback<br/>(authorization code)
    Dialeng->>Google: Exchange code for tokens
    Google->>Dialeng: access_token + refresh_token
    Dialeng->>Dialeng: Save to ~/.dialeng/colab_tokens.json

    Note over Dialeng: Token auto-refreshes before expiry
```

**Key details:**
- Built-in OAuth credentials work out of the box (validated and auto-updated at startup)
- `COLAB_CLIENT_ID` and `COLAB_CLIENT_SECRET` env vars are optional overrides
- Tokens persist in `~/.dialeng/colab_tokens.json` (file permissions `0600`)
- Access tokens auto-refresh 5 minutes before expiry
- Scopes: `profile`, `email`, `https://www.googleapis.com/auth/colaboratory`

## Credential Resolution & Validation

At startup, Dialeng validates that the OAuth client credentials are still accepted by Google. If the built-in defaults have been rotated, it automatically extracts updated credentials from the published Colab VS Code extension.

### Resolution Cascade

```mermaid
flowchart TD
    Start[Startup: Colab enabled] --> EnvCheck{COLAB_CLIENT_ID +<br/>COLAB_CLIENT_SECRET<br/>in .env?}

    EnvCheck -->|Yes| UseEnv[Use env vars — skip validation]
    EnvCheck -->|No| ValidateDefaults[Validate built-in defaults<br/>POST to Google token endpoint]

    ValidateDefaults -->|"invalid_grant = valid client"| UseDefaults[Use defaults]
    ValidateDefaults -->|"invalid_client = rotated"| CheckCache[Check ~/.dialeng/colab_oauth_client.json]

    CheckCache -->|Cache hit + valid| UseCache[Use cached credentials]
    CheckCache -->|Miss or invalid| VSIX[Download VSIX from Open VSX<br/>Extract from extension/out/extension.js]

    VSIX -->|Success + valid| CacheAndUse[Cache + use extracted credentials]
    VSIX -->|Failure| Fallback[Use defaults + warn user]

    ValidateDefaults -->|Network error| UseDefaults
```

### How Validation Works

A POST to `https://oauth2.googleapis.com/token` with a dummy refresh token distinguishes between valid and invalid clients:

| Google Response | Meaning | Action |
|----------------|---------|--------|
| `"error": "invalid_grant"` | Client exists, token is bad | Credentials are valid |
| `"error": "invalid_client"` | Client ID/secret unknown | Credentials rotated — cascade to next source |
| Network error / timeout | Can't reach Google | Assume valid (fail-open) |

### VSIX Extraction

The Colab VS Code extension is published on [Open VSX](https://open-vsx.org/extension/Google/colab). A VSIX is a ZIP archive containing the bundled extension JavaScript. Dialeng:

1. Fetches the latest version from the Open VSX API
2. Downloads the VSIX archive
3. Reads `extension/out/extension.js` from the ZIP
4. Extracts credentials via regex (Google OAuth client IDs/secrets have distinctive formats)
5. Validates the extracted credentials
6. Caches them to `~/.dialeng/colab_oauth_client.json` (7-day TTL, `0600` permissions)

### Implementation

All credential resolution logic lives in `services/colab/colab_auth.py`:

| Function | Purpose |
|----------|---------|
| `resolve_oauth_credentials()` | Main orchestrator — runs the cascade, returns `OAuthClientCredentials` |
| `validate_oauth_client()` | Async POST to Google token endpoint |
| `extract_credentials_from_vsix()` | Download + extract from Open VSX |
| `load_cached_credentials()` / `save_cached_credentials()` | Cache management |
| `print_colab_credential_status()` | Startup status output |

### Manual Setup

If auto-update fails, users can create their own OAuth credentials. See [Colab OAuth Setup Guide](../guides/colab_oauth_setup.md).

## Connection Lifecycle

Connecting to a Colab runtime is a multi-step process involving two API backends, a Jupyter session, and a WebSocket:

```mermaid
sequenceDiagram
    participant CK as ColabKernel
    participant API as ColabAPIClient
    participant CA as Colab API
    participant GP as Colab GAPI
    participant RT as Runtime Proxy

    Note over CK: assign_and_connect()

    CK->>API: cleanup_stale_runtimes()
    API->>GP: GET /v1/assignments
    GP-->>API: [existing assignments]
    API->>CA: POST /tun/m/unassign/{endpoint}

    CK->>API: assign_kernel(variant, accelerator)
    API->>CA: GET /tun/m/assign?nbh=...
    CA-->>API: {token: xsrf_token}
    API->>CA: POST /tun/m/assign?nbh=...<br/>(X-Goog-Colab-Token: xsrf)
    CA-->>API: {endpoint, runtimeProxyInfo}

    CK->>API: create_jupyter_session(proxy_url, proxy_token)
    API->>RT: POST /api/sessions<br/>{kernel: {name: python3}}
    RT-->>API: {id, kernel: {id: kernel_id}}

    CK->>RT: WebSocket connect<br/>wss://.../api/kernels/{kernel_id}/channels

    CK->>RT: kernel_info_request
    RT-->>CK: kernel_info_reply (kernel ready)

    CK->>RT: execute_request (%matplotlib inline)
    RT-->>CK: execute_reply
    CK->>RT: execute_request (%pip install dialoghelper)
    RT-->>CK: execute_reply
    CK->>RT: execute_request (dialoghelper proxy setup)
    RT-->>CK: execute_reply (init done)

    Note over CK: Start keep-alive + token refresh tasks
    Note over CK: connection_state = "connected"
```

### Steps in Detail

1. **Cleanup stale runtimes** — Colab limits active runtimes per user. We unassign any existing ones to avoid `TooManyAssignmentsError`.
2. **Assign runtime** — Two-step XSRF pattern: GET to obtain an XSRF token, POST with that token to create the assignment. Returns a runtime endpoint and proxy connection info.
3. **Create Jupyter session** — POST to the runtime proxy's Jupyter API to start a Python 3 kernel. Returns a `kernel_id`.
4. **Connect WebSocket** — Open `wss://<proxy>/api/kernels/<kernel_id>/channels?session_id=<session>` with proxy token in headers.
5. **Kernel readiness handshake** — Send `kernel_info_request`, wait for `kernel_info_reply` (30s timeout).
6. **Initialize kernel** — Multi-step silent init: `%matplotlib inline`, `pip install dialoghelper`, and dialoghelper proxy setup. See [DialogHelper Proxy for Colab](./13_colab_dialoghelper_proxy.md).
7. **Start background tasks** — Keep-alive pings (5 min) and proxy token refresh (before expiry).

### Two API Backends

Colab uses two separate backends:

| Backend | Domain | Purpose |
|---------|--------|---------|
| ColabApiDomain | `colab.research.google.com` | Tunnel management: assign, unassign, keep-alive |
| ColabGapiDomain | `colab.pa.googleapis.com` | User info, proxy tokens, assignment listing |

Both require XSSI protection prefix stripping (`)]}'` prefix on JSON responses).

## Jupyter Wire Protocol over WebSocket

Colab's runtime exposes a standard Jupyter kernel over WebSocket, using the [Jupyter Wire Protocol v5.3](https://jupyter-client.readthedocs.io/en/stable/messaging.html). However, unlike a standard Jupyter deployment (which uses separate ZMQ channels), **Colab multiplexes all channels onto a single WebSocket** with a `"channel"` field in each message.

### Message Format

Every WebSocket message is JSON with this structure:

```json
{
    "header": {
        "msg_id": "<uuid>",
        "msg_type": "execute_request",
        "username": "dialeng",
        "session": "<session_id>",
        "date": "",
        "version": "5.3"
    },
    "parent_header": {},
    "metadata": {},
    "content": { ... },
    "channel": "shell"
}
```

### Channels

| Channel | Direction | Messages |
|---------|-----------|----------|
| `shell` | Client → Kernel | `execute_request`, `kernel_info_request` |
| `shell` | Kernel → Client | `execute_reply`, `kernel_info_reply` |
| `iopub` | Kernel → Client | `stream`, `display_data`, `execute_result`, `error`, `status` |
| `stdin` | Kernel → Client | `input_request` (from `input()` calls, including dialoghelper proxy) |
| `stdin` | Client → Kernel | `input_reply` (response to input_request) |
| `control` | Client → Kernel | `interrupt_request` |

### Message Correlation

Messages are correlated via `parent_header.msg_id`. When we send an `execute_request` with `msg_id = "abc123"`, all response messages (`stream`, `display_data`, `execute_reply`, etc.) will have `parent_header.msg_id = "abc123"`. We filter by this to handle only messages for the current execution.

## Code Execution & Streaming Output

When user code is executed, `ColabKernel.execute_streaming()` sends an `execute_request` and yields `CellOutput` objects as Jupyter messages arrive:

```mermaid
sequenceDiagram
    participant App as app.py
    participant CK as ColabKernel
    participant RT as Colab Runtime

    App->>CK: execute_streaming(code)
    CK->>RT: execute_request (shell)

    RT-->>CK: status: busy (iopub)
    RT-->>CK: stream: "hello\n" (iopub)
    CK-->>App: yield CellOutput(stream, "hello\n")

    RT-->>CK: display_data: {image/png: ...} (iopub)
    CK-->>App: yield CellOutput(display_data, {...})

    RT-->>CK: execute_reply (shell)
    Note over CK: Save execution_count, don't break

    RT-->>CK: status: idle (iopub)
    Note over CK: Break — all outputs delivered
```

### Output Type Mapping

| Jupyter msg_type | CellOutput.output_type | Content |
|------------------|----------------------|---------|
| `stream` | `stream` | Text from stdout/stderr |
| `display_data` | `display_data` | MIME bundle dict (`{image/png: ..., text/plain: ...}`) |
| `update_display_data` | `update_display_data` | Updated MIME bundle (tqdm, widgets) |
| `execute_result` (text only) | `execute_result` | Plain text repr of result |
| `execute_result` (rich) | `display_data` | Promoted to display_data for rendering |
| `error` | `error` | Exception name, value, traceback |
| `clear_output` | `clear_output` | Clear previous outputs |

### Rich `execute_result` Promotion

When `execute_result` contains rich MIME types (`image/png`, `image/jpeg`, `image/svg+xml`, `text/html`, `image/gif`), we promote it to `display_data` so the rendering pipeline handles it correctly. This is important because some libraries produce rich content as the cell's return value rather than via `display()`.

## Rich Output Handling

The full pipeline for rich output (plots, HTML, images):

```mermaid
graph LR
    subgraph "Colab Runtime"
        K[Kernel] -->|"display_data<br/>{image/png: base64}"| WS[WebSocket]
    end

    subgraph "Dialeng Backend"
        WS --> CK[ColabKernel]
        CK -->|"CellOutput<br/>content={image/png: ...}"| BC[broadcast_cell_output]
        BC --> RM[render_mime_bundle]
        RM -->|"&lt;img src=data:image/png;base64,...&gt;"| WSB[WebSocket broadcast]
    end

    subgraph "Browser"
        WSB -->|"code_display_data"| JS[appendDisplayData]
        JS --> DOM[DOM insertion]
    end
```

### `render_mime_bundle()` Priority Chain

The MIME bundle is converted to HTML using this priority:

1. `text/html` → rendered directly (trusted user code)
2. `image/svg+xml` → inline SVG
3. `image/png` → `<img src="data:image/png;base64,...">` (newlines stripped from base64)
4. `image/jpeg` → `<img src="data:image/jpeg;base64,...">`
5. `image/gif` → `<img src="data:image/gif;base64,...">`
6. `text/markdown` → raw markdown wrapper
7. `text/latex` → wrapper for MathJax/KaTeX
8. `application/json` → pretty-printed JSON
9. `text/plain` → escaped plain text fallback

### Kernel Initialization

When connecting via raw WebSocket (bypassing Colab's frontend), several setup steps run during `_initialize_kernel()`:

1. **`%matplotlib inline`** — Sets matplotlib to use the `module://matplotlib_inline.backend_inline` renderer, registers `flush_figures` as a `post_execute` hook, and registers PNG format handlers. This ensures `plt.show()` produces `display_data` messages with `image/png` content.
2. **`%pip install -q dialoghelper`** — Auto-installs the dialoghelper package on the Colab VM (120s timeout).
3. **DialogHelper proxy setup** — Injects a monkey-patch that redirects dialoghelper's HTTP calls through Jupyter's stdin channel back to the local Dialeng server.

Each step runs as a separate `execute_request` with its own timeout and logging. Progress is exposed via `get_status().connection_state` (e.g., `"initializing: Installing dialoghelper"`).

For full details on the dialoghelper proxy mechanism, see [DialogHelper Proxy for Colab](./13_colab_dialoghelper_proxy.md).

## Multiplexed WebSocket Subtlety

This is the most important implementation detail for correctness.

### The Problem

Standard Jupyter uses separate ZMQ sockets for Shell and IOPub channels. The protocol guarantees IOPub messages arrive before `execute_reply`. But **Colab multiplexes both channels onto a single WebSocket**. The proxy reads from two ZMQ sockets and forwards to one WebSocket, so **ordering between channels is not guaranteed**.

This means `execute_reply` (Shell) can arrive **before** `display_data` (IOPub):

```
Expected order:          What can actually happen:
  stream (iopub)           stream (iopub)
  display_data (iopub)     execute_reply (shell)  ← arrives early!
  status: idle (iopub)     display_data (iopub)   ← we'd miss this
  execute_reply (shell)    status: idle (iopub)
```

### The Solution

We break the receive loop on `status: idle` (IOPub) instead of `execute_reply` (Shell). Within the IOPub channel, message ordering IS preserved (same ZMQ PUB socket), so `display_data` is guaranteed to arrive before `status: idle`.

```python
# In execute_streaming():
elif msg_type == "execute_reply":
    # Save execution_count but do NOT break
    self._execution_count = content.get("execution_count", ...)

elif msg_type == "status":
    if content.get("execution_state") == "idle":
        break  # All IOPub outputs guaranteed delivered
```

This is the same approach used by proper Jupyter clients like VS Code's Jupyter extension.

### Why Stream Output Wasn't Affected

`stream` messages (from `print()`) are sent **during** cell execution, well before `execute_reply`. The large time gap makes reordering unlikely. But `display_data` from matplotlib is sent very close to the end of execution (via the `flush_figures` post-execute hook), making it susceptible to being overtaken by `execute_reply`.

## Background Tasks

Two asyncio tasks run continuously while the kernel is connected:

### Keep-Alive Pings

Every 5 minutes, we send an HTTP GET to `/tun/m/{endpoint}/keep-alive/`. Without this, Colab's idle timeout (default 30 minutes) will disconnect the runtime.

### Proxy Token Refresh

The proxy token has a TTL (typically 1 hour). We refresh it 5 minutes before expiry via the GAPI domain endpoint `/v1/runtime-proxy-token`. The new token is injected into the existing assignment's `proxy_info`.

```mermaid
graph LR
    KA[Keep-Alive Task<br/>every 5 min] -->|"HTTP GET"| CA[Colab API]
    TR[Token Refresh Task<br/>before expiry] -->|"GET /v1/runtime-proxy-token"| GP[Colab GAPI]
    TR -->|"update"| PI[proxy_info.token]
```

## Integration with Multi-Kernel System

`ColabSessionManager` manages `ColabKernel` instances per notebook and integrates with `KernelService`:

```mermaid
sequenceDiagram
    participant UI as Browser
    participant KS as KernelService
    participant SM as ColabSessionManager
    participant CK as ColabKernel

    UI->>KS: Set kernel to "colab"
    KS->>SM: get_kernel(notebook_id, "gpu")
    SM->>SM: Create ColabKernel(api_client, "gpu")
    SM-->>KS: ColabKernel instance

    UI->>KS: Execute cell
    KS->>CK: execute_streaming(code)
    Note over CK: Auto-connects on first execution
    CK-->>KS: yield CellOutput(...)
```

Users can switch between local and Colab kernels per notebook. The `KernelService` delegates to either `SubprocessKernel` or `ColabKernel` through the `BaseKernel` interface.

### Runtime Type Selection

Users can choose between CPU, GPU (T4), and TPU runtimes. Changing runtime type shuts down the existing kernel and creates a new one:

```python
await session_manager.set_runtime_type(notebook_id, "gpu")
```

## How to Extend

### Adding New Colab-Specific Message Types

Colab's kernel can send custom message types (e.g., `colab_request` for Drive mount authentication). To handle these:

1. Add a handler in `ColabKernel.execute_streaming()`:
```python
elif msg_type == "colab_request":
    # Handle Colab-specific request
    colab_type = content.get("colab_request_type")
    if colab_type == "request_auth":
        # Handle Drive mount auth
        ...
```

2. Create a new `CellOutput` type or use an existing one to communicate to the frontend.

### Supporting Colab-Specific Features

Features that could be added using the existing architecture:

- **Drive mount** — Handle `colab_request` messages for authentication during `google.colab.drive.mount()`
- **File upload/download** — Use the runtime proxy's HTTP API
- **Form fields** — Parse `@param` annotations in cell comments
- **Runtime monitoring** — Query resource usage via the runtime proxy

### Adding a New Remote Kernel Backend

Follow the same pattern as `ColabKernel`:

1. Create a new directory under `services/` (e.g., `services/sagemaker/`)
2. Implement `BaseKernel` with `execute_streaming()`, `interrupt()`, etc.
3. Create a session manager (like `ColabSessionManager`)
4. Register with `KernelService`

## See Also

- [DialogHelper Proxy for Colab](./13_colab_dialoghelper_proxy.md) — How dialoghelper functions are proxied through stdin on Colab
- [DialogHelper Integration](./05_dialoghelper_integration.md) — General dialoghelper integration (endpoints, magic variables, JavaScript injection)
- [Kernel Execution](./04_kernel_execution.md) — Local subprocess kernel and execution queue
