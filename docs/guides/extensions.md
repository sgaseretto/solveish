# Building Dialeng Extensions

Dialeng extensions are trusted Python contribution bundles. They can add commands,
tools, settings, events, actions, toolbar controls, panels, status items, browser
assets, cell types, kernels, providers, skills, and templates without modifying
Dialeng core.

Project extensions live under `.dialeng/extensions/` and are disabled until a
user reviews and trusts them. Trusted extensions execute in the Dialeng server
process with the server user's operating-system permissions. Manifest permissions
are review metadata and API gates, not a security sandbox.

## Create An Extension

Use the CLI:

```bash
uv run dialeng extension create notebook-audit --template basic
uv run dialeng extension build .dialeng/extensions/notebook-audit
uv run dialeng extension prepare .dialeng/extensions/notebook-audit
```

Or open **Extensions**, expand **Extension Studio**, choose a template, and select
**Create notebook**. Available templates are `basic`, `tool`, `cell`, `lifecycle`,
`provider`, and `package`.

The generated layout is:

```text
.dialeng/extensions/notebook-audit/
├── extension.toml
├── extension.ipynb       # source of truth
├── extension.py          # generated from #| export cells
├── README.md
└── tests/test_extension.py
```

`build` exports the notebook and performs static validation. `prepare` additionally
activates the extension in an isolated process, runs its tests, optionally runs
nbdev verification, and writes private validation evidence under
`.dialeng/extension-builds/`. If source changes afterward, activation is rejected
until `prepare` succeeds again.

## Minimal Contract

`extension.toml` describes metadata without importing Python:

```toml
schema_version = 1
id = "acme.notebook-audit"
name = "Notebook Audit"
version = "0.1.0"
description = "Checks notebook reproducibility."
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "project"

[permissions]
network = false
filesystem_read = ["project"]

[contributes]
commands = true
tools = true
settings = true
events = true
```

The entry point receives the stable facade and managed context:

```python
def setup(api, ctx):
    async def audit(command):
        root = ctx.resolve_project_path(".", must_exist=True)
        await ctx.notify(f"Auditing {root.name}")
        return {"message": "Audit complete"}

    api.register_command(
        "audit",
        title="Audit notebook",
        description="Check the current project.",
        handler=audit,
        availability="notebook.open && kernel.optional",
    )
    api.register_toolbar_item(
        "audit-button", command="audit", title="Audit", icon="scan-search",
    )
```

Public contribution names are qualified as `<extension-id>:<local-name>`.
Extensions should not import application globals or mutate registries directly.

## Public API

| API | Purpose |
|---|---|
| `register_command` | Reusable, validated operation for toolbar/shortcut/UI invocation |
| `register_shortcut` | Keyboard binding to a namespaced command |
| `register_action` | Typed HTTP action under the extension namespace |
| `register_toolbar_item` | Dynamic command button or server-rendered built-in control |
| `register_panel` | Left, right, bottom, or modal server-rendered surface |
| `register_status_item` | Compact status renderer |
| `register_setting` | Namespaced typed setting with apply mode |
| `register_tool` | Provider-neutral LLM tool |
| `register_cell_type` | Complete render/serialize/deserialize/LLM bundle |
| `register_callback` | Legacy execution callback, now owner tracked |
| `register_renderer` | Renderer for an existing cell type |
| `register_llm_converter` | LLM context converter for an existing cell type |
| `register_kernel` | Kernel factory registration |
| `register_provider` | LLM provider registration |
| `register_asset` | Declared CSS or JavaScript module |
| `on` | Ordered observe, transform, or guard event handler |
| `add_cleanup` | Deterministic synchronous or async cleanup |

`ExtensionContext` provides `logger`, persistent `state`, `setting`, `notify`,
`resolve_project_path`, `watch_path`, permission-gated `open_http_client`,
backend-authoritative `change_cell_type`, `cancel_signal`, and `create_task`.
Always use `ctx.create_task`; unmanaged tasks created by extension setup cause
activation to fail.

## Commands And Actions

Commands receive a dictionary containing:

```python
{
    "operation_id": "ext-command-...",
    "notebook_id": "folder~notebook",
    "project_root": "/absolute/project",
    "args": {},
    "cancel_event": asyncio.Event(),
    "report_progress": async_callable,
}
```

Use `await command["report_progress"]("Indexing", current=2, total=10)` for
progress and stop promptly on cancellation. The browser facade returns an
invocation promise with `.operationId` and `.cancel()`.

Command availability supports conjunctions of `always`, `workspace`,
`notebook.open`, `kernel.optional`, and `kernel.required`.

Actions use JSON-schema-shaped request and response declarations and run at:

```text
POST /dialeng/{notebook}/extensions/{extension}/actions/{action}
```

## Events

Register events with `api.on(event, handler, mode=..., order=...)`.

- `observe` records a side effect without changing payload.
- `transform` returns the next payload.
- `guard` returns a reason or `{block: true, reason: ...}`.

Initial event families cover app/project lifecycle, notebook and cell mutations,
execution, prompt/LLM flow, tools, kernels, settings, trust, and extension lifecycle.
Handlers run by order and qualified name, have deadlines, and are disabled after
three consecutive failures. A failing guard fails closed; ordinary observers do
not stop unrelated core work.

## Settings

Settings are automatically scoped under `extensions.<extension-id>`:

```python
from dialeng.services.settings_service import SettingDefinition

api.register_setting(SettingDefinition(
    "limit", "int", 100,
    label="Result limit",
    description="Maximum items returned by the audit.",
    minimum=1,
    maximum=1000,
    apply_mode="live",
))
```

Apply modes are `live`, `new_kernel`, `extension_reload`, and `server_restart`.
Secrets are redacted from settings snapshots and diagnostics. Settings survive
disable/re-enable.

## Browser Assets

Declare assets in the manifest and register them in setup:

```toml
[permissions]
browser_assets = true

[assets]
css = ["assets/extension.css"]
js = ["assets/extension.js"]
```

```python
api.register_asset("styles", kind="css", path="assets/extension.css")
api.register_asset("client", kind="js", path="assets/extension.js")
```

Modules use `window.dialeng.extensions` to subscribe, invoke commands/actions,
open registered surfaces, show notifications/progress, inspect public state, and
register cleanup. `registerToolRenderer(extensionId, toolName, renderer)` provides
owned `renderCall` and `renderResult` hooks for custom tool UI. Assets are
source-hash versioned, served only when declared,
and removed on disable/reload.

## Skills And Templates

Packages may declare independently enabled resources:

```toml
[resources]
skills = ["resources/skills/project-inspector"]
templates = ["resources/templates"]
```

Enabling a resource does not activate runtime Python. Project-local skills take
precedence over extension skills with the same name.

## Development, Packaging, And nbdev

Test a source directory for one server session:

```bash
uv run dialeng notebooks --extension /path/to/extension
```

Create a package template, then build a wheel:

```bash
uv run dialeng extension create project-inspector --template package
uv run dialeng extension prepare .dialeng/extensions/project-inspector
uv run dialeng extension package .dialeng/extensions/project-inspector
```

Installed distributions expose `[project.entry-points."dialeng.extensions"]` and
must package `extension.toml`. Discovery reads package metadata without importing
the entry point. Optional nbdev projects run export, test, and clean checks, but
Dialeng's deterministic exporter remains canonical for runtime Python.

## Trust And Recovery

The manager supports one-session trust, exact-version trust, and project trust.
Source, identity, origin, or permission changes invalidate applicable trust.
Activation is transactional: a candidate imports and registers into staging,
validates, then atomically replaces the current version. A failed candidate leaves
the prior version active. Disable cancels managed tasks, runs cleanup, unregisters
all owned contributions, removes browser assets/listeners, and broadcasts one
authoritative snapshot to every tab.

## Reference Extensions

Disabled fixtures under `examples/extensions/` demonstrate the contracts:

- `hello-workspace`: commands, shortcut, action, settings, toolbar, status, state.
- `execution-guard`: guard/transform events, observer metrics, managed task.
- `diagram-cell`: custom-cell round trip, collaboration, command, CSS/JS cleanup.
- `project-inspector-package`: wheel entry point, read-only tool, panel, skill,
  template, and independent resource activation.

Run their focused tests with:

```bash
uv run pytest -q tests/test_extension_reference_fixtures.py
uv run pytest -q tests/e2e/test_extension_platform.py
```
