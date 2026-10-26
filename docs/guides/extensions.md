# Building Dialeng Extensions

Dialeng extensions are reviewed Python contribution bundles. They can add commands,
tools, settings, events, actions, toolbar controls, panels, status items, browser
assets, cell types, kernels, providers, skills, and templates without modifying
Dialeng core.

Project extensions live under `.dialeng/extensions/` and are disabled until a
user reviews and trusts them. RPC-safe extensions execute in one supervised
`python -I` host; kernel/provider/custom-renderer contributions are visibly marked
as privileged and execute in the server process. Manifest permissions are both
review metadata and core-broker gates, but neither mode is a complete OS security
sandbox.

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

Choose the smallest template that matches the contribution:

| Requirement | Template | Expected runtime |
|---|---|---|
| Command, action, setting, toolbar item | `basic` | Isolated |
| Provider-neutral LLM tool | `tool` | Isolated |
| Event observer, transform, or guard | `lifecycle` | Isolated |
| New notebook cell representation | `cell` | Privileged in-process |
| Kernel or LLM provider | `provider` | Privileged in-process |
| Installable wheel with resources | `package` | Contribution-dependent |

The manager reports the actual runtime mode before trust. A package is not
automatically isolated: privileged contribution types remain privileged after
packaging.

`build` exports the notebook and performs static validation. `prepare` additionally
activates the extension in an isolated process, runs its tests, optionally runs
nbdev verification, and writes private validation evidence under
`.dialeng/extension-builds/`. If source changes afterward, activation is rejected
until `prepare` succeeds again.

## End-To-End Authoring Workflow

Use this sequence for both CLI and Extension Studio development:

1. Define one narrow user outcome and select the smallest template above.
2. Create the scaffold. Open `extension.ipynb`; do not treat generated
   `extension.py` as editable source.
3. Implement with public `dialeng.extensions` declarations and explicit notebook
   roles. Add unit tests in `#| extension test` cells.
4. Declare the minimum exact permissions, assets, contributions, and resources in
   `extension.toml`.
5. Select **Build**. Resolve generated diagnostics at their mapped notebook cell
   and source line.
6. Select **Validate** and **Test**. Use browser tests for UI, focus,
   collaboration, custom-cell, or asset behavior.
7. Select **Preview** and confirm it says `imported: false`. Compare the listed
   commands, tools, events, settings, assets, resources, permissions, and runtime
   mode with the design.
8. Select **Profile** and require fresh notebook, manifest, module, documentation,
   and test hashes.
9. Review source and permissions, then explicitly trust and select **Activate
   reviewed build**. No build, preview, skill, or lockfile can grant trust.
10. Exercise the real contribution, disable it, re-enable it, reload it, and test
    cleanup. Package and clean-install it when distribution is required.

For LLM-assisted authoring, the bundled `extension-creator` skill follows the
same sequence through constrained tools. It may prepare a review request but
cannot trust or activate the result.

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
filesystem_write = []
subprocess = false
notebook = false
kernel = false

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
        readme = await ctx.read_text("README.md")
        await ctx.notify(f"Audited {len(readme)} characters")
        return {"message": "Audit complete", "characters": len(readme)}

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
`resolve_project_path`, `watch_path`, `read_text`/`read_bytes`,
`write_text`/`write_bytes`, permission-gated `open_http_client`/`request_http`,
`run_process`, backend-authoritative `change_cell_type`/`invoke_kernel`,
`cancel_signal`, and `create_task`.
Always use `ctx.create_task`; unmanaged tasks created by extension setup cause
activation to fail.

Public declarations such as `ExtensionSetting`, `ExtensionCellType`,
`ExtensionKernel`, and `ExtensionProvider` are exported from `dialeng.extensions`.
New extensions should use these declarations instead of importing classes from
`dialeng.core`. Kernel, provider, custom-cell, FastHTML renderer, callback, and
service contributions remain privileged in-process capabilities in API 1. Commands,
actions, settings, tools, events, shortcuts, and declared assets are designed to
cross the isolated-host protocol.

## Compatibility And Diagnostics

Extension API `1.0.0` follows this compatibility policy:

- `>=1,<2` accepts additive API 1 minor releases and corrective patch releases.
- Breaking public contracts require API 2.
- Deprecated public entry points emit actionable diagnostics for at least one
  minor release before removal.
- Manifest schema 1 remains independently versioned from the Python API.

Check one source without importing its entry point:

```bash
uv run dialeng extension check .dialeng/extensions/notebook-audit
```

Inspect discovery, precedence, source hashes, trust, enabled state, build
freshness, registrations, tasks, failures, and cleanup health:

```bash
uv run dialeng extension doctor --path .
```

Export the canonical JSON Schema catalog used by Dialeng and CI:

```bash
uv run dialeng extension schema --output extension-api-v1.schema.json
```

`check` reports direct `dialeng.core` imports and FastHTML renderers as privileged
compatibility paths. It rejects incompatible API ranges, malformed manifests,
missing entry points, and syntax errors before extension code is imported.

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
from dialeng.extensions import ExtensionSetting

api.register_setting(ExtensionSetting(
    path="limit",
    kind="int",
    default=100,
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

`package` reruns preparation and writes artifacts under the extension's `dist/`.
Verify the wheel without source-checkout imports:

```bash
uv venv /tmp/dialeng-extension-check
uv pip install --python /tmp/dialeng-extension-check/bin/python \
  .dialeng/extensions/project-inspector/dist/*.whl pytest
/tmp/dialeng-extension-check/bin/python -m pytest \
  .dialeng/extensions/project-inspector/tests
```

The wheel must include `extension.toml`, entry-point metadata, declared browser
assets, skills/templates, generated documentation, and tests where configured.
Removing the source directory before the final import/test probe catches accidental
checkout imports.

Installed distributions expose `[project.entry-points."dialeng.extensions"]` and
must package `extension.toml`. Discovery reads package metadata without importing
the entry point. Optional nbdev projects run export, test, and clean checks, but
Dialeng's deterministic exporter remains canonical for runtime Python.

The packaged `project-inspector-package` reference is retained as the independent
API 1 compatibility fixture. Its wheel is built and activated in an isolated `uv`
environment so source-checkout imports cannot mask packaging defects.

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

## Command Palette

Press `Cmd/Ctrl+Shift+P` to search every core and extension command. Unavailable
commands remain visible with a backend-provided reason. Parameterized commands
open controls generated from their argument schema. Use `Alt+Enter` on a selected
command to set or clear a browser-local shortcut override. Favorites and recents
are local to the browser and do not modify project state.

Extension authors should register one backend command and reuse it everywhere:

```python
api.register_command(
    "audit",
    title="Audit notebook",
    description="Check the current project for reproducibility issues.",
    category="Quality",
    keywords=("check", "reproducible"),
    availability="notebook.open && kernel.optional",
    args_schema={
        "type": "object",
        "properties": {"strict": {"type": "boolean", "default": False}},
        "additionalProperties": False,
    },
    handler=audit,
)
```

Toolbar items, shortcuts, the palette, browser facade, and HTTP API must converge
on this registration rather than duplicate logic in JavaScript. Long operations
report progress through `command["report_progress"]` and stop when
`command["cancel_event"]` is set. The server revalidates availability and argument
types when invoked; client-side disabled state is not authorization.

## Lock And Restore An Extension Environment

After enabling the intended extension and resource set:

```bash
uv run dialeng extension lock --path .
uv run dialeng extension check --path .
```

Commit `.dialeng/extensions.lock`. A new checkout restores it with:

```bash
uv sync
uv run dialeng extension sync --path .
```

After one online lock/sync has populated the content-addressed source, wheel, and
uv caches, an offline checkout can use:

```bash
uv run dialeng extension sync --offline --path .
```

`sync` installs/verifies immutable artifacts and reconciles enabled IDs. It never
grants trust. Review and trust restored sources locally. To update deliberately:

```bash
uv run dialeng extension update dev.example.extension --path .
```

The output lists version, source, API, permission, and resource changes and marks
changes that require renewed trust. `dialeng doctor --path .` combines Python,
configuration, provider, kernel, extension runtime, generated-build, and lock
diagnostics.

## Notebook Development Profile

Extension notebooks display an **Extension Development** badge. Use the Extension
Manager or command palette for Profile, Build, Validate, Test, Preview, Package,
and Activate Reviewed Build. Preview never imports the candidate. Build and tests
run outside the server, and activation is rejected if the notebook, manifest,
generated module, or documentation differs from the reviewed hashes.

Use explicit cell roles:

```python
#| extension implementation
def setup(api, ctx):
    ...
```

```python
#| extension test
def test_behavior():
    ...
```

Documentation/example/non-export roles are `#| extension documentation`,
`#| extension example`, and `#| extension non-export`. Monaco completion for
`api.` and `ctx.` provides the common public methods and contextual documentation.

## Isolated And Privileged Modes

The manager shows `isolated` or `privileged in-process` before activation.
RPC-safe contributions run in one authenticated private-socket subprocess per extension. Custom
cells, Python-rendered panels/status, kernels, providers, callbacks, and internal
core/FastHTML imports currently require privileged mode. Subprocess isolation
protects server availability but is not a complete OS security sandbox; only run
code you have reviewed. Project filesystem, network, process, notebook, and kernel
operations cross the core broker and are checked against the manifest on every
request.

Isolated extensions must use `ctx.read_text`/`read_bytes`,
`write_text`/`write_bytes`, `request_http`, `run_process`, `change_cell_type`, and
`invoke_kernel`. Prefer exact paths or narrow globs such as `docs/**`; path
traversal and symlink escapes are rejected. Direct project-file, socket, and
process access from the child is denied even when the related capability is
declared. Privileged mode bypasses process isolation, so its source review must be
more stringent.

## Troubleshooting

| Symptom | Action |
|---|---|
| Syntax error references generated Python | Open the mapped notebook cell/line, edit the notebook, and rebuild |
| Build is stale after a successful validation | Rebuild, rerun tests, preview again, and request a new review |
| Manifest or API range is rejected | Run `extension check`, inspect the API schema, and correct metadata before import |
| Broker denies a file/network/process/notebook/kernel operation | Declare only the actual capability, explain the increase, and renew local trust |
| Isolated host exits, times out, or exceeds limits | Inspect `extension doctor`, bound the operation, and make cleanup tolerate partial setup |
| Browser asset does not update or duplicates behavior | Check manifest declaration, source-hash URL, public browser facade use, and cleanup registration |
| Candidate upgrade fails | Confirm the previous healthy version remains active, then correct and retry the candidate |
| `sync --offline` cannot restore | Populate source/wheel and uv caches online first; no partial enabled-state change should occur |
| Custom-cell data looks unsupported after disable | Re-enable the extension; the fallback intentionally preserves raw Jupyter data |

Use both reports when diagnosis crosses layers:

```bash
uv run dialeng extension doctor --path .
uv run dialeng doctor --path .
```

## Release Checklist

An extension is ready only when all applicable checks pass:

- `extension check`, deterministic Build, isolated Validate, and unit tests pass.
- Preview imports no code and lists only intended contributions and permissions.
- Browser tests pass for UI, custom cells, assets, focus, and collaboration.
- Disable, reload, failed upgrade, cancellation, and cleanup preserve core behavior.
- Source or permission changes invalidate review as expected.
- A package wheel installs and tests in a clean `uv` environment without the
  source checkout.
- `.dialeng/extensions.lock` reproduces artifacts/resources while trust remains
  local and explicit.
- User documentation explains configuration, commands, recovery, runtime mode,
  and security limitations.
