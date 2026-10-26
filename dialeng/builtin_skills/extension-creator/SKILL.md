---
name: extension-creator
description: Create, refine, validate, test, package, and prepare a Dialeng extension when the user asks to extend Dialeng with tools, commands, lifecycle hooks, UI, settings, providers, resources, or custom cells.
---
# Dialeng Extension Creator

Create project extensions under `.dialeng/extensions/` with Dialeng's constrained
authoring tools. Keep `extension.ipynb` as the source of truth. Never trust,
activate, or increase permissions on the user's behalf.

RPC-safe commands, actions, tools, events, settings, shortcuts, toolbar items, and
declared assets run in one supervised `python -I` host per active extension.
Kernels, providers, callbacks, custom cells, Python renderers, services, and
internal imports are privileged and run inside the Dialeng server after explicit
review. Isolation protects server availability but is not a complete hostile-code
OS sandbox.

## Tool Map

- `inspect_extension_api(topic)` returns current compatibility, runtime modes,
  public methods, examples, permissions, validation checks, templates, and the
  complete workflow. Call it again for each relevant topic instead of guessing.
- `list_extension_templates()` describes `basic`, `tool`, `cell`, `lifecycle`,
  `provider`, and `package` templates.
- `create_extension(...)` scaffolds and builds without trust or activation.
- `read_extension(...)` and `update_extension_file(...)` are path-constrained;
  use the returned SHA-256 as `expected_sha256` for every update.
- `build_extension`, `validate_extension`, `run_extension_tests`,
  `preview_extension`, `show_extension_profile`, and
  `show_extension_diagnostics` implement the diagnosis loop.
- `package_extension` prepares and builds a package-template wheel. It never
  installs, trusts, or activates the result.
- `request_extension_activation` only creates review instructions. It cannot
  trust or activate code.

## Workflow

1. Call `inspect_extension_api()` and `list_extension_templates()`. Then inspect
   each relevant topic, especially `permissions`, `commands`, `packaging`, or
   `runtime`.
2. Choose the smallest template that matches the requested contribution. Explain
   whether the design will run isolated or privileged before creating it.
3. Call `create_extension`. Read `extension.ipynb`, `extension.toml`, README, and
   tests before editing.
4. Edit the notebook, not generated `extension.py`. Mark cells as
   `#| extension implementation`, `test`, `documentation`, `example`, or
   `non-export`; legacy `#| export` is accepted as implementation.
5. Use `update_extension_file` with optimistic hashes. Declare only the exact
   permissions required by the implementation.
6. Call `build_extension`, `validate_extension`, and
   `run_extension_tests(level="unit")`. Use browser tests for UI, custom-cell,
   collaboration, focus, or asset behavior.
7. Call `preview_extension` before activation. It must report `imported: false`.
   Review every command, tool, event, setting, asset, resource, permission, and
   runtime mode. Then inspect the profile and diagnostics.
8. If distributable, call `package_extension` and instruct the user to verify the
   wheel in a clean `uv` environment. Do not package an ordinary non-package
   template by silently adding package metadata.
9. Call `request_extension_activation` only when validation is fresh and the user
   explicitly asked to enable the extension. Tell the user to review and activate
   it in the Extensions panel.
10. After activation, test the contribution through its real surface, disable and
    re-enable it, reload it, and confirm cleanup. Never report success from static
    validation alone.

## Template Selection

| Need | Template | Runtime expectation |
|---|---|---|
| Command, action, setting, toolbar item | `basic` | Isolated |
| LLM-callable function | `tool` | Isolated |
| Event observer, transform, or guard | `lifecycle` | Isolated |
| New notebook cell representation | `cell` | Privileged |
| Kernel or LLM provider integration | `provider` | Privileged |
| Installable wheel with resources | `package` | Usually isolated; contribution-dependent |

If one extension mixes privileged and RPC-safe contributions, trust review must
show the privileged mode. Do not split behavior merely to hide privilege.

## Authoring Rules

### Public API

- Import declarations from `dialeng.extensions`; never import registries,
  notebook managers, FastHTML application objects, requests, or kernels from
  internal modules.
- Keep module import deterministic and side-effect free. Perform bounded setup in
  `setup(api, ctx)`.
- Use `ctx.create_task` and `ctx.add_cleanup` for long-lived work. Cleanup must be
  idempotent and tolerate partial setup.
- Use public transport-neutral declarations such as `ExtensionSetting`,
  `ExtensionCellType`, `ExtensionKernel`, and `ExtensionProvider`.

### Permissions

- Prefer exact project paths or narrow globs such as `docs/**`; avoid `**` unless
  the requirement is genuinely project-wide.
- Use brokered `ctx.read_text`, `read_bytes`, `write_text`, `write_bytes`,
  `request_http`, `run_process`, `change_cell_type`, and `invoke_kernel`.
- Declare `network`, `subprocess`, `notebook`, `kernel`, and `browser_assets` only
  when used. Direct project-file, socket, and subprocess access is denied in the
  isolated host even when a related capability is declared.
- Treat source, identity, origin, and permission changes as requiring renewed
  local review.

### Commands And The Palette

- Register one backend command and reuse it from the command palette, toolbar,
  shortcut, browser API, or action. Do not create a second client-only handler.
- Supply title, description, category, keywords, availability, and a JSON object
  `args_schema`; the palette generates controls from this declaration.
- For long work, call `command["report_progress"](...)` and stop promptly when
  `command["cancel_event"]` is set.
- Availability in the browser is informative. The server revalidates it when
  invoked, so handlers must not trust stale client state.
- Test unavailable, malformed-argument, cancellation, failure, reload, and
  extension-unload behavior.

### Tools, Events, Settings, UI, And Cells

- Tool parameters need type annotations and a useful docstring. Keep results
  serializable and bounded.
- Choose event mode deliberately: `observe` for side effects, `transform` for a
  replacement payload, and `guard` to block. Keep handlers within deadlines.
- Use `ExtensionSetting`; namespace paths are added automatically. Mark secrets
  and test `live`, `new_kernel`, `extension_reload`, or `server_restart` apply
  behavior.
- Declare browser CSS/JS in the manifest and register it. Use only
  `window.dialeng.extensions`, register cleanup, and render untrusted text safely.
- Custom cells require renderer, serializer, deserializer, and LLM converter
  coverage. Verify Jupyter round trips, collaboration, disable fallback, and
  re-enable recovery.
- Skills/templates are independently enabled resources. Enabling them must not
  implicitly activate runtime Python.

## Reproducibility And Packaging

For a team-reproducible project, recommend:

```bash
uv run dialeng extension lock --path .
uv run dialeng extension check --path .
uv run dialeng extension sync --path .
# Offline only after the source/wheel and uv caches are populated:
uv run dialeng extension sync --path . --offline
```

Commit `.dialeng/extensions.lock`; never commit `~/.dialeng` trust records.
Lockfiles reproduce artifacts; sync never grants trust. Sync restores artifacts
and the enabled resource set. Use
`dialeng extension update ID --path .` deliberately and highlight source,
permission, compatibility, and dependency changes before relocking.

A package must include `extension.toml`, entry-point metadata, declared assets,
skills/templates, generated tests, and documentation. A clean environment must be
able to install its wheel and run tests without the Dialeng source checkout.

## Diagnosis And Recovery

- Syntax/export failure: inspect mapped notebook cell and source line, edit the
  notebook, rebuild, and never patch generated Python.
- Stale build: rebuild, rerun tests, preview again, and request a new review.
- Compatibility or manifest failure: inspect `inspect_extension_api`, profile,
  diagnostics, and the generated API schema before changing code.
- Permission denial: narrow or add the actual capability, explain the increase,
  then require renewed review. Do not bypass the broker.
- Host crash/timeout: inspect diagnostics and logs, bound the operation, make
  cleanup safe, and verify an unrelated extension still works.
- Failed upgrade: verify the previous healthy version remains active before
  correcting and retrying the candidate.
- Browser failure: inspect browser console, asset declaration, cleanup hooks, and
  two-tab behavior.

## Completion Checklist

Do not call the extension complete until all applicable checks pass:

- Build output is deterministic and the notebook/generated hashes are fresh.
- Manifest, API compatibility, permissions, assets, and resources validate.
- Unit tests pass; browser tests pass for user-facing behavior.
- Import-free preview matches intended contributions and runtime mode.
- Disable, reload, upgrade failure, and cleanup preserve core behavior.
- Package wheels install and test cleanly when packaging is requested.
- The user explicitly reviews trust and activation.

Never claim extension code is safe merely because it is isolated or declares
limited permissions. Never silently trust, activate, sync, install, or elevate an
extension.
