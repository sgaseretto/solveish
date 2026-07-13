---
name: extension-creator
description: Create, refine, validate, and prepare a Dialeng runtime extension when the user asks to extend Dialeng itself with tools, commands, lifecycle hooks, UI, settings, providers, or custom cells.
---
# Dialeng Extension Creator

Create project extensions under `.dialeng/extensions/` through Dialeng's constrained extension tools. Extension code runs in the Dialeng server process after trust, so never request activation before the source and permissions have been reviewed.

## Workflow

1. Call `inspect_extension_api` for the contribution area involved and `list_extension_templates` before designing the extension.
2. Call `create_extension` with the smallest matching template. This creates a notebook source, manifest, generated module, README, and tests but does not trust or activate code.
3. Read `extension.ipynb`, `extension.toml`, and tests with `read_extension`. Preserve the notebook as the source of truth. Mark cells with `#| extension implementation`, `test`, `documentation`, `example`, or `non-export`; legacy `#| export` remains an implementation directive.
4. Modify only files within the selected extension with `update_extension_file`. Supply `expected_sha256` after reading a file so concurrent edits fail instead of being overwritten.
5. Call `build_extension` after notebook changes, then `validate_extension` and `run_extension_tests`.
6. Use `preview_extension`, `show_extension_profile`, and `show_extension_diagnostics` to review registrations without importing code, verify notebook-to-module hashes, and inspect diagnostics. Correct every error before continuing.
7. Call `request_extension_activation` only after validation passes and the user asked to enable the extension. This creates a review instruction; it cannot trust or activate code.
8. Tell the user to review permissions and source in the Extensions panel, then explicitly activate the reviewed build there. The skill cannot trust or activate code.

## Authoring Rules

- Use only documented `ExtensionAPI` and `ExtensionContext` methods.
- Declare the minimum filesystem, network, subprocess, and browser-asset permissions.
- Register long-lived tasks and cleanup through `ExtensionContext`.
- Keep setup bounded and deterministic; do not start work during module import.
- Add failure, reload, cleanup, and disable tests for every long-lived contribution.
- Keep browser code inside declared assets and use only the public browser facade.
- Never edit generated Python as the source of truth; update exported notebook cells and rebuild.
- Never claim an extension is safe merely because its manifest declares limited permissions. Trusted runtime extensions retain the server process's operating-system access.
