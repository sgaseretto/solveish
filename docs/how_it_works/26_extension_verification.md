# Extension Platform Verification Baseline

This document is the repeatable acceptance baseline for Extension API 1.

## Automated Layers

```mermaid
flowchart LR
    U["Unit contracts"] --> S["Stress and fault tests"]
    S --> I["Isolated activation"]
    I --> B["Playwright real-server E2E"]
    B --> P["Clean uv package probe"]
```

Run the complete baseline:

```bash
uv run python -m compileall -q dialeng tests
node --check dialeng/static/js/app.js
uv run pytest -q
uv run pytest -q tests/e2e
```

Focused extension commands:

```bash
uv run pytest -q \
  tests/test_extension_platform.py \
  tests/test_extension_project_service.py \
  tests/test_extension_reference_fixtures.py \
  tests/test_extension_stress.py
uv run pytest -q tests/e2e/test_extension_platform.py
```

## Contract Coverage

- Metadata parsing, compatibility, path safety, hashing, malformed diagnostics,
  duplicate precedence, and no-import discovery.
- One-session/version/project trust and source/permission invalidation.
- Atomic registration, rollback, removal, cleanup, unique modules, and 500 reloads.
- Event ordering, transforms, guards, deadlines, failure isolation, and threshold.
- Command schema, availability, operation IDs, progress, and cancellation.
- Action schema and namespace isolation.
- Settings namespacing, persistence, apply mode, and secret redaction.
- Managed tasks, project path boundaries, network permission, and cleanup deadline.
- Custom-cell Jupyter/LLM round trip and unsupported fallback preservation.
- Asset declaration/path checks, source-hash cache keys, and listener cleanup.
- Studio scaffold/build/isolated activation/test/stale-build rejection.
- Deterministic export and optional nbdev verification.
- Installed wheel metadata discovery without pre-trust import.
- Independent skill/template resource activation.

## Browser Scenarios

The deterministic Playwright suite verifies:

1. `hello-workspace`: review/trust, enable, command, shortcut surface, live setting,
   persistent state, reload, disable, and two-tab inventory convergence.
2. `execution-guard`: local kernel attach, blocked source never reaching the kernel,
   visible error, disable cleanup, and normal execution afterward.
3. `diagram-cell`: type conversion, editing, save/reload, collaboration, CSS/JS
   load once, cleanup once, and data-preserving disabled fallback.
4. `project-inspector-package`: panel, command, and independently enabled skill.
5. Extension Studio: scaffold, development profile, deterministic build, import-free preview, isolated validation/tests, trust review, and reviewed activation.
6. Broken update rollback followed by corrected update activation.
7. Command palette: keyboard/focus behavior, argument schemas, live availability,
   cancellation, unload reconciliation, shortcut conflict, safe text, and 1,000 commands.

The shared E2E harness captures screenshots, Playwright traces, browser console and
page errors, redacted server logs, diagnostics, and final project/notebook artifacts
on failure.

## Performance And Fault Baseline

Current non-regression checks assert 25 disabled local manifests discover within a
generous two-second shared-CI bound, 500 isolated source reloads retain one contribution set,
one event handler, one browser asset, and one live host while every previous host exits; concurrent mutations serialize, and repeated event failures stop
at the third call. Package tests build a wheel and use an isolated `uv` environment
to confirm entry-point metadata and bundled resources.

The target is not a universal wall-clock promise. It detects material regressions
in count, lifecycle, and bounded discovery behavior across machines.

## Manual Acceptance

Before releasing changes to extension internals:

1. Start a clean project and create each Studio template.
2. Inspect trust text and verify source is not imported before trust.
3. Enable the four reference fixtures with two tabs and a long notebook open.
4. Reload each while editors are active; confirm scroll, selection, modal state,
   outputs, and unsaved settings remain intact.
5. Change source after `prepare`; confirm activation requires revalidation.
6. Introduce syntax, setup, event, action, asset, and cleanup failures; verify the
   prior extension or core notebook remains usable.
7. Restart Dialeng and verify trusted enabled state and resources restore.
8. Run local and Colab notebooks while unrelated extensions reload.
9. Run Mock, Codex, and Bedrock prompts with an extension tool when credentials
   exist; provider and Colab checks remain opt-in.
10. Disable all fixtures and compare registry, handler, task, browser asset, and
    listener counts with baseline.

Real Codex, Bedrock, and Colab validation is environment-dependent and must be
reported as skipped, not silently assumed, when credentials are unavailable.

## Implementation Acceptance Record

On 2026-07-13 the completed Extension API 1 platform was verified with the
following evidence:

- CPython 3.11 passed 383 non-browser unit/integration tests with two optional
  tests skipped. The focused extension matrix also passed 77 tests on CPython
  3.12 and 77 tests on CPython 3.13.
- The complete deterministic Playwright suite passed 19 scenarios with seven
  authenticated-provider scenarios explicitly skipped. This includes the
  command palette, Extension Studio, collaboration, rollback, and all reference
  extension browser flows.
- All four reference extensions passed initial activation, disable, reload,
  upgrade, downgrade, source removal, reinstall, and shutdown lifecycle tests
  in a nine-test fixture suite.
- A 500-reload isolated-host stress test passed in 28.49 seconds while retaining
  one contribution set and one live host. Twenty-five crash/restart cycles left
  no socket artifacts or child processes.
- Adversarial host tests covered malformed and oversized protocol messages,
  stdout noise, timeout, cancellation, memory pressure, direct file/network/
  process denial, brokered permissions, cleanup races, and failed-upgrade
  rollback.
- A clean `uv` environment built and installed the Dialeng wheel plus an
  independently packaged extension in two passing smoke tests. Git/project-local
  installation is checked separately after committing because that probe installs
  the repository's committed `HEAD`.
- Python compilation, JavaScript syntax, generated-schema synchronization,
  standalone extension-module lint, and `git diff --check` passed. Repository-wide
  Ruff remains outside this gate because historical FastHTML modules, demos, and
  notebooks do not yet conform to that lint profile.

Real Codex, Bedrock, and Colab tests remain opt-in. They are not required to prove
the provider-independent extension lifecycle and are reported as skipped when
credentials or runtime allocation are unavailable.
