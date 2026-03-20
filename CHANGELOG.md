# Changelog

All notable changes to Dialeng will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Colab Kernel Snapshot & Auth Visibility
- Added a backend-authoritative `kernel_snapshot` payload carrying kernel liveness, queue state, notebook setup phase, Colab auth state, runtime id, and account email
- Browser now polls `/dialeng/{nb_id}/kernel/snapshot` and also receives the same snapshot over WebSocket to recover cleanly after reconnects
- Kernel modal now shows the signed-in Colab account email when a persisted Google session is active
- OAuth callback now validates single-use `state` tokens before exchanging the authorization code

#### Colab Resilience Tests
- Added `tests/test_colab_resilience.py` covering per-notebook kernel serialization, Colab auth state handling, degraded Colab connection liveness, connection recycle behavior, and async kernel teardown
- Added `tests/test_logging_config.py` covering console log filtering and Dialeng logger registration

#### Solveit-style Keyboard Shortcuts
- **Navigation**: Arrow keys / `j`/`k` to navigate cells, `Shift+Arrow` for multi-selection, `Cmd+A` select all, `Enter` to edit, `Escape` to exit
- **Multi-selection**: Click, Shift+click (range), Cmd+click (toggle) — multi-select applies to h/p/e toggles, clipboard ops, and clear outputs
- **Cell clipboard**: `x`/`c`/`v` to cut/copy/paste cells, `,`/`.` to copy input/output to system clipboard, `q` to duplicate
- **Cell operations**: `a`/`b` to add cell above/below, `Shift+M` to merge with cell below, `Cmd+Shift+J/K/L/;` to switch cell type
- **Execution**: `r` re-run all code, `Shift+A`/`Shift+B` run above/below, `Shift+R` restart kernel, `Shift+S` stop all, `Alt+Enter` run and create new cell
- **Display**: `i`/`o` toggle input/output collapse (replaces `Z`/`Shift+Z`), `Shift+O` clamp output, `S S` save, `←`/`→` collapse/expand cell
- **Content**: `m` copy code blocks from AI response, `n` edit AI response, `w` extract fenced code blocks to new code cells
- **Keyboard shortcuts modal**: `?` or toolbar keyboard icon shows scrollable modal listing all shortcuts, categorized with `<kbd>` styling
- New backend endpoints: `/cell/{cid}/duplicate`, `/cell/{cid}/clear-output`, `/cell/{cid}/merge-below`, `/cell/{cid}/extract-code-blocks`

#### CLI `--init` Flag
- `dialeng --init` initializes the reuse workflow (CRAFT.ipynb, pyproject.toml, package dir) on startup
- `dialeng --init my_pkg` uses an explicit package name
- Auto-detects package name from existing `[tool.dialeng]` or `[tool.nbdev]` config, or derives from directory name
- Safe to re-run: updates generated cells, preserves user-created cells
- Non-existent directories are created automatically (e.g., `dialeng new_folder --init`)
- Enhanced `dialeng -h` with usage examples and phase documentation

#### CRAFT Init Extension
- New toolbar button (square-library icon) to initialize a package-aware CRAFT.ipynb
- Prompts for a package name (valid Python identifier), pre-filled from existing config
- Confirmation dialog when CRAFT.ipynb already exists
- Creates `pyproject.toml`, `CRAFT.ipynb`, and package directory with `__init__.py`
- Merges existing CRAFT.ipynb content when re-initializing (user cells preserved)
- CRAFT.ipynb now includes a "Transitioning to a Python Package" guide section
- Core extension at `dialeng/extensions/craft_init.py`, logic in `dialeng/services/craft_init_service.py`

#### Configurable Export Folder
- Save-hook extraction reads lib name from `pyproject.toml`, checking `[tool.dialeng]` then `[tool.nbdev]`
- Seamless integration with existing nbdev projects (no `[tool.dialeng]` section needed)
- Defaults to `_lib` when no configuration exists (backward compatible)
- `_inject_lib_syspath()` uses the configured folder name

#### Colab Module Upload
- Exported module files are automatically uploaded to Colab VM during CRAFT init
- Re-upload triggered on kernel restart and on every save that produces an export
- Uses base64 encoding for safe file transfer to remote kernel

### Fixed

#### Colab Connection Robustness
- Removed the global "cleanup all runtimes before attach" behavior that could tear down unrelated Colab sessions on the same Google account
- Colab init steps now wait for `status: idle` instead of treating `execute_reply` as completion, avoiding premature setup transitions on Colab's multiplexed WebSocket
- Notebook setup work (`sys.path` injection, Colab module upload, CRAFT execution, restart setup) now shares the same per-notebook execution lock as regular cell execution, preventing races on the Colab kernel transport
- Notebook setup and save-triggered sync work now carry a per-notebook generation id, so kernel switches, runtime changes, restarts, and notebook teardown cancel stale background tasks before they touch the new kernel
- Notebook deletion/removal now tears down the attached kernel plus pending setup/sync work instead of leaving remote sessions and execution state behind
- Quiet long-running code cells no longer get marked finished by a 30-second browser inactivity timeout; completion now comes from backend kernel state
- WebSocket reconnect now rehydrates queue/kernel state from the backend snapshot instead of inferring readiness from queue emptiness alone
- Stored Colab sessions are validated on startup before Dialeng treats them as authenticated, and invalid/revoked sessions are cleared instead of being reused silently
- Keep-alive is now activity-aware, using live browser-client counts and recent kernel activity to avoid keeping long-idle Colab runtimes warm unnecessarily
- Repeated keep-alive and proxy-token-refresh failures now trigger a Colab connection recycle so the next execute reconnects against a clean runtime/session
- Project-path setup and exported-module sync now flow through `BaseKernel` / `KernelService` abstractions, so local and Colab kernels share one setup contract and future remote kernels can plug into the same hooks
- Server logs for CRAFT discovery, LIB path injection, LIB sync, and per-CRAFT execution now include runtime id, generation, file counts, byte counts, sample paths, and durations
- Final OOB rendering and notebook saves now normalize `update_display_data` and `clear_output` events, so fastprogress/tqdm progress bars keep their last visible state instead of collapsing into blank `<progress value="0">` placeholders after execution
- Formatter-only IPython `__repr__ returned non-string` errors are now treated as benign when the cell also produced rich display output, preventing fastai/fastprogress visual cells from ending in duplicate error tracebacks while still rendering their images/widgets

#### Interactive HTML Widgets Not Rendering on Re-run
- Code cells producing HTML with `<script>` tags (e.g., YouTube embeds, custom JS visualizations) only worked on the first execution — subsequent runs showed an empty output
- Root cause: the OOB swap that finalizes cell output uses `replaceWith()`, which does not execute `<script>` tags. On first run, async API loading (e.g., YouTube IFrame API) happened to fire after the OOB swap. On re-runs, the API was already loaded and created the widget synchronously during streaming, but the OOB swap then destroyed it
- Fix: `processOOBSwap()` now clones `<script>` tags into fresh elements after replacing `output-*` divs, mirroring the pattern already used in `appendDisplayData()` during streaming

#### Uvicorn Reload Loop on Generated Files
- Running `!pip install` or other commands that create files in `.venv`/`.autorun_modules` triggered uvicorn's file watcher, causing infinite reload loops
- Fix: `serve()` now uses `reload_dirs=["dialeng"]` to only watch the source code directory instead of excluding patterns from a broad watch

#### Terminal Logging Visibility
- Dialeng now installs a custom Uvicorn log config so `dialeng.*` logger output appears in the terminal alongside startup/server logs
- Access-log noise from static assets, `/render-markdown`, `/favicon.ico`, and repeated `/kernel/snapshot` polling is filtered so Colab/kernel setup logs remain readable during interactive sessions
- Access logs now render with a clearer `[http]` prefix while Dialeng runtime logs include the logger name

#### Floating Kernel Notifications
- Kernel/setup status messages now render as floating toast-style notifications instead of inline messages under the toolbar
- WebSocket reconnect, Colab attach/setup phases, and HTMX responses targeting `#status` all use the same floating notification region
- Backend snapshot setup details now surface as progressive Colab setup toasts, and the inline status area no longer pushes notebook content down while the kernel is attaching
- The toast region now anchors to the bottom-right corner of the window instead of the top-right corner

#### Server Shutdown Kernel Cleanup
- Added an app-level shutdown hook that cancels notebook setup/sync work, cancels queued execution, and asynchronously shuts down all kernels during server exit
- `Ctrl+C` in the terminal is now the supported clean-exit path for Dialeng, and Colab kernels should release their Jupyter sessions and runtime assignments during that shutdown instead of being left active
- `KernelService` now has async `shutdown_all_async()` support, and Colab kernel removal also clears the `ColabSessionManager` cache so later reconnects do not reuse a stale shutdown kernel object

#### Cell Structure Consistency
- Cell move broadcasts now send the backend-authoritative notebook order instead of only an `up`/`down` direction
- The browser now reorders existing cell DOM nodes from that canonical order, which fixes prompt cells with generated output getting stuck when moved down and fixes neighboring cells being unable to cross back above them
- Cell move requests now return an empty HTMX response, and the browser rebuilds `#cells` from a stable snapshot of existing cell/add-row pairs, preventing duplicate add-row controls, move-induced scroll wiggles, and broken follow-up reordering after prompt-cell moves
- Cell add/delete broadcasts now also carry `ordered_cell_ids`, and the browser reconciles add/delete/move through the same structure-sync path instead of three separate DOM heuristics
- Keyboard add shortcuts no longer use a local HTMX `#cells` swap; they now use the same WebSocket-backed add flow as button clicks and prompt/code auto-insertions

#### Toolbar Dropdown Clipped by Overflow
- YT capture button's dropdown panel was invisible because it rendered inside `.toolbar-right` which has `overflow-y: hidden`
- Fix: panel is now appended to `document.body` with `position: fixed` and positioned via `getBoundingClientRect()`

#### Code Cell Default Output Collapse
- New code cells were created with `output_collapse=1` (scrollable), which is a rendering detail that shouldn't be a default
- Removed `output_collapse=1` from all `Cell()` creation sites (add cell, extract code blocks, auto-add after run)

#### Progress Bar and Streaming Output Improvements
- **tqdm Unicode bars**: `StreamingStdout` now exposes `encoding='utf-8'`, so tqdm uses Unicode block characters (`█`) instead of ASCII (`#`)
- **tqdm bar width**: Sets `COLUMNS=120` in the kernel environment so tqdm renders a reasonable bar width (fallback when `ioctl(TIOCGWINSZ)` fails on non-real FDs)
- **ANSI escape stripping**: Both server-side (`ansi_to_html` in `mime.py`) and client-side (`ansiToHtml` in `app.js`) now strip non-SGR ANSI sequences (cursor control `\x1b[A`, erase `\x1b[2K`, private modes `\x1b[?25h`) that were rendering as visible garbage text
- **Mixed `\n`/`\r` streaming**: `appendCodeOutput()` now correctly handles chunks containing both newlines and carriage returns (e.g., pip output with "Collecting...\n" followed by `\r`-based progress bars), preventing permanent output from being overwritten

#### Colab Kernel Not Showing in Modal on Startup
- Colab kernel option was missing from the "Select Kernel" modal even when `colab.enabled: true` in config
- Root cause: Colab initialization ran at module import time, before the real config file was loaded — it always read in-memory defaults where `colab.enabled=false`
- Extracted initialization into `_init_colab()`, called from both `set_root_dir()` (main process) and `_autorun_startup()` (Uvicorn worker process) after the real config is loaded
- The Uvicorn reloader spawns a worker process that re-imports the module but doesn't call `set_root_dir()`, so `_autorun_startup()` also needed the `_init_colab()` call

#### Colab Kernel Restart 500 Error
- Restarting a Colab kernel when Google's API returns 503 (or other errors) crashed the endpoint with a 500 Internal Server Error
- Added error handling around `kernel_service.restart_async()` — now broadcasts `kernel_error` status (red dot) and returns a user-friendly error message

#### Kernel Restart Status Dot
- Kernel status dot now turns green after restart completes (previously stayed yellow)
- Removed "Kernel restarted" flash message (dot color is sufficient feedback)

#### Package Scaffold with Existing pyproject.toml
- `dialeng package init` now merges into existing `pyproject.toml` instead of erroring
- Skips gracefully if `[tool.nbdev]` section already exists
- `scan_notebook_modules` skips the configured lib directory, not just hardcoded `_lib`

#### Extension Action Endpoint
- Fixed `/dialeng/{nb_id}/ext/{action_name}` endpoint not passing form parameters to action handlers (FastHTML ignores `**kwargs`; now uses `request.form()`)

#### Kernel-First Dialog Opening
- Kernel selection modal now shows automatically when opening a notebook without an attached kernel
- Users can browse notebook content without selecting a kernel (view-only mode)
- Attempting to run a code or prompt cell without a kernel triggers the modal; after selection, CRAFT code cells execute first, then the pending cell runs
- Kernel status dot (next to notebook title) now reflects initialization state: grey (no kernel), yellow (initializing/CRAFT running), green (connected), red (error)
- New `POST /dialeng/{nb_id}/kernel/craft-init` endpoint for on-demand CRAFT code cell execution after kernel selection
- Server-side guard on cell run endpoint returns `HX-Trigger: kernel-required` if no kernel is attached

### Changed

#### Toolbar Layout
- `.toolbar-right` now uses horizontal scroll (`overflow-x: auto`) with hidden scrollbar instead of `flex-wrap`
- Settings button moved to first position in right toolbar group for easier access
- Mobile responsive styles simplified (removed `flex-direction: column`)

#### Notebook Rename
- `notebooks/yt-companion/` renamed to `notebooks/yt-insights/`

#### URL Path Rename
- All routes changed from `/notebook/` to `/dialeng/` (e.g., `http://localhost:8000/dialeng/?name=test_capture`)

#### Kernel Selection UX
- Kernel modal closes immediately on Apply (optimistic UI) — kernel setup runs in background
- Removed "Kernel: Local Python" status bar message after kernel selection (redundant with toolbar indicator)
- Kernel status dot increased from 8px to 12px for better visibility
- CRAFT code cells are no longer auto-executed on page load; deferred until explicit kernel selection

### Added (previous)

#### Output Rendering Utilities (`dialeng/ui/mime.py`)
- Extracted `ansi_to_html()` from `app.py` into shared `dialeng/ui/mime.py` module, used by both server-side OOB rendering and WebSocket streaming
- Extracted `render_mime_bundle()` from `app.py` into same shared module for MIME bundle → HTML conversion (text/html, images, SVG, markdown, LaTeX, JSON)

#### Prompt Cell Streaming Feedback
- Added `prompt_stream_start` WebSocket message sent when a prompt cell begins LLM generation
- Provides visual "Generating..." indicator and cancel button for prompt cells created programmatically (e.g., via `_add_msg_unsafe` with `run_mode='run'`), where no UI button click would otherwise trigger the streaming state

### Changed

- `CellOutputOOB` now uses `_render_cell_outputs()` for structured output rendering (display_data, stream, error) instead of wrapping `cell.output` in a single `<pre>` tag
- `finalize_cell_execution()` no longer flattens `cell.outputs` to a string — preserves structured `CellOutput` objects (display_data with MIME bundles, etc.)
- Monaco editor height auto-resize now uses `requestAnimationFrame` debouncing to prevent layout feedback loops when typing bracket-completing characters

### Fixed

- **`IPython.display.HTML()` producing empty output cells** — `finalize_cell_execution()` was using `cell.output =` setter which destroyed structured `CellOutput` objects (display_data, MIME bundles) by replacing them with a single stream output. The OOB swap then re-rendered the destroyed state. Fixed by preserving the original `cell.outputs` list and using `_render_cell_outputs()` in both `CellOutputOOB` and `CodeCellView`
- **`execute_result` with rich MIME types not rendering** — `subprocess_kernel.py` now promotes `execute_result` containing rich types (text/html, image/png, etc.) to `display_data`, ensuring objects returned by `IPython.display.HTML()`, pandas DataFrames, and PIL Images render inline
- **tqdm progress bar showing full history in final output** — Added `_process_carriage_returns()` that processes `\r` (carriage return) to collapse intermediate progress updates, showing only the final completed bar
- **tqdm/stderr output appearing in red during streaming** — Removed incorrect `error` CSS class addition on stderr stream chunks; many tools (tqdm, warnings, logging) write to stderr without it being an error
- **Monaco editor visual glitch when typing `"`** — Auto-bracket completion triggered a layout feedback loop (resize → automaticLayout → word-wrap recalc → content size change → resize). Fixed by debouncing `updateEditorHeight` via `requestAnimationFrame` and skipping no-op height changes

#### Subdirectory Notebook Navigation & Clean URLs
- Subdirectory notebooks now open correctly from the file explorer via `?name=path/to/notebook` query parameter URLs instead of `_`-encoded path IDs
- Added `nbApiPath()` JS helper that uses `window.NOTEBOOK_ID` for reliable API calls regardless of URL format
- Added `_render_notebook_page()` shared rendering function used by both `/dialeng/{nb_id}` and `/dialeng/?name=...` routes
- Added `_nb_id_from_path()` and `_find_notebook_by_name()` for bidirectional path/ID resolution

#### Demo Project for CRAFT/TEMPLATE
- Created `notebooks/demo_project/` with hierarchical CRAFT and TEMPLATE examples
- Parent-level `CRAFT.ipynb` with project banner and shared imports
- Child `data_analysis/CRAFT.ipynb` with data science stack (numpy, pandas, matplotlib)
- Parent and child `TEMPLATE.ipynb` files demonstrating cell prepending hierarchy

### Changed

- File explorer links now use `href="/dialeng/?name=..."` with human-readable paths instead of `_`-encoded IDs
- `_jupyter_to_cell()` in `document/serialization.py` now resolves cell IDs with fallback chain: `metadata.id → cell-level id → random UUID` for stable cell ID round-tripping with standard Jupyter notebooks

### Fixed

- **Subdirectory notebooks not opening** — `_find_notebook_path()` now uses rglob + `_nb_id_from_path()` comparison instead of searching for `{encoded_id}.ipynb` on disk
- **Cell buttons and Shift+Enter broken after URL change** — replaced all 10 `window.location.pathname` usages in `app.js` with `nbApiPath()` helper
- **Subdirectory notebooks connecting to wrong kernel** — `get_notebook()` and `_find_notebook_by_name()` now set `nb.id` to the `_`-encoded notebook ID after loading, preventing identity mismatch (e.g., `template_test_hello_test` vs `test`)
- **CRAFT double execution on page load** — HTMX causes two GET requests per navigation; both would race to launch CRAFT execution. Fixed by marking all CRAFT cell IDs in `_executed_craft` synchronously before dispatching the async task
- **CRAFT.ipynb executing its own code** — opening a CRAFT notebook no longer auto-executes its own code cells; only parent CRAFT files in the hierarchy are executed (self-exclusion via path comparison)
- **CRAFT cell ID instability** — `_jupyter_to_cell()` previously only checked `metadata.id`, generating random UUIDs for standard Jupyter cells that store `id` at cell level. Fixed fallback chain prevents CRAFT execution tracking from breaking across reloads
- **CRAFT not re-executing after kernel restart** — kernel restart route now calls `reset_craft_tracking()` and re-triggers CRAFT execution with the same self-exclusion and synchronous marking protections
- **Layout broken after Titled() removal** — restored `Titled()` in `ui/layout.py` (provides `<main class="container">` wrapper needed for sidebar layout), added CSS to hide the H1

#### safepyrun Integration
- Added `pyrun` as a built-in LLM tool for safe sandboxed Python execution via [safepyrun](https://github.com/AnswerDotAI/safepyrun)
- The AI can now execute Python code safely during prompt responses without the `&` prefix
- Allowlist-based sandbox with curated stdlib access, `_` suffix variable persistence, and controlled write permissions (`ok_dests=['.']`)
- Added async support to `execute_builtin()` in `tool_registry.py` for async callable tools
- New demo notebook: `notebooks/safepyrun_demo.ipynb`
- New docs: `docs/how_it_works/18_safepyrun_integration.md`

#### mistlefoot Markdown Rendering
- Replaced `markdown-it-py` with [mistlefoot](https://github.com/AnswerDotAI/mistlefoot) for extended markdown rendering
- New markdown features: subscript (`H~2~O`), superscript (`E=mc^2^`), `==highlighting==`, `~~strikethrough~~`, emojis (`:rocket:`), footnotes, task lists, heading attributes
- Added `/render-markdown` server endpoint for full-fidelity rendering
- Hybrid rendering strategy: fast client-side JS preview during editing + server-side mistlefoot for final output
- New demo notebook: `notebooks/mistlefoot_demo.ipynb`
- New docs: `docs/how_it_works/19_mistlefoot_rendering.md`

### Changed

#### GUI Smoothness Optimizations (Phase 1, 2 & 3)

> **FOUST** = Flash of Unstyled Text. Monaco Editor renders code as plain white text first, then asynchronously tokenizes via a web worker to apply syntax highlighting. If the editor DOM is destroyed and recreated (e.g., by an HTMX swap), there's a visible flash of white text before highlighting reappears.

- **Targeted OOB swaps for execution** — code cell execution now broadcasts `CellOutputOOB` + `CellHeaderOOB` instead of full `CellViewOOB`, preserving the Monaco editor DOM and eliminating FOUST on run
- **JSON WebSocket messages for source edits** — dialoghelper source operations (`msg_insert_line_`, `msg_str_replace_`, etc.) now send `cell_source_update` JSON messages that update Monaco via `editor.setValue()` instead of replacing the cell DOM
- **JSON WebSocket messages for state/class updates** — state toggles and collapse send `cell_class_update` JSON messages instead of full cell replacements
- **Granular cell add** — `cell_add` JSON message inserts a single cell + add-row via `insertAdjacentHTML` instead of replacing the entire `#cells` container. Applied to `/cell/add`, `/add_relative_`, `/msg_paste_`, and auto-add after last cell run
- **Granular cell delete** — `cell_delete` JSON message removes a single cell + adjacent add-row from DOM instead of replacing all cells. Applied to `DELETE /cell/{cid}`, `/rm_msg_`, and clipboard cut
- **Granular cell move** — `cell_move` JSON message swaps two adjacent cells in DOM via `insertBefore` (which moves nodes without copying, preserving Monaco editors) instead of replacing all cells
- **Collapse-section via JSON** — `cell_collapse_update` JSON message reuses the existing `setCollapseLevel()` function to update CSS classes instead of full cell OOB swap
- **Prompt cell completion via targeted OOB** — prompt cell completion now broadcasts `CellHeaderOOB` + `cell_class_update` instead of full `CellViewOOB` (output already streamed via `stream_chunk`/`stream_end`)
- **Stable header IDs** — cell headers now have `id="header-{cell.id}"` enabling targeted header-only OOB swaps
- **Removed inline Script tags** — code and shell cells no longer emit `<Script>` tags for editor initialization; editors initialize via `initCell()` called from `htmx:afterSettle`
- **Debounced streaming output** — code cell output uses `requestAnimationFrame`-based batching instead of per-chunk DOM writes, with smart auto-scroll
- **AbortController for event listener cleanup** — `initCell()` uses `AbortController` to prevent listener accumulation on re-init
- **WebSocket exponential backoff** — reconnect delay starts at 1s, doubles per failure (capped at 30s), resets on successful connection
- **CSS performance** — `will-change: opacity` on thinking indicator, child combinator (`>`) instead of universal selector (`*`) in `.collapse-summary`, outline-based focus indicator instead of `box-shadow`

#### Monaco Editor Migration

- **Replaced Ace Editor with Monaco Editor** — migrated from Ace 1.32.6 (4 CDN scripts) to Monaco 0.52.2 (AMD loader) for code and shell cells
- **Auto-resize editor** — Monaco editors grow/shrink with content (60px min, 600px max) using `onDidContentSizeChange`, replacing Ace's `minLines`/`maxLines`
- **Improved theme integration** — Monaco switches between `vs-dark` and `vs` themes in sync with Dialeng's dark/light theme toggle
- **Keyboard shortcuts preserved** — Shift+Enter (run + move), Ctrl/Cmd+Enter (run), Ctrl/Cmd+S (save) all work in Monaco via `addAction()` (not `addCommand()`) to properly override built-in Monaco keybindings
- **HTMX lifecycle integration** — editors properly disposed on `htmx:beforeSwap` and reinitialized on `htmx:afterSettle`
- **Pending init queue** — editors requested before Monaco AMD loader finishes are queued and initialized once ready

### Fixed

- **FOUST eliminated for add/delete/move** — granular JSON WebSocket messages (`cell_add`, `cell_delete`, `cell_move`) manipulate DOM directly via `insertAdjacentHTML`/`remove`/`insertBefore`, preserving all Monaco editors across all cells
- **FOUST eliminated for collapse-section** — JSON `cell_collapse_update` message updates CSS classes in-place via `setCollapseLevel()` instead of replacing the full cell DOM
- **FOUST eliminated for prompt completion** — targeted `CellHeaderOOB` + `cell_class_update` instead of full `CellViewOOB`
- **FOUST eliminated for code cell execution** — targeted OOB swaps (`CellOutputOOB` + `CellHeaderOOB`) replace only the output and header sections, leaving the Monaco editor DOM untouched
- **FOUST eliminated for `htmx:beforeSwap` on no-swap responses** — `htmx:beforeSwap` no longer destroys Monaco editors when `hx_swap="none"` (code cell run returns empty response); previously the editor was destroyed and recreated for no reason
- **FOUST eliminated for `htmx:afterSettle` re-initialization** — `initMonacoEditor()` now checks if the container already has a live `.monaco-editor` element and skips re-initialization, preventing unnecessary editor destruction
- **Race condition: debounced RAF vs OOB output** — pending `requestAnimationFrame` from streaming is cancelled in `finishCodeStreaming()` before OOB output arrives, preventing empty content from overwriting server-rendered output
- **Cell focus not moving to non-code cells** — `focusNextCell()` now explicitly moves DOM focus (`cell.focus()`) for note and prompt cells, so Shift+Enter correctly advances past non-code cells instead of re-running the previous code cell
- **Shell cells not initialized after inline Script removal** — `initCell()`, `processOOBSwap()`, and `reinitializeMonacoEditors()` now handle `data-type="shell"` cells
- **Scroll position preserved on cell operations** — adding, deleting, and moving cells no longer jumps the notebook to the bottom of the page; scroll position is saved before HTMX `outerHTML` swaps and restored after Monaco editors reinitialize
- **Scroll passthrough from Monaco editors** — mouse wheel events now propagate to the notebook when the editor content is fully scrolled, via `alwaysConsumeMouseWheel: false`

### Known Issues

- **FOUST on cell type change** — this still uses `CellViewOOB` which replaces the full cell DOM; this is inherent since the input section fundamentally changes (Monaco ↔ textarea)

### Added

#### Kernel-Backed Code Completion

- **Python code completion in Monaco** — `CompletionItemProvider` registered for Python language, triggered on `.` character and manual invoke
- **Completion endpoint** — `POST /api/complete/{nb_id}` accepts `code` and `cursor_pos`, returns matching completions from the kernel
- **Kernel completion pipeline** — `KernelService.complete()` → `SubprocessKernel.complete()` → kernel worker → `CaptureShell.complete()`
- **Busy guard** — completions return empty while the kernel is executing code, preventing queue message interleaving
- **Fixed kernel_worker.py completion handler** — `CaptureShell.complete()` takes 1 argument (code string) and returns `list[str]`, not a tuple; fixed the handler to match

#### Package Restructuring

- **Moved all source code into `dialeng/` package directory** — proper Python package layout with a single top-level package instead of multiple loose packages (`core`, `document`, `services`, `ui`, `extensions`) at the repo root
- **Moved test files into `tests/` directory** — `test_integration.py`, `test_kernel.py`, `test_stateless_dialoghelper.py`
- **Simplified `pyproject.toml`** — entry point is now `dialeng.app:main`, build config is `packages = ["dialeng"]` (no more `include` hacks)
- **Updated all imports** — cross-package imports now use `from dialeng.X import ...` (e.g., `from dialeng.core.registry import registry`)
- **Static file serving** — now resolves paths relative to the package directory via `Path(__file__).parent`

#### Breaking Changes

- **User-authored extensions** in `AUTORUN/` or `extensions/` that import from the project must update their imports: `from core.registry import ...` becomes `from dialeng.core.registry import ...`. The same applies to all modules (`services`, `document`, `ui`, `extensions`, `app`, `state`).

### Added

#### GUI Aesthetic Overhaul & Display Settings

**Display Settings**
- **Dialeng Display Settings** group added to settings sidebar (positioned at the top for easy access):
  - **Notebook Width (px)**: Configurable container max-width (600-3000px, default 1400)
  - **Button Size**: Compact / Normal / Large — scales all buttons across toolbar, cells, file explorer, and dropdowns via CSS custom properties (`--btn-padding`, `--btn-sm-padding`, `--btn-font-size`)
  - **Font Size (px)**: Base UI font size (10-24px, default 15)
  - **Reasoning Text Limit**: Max characters for LLM reasoning display
- All display settings persist in `dialeng_config.json` under the `display` section
- CSS uses `var()` references so new buttons automatically inherit sizing — `button, select { font-size: var(--btn-font-size, inherit); }` base rule

**Kernel Status Dot**
- Green dot next to notebook title in toolbar indicating kernel connection state:
  - **Grey** (default): No kernel connected
  - **Green**: Kernel alive and idle
  - **Yellow**: Cells running/queued, or kernel restarting
  - **Red**: Execution error (flashes for 3s, then returns to green)
- Status is set server-side on page load (checks `kernel_service.kernel_is_alive()`), so returning to a notebook with a running kernel shows green immediately
- Real-time updates via existing WebSocket messages (`queue_update`, `kernel_connected`, `kernel_restarting`, `code_stream_end`)
- New `broadcast_kernel_status()` helper in `app.py` for kernel lifecycle events

**File Explorer Kernel Indicators**
- Notebook icons in the file explorer turn green when that notebook has a running kernel
- Uses `.has-kernel` CSS class with `stroke: var(--accent-green)` on the SVG icon
- All file list routes (`/files`, `/files/new-folder`, `/files/delete`) query `kernel_service` for alive kernels

**File Explorer Enhancements**
- **Refresh button**: Refresh icon in file explorer header to reload the file list
- **Delete button**: Per-file trash icon (appears on hover) with confirmation modal
- **New Item Modal redesign**: Name input + type toggle (Dialog / Folder) replacing the old two-button approach
- File explorer width increased to 320px (matching outline sidebar)

**Settings UI Improvements**
- Setting rows now have separator borders and more padding for readability
- Toggle rows use CSS Grid layout: label+badge on left, toggle on right, help text spanning full width below
- Restart badge restyled as a compact pill
- Group content has increased top padding

**CSS Architecture**
- Sidebar borders only appear when open (no visible line when collapsed)
- Toolbar is a rounded card container with sticky positioning and backdrop blur
- Theme toggle SVG uses `stroke: var(--text-primary)` for visibility in both themes
- All keyboard shortcuts use `e.ctrlKey || e.metaKey` for macOS Cmd key compatibility

#### Extensibility & Packaging (Phases 1-8)

**Phase 1: Package Structure (uv project)**
- **`pyproject.toml`** — Project is now a proper `uv` project with hatchling build backend. All dependencies moved from `requirements.txt`. Entry point: `uv run dialeng` or `uv run python -m dialeng`.
- **`__main__.py`** — Enables `python -m dialeng` execution.
- **Configurable paths** — `DIALENG_NOTEBOOKS_DIR` and `DIALENG_CONFIG_PATH` environment variables for customizing notebook/config directories.
- **Removed `requirements.txt`** — Replaced by `pyproject.toml` dependencies.

**Phase 2: Registry Extensions (Kernel + Provider + Toolbar + Settings)**
- **Kernel registry** (`core/registry.py`) — `KernelRegistration` dataclass with `register_kernel_type()` method and `@register_kernel` decorator. Local and Colab kernels self-register on import.
- **LLM provider registry** — `ProviderRegistration` dataclass with priority-based selection. Claudette, ClaudetteAgent, and ClaudeAgentSdk providers self-register.
- **Toolbar extension point** — `ToolbarItemRegistration` with `position` and `order` fields. Extensions can add toolbar buttons via `@register_toolbar_item`.
- **Settings extension point** — `SettingsSectionRegistration` rendered after built-in settings groups. Extensions can add settings sections via `@register_settings_section`.
- **Registry-based kernel switching** — `KernelService.get_kernel()` uses registry lookup instead of hardcoded if/elif.
- **Registry-based provider selection** — `LLMService._ensure_initialized()` uses registry lookup.

**Phase 3: TEMPLATE.ipynb Support**
- **`services/template_service.py`** — `find_templates()` walks up directory tree collecting `TEMPLATE.ipynb` files (parent-first). `load_template_cells()` loads cells with fresh UUIDs.
- **New notebook creation** — `/dialeng/new?dir=` uses template cells when available.

**Phase 4: CRAFT.ipynb Support**
- **`services/craft_service.py`** — `find_craft_files()` walks up directory tree. `get_craft_context()` extracts note/prompt cells as LLM messages. `get_craft_code_cells()` extracts code cells for kernel execution.
- **Context prepending** — CRAFT context is prepended to LLM messages in `build_context_messages()`.
- **Auto-execute** — CRAFT code cells auto-execute in background on notebook open.

**Phase 5: nbdev Export Integration**
- **`#| export` sync** — `Cell.sync_export_directive()` keeps `#| export` in sync with `is_exported` flag.
- **Load detection** — Loading notebooks with `#| export` in source auto-sets `is_exported=True`.
- **`default_export_module` property** — `Notebook.default_export_module` parses `#| default_exp` directive.
- **Extension extraction** — `extract_extension()` now uses `#| export` as default marker.

**Phase 6: AUTORUN Folder Support**
- **`services/autorun_service.py`** — Two-phase startup: extract `#| export` cells from AUTORUN notebooks to `.autorun_modules/`, load `.py` extensions, then run notebooks in background kernels.
- **Startup integration** — `@app.on_event("startup")` calls `process_autorun()`.
- **`.autorun_modules/`** added to `.gitignore`.

**Phase 7: Kernel Selection Redesign**
- **`ui/kernel_modal.py`** — `KernelStatusBar` (bottom of page, shows kernel type and connection state) and `KernelModal` (overlay with all registered kernels, runtime options, auth status).
- **New routes** — `GET /dialeng/{nb_id}/kernel/info` (status bar refresh), `GET /dialeng/{nb_id}/kernel/modal` (modal content).
- **Removed old toolbar dropdown** — Kernel type/runtime select removed from toolbar in favor of status bar + modal.
- **kernel-connected event** — WebSocket broadcasts `kernel_connected` after first successful execution; client refreshes status bar.

**Phase 8: File Explorer**
- **`ui/file_explorer.py`** — `FileExplorerSidebar` with directory navigation, breadcrumbs, folder/notebook items with lucide icons. `NewItemModal` for creating notebooks/folders.
- **New routes** — `GET /files?path=` (directory listing), `POST /files/new-folder` (create folder).
- **Keyboard shortcut** — `Ctrl+Shift+E` toggles file explorer.
- **Replaced flat file list** — Old `Div.file-list` replaced by collapsible sidebar with breadcrumb navigation.
- **Lucide icons** — Added `house-plug`, `microchip`, `cpu`, `monitor`, `zap`, `notebook`, `notebook-text`, `folder`, `folder-open`, `file-plus`, `folder-plus`, `panel-left-close`, `panel-left-open`, `chevron-right`, `circle`, `check`.

### Changed

#### LLM Service Provider-Based Architecture Refactor
- **Modularized `services/llm_service.py`** — Refactored the ~1540-line monolithic LLM service into a provider-based architecture under `services/llm/` package. Each LLM backend (claudette, claudette-agent, claude-agent-sdk) is now a separate provider class implementing `BaseLLMProvider` ABC.
- **`BaseLLMProvider` ABC** (`services/llm/base_provider.py`) — Abstract base class with `ProviderInfo`/`LLMResult` dataclasses, following the existing `BaseKernel` pattern. Defines the contract for `stream()`, `stream_with_tools()`, `check_thinking_support()`, `get_info()`, and `initialize()`.
- **Three provider implementations** — `ClaudetteProvider` (API/Bedrock), `ClaudetteAgentProvider` (Claude Code subscription wrapper), `ClaudeAgentSdkProvider` (direct SDK with MCP tool support and real-time `StreamEvent` streaming).
- **Slim coordinator** (`services/llm/llm_service.py`) — `LLMService` class with the same public API (`stream_response`, `stream_response_with_tools`, `get_provider`, `last_usage`, `last_cost`). Owns provider selection, mode/model resolution, prompt parsing, and tool registry orchestration.
- **Shared utilities** (`services/llm/utils.py`) — Extracted stateless helper functions: `build_prompt_with_context`, `execute_tool`, `format_tool_result_for_llm`, `save_debug_log`, `chunk_text`, `build_text_tool_definitions`, `parse_text_tool_calls`, `build_tool_results_prompt`.
- **Compatibility shim** — `services/llm_service.py` remains as a thin re-export so existing `from services.llm_service import ...` imports continue to work.
- **Replaced `print()` debug statements** — Provider implementations use `logger.debug()` instead of `print()` for debug output.
- **Updated documentation** — `docs/how_it_works/06_llm_integration.md` updated with provider architecture diagrams, `BaseLLMProvider` ABC docs, event dict protocol, and a step-by-step "How to Add a New LLM Provider" guide.

#### Image Handling in LLM Context
- **Multimodal context messages** (`core/dispatch.py`) — Code cell outputs with images (screenshots, plots) now produce Anthropic-format image content blocks instead of raw base64 HTML. `_extract_image_blocks()` extracts from both structured `display_data` outputs and post-finalization HTML `<img>` tags. `_resize_base64_image()` resizes to max 1024px and re-encodes as JPEG to keep prompts within token limits.
- **Image-aware providers** (`claudette_provider.py`, `claudette_agent_provider.py`) — Both providers use `_split_context_images()` to separate image blocks from context messages and attach them to the current prompt (user turn only), respecting the Anthropic API constraint that images cannot appear in assistant turns.
- **claudette-agent non-streaming fallback** — When images are present, the provider uses `chat()` (non-streaming) instead of `chat.stream()`, because `chat.stream()` flattens all messages to text via `_build_conversation_prompt()`. `chat()` routes to `_call_with_images()` which sends structured content via stdin transport, avoiding the "Argument list too long" OS limit from CLI arguments.
- **Base64 stripping in text output** — `_get_text_output()` and `_strip_base64_images()` ensure base64 `<img>` tags are removed from text context, preventing prompt bloat after `finalize_cell_execution` flattens structured outputs into HTML.
- **Vision test notebook** (`notebooks/test_capture.ipynb`) — Added vision test cells: capture screen with `capture_tool()`, then prompt the LLM to describe what was captured.
- **Updated documentation** — `docs/how_it_works/06_llm_integration.md` updated with image handling architecture, SDK limitations, and future improvement notes.

### Added

#### DialogHelper v2 Compatibility Update
- **Expanded `find_msgs` endpoint** — New search parameters: `use_case` (case-sensitive), `use_regex` (regex/literal toggle), `only_err` (error cells), `only_exp` (exported cells), `only_chg` (changed cells), `ids` (filter by IDs), `include_output`, `include_meta`, `as_xml`, `nums`, `trunc_out`, `trunc_in`, `headers_only`, `header_section`. Supports both XML and JSON response formats.
- **New text editing endpoints** — `msg_del_lines_` (delete line range with regex filter) and `msg_pyrun_` (execute Python code against cell text)
- **Clipboard operations** — `msg_clipboard_` (copy/cut cells) and `msg_paste_` (paste cells) with per-notebook clipboard storage
- **UI toggle endpoints** — `toggle_header_collapse_` (toggle heading collapse), `bookmark_` (numbered bookmarks 1-9), `toggle_comment_` (toggle line comments on code)
- **Dialog management endpoints** — `create_dialog_` (create/load notebook), `stop_kernel_` (stop execution queue), `rm_dialog_` (delete notebook from memory)
- **RAW cell type** — New `CellType.RAW` for raw content cells
- **Cell model fields** — Added `heading_collapsed` (persisted), `bookmark` (numbered 1-9)
- **Change logging** — `log_changed` parameter on `update_msg_` and `rm_msg_` for audit trail

#### Tracetools Support
- **`tracefunc` dependency** — Added `tracefunc` package to `requirements.txt`, enabling `dialoghelper.tracetools` (function execution tracing via `sys.monitoring`).
- **Markdown rendering for `text/markdown` MIME type** — `render_mime_bundle()` in `app.py` now converts markdown to HTML using `markdown-it-py` (tables, formatting). Previously raw markdown text was wrapped in a plain div.
- **`_repr_markdown_()` rich result promotion** (`kernel_worker.py`) — Objects with `_repr_markdown_()` (e.g., IPython's `Markdown` display) are now auto-promoted to `display_data` with HTML conversion, alongside existing `_repr_png_()` and `_repr_html_()` support.
- **Test notebook** — `notebooks/test_tracetools.ipynb` with 7 cells covering `tracetool()`, `fmt_trace()`, stdlib tracing, `target_func`, and recursive function tracing.
- **Markdown table styling** (`static/css/components.css`) — Added `.mime-markdown` CSS rules for rendered markdown tables: themed borders, header styling with blue accent, alternating row shading, hover highlights, and monospace font. Works in both dark and light themes.

#### Exhash (Hash-Addressed Editor) Support
- **Test notebook** — `notebooks/test_exhash.ipynb` with 13 cells covering `lnhashview()`, `lnhash()`, `line_hash()`, `exhash()` (substitute, delete, insert, append, change, indent, global commands), `exhash_result()`, hash verification, multi-command edits, and file editing.

#### Tmux Tools Support
- **Test notebook** — `notebooks/test_tmux.ipynb` with 14 cells covering `shell_ret()`, `pane()`, `panes()`, `windows()`, `sessions()`, `flatten_dict()`, `set_default_history()`, and cross-pane keyword search. Auto-creates/cleans up a test tmux session.

#### Screen Capture Support
- **Global `pushData()` JS function** (`static/js/app.js`) — Enables dialoghelper's `screenshot.js` to send captured image data back to Python via `/push_data_blocking_`. Uses `URLSearchParams` for proper encoding of base64 data.
- **Rich result promotion** (`kernel_worker.py`) — Cell return values with `_repr_png_()` or `_repr_html_()` (PIL Images, DataFrames, etc.) are automatically promoted from `execute_result` to `display_data` with full MIME bundle, so they render inline.
- **Test notebook** — `notebooks/test_capture.ipynb` with 7 cells covering `setup_share()`, `start_share()`, `capture_screen()`, `capture_tool()`, multiple captures, and file saving.

#### Cell Class Unification
- **Unified Cell class** — Merged the duplicate `Cell` from `app.py` into the single `document/cell.py` Cell, eliminating the fragile duck-typing bridge between UI and execution layers
- **Extracted `document/prompt_utils.py`** — Prompt separator constants and split/join functions moved from `app.py` to a shared module

### Changed

#### Cell Class Unification
- **`CellType` enum** — Added `__str__`/`__format__` so f-string interpolation produces `"code"` not `"CellType.CODE"`
- **Cell `__post_init__` coercion** — Accepts both string (`"code"`) and enum (`CellType.CODE`) for `cell_type`, and both int (`1`) and enum (`CollapseLevel.SCROLLABLE`) for collapse levels
- **Cell ID generation** — Removed underscore prefix from auto-generated IDs to match existing notebook format
- **`output` property setter** — Empty string now clears the outputs list instead of creating an empty CellOutput
- **`clear_outputs()`** — Now also resets `execution_count` and `time_run` (matching original app.py behavior)
- **Removed ~150 lines** of duplicate `Cell`, `CellType`, and `CollapseLevel` definitions from `app.py`

#### DialogHelper Endpoint Updates
- **`add_relative_` placement values** — Now supports `add_after`, `add_before`, `at_start`, `at_end` (with backwards compat for `after`/`before`). Added `run_mode` param alongside existing `run`.
- **`msg_str_replace_` enhanced** — New parameters: `start_line`, `end_line` (line range restriction), `n_matches` (replacement count), `re_filter`, `invert_filter` (regex line filtering)
- **`rm_msg_` response** — Now returns JSON with `id` field alongside `status`
- **`cell_to_dict` expanded** — Now includes `heading_collapsed`, `bookmark`, `is_exported` fields
- **Serialization** — `heading_collapsed` and `bookmark` now persist in `.ipynb` metadata

#### Google Colab Kernel Integration
- **Remote execution on Colab runtimes** — Execute notebook cells on Google's cloud infrastructure (CPU, GPU T4, TPU) via the same APIs as the Colab VS Code extension
- **OAuth2 authentication** — Google sign-in via OAuth2 with built-in credentials (zero configuration); optionally override with `COLAB_CLIENT_ID` and `COLAB_CLIENT_SECRET` environment variables; user tokens persist in `~/.dialeng/colab_tokens.json`
- **Credential auto-validation & auto-update** — At startup, validates OAuth credentials against Google's token endpoint. If Google rotates the built-in credentials, Dialeng automatically extracts updated ones from the published Colab VS Code extension VSIX on Open VSX. Resolution cascade: env vars → validated defaults → cached extraction → live VSIX extraction → fallback with warning
- **Jupyter wire protocol over WebSocket** — Full implementation of Jupyter v5.3 protocol over Colab's multiplexed WebSocket
- **Rich output support** — Matplotlib plots, HTML, images, tqdm progress bars, and interactive widgets render correctly from Colab kernels
- **Kernel initialization** — Automatic `%matplotlib inline` setup on connect so plot rendering works out of the box
- **Background keep-alive & token refresh** — Prevents runtime idle timeout (5-min pings) and refreshes proxy tokens before expiry
- **Stale runtime cleanup** — Automatically unassigns existing runtimes on connect to avoid `TooManyAssignmentsError`
- **Namespace introspection** — Variables/functions panel works with Colab kernels via remote code introspection
- **Multi-kernel switching** — Users can switch between local Python and Colab per notebook through the `BaseKernel` interface

#### Course Study Assistant Notebook
- **YouTube transcript pipeline** — Download any YouTube video, extract subtitles, chunk transcript, and use LLM to identify chapters/sections with timestamps (parallel processing with 32 workers)
- **9 interactive study tools** — `load_course_video`, `search_transcript`, `list_chapters`, `get_chapter`, `play_video_at`, `load_fastbook_chapter`, `load_docs`, `create_experiment`, `search_notebook`
- **Video embedding** — Embed YouTube video at specific timestamps directly in the notebook via `play_video_at`
- **Fastbook integration** — Load any of the 20 fastbook chapters on demand with local caching (`~/.dialeng/cache/`)
- **Documentation loading** — Access fastcore, fasthtml, fastlite, docker, and claudette docs via `contextpack` with compatibility fix for contextkit's `read_url` → `read_link` rename
- **LLM provider abstraction** — `llm_complete()` wrapper that uses claudette first, falls back to direct Anthropic SDK for transcript processing

### Changed
- **Colab OAuth credentials** — Built-in credentials validated at startup and auto-updated if rotated; env vars are now optional overrides; added `.env.example` template and `.env` to `.gitignore`
- **"Connect Google" → "Connect Colab"** — Renamed the authentication button; button now auto-updates to "Disconnect" after OAuth without page refresh (via `postMessage` from popup)

### Fixed
- **Matplotlib plots not rendering from Colab kernel** — Fixed race condition where `execute_reply` (Shell channel) could arrive before `display_data` (IOPub channel) on Colab's multiplexed WebSocket. The execution loop now waits for `status: idle` instead of `execute_reply`, ensuring all IOPub outputs (including plots) are delivered before the loop exits.

### Documentation
- Added `docs/how_it_works/12_colab_kernel.md` — Comprehensive technical documentation covering the Colab kernel architecture, authentication flow, credential resolution, connection lifecycle, Jupyter wire protocol details, the multiplexed WebSocket subtlety, and rich output pipeline
- Added `docs/guides/colab_oauth_setup.md` — Step-by-step guide for users to create their own Google OAuth2 credentials when auto-update is not possible
- Updated `docs/how_it_works/04_kernel_execution.md` — Added "Rich Result Promotion" subsection documenting how PIL Images and DataFrames auto-promote to `display_data`
- Updated `docs/how_it_works/05_dialoghelper_integration.md` — Added `pushData()` global function docs, Screen Capture section, Tracetools section, Tmux Tools section, Markdown Rendering Pipeline with Mermaid diagram, and test notebook entries

#### Shell Command Execution (pshnb + safecmd)
- **Shell cells (optional)** - New cell type for dedicated bash command execution:
  - Disabled by default - enable via Settings > Shell Settings > "Enable Shell Cells"
  - Bash syntax highlighting (Ace editor `sh` mode)
  - Fresh shell session per cell (no persistent state across cells)
  - "Shell" badge in cell header
  - Safe mode indicator when enabled
- **%bash magic in code cells** - pshnb extension enables `%bash`, `%%bash`, and `!command` in regular code cells (always available, even when shell cells are disabled)
- **Variable expansion** - Use Python variables in shell commands with `@{variable}` syntax:
  ```python
  name = "World"
  %bash echo "Hello, @{name}!"  # Outputs: Hello, World!
  ```
- **Safe Mode** - Notebook-level toggle to validate shell commands against safecmd allowlist:
  - Enable via "Safe" checkbox in toolbar
  - Blocks dangerous commands like `rm`, `sudo`, `chmod`
  - Allows read-only operations: `ls`, `cat`, `grep`, `git status`, etc.
  - Blocks output redirection (`>`, `>>`) to prevent file overwrites
- **shfmt dependency check** - Server startup checks for `shfmt` binary:
  - Warning message with installation instructions if missing
  - Safe Mode toggle disabled in UI when shfmt unavailable
  - Shell execution still works (pshnb doesn't require shfmt)
- **SSH remote execution** - Support for running commands on remote hosts via pshnb's SSH integration

#### New Files
- `services/shell_service.py` - Shell execution service wrapping pshnb and safecmd
- `ui/cells/shell_cell.py` - Shell cell UI component
- `extensions/shell_cell.py` - Shell cell type registration and execution callback
- `notebooks/pshnb_guide.ipynb` - pshnb usage guide
- `notebooks/safecmd_guide.ipynb` - Safe Mode and security guide
- `notebooks/shell_integration.ipynb` - Complete integration overview
- `docs/how_it_works/11_shell_integration.md` - Technical documentation

#### Dependencies
- Added `pshnb>=0.0.4` - Persistent shell for notebooks (like Jupyter %bash magic)
- Added `safecmd>=0.1.2` - Safe command execution with allowlist validation

#### Settings GUI Sidebar
- **Settings sidebar** - New collapsible sidebar accessible via ⚙️ button in toolbar for viewing and editing `dialeng_config.json` settings
- **Collapsible settings groups** - Settings organized into collapsible sections using `<details>` elements:
  - AWS Settings (region selection)
  - Model Defaults (Bedrock model, Claude Code model, dialog mode)
  - Tool Settings (max steps slider, confirmation toggle, builtin tools toggle)
  - Display Settings (reasoning truncation limit)
  - Shell Settings (enable/disable dedicated shell cell type)
  - Advanced (thinking tokens, SDK direct mode, debug mode, debug log directory)
- **Form controls** - Various input types for different setting types:
  - Dropdown selects for enumerated options (regions, models, modes)
  - Toggle switches for boolean settings with restart warning badges
  - Number inputs for numeric values
  - Range sliders for bounded values (max steps 1-10)
  - Text inputs for string values
- **Settings persistence** - Changes saved to `dialeng_config.json` via POST `/settings` endpoint
- **New functions in `dialeng_config.py`**:
  - `save_config()` - Save complete config dict to file
  - `update_config()` - Deep merge updates into existing config
  - `get_config_dict()` - Get raw config dict for settings UI

#### Model Selection Validation
- **Per-notebook model persistence** - Each notebook remembers its selected model in metadata
- **Model validation on load** - Validates saved model exists in available models; falls back to provider default if invalid
- **Provider-aware defaults** - New notebooks use the default model for the detected provider (Bedrock vs Claude Code)
- **Graceful config changes** - If config changes and saved model becomes invalid, notebooks fall back to defaults instead of breaking

### Fixed

#### Response Deduplication False Positive Fix
- **Fixed truncation of legitimate responses** - The `_deduplicate_response_text()` function was incorrectly truncating responses at ~50 characters due to matching single characters like "," during partial overlap detection. Now requires minimum 20-character overlap for detection
- **Added comprehensive documentation** - Function docstring now includes bug fix history and future improvement suggestions

#### Bedrock Tool Calling Fixes
- **Fixed tools not being passed to claudette** - Tools must be passed to the Chat constructor, not in individual calls
- **Fixed tool call detection** - Tool calls are now extracted from `stream_result.value` after streaming completes (per claudette docs)
- **Fixed tool result format** - Tool results are updated in-place to preserve claudette's AttrDict format instead of replacing with plain dicts

### Changed

#### Provider-Specific Default Models
- **Default models are now per-provider** - Configuration now supports different default models for each provider:
  - `bedrock`: Default to cheaper models (e.g., Haiku) for pay-per-use
  - `anthropic_api`: Configurable per preference
  - `claude_code_subscription`: Can use more capable models with flat-rate subscription
  - `fallback`: Used when provider is unknown
- **Removed `default` flag from model entries** - The new `models.defaults` section replaces per-model default flags
- **Updated startup logging** - Now shows active default model based on detected provider

### Added

#### ContextKit Integration
- **Added ContextKit dependency** - `contextkit` package for reading context from various sources (files, URLs, PDFs, GitHub repos, YouTube transcripts, etc.)
- **Available readers** - `read_file`, `read_url`, `read_pdf`, `read_gh_repo`, `read_gh_file`, `read_gist`, `read_yt_transcript`, `read_gdoc`, `read_google_sheet`, `read_html`, `read_dir`, `read_git_path`

#### Debug Logging for Tool Calling
- **Added extensive debug logging to claudette path** - Logs model info, tools, history, stream results, and tool call detection
- **Added debug logging to claudette-agent path** - Logs provider info, context messages, prompts, responses, and tool execution

#### LLM Steps Display with Proper Reasoning Separation
- **Pre-tool reasoning inside LLM Steps** - Text before tool calls ("I'll use the greet tool...") is captured and displayed inside the collapsible LLM Steps
- **Inter-tool reasoning inside LLM Steps** - Text between tool calls is captured as reasoning steps
- **Post-tool reasoning inside LLM Steps** - Acknowledgment text after the last tool result ("I see that the tool was called...") is now properly captured using paragraph-break heuristic and displayed inside LLM Steps
- **Final response properly separated** - Only the actual final response (paragraphs after the acknowledgment) appears outside the LLM Steps details element
- **Nested collapsible dropdowns for tool inputs/outputs** - Each tool call now has individually expandable `<details>` elements for viewing inputs and outputs
- **Fixed nested HTML block parsing in markdown** - The `extractLeadingHtmlBlocks()` function now properly handles nested `<details>` tags using depth counting instead of non-greedy regex that would break at the first closing tag

#### Tool Loop Output Duplication Fixes
- **Fixed duplicated response content** - The follow-up prompt in tool loops now includes the original user question, preventing the LLM from producing confused/duplicated output
- **Added response deduplication** - New `_deduplicate_response_text()` function detects and removes duplicated content in LLM responses that can occur during tool calling
- **Stripped LLM Steps HTML from context** - Previous prompt cell outputs containing `<details class="tool-steps-container">` HTML are now cleaned before being included in LLM context, preventing the LLM from reproducing formatting HTML

#### Post-Tool Response Placement Fix
- **All post-tool text now outside LLM Steps** - Removed paragraph-break heuristic that incorrectly captured introductory text (like "Based on the analysis, here are the results:") as reasoning inside the LLM Steps. Now all text after the last tool_result is displayed as the final response outside the collapsible section

#### Multi-Paragraph Reasoning Display Fixes
- **Preserved line breaks in reasoning text** - Added `white-space: pre-wrap` to `.step-reasoning .step-text` CSS so multi-paragraph reasoning displays correctly instead of collapsing into a single line
- **Configurable reasoning truncation limit** - Increased default truncation limit from 300 to 500 characters; now configurable via `display.reasoning_truncate_chars` in `dialeng_config.json` (0 = no limit)

#### AWS Bedrock Integration Fixes
- **Fixed AWS region not being passed to Bedrock client** - The `aws_region` from config is now explicitly passed to `AnthropicBedrock()` instead of relying on auto-detection
- **Fixed invalid Bedrock model IDs** - Updated default model mappings to use correct Bedrock model ID format (`anthropic.{model}-{date}-v{n}:{profile}`) instead of cross-region format (`us.anthropic...`)
- **Updated default models** - Changed default models to publicly available Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3 Opus with correct API model IDs

### Changed

- **Renamed "ReAct Steps" to "LLM Steps"** - More user-friendly name for the collapsible section showing tool activity
- **Refactored streaming text categorization** - The prompt cell streaming loop now properly categorizes text chunks:
  - `pre_tool_text`: Text before first tool call → saved as reasoning in LLM Steps
  - `post_tool_text` between tools: Text between tool_result and next tool_call → saved as reasoning in LLM Steps
  - `post_tool_text` at end: Text after last tool_result with no more tool_calls → displayed as final response outside LLM Steps
- **`_format_tool_steps_markdown()`** - Refactored to use nested `<details>/<summary>` elements for tool inputs and outputs instead of flat `<div>` elements
- **`extractLeadingHtmlBlocks()`** - New JavaScript function that properly extracts leading HTML blocks with nested tag support using depth counting
- **`renderMarkdown()`** - Now uses `extractLeadingHtmlBlocks()` instead of simple regex for more robust HTML preservation

### Added

#### Display Configuration
- **`display.reasoning_truncate_chars`** - New config option in `dialeng_config.json` to control reasoning text truncation (default: 500 characters, 0 = no limit)

#### CSS Styles for Nested Tool Details
- **`.step-input-details`, `.step-output-details`** - Styling for the nested collapsible sections within each tool call
- **`.step-toggle`** - Styling for the expandable summary (📥 Input, 📤 Output) with hover effects and arrow indicators
- **`.step-pre`** - Styled pre-formatted code blocks for input JSON and output results with max-height scrolling

---

## [0.10.0] - 2025-01-15

### Added

#### Extensibility Framework (fastai-inspired)
- **New `core/` package** - Foundation for dialeng's hackable, extensible architecture using fastcore-inspired patterns

##### Type Dispatch System (`core/dispatch.py`)
- **Extensible cell rendering** - `render_cell()` dispatch function routes to appropriate renderer based on cell type
- **Extensible LLM conversion** - `cell_to_llm_messages()` dispatch function for converting cells to LLM context
- **Extensible serialization** - `cell_to_jupyter()` and `jupyter_to_cell()` dispatch functions
- **Registration decorators** - `@register_renderer()`, `@register_llm_converter()`, `@register_jupyter_serializer()`

##### 2-Way Callback System (`core/callbacks.py`)
- **ExecutionContext** - Shared mutable context that callbacks can modify during execution
- **Callback base class** - Extensible hooks for `before_execution`, `on_output`, `after_execution`
- **Code transformation** - Callbacks can modify `ctx.source` before execution (e.g., auto-imports)
- **Output filtering** - Callbacks can filter or transform outputs via `on_output` hook
- **Flow control** - `CancelCellException` and `CancelQueueException` for callback-driven cancellation
- **Built-in callbacks** - `TimingCallback`, `LoggingCallback`, `OutputTruncateCallback`

##### Extension Registry (`core/registry.py`)
- **Central registry** - `ExtensionRegistry` class for cell types, callbacks, and services
- **Registration decorators** - `@register_cell_type()`, `@register_callback()`, `@register_service()`
- **Global registry instance** - `registry` singleton accessible throughout the application

##### Extension Loading (`core/extensions.py`)
- **Automatic extension loading** - Extensions in `extensions/` directory loaded at startup
- **Extension extraction** - `extract_extension()` utility to extract code from notebooks marked with `# @extension`
- **Hot reload support** - `reload_extension()` for development without restart

#### Extensions Directory (`extensions/`)
- **Example callbacks** - `extensions/example_callbacks.py` demonstrates:
  - `ExecutionTimingCallback` - Track and log execution time for all cells
  - Commented examples for `AutoImportCallback` and `OutputLimitCallback`

### Changed

- **`ui/cells/base.py`** - `CellView()` now uses `render_cell()` from dispatch system
- **`services/dialoghelper_service.py`** - `cell_to_messages()` now uses `cell_to_llm_messages()` from dispatch
- **`services/kernel/execution_queue.py`** - Integrated 2-way callback system:
  - `ExecutionQueue` now accepts optional `CallbackHandler`
  - `_process_queue()` creates `ExecutionContext` and runs callbacks
  - Callbacks can modify source code before execution
  - Callbacks can filter/transform outputs during streaming
- **`services/kernel/kernel_service.py`** - `execute_cell()` accepts optional `source` parameter for transformed code
- **`app.py`** - Loads extensions at startup and passes callback handler to execution queues

### Technical Details

- **No breaking changes** - Existing code continues to work; extension system is additive
- **Backward compatible callbacks** - Legacy `on_output` and `on_state_change` callbacks still work
- **Lazy imports** - Dispatch functions use lazy imports to avoid circular dependencies

### Developer Experience

- **Experiment in notebooks** - Write extension code in dialeng notebooks with `# @extension` marker
- **Extract to extension** - Use `extract_extension()` to create standalone extension file
- **Hot reload** - Use `reload_extension()` during development

### Documentation

- **`docs/how_it_works/07_code_organization.md`** - Added Section 8: "Architecture Design: `core/` vs `services/`"
  - Explains that `core/` contains extension infrastructure (HOW to extend)
  - Explains that `services/` contains feature implementations (WHAT the system does)
  - Updated architecture diagram to include `core/` and `extensions/` packages
  - Updated directory structure with new packages

- **`docs/how_it_works/05_dialoghelper_integration.md`** - Added "Key Implementation Files" section
  - Lists the core files for headless interaction: `dialoghelper_service.py`, `app.py`, `kernel_worker.py`, `subprocess_kernel.py`
  - Documents file responsibilities and function summaries
  - Provides quick reference for developers working with remote/headless interaction

- **`docs/how_it_works/README.md`** - Updated quick reference table
  - Added entry for "Extension system" pointing to `08_extension_system.md`
  - Added entry for "`core/` vs `services/`" pointing to Section 8 of `07_code_organization.md`

### Fixed

- **Empty notebook ID routing** - Fixed `/dialeng/` (with trailing slash) matching `/dialeng/{nb_id}` with empty string
  - Added explicit `/dialeng/` redirect route to `/dialeng/default`
  - Added guard in `/dialeng/{nb_id}` route to redirect empty notebook IDs

- **Cell `clear_outputs()` method** - Added missing method to Cell class in `app.py`
  - Clears `output`, `execution_count`, and `time_run` when source changes
  - Prevents stale context in subsequent LLM calls

- **Logger undefined error** - Changed `logger.info()` to `print()` in source update handler
  - Matches `app.py`'s existing logging style

---

## [0.9.6] - 2025-01-13

### Changed

#### Major Code Modularization
- **Reduced app.py from ~4,000 to ~1,500 lines** - Extracted CSS, JavaScript, and UI components to separate modules
- **External static assets** - CSS and JavaScript now served from `static/` directory for easier maintenance and browser caching
- **UI component package** - New `ui/` package with modular, reusable FastHTML components

### Added

#### Static Assets (`static/`)
- **`static/css/themes.css`** - CSS custom properties for dark/light themes
- **`static/css/base.css`** - Reset, typography, responsive layout rules
- **`static/css/components.css`** - Cell, button, badge, markdown preview styles
- **`static/css/editor.css`** - Ace editor container and focus styles
- **`static/js/app.js`** - All client-side logic (~1,700 lines) with clear section headers:
  - Global State Management
  - Ace Editor Management
  - Cell Focus & Selection
  - Keyboard Shortcuts
  - Preview/Edit Toggle (with event delegation fix)
  - WebSocket & Streaming
  - Initialization

#### UI Components Package (`ui/`)
- **`ui/__init__.py`** - Public exports for all UI components
- **`ui/base.py`** - Shared utilities (`get_collapse_class`)
- **`ui/controls.py`** - Interactive controls:
  - `TypeSelect` - Cell type dropdown
  - `CollapseBtn` - Collapse level toggle button
  - `AddButtons` - "+ Code", "+ Note", "+ Prompt" buttons
- **`ui/layout.py`** - Page-level components:
  - `NotebookPage` - Complete page with header, cells, footer
  - `AllCells` - Container with all cells
  - `AllCellsContent` - Cells without wrapper ID (for innerHTML swaps)
- **`ui/oob.py`** - Out-of-Band variants for WebSocket:
  - `AllCellsOOB` - AllCells with `hx-swap-oob`
  - `CellViewOOB` - CellView with `hx-swap-oob`
- **`ui/cells/`** - Cell type components:
  - `ui/cells/base.py` - `CellView` dispatcher and `CellHeader`
  - `ui/cells/code_cell.py` - `CodeCellView` (Ace editor + output)
  - `ui/cells/note_cell.py` - `NoteCellView` (markdown preview)
  - `ui/cells/prompt_cell.py` - `PromptCellView` (user input + AI response)

#### Documentation
- **`docs/how_it_works/07_code_organization.md`** - Comprehensive guide covering:
  - Directory structure and architecture overview
  - Static assets organization (CSS/JS)
  - UI component hierarchy and patterns
  - Data flow diagrams (code/prompt execution, OOB updates)
  - How to add new cell types, keyboard shortcuts, themes
  - Common FastHTML patterns and examples
  - Testing procedures

### Fixed

#### GUI Reliability (Double-Click Editing)
- **Fixed unreliable double-click editing on note/prompt cells** - Changed from per-element event listeners to event delegation pattern
- **Root cause**: `setupPreviewEditing()` was not called after WebSocket OOB updates, causing listeners to be lost when cells were replaced
- **Solution**: Single document-level `dblclick` listener using event delegation, which works regardless of DOM changes:
  ```javascript
  document.addEventListener('dblclick', function(e) {
      const preview = e.target.closest('.md-preview, .ai-preview, .prompt-preview');
      if (!preview) return;
      const cellId = preview.dataset.cellId;
      const field = preview.dataset.field;
      if (cellId && field) switchToEdit(cellId, field);
  });
  ```

### Technical Details

- **External JavaScript via static route** - Uses `Script(src="/static/js/app.js")` to load JS externally (not `ScriptX` which embeds inline and escapes backslashes)
- **Static file serving route** - New `@rt("/static/{path:path}")` route serves files from `static/` directory
- **Component pattern** - All UI components follow FastHTML conventions with `*args` for children and `**kwargs` for attributes
- **Import structure** - Clean imports: `from ui import CellView, NotebookPage, AllCells, AddButtons`

---

## [0.9.5] - 2025-01-13

### Added

#### Direct SDK Mode for Maximum Session Isolation
- **New `use_sdk_directly` option** - Uses `claude-agent-sdk.query()` directly instead of claudette-agent wrapper for maximum isolation (disabled by default). Each query:
  - Creates a completely fresh subprocess
  - Uses a unique temporary directory as `cwd`
  - Sets all stateless options explicitly (`continue_conversation=False`, `resume=None`, `setting_sources=[]`)
  - Cleans up the temp directory after completion
- **New `debug_mode` option** - When enabled, saves full prompts and responses to timestamped JSON files in `debug_log_dir`
- **New configuration section** - `llm` section in `dialeng_config.json`:
  ```json
  {
    "llm": {
      "use_sdk_directly": false,
      "debug_mode": false,
      "debug_log_dir": "./debug_logs"
    }
  }
  ```
- **Automated test harness** - `test_stateless_dialoghelper.py` verifies stateless behavior with the John→Mark scenario

#### Cell Version Tracking and Context Freshness
- **Cell version tracking** - Added `version: int` and `last_modified: datetime` fields to `Cell` class to track when cells are modified
- **Automatic output clearing** - When cell source is edited, outputs are automatically cleared to prevent stale context contamination
- **`Cell.update_source()` method** - New method that:
  - Detects if source actually changed
  - Increments version counter
  - Updates `last_modified` timestamp
  - Clears outputs to ensure fresh context
- **Detailed context logging** - Added comprehensive logging to `build_context_messages()` and `_stream_claudette()` to trace exactly what context is sent to LLM

### Changed

- **`llm_service.py`** - Added `_stream_claude_sdk_direct()` method that bypasses claudette-agent wrapper entirely
- **`llm_service.py`** - Added detailed logging to `_stream_claudette()` showing all context messages being sent
- **`dialeng_config.py`** - Added `use_sdk_directly`, `debug_mode`, and `debug_log_dir` fields to `DialengConfig`
- **`dialoghelper_service.py`** - Added comprehensive logging to `build_context_messages()` showing each cell being included
- **`app.py`** - `/cell/{cid}/source` endpoint now clears cell output when source changes
- **`document/cell.py`** - Added version tracking fields and `update_source()` method
- **Startup logging** - Now shows which LLM provider mode is active (SDK direct vs claudette-agent)

### Fixed

- **Stale context contamination** - When editing a prompt cell (e.g., changing "John" to "Mark"), the old assistant response was still included in context for subsequent queries. Now the output is automatically cleared when the source changes.
- **Prompt cell source updates not being sent to server** - The prompt textarea used `name="prompt_source"` but the endpoint expected `source`. Added `hx_include` to include the hidden source input when posting, ensuring the correct parameter is sent. Without this fix, editing a prompt cell after running it would not update the server-side source or clear the output.

### Documentation

- Updated `docs/how_it_works/06_llm_integration.md`:
  - Added SDK direct mode documentation
  - Updated configuration options table

---

## [0.9.4] - 2025-01-13

### Fixed

#### Truly Stateless Queries via claudette-agent
- **Fixed session contamination issue** - Claude was remembering conversation history from previous sessions even after editing notebook cells. For example, changing "John" to "Mark" in a notebook cell would still result in Claude responding with "John" because it loaded the original session.
- **Root cause**: The `--no-session-persistence` flag only prevents SAVING new sessions, but doesn't prevent LOADING existing sessions. Claude Code stores session transcripts per-project based on the working directory (`cwd`).
- **Solution**: Use stateless configuration with `cwd=None`:
  1. `setting_sources=[]` - Prevents loading settings files
  2. `cwd=None` - No working directory specified, SDK creates fresh session each time
  3. `extra_args={'no-session-persistence': None}` - Prevents saving new sessions
  4. claudette-agent's `_build_options()` sets `continue_conversation=False` and `resume=None` to prevent session continuation/resumption
- **Verified stateless behavior** - Test confirms that editing notebook cells (e.g., changing "John" to "Mark") is correctly reflected in subsequent queries

### Changed

- **claudette-agent Chat._build_options** - Fixed missing `extra_args` handling and added explicit stateless options:
  - Merges `self.c.extra_args` into SDK options for CLI flags like `--no-session-persistence`
  - Explicitly sets `continue_conversation=False` to prevent session continuation
  - Explicitly sets `resume=None` to prevent session resumption
- **llm_service.py** - Simplified stateless configuration to use `cwd=None` instead of creating temporary directories

### Updated

- **claudette-agent dependency** - Updated to latest version with `extra_args` parameter support and stateless options

### Documentation

- Updated `docs/how_it_works/06_llm_integration.md`:
  - Updated stateless query code example with `cwd=None` approach
  - Documented four-mechanism stateless configuration
  - Updated architecture diagram to reflect new stateless mechanism

---

## [0.9.3] - 2025-01-12

### Changed

#### Extended Thinking Support
- **Real extended thinking** - Now uses claudette-agent's `maxthinktok` parameter instead of placeholder markers. When a thinking-capable model (Claude Sonnet 3.7+, Sonnet 4+, Opus 4+) is used with thinking enabled, the `stream()` method receives `maxthinktok=N` to enable actual extended reasoning.
- **Model capability checks** - Uses `can_use_extended_thinking()` from claudette-agent to verify model support before enabling thinking. If a model doesn't support thinking, it gracefully disables with a warning log.
- **Configurable token budget** - New `thinking.max_tokens` setting in `dialeng_config.json` (default: 10000) controls the maximum tokens allocated for extended thinking.

#### Usage and Cost Tracking
- **Token usage tracking** - `llm_service.last_usage` now exposes token counts from the last API call after each streaming response completes
- **Cost estimation** - `llm_service.last_cost` provides estimated cost in USD from the last streaming response
- **Automatic logging** - Usage and cost are logged automatically after each streaming response: `claudette-agent: Usage=..., Cost=$...`

#### Truly Stateless Queries via claudette-agent (legacy approach)
- **Stateless by default** - `_stream_claudette_agent()` uses `AsyncChat` with two stateless mechanisms:
  1. `setting_sources=[]` to prevent loading settings files
  2. **Unique temporary `cwd` per query** to prevent loading transcripts from previous sessions (Claude Code stores transcripts per project based on cwd)
- **Notebook as sole source of truth** - Edits to notebook cells are immediately reflected in subsequent LLM queries. No "memory leak" from previous Claude Code sessions.
- **Automatic cleanup** - Temporary directories are cleaned up after each streaming response completes
- **claudette-agent updated** - Now uses updated claudette-agent library (v0.1.0) with `setting_sources` and `cwd` parameter support
- **Note:** This approach was replaced in v0.9.4 with the cleaner `--no-session-persistence` CLI flag approach

### Added

- **`_check_thinking_support()` method** - New method in `LLMService` that checks if a model supports extended thinking using claudette-agent's capability checks
- **`last_usage` property** - Returns usage stats from the last API call
- **`last_cost` property** - Returns cost estimate from the last API call
- **`thinking` config section** - New configuration section in `dialeng_config.json` and `DialengConfig` dataclass

### Fixed

- **Session persistence issue (complete fix)** - Fixed bug where Claude Code's session tracking caused LLM to remember conversation history separately from notebook cells. When users edited notebook cells (e.g., changing a name from "John" to "Mark"), Claude would still remember the original conversation. Root cause: Claude Code stores transcripts per project based on the working directory (`cwd`). **Complete fix uses two mechanisms:**
  1. `setting_sources=[]` - Prevents loading settings files
  2. **Unique temporary `cwd` per query** - Creates a fresh temp directory for each LLM call, preventing Claude Code from loading any previous transcripts. The temp directory is cleaned up after streaming completes.
  This ensures truly stateless queries where the notebook cells are the sole source of truth for conversation history.
- **Conversation history chronological ordering** - Fixed bug where pinned cells were placed before all non-pinned cells regardless of their actual notebook position, breaking chronological order. Now all context cells are sorted by their original notebook index before being converted to messages. This fixes LLM confusion when previous prompt/response pairs appeared out of order.
- **Off-by-one prompt/response issue** - Fixed bug where LLM responses were consistently one prompt behind the current question. Two issues were found: (1) multiple "User:" messages in history confused the SDK, and (2) `AsyncChat._append_pr` is async but `stream()` calls it without await, so prompts were never added to history. Fix: build a single prompt string with context, manually append to `chat.h`, then call `stream(None)` to skip the broken `_append_pr` call.
- **Claude Code credential detection** - Now uses direct `claude_agent_sdk` probe instead of searching for CLI binary. This fixes an issue where credentials weren't detected on first startup even when logged in, but worked after running any script using `claude_agent_sdk` first.
- **System prompt context clarification** - System prompts now include a preamble explaining that conversation history may include code cells and notes from the notebook. This prevents Claude from analyzing/listing notebook cells when the user asks a simple question like "Hello".
- **Streaming block handling** - Properly handles claudette-agent's message block format, distinguishing between thinking blocks (`type='thinking'`) and text blocks (`text` attribute)
- **Thinking state management** - Correctly tracks thinking phase start/end across streaming, ensuring `thinking_end` is always yielded even if no text blocks follow

### Documentation

- Updated `docs/how_it_works/06_llm_integration.md` with:
  - Extended thinking implementation details and configuration
  - Usage and cost tracking documentation
  - Updated claudette-agent code examples with `maxthinktok` parameter
  - New configuration options table entry for `thinking.max_tokens`
  - **Stateless query mechanism** - Explained unique `cwd` approach for truly stateless queries
  - Updated architecture diagram to show unique temp cwd creation step

---

## [0.9.2] - 2024-12-15

### Fixed

#### DialogHelper Response Format Fixes
- **`read_msg_` endpoint** - Now returns `{'msg': {'content': ..., 'id': ..., 'type': ..., 'pinned': ...}}` format expected by dialoghelper library (previously returned flat dict with 'source' field)
- **`find_msgs_` endpoint** - Uses 'content' field instead of 'source' for consistency with read_msg
- **`update_msg_` endpoint** - Maps 'content' to 'source' for cell content updates; now returns cell ID (not `{"status": "ok"}`) so dialoghelper can maintain `__msg_id` for relative operations
- **`update_msg_` boolean parameter fix** - Changed boolean parameters (`pinned`, `skipped`, `is_exported`, `heading_collapsed`) from `int` type to `str` type with `_str_to_bool()` conversion. HTTP form data sends Python's `True` as string `"True"`, and FastHTML couldn't convert `"True"` to `int`, causing a 422 error
- **`add_relative_` boolean parameter fix** - Same fix applied to `is_exported`, `skipped`, `pinned`, `run`, and `heading_collapsed` parameters
- **Test notebook** - Updated to access data via `.msg.content`, `.msg.type`, `.msg.pinned` etc.

#### Real-time WebSocket Broadcasting
- **`add_relative_` (add_msg)** - Now broadcasts `AllCellsOOB` when cells are created, so new cells appear immediately without refresh
- **`rm_msg_` (del_msg)** - Now broadcasts `AllCellsOOB` when cells are deleted
- **`update_msg_`** - Now broadcasts `CellViewOOB` when cell properties are updated
- **OOB swap skip logic fix** - Client-side JavaScript now correctly distinguishes between "user typing" and "cell executing". Previously, the Ace editor's hidden textarea maintaining focus during code execution was incorrectly treated as "user typing", causing `add_msg()` cell updates to be skipped. Now only skips if user is typing in a cell that's NOT streaming (executing)

#### JavaScript Injection (iife/add_scr)
- **Script execution fix** - `processOOBSwap()` now properly handles scripts injected via `beforeend:#js-script` swap strategy. Scripts inserted via `innerHTML` don't execute automatically (browser security), so the function now manually creates `<script>` elements via `document.createElement('script')` which triggers actual execution
- **fire_event() script handling** - Added handling for `<script hx-swap-oob="true">` elements used by `fire_event()`, ensuring they execute when broadcast via WebSocket

#### Dialog Mode Detection
- **Mock mode enforcement** - When no LLM credentials are available, `dialog_mode` is forced to "mock" regardless of saved value in notebook file (fixes issue where loaded notebooks showed "learning" mode without credentials)

### Added

- **Advanced test notebook** - Created `notebooks/test_dialoghelper_advanced.ipynb` with tests for:
  - `msg_strs_replace()` - Multi-string replacement
  - `msg_replace_lines()` - Line range replacement
  - `add_scr()` - Lower-level script injection
  - Advanced `iife()` - Async patterns, progress indicators, Fetch API, DOM queries
  - `fire_event()` + `pop_data()` - Bidirectional browser/Python communication
  - Utility patterns: cell duplication, backup before modify, find/replace all

### Changed

- **Cell 10 typo fix** - Changed `del_msg(msid=...)` to `del_msg(msgid=...)` in test notebook
- **Test notebook pop_data() fix** - Fixed `pop_data()` calls to use correct parameter name `idx` (not `data_id`). The dialoghelper function signature is `pop_data(idx, timeout=15)` which internally maps to `data_id`.

### Documentation

- **05_dialoghelper_integration.md** - Comprehensive documentation of:
  - Script execution mechanism (why `innerHTML` doesn't execute scripts and how `processOOBSwap()` handles it)
  - `add_scr()` function documentation
  - `fire_event()` and `pop_data()` with correct parameter names
  - Complete bidirectional data transfer example (browser calculation pattern)
  - Both test notebooks documented with full feature lists

- **03_real_time_collaboration.md** - Added "Script Injection OOB" section explaining:
  - Swap strategy pattern for `iife()`/`add_scr()`
  - Direct script pattern for `fire_event()`
  - Why `document.createElement('script')` triggers execution

---

## [0.9.1] - 2024-12-15

### Fixed

#### DialogHelper Magic Variable Injection
- **`__dialog_name` and `__msg_id` injection** - Kernel now injects these magic variables into the execution namespace before running each cell, enabling dialoghelper functions like `read_msg(-1)` and `iife()` to work correctly
- **Port configuration** - Updated test notebook to use correct port (8000) instead of 5001

#### ExecutionQueue Missing Method
- **Added `is_cell_queued()` method** - Fixed `AttributeError: 'ExecutionQueue' object has no attribute 'is_cell_queued'` by adding the missing method to check if a cell is queued or currently running

### Changed

- **Test notebook expanded** - `notebooks/test_dialoghelper.ipynb` now includes comprehensive tests for:
  - `curr_dialog()` - Get dialog/notebook info
  - `msg_idx()` - Get cell index by ID
  - `read_msg()` - Read cell content (absolute and relative)
  - `find_msgs()` - Search cells by type, pattern
  - `update_msg()` - Update cell properties
  - `add_msg()` - Create new cells
  - `del_msg()` - Delete cells
  - `msg_str_replace()` - Replace string in cell
  - `msg_insert_line()` - Insert line in cell
  - `iife()` - JavaScript injection (console.log, alert, DOM manipulation)
  - `add_html()` - Direct HTML injection
  - `event_get()` - Bidirectional browser communication
  - `run_msg()` - Queue cells for execution

### Technical Changes

- Modified `services/kernel/subprocess_kernel.py` - `execute_streaming()` now accepts `notebook_id` and `cell_id` parameters
- Modified `services/kernel/kernel_service.py` - Passes notebook_id and cell_id through to `execute_streaming()`
- Modified `services/kernel/kernel_worker.py` - Injects `__dialog_name` and `__msg_id` into `shell.user_ns` before each cell execution

---

## [0.9.0] - 2024-12-15

### Added

#### Complete DialogHelper JavaScript Injection Support
- **`iife()` support** - Execute JavaScript in browser from Python via IIFE pattern
- **`add_html()` broadcast** - Fixed endpoint to broadcast HTML via WebSocket for HTMX OOB swaps
- **`#js-script` container** - Frontend div for receiving injected scripts
- **`window.NOTEBOOK_ID`** - Global JavaScript variable exposing notebook ID

#### Bidirectional Browser Communication
- **`/push_data_blocking_`** - New endpoint for browser-to-Python data transfer
- **`/pop_data_blocking_`** - Fixed with async queue implementation and timeout support
- **Data queue infrastructure** - Per-notebook async queues for `push_data`/`pop_data` operations
- **`event_get()` support** - Fire events and wait for browser response

#### Cell Execution via DialogHelper
- **`/add_runq_`** - Implemented to queue cells for execution using `ExecutionQueue`
- **`run_msg()` support** - Queue and execute code cells programmatically

#### Test Notebook
- **`notebooks/test_dialoghelper.ipynb`** - Comprehensive test notebook demonstrating:
  - Basic cell operations (read_msg, find_msgs, add_msg)
  - JavaScript injection via iife()
  - DOM manipulation from Python
  - Bidirectional data transfer (event_get)
  - Programmatic cell execution (run_msg)

### Changed

- **`/add_html_`** - Now broadcasts via WebSocket instead of returning content
- **Documentation** - Updated `docs/how_it_works/05_dialoghelper_integration.md` with iife and data transfer docs

## [0.8.0] - 2024-12-15

### Added

#### Multi-Provider LLM Support
- **Automatic credential detection** - System detects available LLM credentials at startup in priority order:
  1. Anthropic API key (`ANTHROPIC_API_KEY`)
  2. AWS Bedrock credentials (env vars, profiles, IAM)
  3. Claude Code CLI subscription
- **claudette integration** - Direct Anthropic API and AWS Bedrock support via `claudette` library
- **claudette-agent integration** - Claude Code subscription support via `claudette-agent` library
- **Startup logging** - Clear credential status shown at startup with provider, backend, and source details
- **Dynamic UI** - Mode selector only shows available options based on detected credentials

#### Configurable LLM Settings (`dialeng_config.json`)
- **Auto-generated config file** - `dialeng_config.json` created on first startup with sensible defaults
- **Model configuration** - Define available models for UI picker with customizable names
- **Backend-specific model mappings** - Separate model ID mappings for:
  - `anthropic_api_map` - Direct Anthropic API (with date suffix)
  - `bedrock_map` - AWS Bedrock (with region prefix and version suffix)
  - `claudette_agent_map` - Claude Code subscription (simple names)
- **AWS region configuration** - Configurable region for Bedrock API calls
- **Default mode setting** - Configure default dialog mode for new notebooks
- **Config status logging** - Shows loaded models and defaults at startup

#### New Service Modules
- **services/credential_service.py** - Credential detection with `CredentialStatus` dataclass
- **services/dialeng_config.py** - Configuration management with `DialengConfig` dataclass

### Changed

- **Model selection** - Now supports Claude Sonnet 3.7 (default), Claude Sonnet 4.5, and Claude Haiku 4.5
- **LLM service** - Refactored to use config-based model mappings instead of hardcoded values
- **Error handling** - Improved error messages for streaming failures with detailed logging

### Dependencies

- Added `claudette>=0.2.0` - For direct Anthropic API and AWS Bedrock
- Added `anthropic>=0.40.0` - Anthropic SDK
- Added `boto3>=1.34.0` - AWS SDK for Bedrock
- Added `botocore>=1.34.0` - AWS core library

### Documentation

- Updated `docs/how_it_works/06_llm_integration.md` with:
  - Credential detection flow diagram
  - Configuration options and examples
  - Provider-specific implementation details
  - Model mapping documentation

---

## [0.7.0] - 2024-12-11

### Added

#### FIFO Cell Execution Queue
- **Proper execution queue** - Cells now execute in FIFO order like Jupyter, preventing output mixing when running multiple cells quickly
- **Queue visual feedback** - Queued cells show:
  - Yellow border with `queued` CSS class
  - ⏳ icon on the run button
  - "Queued (position N)..." message in output area
- **Cancel All button** - Red "⏹ Cancel All" button in toolbar (visible when queue has items)
  - Interrupts running cell AND clears entire queue
  - Keyboard shortcut: `Escape Escape` (double-Escape, like Jupyter's `I I`)
- **Real-time queue state via WebSocket** - New message types:
  - `queue_update`: Broadcasts current running cell and queued cell IDs
  - `cell_state_change`: Broadcasts individual cell state changes
- **Duplicate run prevention** - Clicking run on already-queued cell is ignored
- **Queue cleanup on cell delete** - Deleting a queued cell removes it from queue

#### Backend Changes
- **ExecutionQueue integration** - The previously unused `services/kernel/execution_queue.py` is now active
- **New helper method** - Added `is_cell_queued(nb_id, cell_id)` to ExecutionQueue
- **New endpoint** - `POST /dialeng/{nb_id}/queue/cancel_all` - cancels running + clears queue
- **Broadcast functions** - Added `broadcast_queue_state()`, `broadcast_cell_state()`, `broadcast_cell_output()`
- **State callback system** - Queue emits callbacks for output chunks and state changes, enabling WebSocket broadcasting
- **Cancel flag for queue** - Added `_cancelled` flag to ExecutionQueue to properly stop entire queue when Cancel All is triggered

#### Frontend JavaScript
- **Queue state tracking** - `cellQueueState` Map tracks each cell's queue state
- **Visual state updates** - `updateCellVisualState()` manages queued/running/idle UI
- **Cancel All function** - `cancelAllExecution()` calls cancel endpoint
- **Ace editor focus handling** - Clicking inside code editor now properly highlights the cell
- **Global cell selection** - Document-level event delegation ensures clicking anywhere on a cell highlights it
- **Per-cell interrupt clears queue** - The interrupt button on individual cells now also clears the entire queue

### Changed
- **Run endpoint** - Now queues cells instead of executing inline; returns immediately
- **prepareCodeRun()** - Shows "Queuing..." instead of "Running...", skips if already queued

### Fixed
- **Cancel All stops entire queue** - Previously, Cancel All only stopped the running cell; queued cells would still execute. Now the entire queue is properly cleared using a `_cancelled` flag
- **Per-cell interrupt stops entire queue** - The interrupt button (⏹) on individual cells now calls `cancelAllExecution()` to stop the running cell AND clear the queue
- **Shift+Enter moves focus immediately** - Focus now moves to next cell immediately when pressing Shift+Enter, matching Jupyter behavior (previously waited for server response which didn't work with queue system)
- **Cell selection highlighting** - Clicking anywhere on a cell (not just the editor) now properly sets the cell as focused with visual highlighting, using document-level event delegation with capture phase

### Documentation
- Updated `docs/how_it_works/04_kernel_execution.md` with queue integration details

---

## [0.6.1] - 2024-12-11

### Fixed

#### Cell Execution State Management
- **Separated code and prompt cell streaming** - Code cells now use dedicated `prepareCodeRun()` function instead of prompt-cell-specific `startStreaming()`, preventing state corruption between cell types
- **Guaranteed code_stream_end delivery** - Wrapped server-side code execution in try/finally to ensure `code_stream_end` WebSocket message is always sent, even on errors
- **Improved HTMX error handling** - New `resetCellOnError()` function properly resets both code and prompt cells on network errors, timeouts, and request failures
- **Cell type-specific onclick handlers** - Run button now calls appropriate function based on cell type:
  - Prompt cells: `startStreaming()` for LLM response streaming
  - Code cells: `prepareCodeRun()` for kernel execution

#### Code Cell Reliability
- **Fixed HTMX bindings after WebSocket OOB swaps** - Added `htmx.process()` call after replacing DOM elements via WebSocket OOB swaps, ensuring Run buttons and other HTMX-powered elements continue to work after collaborative updates
- **Reduced safety timeout to 30 seconds** - Changed from 2 minutes to 30 seconds for faster recovery from stuck cells
- **Kernel busy detection** - Server now detects when kernel is busy executing another cell and sends "⏳ Kernel busy, waiting..." feedback message to the client
- **Extended HTMX timeout** - Added `hx_timeout="120s"` to run button for long-running cells (e.g., matplotlib plots) that may take longer than the default HTMX timeout
- **Improved interrupt handling** - `interruptCodeCell()` now waits 2 seconds after sending interrupt, then force-resets cell if still stuck
- **Per-cell timeout tracking** - Using `codeStreamingTimeouts` Map to track timeouts individually for each cell, allowing multiple cells to stream independently
- **Timeout reset on activity** - Timeouts reset when WebSocket messages (`code_stream_start`, `code_stream_chunk`, `code_display_data`) are received

### Changed

- **Immediate visual feedback for code cells** - New `prepareCodeRun()` shows "Running..." indicator immediately when user clicks run, before server responds
- **Better error messages** - HTMX errors now display specific messages (network error, timeout, request failed) in cell output
- **Enhanced interrupt UX** - Shows "Stopping..." while waiting for server to respond to interrupt, then "Execution interrupted" if timeout occurs

### Developer/Debug

- **Comprehensive WebSocket logging** - Added detailed console.log statements for tracking message flow:
  - `[WS]` prefix for client-side WebSocket messages
  - `[Code]` prefix for code cell streaming functions
  - `[CODE RUN]` prefix for server-side execution logging
- **Element validation** - `startCodeStreaming()` now validates cell and output elements exist before attempting to update them

## [0.6.0] - 2024-12-10

### Added

#### Real LLM Integration via claudette-agent
- **claudette-agent integration** - Real Claude API access for prompt cells
- **Multiple AI modes** - Mock, Learning, Concise, Standard selectable from toolbar
  - **Mock**: Fake responses for testing (no API calls, backwards compatible)
  - **Learning**: Guides users to discover answers - asks leading questions
  - **Concise**: Brief, code-focused responses with minimal explanation
  - **Standard**: Balanced, helpful assistant (default behavior)
- **Model selection** - Dropdown to choose Claude model (appears for non-Mock modes)
  - **Claude Sonnet 4.5** (default) - Balanced performance and quality
  - **Claude Haiku 4.5** - Faster, more cost-effective
  - Model selection persisted in notebook metadata
- **Context window management** - Up to 25 cells in LLM context
  - Pinned cells always included first
  - Recent non-pinned cells fill remaining slots
  - Skipped cells excluded from context

#### DialogHelper Compatibility
- **Full dialoghelper API support** - All 14 endpoints implemented:
  - Information: `curr_dialog_`, `msg_idx_`, `find_msgs_`, `read_msg_`
  - Modification: `add_relative_`, `rm_msg_`, `update_msg_`, `add_runq_`
  - Content editing: `msg_insert_line_`, `msg_str_replace_`, `msg_strs_replace_`, `msg_replace_lines_`
  - Utility: `add_html_`, `pop_data_blocking_`
- **Shared service layer** - `services/dialoghelper_service.py` provides core logic
  - Reused by HTTP endpoints AND LLM context building
  - Functions: `get_msg_idx()`, `find_msgs()`, `read_msg()`, `build_context_messages()`

#### New Service Modules
- **services/dialoghelper_service.py** - Shared dialoghelper logic and context building
- **services/llm_service.py** - LLM streaming via claudette-agent with mode-specific prompts

### Changed

- **Mode selector** - Added "Mock" option, now shows: Mock | Learning | Concise | Standard
- **Prompt execution** - Routes to mock or real LLM based on selected mode
- **Context building** - Uses dialoghelper functions (`find_msgs()`) for consistency

### Documentation

- **docs/how_it_works/05_dialoghelper_integration.md** - DialogHelper API documentation
- **docs/how_it_works/06_llm_integration.md** - LLM modes and context building docs
- **README.md** - Added AI Modes and DialogHelper Compatibility sections

### Dependencies

- Added `claudette-agent` - LLM integration via Claude API
- Added `dialoghelper` - Solveit compatibility (for reference)

---

## [0.5.1] - 2024-12-10

### Fixed

#### Code Cell Streaming Output
- **Fixed streaming output display** - Output now streams to browser in real-time (was not showing due to CSS selector mismatch)
- **Fixed empty cell output** - Output container always renders, even for cells without prior output
- **Fixed output element ID** - Added `id="output-{cell.id}"` for reliable JavaScript selection

### Added

#### Rich Output Support (Jupyter-like)
- **MIME bundle rendering** - Full support for Jupyter MIME types:
  - `text/html` - Direct HTML rendering (FastHTML components, widgets)
  - `image/png`, `image/jpeg`, `image/gif` - Base64 image display
  - `image/svg+xml` - Inline SVG rendering
  - `text/markdown` - Markdown content
  - `text/latex` - LaTeX math notation
  - `application/json` - Pretty-printed JSON
  - `text/plain` - Escaped text fallback
- **ANSI color code support** - Terminal colors render as styled HTML spans
- **tqdm progress bar support** - Progress bars now work with carriage return handling
- **display_data streaming** - Rich outputs stream via WebSocket (`code_display_data` message type)

### Technical Changes

- Updated CellView HTML: Changed class from `output` to `cell-output`, added `id` attribute
- Added `render_mime_bundle()` function for converting MIME bundles to HTML
- Added `appendDisplayData()` JavaScript function for rich output rendering
- Added `ansiToHtml()` JavaScript function for ANSI-to-HTML conversion
- Updated `appendCodeOutput()` to handle carriage returns for tqdm
- Changed `StreamingStdout.isatty()` to return `True` to enable tqdm
- Added CSS for `.cell-output`, `.stream-output`, `.display-data`, `.mime-*` classes
- Added `code_display_data` WebSocket message type handler

---

## [0.5.0] - 2024-12-10

### Added

#### Streaming Code Cell Execution
- **Real-time stdout/stderr streaming** - Code cell output streams incrementally as it runs, like Jupyter notebooks
- **Subprocess-based kernel** - Code executes in a separate process for hard interrupt support
- **Hard interrupt (SIGINT)** - Stop any running code including C extensions and tight loops via cancel button
- **Rich output support** - Infrastructure for matplotlib plots, images, and HTML displays
- **WebSocket streaming** - Output chunks stream to browser in real-time via WebSocket messages

#### Execution Queue
- **FIFO cell queue** - Queue multiple cells while one is running
- **Responsive UI** - UI stays responsive during execution (async background processing)
- **Cancel queued cells** - Cancel individual cells or all queued cells

#### Project Restructuring (DCS Architecture)
- **Document layer** (`document/`) - Data models for Cell, Notebook, CellOutput
  - `cell.py` - Cell with streaming outputs, state tracking (IDLE, QUEUED, RUNNING, SUCCESS, ERROR)
  - `notebook.py` - Notebook with cell operations
  - `serialization.py` - .ipynb I/O using execnb.nbio
- **Service layer** (`services/kernel/`) - Business logic for code execution
  - `kernel_worker.py` - Subprocess kernel worker with streaming output
  - `subprocess_kernel.py` - Kernel manager with interrupt and restart
  - `kernel_service.py` - Service managing kernels per notebook
  - `execution_queue.py` - FIFO queue with callbacks for streaming

#### UI Enhancements
- **Cancel button for code cells** - Stop running code execution with the interrupt button
- **Streaming visual feedback** - Cell border shows streaming state during execution
- **Per-notebook kernels** - Each notebook has its own isolated kernel process

### Technical Changes

- Replaced `PythonKernel` class with `KernelService` for subprocess-based execution
- Added `multiprocessing.Process` for subprocess-based kernel with SIGINT support
- Added `fastcore.patch` to extend execnb's CaptureShell with streaming method
- Added custom `StreamingStdout` and `StreamingDisplayPublisher` for real-time output capture
- Added async generators for streaming output via multiprocessing.Queue
- Added signal handler in kernel worker for reliable KeyboardInterrupt on SIGINT
- Added `CellState` enum for tracking cell execution state
- Added `CellOutput` dataclass for structured output (stream, execute_result, error, display_data)
- Updated `requirements.txt` to include `fastcore>=1.5.0`
- Added WebSocket message types: `code_stream_start`, `code_stream_chunk`, `code_stream_end`
- Added JavaScript handlers for code cell streaming UI updates
- Added `/dialeng/{nb_id}/kernel/interrupt` route for hard interrupt
- Updated `/dialeng/{nb_id}/kernel/restart` route (was `/kernel/restart`)

### Documentation

- Added `docs/how_it_works/04_kernel_execution.md` - Comprehensive guide to kernel architecture
- Updated `docs/how_it_works/README.md` with kernel execution documentation
- Added test files: `test_kernel.py`, `test_integration.py`

## [0.4.1] - 2024-12-09

### Fixed

#### WebSocket Collaboration Bug Fixes
- **Fixed HTML serialization** - Changed from `str(component)` to `to_xml(component)` for proper HTML serialization when broadcasting updates via WebSocket
- **Fixed connection tracking** - Changed from Dict-based (`id(send)`) to List-based connection tracking following FastHTML Game of Life pattern
- **Fixed WebSocket lifecycle** - Used `conn=` and `disconn=` decorator parameters on `@app.ws` instead of manual registration on first message
- **Fixed prompt cell streaming** - Fixed `.values()` call on list in streaming loop (lists don't have `.values()` method)

### Technical Changes

- Changed `ws_connections` from `Dict[str, Dict[int, Any]]` to `Dict[str, List[Any]]`
- Updated `ws_on_connect` to append send function to list: `ws_connections[nb_id].append(send)`
- Updated `ws_on_disconnect` to remove from list: `ws_connections[nb_id].remove(send)`
- Added `AllCellsOOB()` and `CellViewOOB()` helper functions for generating OOB swap components
- Updated `broadcast_to_notebook()` to use `to_xml()` for HTML serialization
- Added `processOOBSwap()` JavaScript function to handle HTML OOB swaps from WebSocket
- Fixed prompt cell streaming to iterate over list directly instead of calling `.values()`
- Removed unused JSON-based WebSocket message handlers

### Documentation

- Added `docs/` folder with technical documentation
- Added numbered documentation files in `docs/how_it_works/` for recommended reading order:
  1. `01_state_management.md` - comprehensive guide to notebook/cell state management including in-memory storage, lazy loading, persistence to `.ipynb` files, CRUD operations, and state synchronization
  2. `02_cell_types.md` - comprehensive guide to the three cell types (Code, Note, Prompt) including data model, rendering logic, execution behavior, serialization format, collapse system, and how to add new cell types
  3. `03_real_time_collaboration.md` - comprehensive guide to the WebSocket collaboration system including architecture, message types, OOB swaps, cell-specific behavior, conflict avoidance, and improvement suggestions
- Converted ASCII diagrams to Mermaid format in README.md, DEVELOPERS.md, and docs/how_it_works/ for better rendering on GitHub

## [0.4.0] - 2024-12-09

### Added

#### Real-time Collaborative Editing
- **Share notebook URL for collaboration** - Multiple users can view and edit the same notebook simultaneously by sharing the URL
- **Real-time cell operations** - When a collaborator adds, deletes, or moves a cell, all connected users see the change instantly
- **Live code execution output** - When someone runs a code cell, all collaborators see the output in real-time
- **Streaming AI responses** - Prompt cell AI responses are streamed to all connected collaborators simultaneously
- **Collapse state synchronization** - Cell collapse/expand states are broadcast to all users
- **Cell type changes** - Changing a cell's type (code/note/prompt) is reflected for all collaborators

### Technical Changes

- Added `broadcast_to_notebook()` async helper function to broadcast WebSocket messages to all connected clients
- Added `to_html_string()` utility function to convert FastHTML components to HTML strings for WebSocket transmission
- Added two new WebSocket message types for collaborative updates:
  - `cells_updated` - Full cells container replacement (for add/delete/move operations)
  - `cell_updated` - Single cell replacement (for run output, collapse changes, type changes)
- Added JavaScript handlers for collaborative updates:
  - `handleCellsUpdated()` - Replaces entire cells container, reinitializes Ace editors, re-renders previews
  - `handleCellUpdated()` - Replaces single cell (skips if user is editing or cell is streaming)
  - `reinitializeAceEditors()` - Destroys and recreates all Ace editors after DOM update
  - `renderAllPreviews()` - Re-renders all markdown previews after collaborative update
  - `renderCellPreviews()` - Re-renders previews for a specific cell
- Updated routes to async and added broadcast calls:
  - `/dialeng/{nb_id}/cell/add` - Broadcasts cells_updated
  - `/dialeng/{nb_id}/cell/{cid}` (DELETE) - Broadcasts cells_updated
  - `/dialeng/{nb_id}/cell/{cid}/move/{direction}` - Broadcasts cells_updated
  - `/dialeng/{nb_id}/cell/{cid}/type` - Broadcasts cell_updated
  - `/dialeng/{nb_id}/cell/{cid}/collapse` - Broadcasts cell_updated
  - `/dialeng/{nb_id}/cell/{cid}/collapse-section` - Broadcasts cell_updated
  - `/dialeng/{nb_id}/cell/{cid}/run` - Broadcasts cell_updated (for code cells and final prompt state)
- Smart conflict avoidance: Cell updates are skipped if user is actively editing that cell or if it's currently streaming

## [0.3.1] - 2024-12-09

### Added

#### Multi-Level Collapsing for Input/Output
- **Independent input/output collapse** - Each cell now has separate collapse controls for input (code/prompt) and output (result/response)
- **Three collapse levels**:
  - **Expanded (▼)** - Full visibility
  - **Scrollable (◐)** - Limited height (168px) with scrollbar for longer content
  - **Summary (▬)** - Single line (2.25em) with ellipsis for quick preview
- **Code cells default to scrollable output** - Both new code cells and existing code cells loaded from notebooks now start with output in scrollable mode for better screen space usage
- **Section collapse buttons** - Visual indicators in cell header show current collapse state for each section
- **Keyboard shortcuts for collapsing**:
  - `Z` - Cycle input collapse level
  - `Shift+Z` - Cycle output collapse level
  - `Alt+Z` - Cycle both input and output together
- **Persistent collapse state** - Collapse levels are saved to notebook metadata and restored on load

### Technical Changes

- Added `CollapseLevel` enum (EXPANDED=0, SCROLLABLE=1, SUMMARY=2)
- Added `input_collapse` and `output_collapse` fields to Cell dataclass
- Added CSS classes: `.collapse-scrollable`, `.collapse-summary`
- Added CSS for section collapse buttons with level indicators
- Added JavaScript functions: `cycleCollapseLevel()`, `setCollapseLevel()`
- Added new route `/dialeng/{nb_id}/cell/{cid}/collapse-section` for updating section collapse state
- Updated `CellView()` to include collapse controls and data attributes
- Updated `to_jupyter_cell()` and `from_jupyter_cell()` to serialize/deserialize collapse levels
- New code cells default to `output_collapse=1` (scrollable output)
- Code cells loaded from disk without explicit `output_collapse` metadata default to scrollable (1) instead of expanded (0)

## [0.3.0] - 2024-12-09

### Added

#### Cell Folding/Collapsing
- **Collapse toggle button** - Each cell now has a ▼ button in the header to collapse/expand cell content
- **Persistent collapse state** - Collapsed state is saved per cell and persists across page reloads
- **Visual feedback** - Collapsed cells show reduced opacity and the collapse button rotates to indicate state

#### Dark/Light Theme Toggle
- **Theme toggle button** - Sun/Moon icon (☀️/🌙) in the toolbar to switch between dark and light themes
- **Light theme CSS** - Complete light theme with GitHub-inspired color palette
- **Ace Editor theme sync** - Code editor automatically switches between Monokai (dark) and Chrome (light) themes
- **localStorage persistence** - Theme preference is saved and restored across sessions

#### Mobile-Responsive Layout
- **Tablet breakpoint (768px)** - Responsive toolbar, stacked cell headers, and full-width buttons
- **Mobile breakpoint (480px)** - Compact layout with smaller fonts, tighter padding, and optimized touch targets
- **Responsive Ace Editor** - Code editor adjusts minimum height on smaller screens
- **Fluid notebook list** - Notebooks stack vertically on mobile for better navigation

#### Real-time Token Streaming
- **🧠 Thinking indicator** - When `use_thinking` is enabled, shows animated "🧠 Thinking..." in the AI response area and cell header
- **Visual streaming feedback** - Cell border turns orange during generation
- **Smooth animation** - Thinking indicator pulses with a subtle animation

#### Cancel Generation
- **Cancel button (⏹)** - Appears during streaming to stop AI generation mid-stream
- **WebSocket cancellation** - Properly signals the server to stop generation
- **Clean state handling** - Run button re-appears after cancellation or completion
- **Server-side tracking** - Uses a global set to track cancelled cells across async operations

### Technical Changes

- Added light theme CSS variables under `[data-theme="light"]` selector
- Added `cancelled_cells` global set for tracking cancelled generations
- Updated `mock_llm_stream()` to yield dictionaries with `type` field for different message types
- Added WebSocket message handler for `cancel` type messages
- Added new route `/dialeng/{nb_id}/cell/{cid}/collapse` for toggling cell collapse state
- Added Chrome theme CDN for Ace Editor (light mode)
- Added responsive CSS media queries for 768px and 480px breakpoints
- Added JavaScript functions: `toggleTheme()`, `loadTheme()`, `toggleCollapse()`, `cancelStreaming()`, `showThinkingIndicator()`, `hideThinkingIndicator()`

## [0.2.0] - 2024-12-09

### Added

#### Code Editor Improvements
- **Ace Editor with Monokai theme** - Full syntax highlighting for Python code cells using Ace Editor
- **Persistent editor state** - Editor content and syntax highlighting preserved after cell execution
- **Jupyter-style output for last expression** - The last expression in a code cell is automatically displayed (e.g., `x` at the end of a cell shows its value), just like Jupyter notebooks
- **AST-based code execution** - Uses Python's `ast` module to separate statements from trailing expressions for proper output handling

#### Cell Navigation & Execution Flow
- **Auto-select next cell after execution** - After running any cell (code, prompt, or note), focus automatically moves to the next cell
- **Shift+Enter on note cells** - Pressing Shift+Enter on note cells now moves to the next cell (previously did nothing)
- **Smart focus management** - Code cells focus the Ace editor, prompt cells focus the textarea, note cells just select the cell

#### Prompt Cell Enhancements
- **Mock LLM echo** - The mock LLM now echoes the user's prompt in the response for easier testing
- **Markdown preview for user prompts** - After running a prompt cell, the user's input is rendered as markdown (similar to note cells)
- **Double-click to edit** - Both user prompts and AI responses can be edited by double-clicking the rendered preview

#### UI/UX Improvements
- **Cell selection on click** - Clicking anywhere on a cell (not just inside editors) now selects it and adds the focused indicator
- **Efficient partial re-rendering** - Running a cell only re-renders that specific cell, not all cells (improved performance)
- **HTMX Out-of-Band swaps** - New cells are added efficiently using OOB swaps without full page refresh
- **Proper AddButtons cleanup** - Deleting a cell now properly removes associated "Add Cell" buttons (no more orphaned buttons)

### Fixed

- **Ace Editor positioning** - Fixed issue where code appeared at top-left of page after running cell (added `position: relative` to container)
- **Syntax highlighting after run** - Fixed issue where syntax highlighting was lost after cell execution (async theme/mode loading)
- **Keyboard shortcuts with unfocused cells** - Fixed Shift+Enter not working when cell was selected but no element was focused

### Technical Changes

- Added `ast` module import for code execution parsing
- Added `focusNextCell(cellId)` JavaScript function for programmatic cell navigation
- Added `moveToNextCell(currentCell)` JavaScript function for DOM-based next cell lookup
- Modified HTMX event handlers (`htmx:beforeSwap`, `htmx:afterSettle`) to only process affected cells
- Changed run endpoint to return `Script` elements for client-side focus management
- Updated keyboard handlers to use `getFocusedCellId()` for reliable cell identification

## [0.1.0] - 2024-XX-XX

### Added

- Initial MVP release
- Three cell types: Code, Note, Prompt
- Editable AI responses in prompt cells
- Python kernel with persistent namespace
- Solveit-compatible .ipynb serialization
- HTMX reactive UI
- Basic keyboard shortcuts (Shift+Enter, Ctrl+S, D D, etc.)
- Mock LLM for testing
- File-based notebook storage
- FastHTML web framework integration
