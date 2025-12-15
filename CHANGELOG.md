# Changelog

All notable changes to LLM Notebook will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **New endpoint** - `POST /notebook/{nb_id}/queue/cancel_all` - cancels running + clears queue
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
- Added `/notebook/{nb_id}/kernel/interrupt` route for hard interrupt
- Updated `/notebook/{nb_id}/kernel/restart` route (was `/kernel/restart`)

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
  - `/notebook/{nb_id}/cell/add` - Broadcasts cells_updated
  - `/notebook/{nb_id}/cell/{cid}` (DELETE) - Broadcasts cells_updated
  - `/notebook/{nb_id}/cell/{cid}/move/{direction}` - Broadcasts cells_updated
  - `/notebook/{nb_id}/cell/{cid}/type` - Broadcasts cell_updated
  - `/notebook/{nb_id}/cell/{cid}/collapse` - Broadcasts cell_updated
  - `/notebook/{nb_id}/cell/{cid}/collapse-section` - Broadcasts cell_updated
  - `/notebook/{nb_id}/cell/{cid}/run` - Broadcasts cell_updated (for code cells and final prompt state)
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
- Added new route `/notebook/{nb_id}/cell/{cid}/collapse-section` for updating section collapse state
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
- Added new route `/notebook/{nb_id}/cell/{cid}/collapse` for toggling cell collapse state
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
