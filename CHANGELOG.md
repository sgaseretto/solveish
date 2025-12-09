# Changelog

All notable changes to LLM Notebook will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
