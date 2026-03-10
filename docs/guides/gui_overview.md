# Dialeng GUI Overview

This guide walks through every section of the Dialeng notebook interface, explaining what each element does and how to use it.

## Table of Contents

- [Page Layout](#page-layout)
- [Header & Toolbar](#header--toolbar)
- [File List](#file-list)
- [Cell Area](#cell-area)
- [Cell Types](#cell-types)
  - [Code Cell](#code-cell)
  - [Note Cell](#note-cell)
  - [Prompt Cell](#prompt-cell)
  - [Shell Cell](#shell-cell)
- [Cell Header Controls](#cell-header-controls)
  - [State Indicators](#state-indicators)
  - [State Toggle Buttons](#state-toggle-buttons)
  - [Collapse Controls](#collapse-controls)
  - [Action Buttons](#action-buttons)
- [Add Cell Buttons](#add-cell-buttons)
- [Outline Sidebar](#outline-sidebar)
- [Settings Sidebar](#settings-sidebar)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Theme](#theme)

---

## Page Layout

The interface is split into three main areas arranged horizontally:

```
┌──────────────────────────────────────────────────────────┐
│                        Header & Toolbar                   │
├──────────┬───────────────────────────────────────────────┤
│          │  File List (notebook tabs)                     │
│ Outline  ├───────────────────────────────────────────────┤
│ Sidebar  │                                               │
│ (toggle) │  Cell Area                                    │
│          │  ┌─────────────────────────────────────────┐  │
│          │  │ [+ Code] [+ Note] [+ Prompt] [+ Shell] │  │
│          │  ├─────────────────────────────────────────┤  │
│          │  │ Cell 1                                  │  │
│          │  ├─────────────────────────────────────────┤  │
│          │  │ [+ Code] [+ Note] [+ Prompt] [+ Shell] │  │
│          │  ├─────────────────────────────────────────┤  │
│          │  │ Cell 2                                  │  │
│          │  ├─────────────────────────────────────────┤  │
│          │  │ ...                                     │  │
│          │  └─────────────────────────────────────────┘  │
├──────────┴──────────────────────────────────┬────────────┤
│                                             │  Settings  │
│                                             │  Sidebar   │
│                                             │  (toggle)  │
└─────────────────────────────────────────────┴────────────┘
```

- **Outline Sidebar** (left): Push-style sidebar showing headings, variables, and functions. Toggle with the outline button or `Ctrl+Shift+O`.
- **Main Content** (center): Header, file list, and all notebook cells.
- **Settings Sidebar** (right): Overlay-style panel for configuration. Toggle with the `⚙️` button.

---

## Header & Toolbar

The header bar sits at the top of the page and contains the notebook title and a toolbar with all global controls.

### Title Bar

Displays the notebook icon (📓) and title on the left side.

### Toolbar Controls (left to right)

| Control | Description |
|---------|-------------|
| **Outline toggle** | Opens/closes the outline sidebar. Keyboard: `Ctrl+Shift+O` |
| **Theme toggle** (☀️/🌙) | Switches between dark and light themes. Preference is saved to your browser. |
| **Dialog Mode** selector | Dropdown to choose the AI conversation mode: **Mock** (no LLM calls), **Learning**, **Concise**, **Standard**. |
| **Model** selector | Dropdown to pick which Claude model to use. Hidden when mode is "Mock". |
| **Kernel Type** selector | Choose **Local Python** or **Google Colab**. Only visible if Colab integration is enabled. |
| **Colab status dot** | Green = connected, gray = disconnected. Only visible in Colab mode. |
| **Colab Runtime** selector | Choose CPU, GPU (T4), or TPU. Only visible when Colab is selected AND authenticated. |
| **Colab Auth** button | "Connect Colab" (opens Google sign-in popup) or "Disconnect". |
| **Safe Mode** checkbox | When checked, shell commands are validated against an allowlist before execution. Requires `shfmt` to be installed. |
| **Restart** button | Restarts the Python/Colab kernel. All variables in memory are lost. |
| **Cancel All** button (⏹) | Cancels the currently running cell and clears the execution queue. Only visible when cells are running. Keyboard: `Esc Esc` (double press). |
| **Save** button (💾) | Saves the notebook to disk. Keyboard: `Ctrl+S` / `Cmd+S`. |
| **Export** button (📥) | Downloads the notebook as an `.ipynb` file (Jupyter-compatible). |
| **Settings** button (⚙️) | Opens the settings sidebar on the right. |

---

## File List

Below the header, a horizontal tab bar shows all notebooks in the workspace. The currently open notebook is highlighted. Click any name to switch notebooks. A **"+ New"** link at the end creates a new notebook.

---

## Cell Area

The main content area contains all cells in the notebook, with "add cell" button rows between each cell.

---

## Cell Types

### Code Cell

Code cells contain Python code that runs in the kernel.

```
┌──────────────────────────────────────────────────────────┐
│ [▼] CODE  [execution count] [run time]                   │
│      In: [▼/◐/▬]  Out: [▼/◐/▬]                          │
│                     [👁] [📌] [📤] [type ▾] [▶] [↑] [↓] [×]│
├──────────────────────────────────────────────────────────┤
│ ┌── Ace Editor (Python) ──────────────────────────────┐  │
│ │ import pandas as pd                                 │  │
│ │ df = pd.read_csv("data.csv")                        │  │
│ │ df.head()                                           │  │
│ └─────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│ Output:                                                   │
│   <rendered output from kernel>                          │
└──────────────────────────────────────────────────────────┘
```

- **Editor**: Ace editor with Python syntax highlighting, auto-indent, and bracket matching. Theme follows the global dark/light setting (Monokai / Chrome).
- **Output**: Shows `stdout`, `stderr`, rendered HTML, images, and errors. Errors are highlighted with a red accent.
- **Run**: Press `Shift+Enter` to run and advance to the next cell, or `Ctrl+Enter` to run and stay.

### Note Cell

Note cells hold Markdown text. They serve as documentation, headings, or annotations.

```
┌──────────────────────────────────────────────────────────┐
│ [▼] NOTE                                                  │
│      [▼/◐/▬]                                              │
│                     [👁] [📌] [📤] [type ▾] [↑] [↓] [×]    │
├──────────────────────────────────────────────────────────┤
│ ┌── Markdown Preview ─────────────────────────────────┐  │
│ │ ## My Section Heading                               │  │
│ │ Some explanation text with **bold** and *italic*.   │  │
│ └─────────────────────────────────────────────────────┘  │
│ (Double-click to edit | Escape to finish | Z to collapse)│
└──────────────────────────────────────────────────────────┘
```

- **Preview mode** (default): Rendered Markdown. Headings, bold, italic, code blocks (with copy button), lists, and links are supported.
- **Edit mode**: Double-click the preview to switch to a raw Markdown textarea. Press `Escape` or click outside to return to preview.
- Note cells have no run button and no output section.

### Prompt Cell

Prompt cells let you interact with the AI. You write a prompt and the AI generates a response.

```
┌──────────────────────────────────────────────────────────┐
│ [▼] PROMPT                                                │
│      In: [▼/◐/▬]  Out: [▼/◐/▬]                          │
│                     [👁] [📌] [📤] [type ▾] [▶] [⏹] [↑][↓][×]│
├──────────────────────────────────────────────────────────┤
│ 👤 Your Prompt                                            │
│ ┌─────────────────────────────────────────────────────┐  │
│ │ Explain how pandas groupby works with an example    │  │
│ └─────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│ 🤖 AI Response (double-click to edit)                     │
│ ┌─────────────────────────────────────────────────────┐  │
│ │ The `groupby()` method splits data into groups...   │  │
│ │ ```python                                           │  │
│ │ df.groupby('category').mean()                       │  │
│ │ ```                                                 │  │
│ └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

- **User prompt**: Editable textarea before running. After running, it switches to a Markdown preview (double-click to re-edit).
- **AI response**: Streams in real-time as the model generates. Rendered as Markdown with syntax-highlighted code blocks.
- **Cancel** (⏹): Stops the AI generation mid-stream.
- The AI sees the full notebook context (respecting pinned/skipped states) when generating a response.

### Shell Cell

Shell cells execute bash commands. Only available if shell cells are enabled in settings.

```
┌──────────────────────────────────────────────────────────┐
│ [V] SHELL  [Safe]  [execution count] [run time]          │
│      In: [▼/◐/▬]  Out: [▼/◐/▬]                          │
│                     [👁] [📌] [📤] [type ▾] [▶] [↑] [↓] [×]│
├──────────────────────────────────────────────────────────┤
│ ┌── Ace Editor (Bash) ────────────────────────────────┐  │
│ │ ls -la /tmp                                         │  │
│ │ echo "Hello from shell"                             │  │
│ └─────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│ Output:                                                   │
│   total 128                                              │
│   drwxrwxrwt  12 root root ...                           │
│   Hello from shell                                       │
└──────────────────────────────────────────────────────────┘
```

- **Editor**: Ace editor with bash syntax highlighting.
- **Safe mode badge**: Shows "Safe" in the header when safe mode is enabled. Commands are validated against an allowlist before execution.
- **Output**: Shows command output. Errors and `DisallowedCmd` messages are highlighted.

---

## Cell Header Controls

Every cell has a header bar with controls split into two groups: info on the left, actions on the right.

### State Indicators

When a cell has an active state, colored badges appear in the header:

| Badge | Meaning | Visual Effect on Cell |
|-------|---------|----------------------|
| **HIDDEN** (red) | Cell is hidden from AI context. The AI will not see this cell's content when generating responses. | Cell dims to 50% opacity with a red left border. |
| **PINNED** (blue) | Cell is pinned. Its content is always included in the AI context window, regardless of how many cells are in the notebook. | Blue left border. |
| **EXPORT** (green) | Cell is marked for export. Used to tag cells for inclusion in exported artifacts. | Green left border. |

### State Toggle Buttons

Three icon buttons control cell states. They appear slightly transparent when inactive and fully opaque when active:

| Button | Icon (inactive → active) | Keyboard | What it does |
|--------|--------------------------|----------|--------------|
| **Visibility** | eye → eye-closed | `h` | Toggles whether the AI can see this cell. Hidden cells are excluded from the context sent to the LLM. |
| **Pin** | pin-off → pin | `p` | Toggles pin. Pinned cells are always included in the AI context, even if they fall outside the normal context window. |
| **Export** | bookmark → bookmark-check | `e` | Toggles export flag. Marks the cell for inclusion in exports. |

These buttons update instantly across all connected browser tabs via WebSocket.

### Collapse Controls

Cells with input/output sections have collapse buttons that cycle through three levels:

| Symbol | Level | Effect |
|--------|-------|--------|
| **▼** | Expanded | Full content visible |
| **◐** | Scrollable | Content shown in a scrollable container with max height |
| **▬** | Summary | Minimal height, just a sliver of content visible |

- **Code/Shell/Prompt cells**: Separate collapse for "In:" (input/editor) and "Out:" (output/response).
- **Note cells**: Single collapse control for the content.
- The **▼** button on the far left collapses the entire cell (both input and output).

Keyboard shortcuts for collapse:
- `Z` — Cycle input collapse
- `Shift+Z` — Cycle output collapse
- `Alt+Z` — Cycle both
- `0`-`3` — Set input to specific level
- `Shift+0`-`3` — Set output to specific level
- `Alt+0`-`3` — Set both to specific level

### Action Buttons

| Button | What it does |
|--------|-------------|
| **Type selector** (dropdown) | Change the cell type (code, note, prompt, shell). The cell is re-rendered as the new type. |
| **Run** (▶) | Execute the cell. For code/shell: runs in kernel. For prompt: sends to AI. |
| **Cancel** (⏹) | Stop execution (code/shell) or cancel AI generation (prompt). Only visible while running. |
| **Move Up** (↑) | Move the cell one position up in the notebook. Keyboard: `Alt+↑` |
| **Move Down** (↓) | Move the cell one position down. Keyboard: `Alt+↓` |
| **Delete** (×) | Remove the cell from the notebook. Keyboard: `D D` (press D twice) or `Ctrl+Backspace` |

---

## Add Cell Buttons

Between every pair of cells (and at the top), a row of buttons lets you insert new cells at that position:

```
[+ Code]  [+ Note]  [+ Prompt]  [+ Shell]
```

- **+ Code**: Insert a Python code cell
- **+ Note**: Insert a Markdown note cell
- **+ Prompt**: Insert an AI prompt cell
- **+ Shell**: Insert a bash shell cell (only visible if shell cells are enabled in settings)

You can also add cells at the end of the notebook with keyboard shortcuts:
- `Ctrl+Shift+C` — Add code cell
- `Ctrl+Shift+N` — Add note cell
- `Ctrl+Shift+P` — Add prompt cell

---

## Outline Sidebar

The outline sidebar is a left-side panel that provides a table-of-contents view of your notebook. Toggle it with the outline button in the toolbar or `Ctrl+Shift+O`.

It has three sections:

### Headings

Lists all Markdown headings from note cells. Click a heading to scroll to that cell. Headings are indented by level (H1, H2, H3, etc.).

### Variables

Shows variables currently defined in the kernel (from code cell execution). Displays name, type, and a short preview of the value. Click to jump to the cell where the variable was defined.

### Functions

Shows functions defined in the kernel. Displays name and signature. Click to jump to the defining cell.

The outline auto-refreshes when cells are executed or modified.

---

## Settings Sidebar

Click the **⚙️** button in the toolbar to open the settings panel. It slides in from the right with a semi-transparent overlay behind it (click the overlay to close).

Settings are organized in collapsible groups:

### AWS Settings
- **Region**: Select the AWS region for Bedrock API calls (e.g., us-east-1, us-west-2).

### Model Defaults
- **Bedrock Model**: Default model for Bedrock-based calls.
- **Claude Code Model**: Model used for code-related tasks.
- **Default Dialog Mode**: Which mode new notebooks start in (Mock, Learning, Concise, Standard).

### Tool Settings
- **Max Tool Steps**: Slider (1-10) controlling how many tool-use rounds the AI can perform per prompt.
- **Require Confirmation**: Whether to ask before executing tool calls.
- **Enable Built-in Tools**: Toggle built-in tool definitions on/off.

### Display Settings
- **Reasoning Text Limit**: Max characters of AI reasoning/thinking text to display (0-10000).

### Shell Settings
- **Enable Shell Cells**: Toggle shell cell type availability. Requires app restart.

### Google Colab
- **Enable Colab**: Toggle Colab kernel integration. Requires app restart.

### Advanced
- **Thinking Max Tokens**: Maximum tokens for AI thinking/reasoning (0-50000).
- **Use SDK Directly**: Bypass the service layer and call the Claude SDK directly. Requires restart.
- **Debug Mode**: Enable verbose logging.
- **Debug Log Directory**: Path where debug logs are written.

Click **Save Settings** at the bottom to apply changes.

---

## Keyboard Shortcuts

### When focused on a cell (outside the editor)

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell and move to next |
| `Ctrl/Cmd+Enter` | Run cell (stay in place) |
| `Ctrl/Cmd+S` | Save notebook |
| `D D` | Delete cell (press D twice quickly) |
| `Ctrl/Cmd+Backspace` | Delete cell |
| `Esc` | Exit edit mode / blur active element |
| `Esc Esc` | Cancel all running cells (double press within 500ms) |
| `h` | Toggle hidden (skip from AI) |
| `p` | Toggle pinned |
| `e` | Toggle export |
| `Z` | Cycle input collapse |
| `Shift+Z` | Cycle output collapse |
| `Alt+Z` | Cycle both input and output collapse |
| `0`-`3` | Set input collapse level |
| `Shift+0`-`3` | Set output collapse level |
| `Alt+0`-`3` | Set both collapse levels |
| `Alt+↑` or `Ctrl+Shift+↑` | Move cell up |
| `Alt+↓` or `Ctrl+Shift+↓` | Move cell down |
| `Ctrl+Shift+C` | Add code cell at end |
| `Ctrl+Shift+N` | Add note cell at end |
| `Ctrl+Shift+P` | Add prompt cell at end |

### Inside the Ace editor

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell and move to next |
| `Ctrl/Cmd+Enter` | Run cell |
| `Ctrl/Cmd+S` | Save notebook |

> **Note**: Single-key shortcuts (`h`, `p`, `e`, `D D`, `Z`, number keys) only work when you are **not** typing in an editor or text input. Click outside the editor or press `Esc` first.

---

## Theme

Dialeng supports dark and light themes. Click the theme toggle button (☀️ in dark mode, 🌙 in light mode) in the toolbar to switch.

- **Dark theme**: Dark background with Monokai syntax highlighting in editors.
- **Light theme**: Light background with Chrome syntax highlighting in editors.

Your preference is saved in the browser and persists across sessions.
