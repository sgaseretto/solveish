# LLM Notebook 📓

An open-source Solveit-like notebook built entirely with **FastHTML**. Features prompt cells with editable LLM responses, Python code execution, and markdown notes.

**Full `.ipynb` compatibility** - Uses the Solveit convention for prompt cells, making notebooks interoperable with Jupyter and Solveit.

## ✨ Key Features

- **Three cell types**: Code, Note, and Prompt
- **Editable AI responses**: Both user prompts AND AI responses are fully editable
- **Solveit-compatible**: Standard `.ipynb` format with Solveit metadata conventions
- **Pure Python**: Built with FastHTML, no JavaScript frameworks
- **Mock LLM included**: Works out of the box without API keys
- **Dark/Light theme**: Toggle between themes with persistent localStorage saving
- **Mobile responsive**: Works on tablets and phones with adaptive layout
- **Cell folding**: Collapse cells to focus on what matters
- **Cancel generation**: Stop AI responses mid-stream with the ⏹ button
- **🧠 Thinking mode**: Visual indicator during AI thinking process

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Quick start and overview (this file) |
| **[DEVELOPERS.md](DEVELOPERS.md)** | Developer guide for extending the project |
| **[ROADMAP.md](ROADMAP.md)** | Planned features and contribution opportunities |

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo>
cd llm_notebook
pip install -r requirements.txt

# Run (no API key needed - uses mock LLM)
python app.py

# Open http://localhost:8000
```

Notebooks are saved to `./notebooks/*.ipynb`.

---

## 🎯 The Prompt Cell Innovation

Unlike traditional notebooks (Jupyter) or chat interfaces (ChatGPT), LLM Notebook introduces a **prompt cell** that bridges the gap:

| Cell Type | Purpose | Output | Editable? |
|-----------|---------|--------|-----------|
| **Code** | Execute Python | Stdout/stderr | Source only |
| **Note** | Documentation | Rendered markdown | Yes |
| **Prompt** | Chat with AI | LLM response | **Both parts!** |

```
┌─────────────────────────────────────────┐
│ PROMPT                               ▶  │
├─────────────────────────────────────────┤
│ 👤 Your Prompt                          │
├─────────────────────────────────────────┤
│ How do I reverse a list in Python?      │ ← Editable
├─────────────────────────────────────────┤
│ 🤖 AI Response (editable)               │
├─────────────────────────────────────────┤
│ You can reverse a list using:           │ ← Also editable!
│ - `list.reverse()` (in-place)           │
│ - `list[::-1]` (creates new list)       │
└─────────────────────────────────────────┘
```

---

## 📄 .ipynb Format (Solveit-Compatible)

LLM Notebook uses the Solveit convention for storing prompt cells in standard `.ipynb` files:

### Cell Type Mapping

| LLM Notebook | Jupyter Cell | Key Metadata |
|--------------|--------------|--------------|
| **Code** | `cell_type: "code"` | `time_run`, `skipped` |
| **Note** | `cell_type: "markdown"` | *(no `solveit_ai`)* |
| **Prompt** | `cell_type: "markdown"` | `solveit_ai: true` |

### Prompt Cell Format

Prompt cells store both user input and AI response in one markdown cell:

```markdown
What is the capital of France?

##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_7f3a9b2c -->

The capital of France is **Paris**.
```

The separator `##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_xxx -->` divides:
- **Before**: User's prompt (editable)
- **After**: AI's response (also editable!)

### Notebook Metadata

```json
{
  "metadata": {
    "solveit_dialog_mode": "learning",
    "solveit_ver": 2
  }
}
```

### Round-Trip Compatibility

```python
from app import Notebook

# Load any Solveit notebook
nb = Notebook.load("solveit_dialog.ipynb")

# Edit cells programmatically
nb.cells[2].output = "Updated AI response"

# Save - preserves Solveit format
nb.save("solveit_dialog.ipynb")
```

---

## ⌨️ Keyboard Shortcuts (Jupyter-style)

Works on both Windows/Linux (Ctrl) and macOS (Cmd):

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run current cell (Jupyter style) |
| `Ctrl/Cmd+Enter` | Run current cell (alternative) |
| `Ctrl/Cmd+S` | Save notebook |
| `D D` | Delete cell (press D twice, Jupyter style) |
| `Ctrl/Cmd+Backspace` | Delete current cell |
| `Ctrl/Cmd+Shift+C` | Add code cell at end |
| `Ctrl/Cmd+Shift+N` | Add note cell at end |
| `Ctrl/Cmd+Shift+P` | Add prompt cell at end |
| `Alt+↑` | Move cell up |
| `Alt+↓` | Move cell down |
| `Escape` | Exit edit mode |
| `Double-click` | Edit markdown/AI response |
| `Z` | Cycle input collapse level |
| `Shift+Z` | Cycle output collapse level |
| `Alt+Z` | Cycle both input and output collapse |

## 🎨 Ace Editor

Code cells use the **Ace Editor** for proper syntax highlighting:
- Python syntax highlighting
- Auto-indentation
- Bracket matching
- Keyboard shortcuts work inside the editor
- **Theme-aware**: Monokai (dark) / Chrome (light) based on app theme

## 🔌 WebSocket Streaming

LLM responses stream in real-time via WebSocket:

```javascript
// Automatic connection on page load
connectWebSocket(notebookId);

// Receives chunks as they're generated
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'stream_chunk') {
        appendToResponse(data.cell_id, data.chunk);
    }
};
```

## 📝 Markdown Features

- **Code highlighting** with copy button
- **Double-click to edit** rendered markdown
- **Escape to exit** edit mode
- Supports headers, lists, bold, italic, code blocks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastHTML Frontend                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐    │
│  │Code Cell │  │Note Cell │  │     Prompt Cell        │    │
│  │[Python]  │  │[Markdown]│  │[User Input]            │    │
│  │[Output]  │  │[Preview] │  │[Editable AI Response]  │    │
│  └──────────┘  └──────────┘  └────────────────────────┘    │
│                         │                                   │
│                    HTMX Requests                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastHTML Backend                         │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Python Kernel  │  │ Context Builder │  │ LLM Client  │  │
│  │ (exec/eval)    │  │ (aggregates     │  │ (Mock/API)  │  │
│  │                │  │  visible cells) │  │             │  │
│  └────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Notebook Storage (.ipynb files)           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User types in cell** → HTMX posts to `/cell/{id}/source` → Updates in-memory state
2. **User clicks Run (Code)** → Python kernel executes → Returns output
3. **User clicks Run (Prompt)** → Context aggregated → LLM generates response
4. **User edits AI response** → `/cell/{id}/output` → Updates stored response
5. **User saves** → Notebook serialized to `.ipynb` file

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run current cell |
| `Ctrl+S` | Save notebook |

---

## 🔧 Extending LLM Notebook

### Swap LLM Provider

Replace the mock LLM with a real provider:

```python
# In app.py, replace mock_llm_stream with:

import anthropic

async def real_llm_stream(prompt: str, context: str):
    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"{context}\n\n{prompt}"}]
    ) as stream:
        for text in stream.text_stream:
            yield text
```

### Add New Cell Type

1. Add to `CellType` enum
2. Add rendering in `CellView()`
3. Add execution in `/cell/{id}/run`
4. Add serialization in `to_jupyter_cell()`

### Add Persistence Backend

```python
class MyStorage:
    def save(self, nb: Notebook): ...
    def load(self, id: str) -> Notebook: ...
    def list(self) -> list[str]: ...
```

See **[DEVELOPERS.md](DEVELOPERS.md)** for complete examples and patterns.

---

## 🌗 Theme & UI Features

### Theme Toggle
Click the ☀️/🌙 button in the toolbar to switch between dark and light themes. Your preference is saved to localStorage.

### Cell Folding
Click the ▼ button on any cell to collapse it completely. Collapsed cells show only the header. Click again to expand.

### Multi-Level Section Collapse
Each cell has independent collapse controls for input and output sections:

| Icon | Level | Description |
|------|-------|-------------|
| ▼ | Expanded | Full visibility |
| ◐ | Scrollable | 168px max height with scrollbar (default for code cells) |
| ▬ | Summary | Single line with ellipsis |

**Keyboard shortcuts:**
- `Z` - Cycle input collapse level
- `Shift+Z` - Cycle output collapse level
- `Alt+Z` - Cycle both together

New code cells default to scrollable output mode for better screen space usage. Collapse states are saved per cell and persist across page reloads.

### Mobile Support
The interface adapts to smaller screens:
- **Tablet (768px)**: Stacked toolbar, full-width buttons
- **Mobile (480px)**: Compact layout, optimized touch targets

### Cancel Generation
During AI response streaming, click the ⏹ button to stop generation. The run button (▶) returns after cancellation.

## 🗺️ Roadmap

| Phase | Key Features |
|-------|--------------|
| **v0.2** ✓ | Ace Editor, cell navigation, output improvements |
| **v0.3** ✓ | Theme toggle, mobile responsive, cell folding, cancel streaming |
| **v0.4** | Real LLM integration, context management, rich outputs |
| **v0.5** | Real-time collaboration, authentication, cloud storage |
| **v1.0** | Module export, full Solveit feature parity |

See **[ROADMAP.md](ROADMAP.md)** for detailed plans and contribution opportunities.

---

## 📁 Project Structure

```
llm_notebook/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # This file
├── DEVELOPERS.md      # Developer guide
├── ROADMAP.md         # Feature roadmap
└── notebooks/         # Saved notebooks (created at runtime)
    └── *.ipynb
```

---

## 🤝 Contributing

This is an open-source alternative to Solveit. Contributions welcome!

**Good first issues:**
- Add real LLM provider integration (Claude, OpenAI)
- Write unit tests
- Add cell execution queue
- Implement cell pinning UI
- Add notebook title editing

**See [ROADMAP.md](ROADMAP.md)** for the full list of planned features.

---

## 📚 References

- [Solveit](https://solve.it.com) - The original inspiration
- [FastHTML](https://docs.fastht.ml) - The framework
- [HTMX](https://htmx.org) - For interactivity
- [Answer.AI](https://www.answer.ai) - FastHTML creators

---

## License

MIT
