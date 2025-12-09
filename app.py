"""
LLM Notebook - Open source Solveit-like notebook with FastHTML

Features:
- Three cell types: Code, Note, Prompt
- Prompt cells with BOTH user input and AI response editable
- Full .ipynb serialization/deserialization following Solveit conventions
- Python kernel with persistent state
- WebSocket streaming for LLM responses
- Keyboard shortcuts (Ctrl+Enter, Ctrl+S, etc.)
- Code highlighting with copy button
- Markdown rendering with double-click to edit
"""

from fasthtml.common import *
from fastcore.utils import *
import uuid, json, os, sys, io, traceback, asyncio, re, ast
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from enum import Enum
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

SOLVEIT_VER = 2
SEPARATOR_PREFIX = "##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_"
SEPARATOR_SUFFIX = " -->"
SEPARATOR_PATTERN = re.compile(r'##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_([a-f0-9]+) -->')

def make_separator() -> str:
    """Generate a new separator with random ID"""
    sep_id = uuid.uuid4().hex[:8]
    return f"{SEPARATOR_PREFIX}{sep_id}{SEPARATOR_SUFFIX}"

def split_prompt_content(content: str) -> tuple[str, str]:
    """Split prompt cell content into (user_prompt, ai_response)"""
    match = SEPARATOR_PATTERN.search(content)
    if match:
        idx = match.start()
        user_prompt = content[:idx].strip()
        after_sep = content[match.end():]
        ai_response = after_sep.strip()
        return user_prompt, ai_response
    else:
        return content.strip(), ""

def join_prompt_content(user_prompt: str, ai_response: str) -> str:
    """Join user prompt and AI response with separator"""
    if not ai_response:
        return user_prompt
    return f"{user_prompt}\n\n{make_separator()}\n\n{ai_response}"

# ============================================================================
# Data Models
# ============================================================================

class CellType(str, Enum):
    CODE = "code"
    NOTE = "note"
    PROMPT = "prompt"

@dataclass
class Cell:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    cell_type: str = CellType.CODE.value
    source: str = ""
    output: str = ""
    execution_count: Optional[int] = None
    time_run: str = ""
    skipped: bool = False
    use_thinking: bool = False
    collapsed: bool = False
    pinned: bool = False
    is_exported: bool = False
    
    def to_jupyter_cell(self) -> Dict[str, Any]:
        """Convert to Jupyter .ipynb cell format"""
        if self.cell_type == CellType.CODE.value:
            cell = {
                "cell_type": "code",
                "id": self.id,
                "metadata": {},
                "source": self._to_source_lines(self.source),
                "execution_count": self.execution_count,
                "outputs": self._format_outputs(self.output) if self.output else []
            }
            if self.time_run: cell["metadata"]["time_run"] = self.time_run
            if self.skipped: cell["metadata"]["skipped"] = True
            if self.is_exported: cell["metadata"]["is_exported"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
        elif self.cell_type == CellType.NOTE.value:
            cell = {
                "cell_type": "markdown",
                "id": self.id,
                "metadata": {},
                "source": self._to_source_lines(self.source)
            }
            if self.collapsed: cell["metadata"]["collapsed"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
        else:  # Prompt
            combined = join_prompt_content(self.source, self.output)
            cell = {
                "cell_type": "markdown",
                "id": self.id,
                "metadata": {"solveit_ai": True},
                "source": self._to_source_lines(combined)
            }
            if self.use_thinking: cell["metadata"]["use_thinking"] = True
            if self.time_run: cell["metadata"]["time_run"] = self.time_run
            if self.collapsed: cell["metadata"]["collapsed"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
        return cell
    
    @classmethod
    def from_jupyter_cell(cls, cell: Dict[str, Any]) -> "Cell":
        cell_id = cell.get("id", uuid.uuid4().hex[:8])
        metadata = cell.get("metadata", {})
        source = cls._from_source_lines(cell.get("source", []))
        
        if cell["cell_type"] == "code":
            output = cls._extract_output(cell.get("outputs", []))
            return cls(
                id=cell_id, cell_type=CellType.CODE.value, source=source, output=output,
                execution_count=cell.get("execution_count"),
                time_run=metadata.get("time_run", ""),
                skipped=metadata.get("skipped", False),
                is_exported=metadata.get("is_exported", False),
                pinned=metadata.get("pinned", False)
            )
        else:
            if metadata.get("solveit_ai"):
                user_prompt, ai_response = split_prompt_content(source)
                return cls(
                    id=cell_id, cell_type=CellType.PROMPT.value,
                    source=user_prompt, output=ai_response,
                    use_thinking=metadata.get("use_thinking", False),
                    time_run=metadata.get("time_run", ""),
                    collapsed=metadata.get("collapsed", False),
                    pinned=metadata.get("pinned", False)
                )
            else:
                return cls(
                    id=cell_id, cell_type=CellType.NOTE.value, source=source,
                    collapsed=metadata.get("collapsed", False),
                    pinned=metadata.get("pinned", False)
                )
    
    @staticmethod
    def _to_source_lines(text: str) -> List[str]:
        if not text: return []
        lines = text.split('\n')
        return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    
    @staticmethod
    def _from_source_lines(source) -> str:
        if isinstance(source, list): return ''.join(source)
        return source or ""
    
    def _format_outputs(self, output: str) -> List[Dict]:
        if not output: return []
        lines = output.split('\n')
        text = [line + '\n' for line in lines[:-1]]
        if lines[-1]: text.append(lines[-1])
        return [{"output_type": "stream", "name": "stdout", "text": text if text else [output]}]
    
    @staticmethod
    def _extract_output(outputs: List[Dict]) -> str:
        result = []
        for out in outputs:
            if out.get("output_type") == "stream":
                text = out.get("text", [])
                result.append(''.join(text) if isinstance(text, list) else text)
            elif out.get("output_type") == "execute_result":
                data = out.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    result.append(''.join(text) if isinstance(text, list) else text)
            elif out.get("output_type") == "error":
                result.append('\n'.join(out.get("traceback", [])))
        return '\n'.join(result)


@dataclass
class Notebook:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "Untitled Notebook"
    cells: List[Cell] = field(default_factory=list)
    dialog_mode: str = "learning"
    
    def to_ipynb(self) -> Dict[str, Any]:
        return {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11.0"},
                "solveit_dialog_mode": self.dialog_mode,
                "solveit_ver": SOLVEIT_VER
            },
            "cells": [cell.to_jupyter_cell() for cell in self.cells]
        }
    
    @classmethod
    def from_ipynb(cls, data: Dict[str, Any], notebook_id: str = None) -> "Notebook":
        metadata = data.get("metadata", {})
        cells = [Cell.from_jupyter_cell(c) for c in data.get("cells", [])]
        return cls(
            id=notebook_id or uuid.uuid4().hex[:8],
            title="Imported Notebook", cells=cells,
            dialog_mode=metadata.get("solveit_dialog_mode", "learning")
        )
    
    def save(self, path: str):
        with open(path, 'w') as f: json.dump(self.to_ipynb(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Notebook":
        with open(path) as f: data = json.load(f)
        nb_id = Path(path).stem
        nb = cls.from_ipynb(data, nb_id)
        nb.title = Path(path).stem
        return nb

# ============================================================================
# Python Kernel
# ============================================================================

class PythonKernel:
    def __init__(self):
        self.namespace = {"__name__": "__main__"}
        self.execution_count = 0
        self._setup_builtins()
    
    def _setup_builtins(self):
        exec("import sys, os, json, math, random, datetime", self.namespace)
        
    def execute(self, code: str) -> tuple[str, str, bool, int]:
        """Execute code with Jupyter-style output for the last expression."""
        self.execution_count += 1
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        success = True
        last_expr_result = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Parse the code into an AST
                tree = ast.parse(code, '<cell>', 'exec')

                if tree.body:
                    last_stmt = tree.body[-1]

                    # Check if the last statement is an expression (not assignment, etc.)
                    if isinstance(last_stmt, ast.Expr):
                        # Execute all statements except the last
                        if len(tree.body) > 1:
                            exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                            exec(compile(exec_tree, '<cell>', 'exec'), self.namespace)

                        # Evaluate the last expression and capture its result
                        expr_tree = ast.Expression(body=last_stmt.value)
                        last_expr_result = eval(compile(expr_tree, '<cell>', 'eval'), self.namespace)
                    else:
                        # Last statement is not an expression, execute everything
                        exec(compile(tree, '<cell>', 'exec'), self.namespace)

        except Exception:
            success = False
            stderr_capture.write(traceback.format_exc())

        # Append the last expression result to stdout (Jupyter-style)
        stdout_val = stdout_capture.getvalue()
        if last_expr_result is not None:
            if stdout_val and not stdout_val.endswith('\n'):
                stdout_val += '\n'
            stdout_val += repr(last_expr_result)

        return (stdout_val, stderr_capture.getvalue(), success, self.execution_count)
    
    def restart(self):
        self.namespace = {"__name__": "__main__"}
        self.execution_count = 0
        self._setup_builtins()

kernel = PythonKernel()

# ============================================================================
# Mock LLM with Streaming
# ============================================================================

async def mock_llm_stream(prompt: str, context: str):
    """Mock LLM for demo (replace with real API)"""
    # Always echo the user's prompt first, then provide a response
    response = f"""You said:

> {prompt}

---

This is a **demo response**. In production, connect to Claude, OpenAI, or local models.

**Key features**:
- Both prompt AND response are editable
- Double-click to edit rendered markdown
- Press `Escape` to finish editing
- `Ctrl+Enter` runs cells"""
    
    # Stream word by word
    words = response.split(' ')
    for i, word in enumerate(words):
        yield word + (' ' if i < len(words) - 1 else '')
        await asyncio.sleep(0.02)

# ============================================================================
# Storage
# ============================================================================

notebooks: Dict[str, Notebook] = {}
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Track active WebSocket connections per notebook
ws_connections: Dict[str, Dict[int, Any]] = {}

def get_notebook(notebook_id: str) -> Notebook:
    """Get or create a notebook - ALWAYS requires notebook_id"""
    if notebook_id not in notebooks:
        path = NOTEBOOKS_DIR / f"{notebook_id}.ipynb"
        if path.exists():
            notebooks[notebook_id] = Notebook.load(str(path))
        else:
            nb = Notebook(id=notebook_id, title=notebook_id)
            nb.cells = [
                Cell(cell_type="note", source="# Welcome to LLM Notebook! 🚀\n\nAn open-source notebook with **prompt cells** for AI interaction.\n\n**Keyboard Shortcuts (Jupyter-style):**\n- `Shift+Enter` - Run cell (recommended)\n- `Ctrl/Cmd+Enter` - Run cell (alternative)\n- `Ctrl/Cmd+S` - Save notebook\n- `D D` - Delete cell (press D twice)\n- `Ctrl/Cmd+Shift+C` - Add code cell\n- `Ctrl/Cmd+Shift+N` - Add note cell\n- `Ctrl/Cmd+Shift+P` - Add prompt cell\n- `Alt+↑/↓` - Move cell up/down\n- `Escape` - Exit edit mode\n- Double-click - Edit markdown/response"),
                Cell(cell_type="code", source="# Try running some Python (Shift+Enter)\nx = [1, 2, 3, 4, 5]\nprint(f'Sum: {sum(x)}')\nprint(f'Average: {sum(x)/len(x)}')\nx"),
                Cell(cell_type="prompt", source="Hello! What can you help me with?"),
            ]
            notebooks[notebook_id] = nb
    return notebooks[notebook_id]

def save_notebook(notebook_id: str):
    if notebook_id in notebooks:
        path = NOTEBOOKS_DIR / f"{notebook_id}.ipynb"
        notebooks[notebook_id].save(str(path))

def list_notebooks() -> List[str]:
    return [p.stem for p in NOTEBOOKS_DIR.glob("*.ipynb")]

# ============================================================================
# CSS
# ============================================================================

css = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-cell: #21262d;
    --bg-input: #0d1117;
    --text-primary: #c9d1d9;
    --text-muted: #8b949e;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-purple: #bc8cff;
    --accent-orange: #d29922;
    --accent-red: #f85149;
    --border: #30363d;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

.container { max-width: 960px; margin: 0 auto; padding: 20px; }

/* Header */
.header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.title { font-size: 1.3rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.title-icon { font-size: 1.5rem; }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

/* Buttons */
.btn {
    padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px;
    background: var(--bg-secondary); color: var(--text-primary);
    cursor: pointer; font-size: 0.85rem; transition: all 0.15s;
    display: inline-flex; align-items: center; gap: 4px;
}
.btn:hover { background: var(--bg-cell); border-color: var(--accent-blue); }
.btn-sm { padding: 4px 8px; font-size: 0.75rem; }
.btn-icon { padding: 4px 6px; font-size: 0.9rem; }
.btn-run { background: var(--accent-green); border-color: var(--accent-green); color: #000; }
.btn-run:hover { background: #2ea043; }
.btn-save { background: var(--accent-blue); border-color: var(--accent-blue); }

.mode-select {
    padding: 4px 8px; background: var(--bg-cell); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text-primary); font-size: 0.8rem;
}

/* Cells */
.cell {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 12px;
    overflow: hidden;
    transition: border-color 0.15s;
}
.cell:focus-within { border-color: var(--accent-blue); }
.cell.streaming { border-color: var(--accent-orange); }

.cell-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; background: var(--bg-cell);
    border-bottom: 1px solid var(--border);
}

.cell-badge {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px; letter-spacing: 0.5px;
}
.cell-badge.code { background: var(--accent-green); color: #000; }
.cell-badge.note { background: var(--accent-purple); color: #000; }
.cell-badge.prompt { background: var(--accent-orange); color: #000; }

.cell-meta { font-size: 0.7rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px; }
.cell-actions { display: flex; gap: 4px; }
.cell-body { padding: 0; }

/* Source textarea */
.source {
    width: 100%; min-height: 60px; padding: 12px;
    background: var(--bg-input); border: none; 
    color: var(--text-primary);
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.9rem; resize: vertical;
    border-bottom: 1px solid var(--border);
}
.source:focus { outline: none; background: #0a0c10; }
.source:last-child { border-bottom: none; }

/* Code output */
.output {
    padding: 12px; background: var(--bg-input);
    font-family: monospace; font-size: 0.85rem;
    white-space: pre-wrap; border-top: 1px solid var(--border);
}
.output.error { color: var(--accent-red); }
.output:empty { display: none; }

/* Prompt cell sections */
.prompt-section { border-bottom: 1px solid var(--border); }
.prompt-section:last-child { border-bottom: none; }

.prompt-label {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 12px; background: var(--bg-cell);
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    color: var(--text-muted); border-bottom: 1px solid var(--border);
}
.prompt-label.user { color: var(--accent-blue); }
.prompt-label.ai { color: var(--accent-orange); }

.prompt-content {
    width: 100%; min-height: 50px; padding: 12px;
    background: var(--bg-input); border: none;
    color: var(--text-primary); font-size: 0.9rem;
    font-family: inherit; resize: vertical;
}
.prompt-content:focus { outline: none; background: #0a0c10; }

/* Markdown preview - rendered content */
.md-preview, .ai-preview {
    padding: 12px; min-height: 40px;
    cursor: pointer;
}
.md-preview:hover, .ai-preview:hover { background: rgba(88, 166, 255, 0.05); }
.md-preview h1, .ai-preview h1 { font-size: 1.5em; margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.md-preview h2, .ai-preview h2 { font-size: 1.3em; margin-bottom: 8px; }
.md-preview h3, .ai-preview h3 { font-size: 1.1em; margin-bottom: 6px; }
.md-preview p, .ai-preview p { margin-bottom: 10px; }
.md-preview ul, .md-preview ol, .ai-preview ul, .ai-preview ol { margin: 10px 0; padding-left: 24px; }
.md-preview li, .ai-preview li { margin-bottom: 4px; }
.md-preview strong, .ai-preview strong { color: var(--accent-blue); }
.md-preview em, .ai-preview em { color: var(--accent-purple); }

/* Code blocks in markdown */
.md-preview pre, .ai-preview pre {
    background: var(--bg-cell); padding: 12px; border-radius: 6px;
    overflow-x: auto; margin: 10px 0; position: relative;
}
.md-preview code, .ai-preview code {
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.9em;
}
.md-preview p code, .ai-preview p code {
    background: var(--bg-cell); padding: 2px 6px; border-radius: 3px;
}

/* Copy button for code blocks */
.copy-btn {
    position: absolute; top: 8px; right: 8px;
    padding: 4px 8px; font-size: 0.7rem;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text-muted); cursor: pointer;
    opacity: 0; transition: opacity 0.2s;
}
.md-preview pre:hover .copy-btn, .ai-preview pre:hover .copy-btn { opacity: 1; }
.copy-btn:hover { background: var(--accent-blue); color: #000; }
.copy-btn.copied { background: var(--accent-green); color: #000; }

/* Edit hint */
.edit-hint {
    font-size: 0.7rem; color: var(--text-muted); opacity: 0.5;
    padding: 4px 12px; text-align: right;
}

/* Add cell buttons */
.add-row {
    display: flex; justify-content: center; gap: 8px;
    padding: 6px; opacity: 0.3; transition: opacity 0.2s;
}
.add-row:hover { opacity: 1; }

.type-select {
    padding: 3px 6px; background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text-primary); font-size: 0.7rem;
}

/* Status messages */
.status {
    padding: 8px 12px; border-radius: 4px; font-size: 0.85rem;
    display: inline-flex; align-items: center; gap: 6px; margin-bottom: 12px;
}
.status.success { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
.status.error { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }

/* File list */
.file-list { 
    display: flex; flex-wrap: wrap; gap: 8px; 
    padding: 12px; background: var(--bg-secondary);
    border-radius: 6px; margin-bottom: 16px;
}
.file-item {
    padding: 6px 12px; background: var(--bg-cell);
    border: 1px solid var(--border); border-radius: 4px;
    font-size: 0.85rem; cursor: pointer; transition: all 0.15s;
    text-decoration: none; color: var(--text-primary);
}
.file-item:hover { border-color: var(--accent-blue); }
.file-item.active { border-color: var(--accent-blue); background: rgba(88, 166, 255, 0.1); }

/* Streaming indicator */
.streaming-indicator {
    display: inline-block; width: 8px; height: 8px;
    background: var(--accent-orange); border-radius: 50%;
    animation: pulse 1s infinite;
    margin-left: 8px;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Keyboard shortcut hints */
.kbd {
    display: inline-block; padding: 2px 6px;
    background: var(--bg-cell); border: 1px solid var(--border);
    border-radius: 3px; font-size: 0.75rem; font-family: monospace;
}

/* Ace Editor container */
.ace-container {
    width: 100%;
    min-height: 80px;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.ace-container .ace_editor {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
}

/* Focused cell indicator */
.cell.focused {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 1px var(--accent-blue);
}
"""

# ============================================================================
# JavaScript
# ============================================================================

js = """
// ==================== Ace Editor Management ====================
const aceEditors = {};

function initAceEditor(cellId) {
    const container = document.getElementById(`ace-${cellId}`);
    if (!container) return null;

    // If editor already exists, destroy it first to ensure fresh state
    if (aceEditors[cellId]) {
        aceEditors[cellId].destroy();
        delete aceEditors[cellId];
    }

    // Also check if Ace left any state on the container
    if (container.env && container.env.editor) {
        try {
            container.env.editor.destroy();
        } catch (e) {}
    }

    // Get initial content from hidden textarea
    const textarea = document.getElementById(`source-${cellId}`);
    const initialContent = textarea ? textarea.value : '';

    // Clear container completely before Ace takes over
    container.innerHTML = '';
    container.className = 'ace-container';  // Reset classes

    const editor = ace.edit(container);
    editor.setTheme("ace/theme/monokai");
    editor.setOptions({
        fontSize: "14px",
        showPrintMargin: false,
        highlightActiveLine: true,
        wrap: true,
        minLines: 3,
        maxLines: 30,
        tabSize: 4,
        useSoftTabs: true,
    });

    // Set content first, then apply mode (mode triggers re-highlighting)
    editor.setValue(initialContent, -1);

    // Apply mode after setValue - this ensures syntax highlighting works
    // Use a small delay to let Ace settle
    editor.session.setMode("ace/mode/python");

    // Force a complete re-render after a brief delay
    setTimeout(() => {
        editor.session.setMode("ace/mode/python");
        editor.renderer.updateFull();
    }, 50);
    
    // Sync to hidden textarea on change
    if (textarea) {
        editor.session.on('change', () => {
            textarea.value = editor.getValue();
        });
    }
    
    // Shift+Enter to run (Jupyter style)
    editor.commands.addCommand({
        name: 'runCell',
        bindKey: {win: 'Shift-Enter', mac: 'Shift-Enter'},
        exec: function(editor) {
            const cell = editor.container.closest('.cell');
            if (cell) {
                // Sync content first
                const cellId = cell.id.replace('cell-', '');
                syncAceToTextarea(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
            }
        }
    });
    
    // Ctrl/Cmd+Enter also runs
    editor.commands.addCommand({
        name: 'runCellAlt',
        bindKey: {win: 'Ctrl-Enter', mac: 'Cmd-Enter'},
        exec: function(editor) {
            const cell = editor.container.closest('.cell');
            if (cell) {
                const cellId = cell.id.replace('cell-', '');
                syncAceToTextarea(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
            }
        }
    });
    
    // Ctrl/Cmd+S to save
    editor.commands.addCommand({
        name: 'saveNotebook',
        bindKey: {win: 'Ctrl-S', mac: 'Cmd-S'},
        exec: function() {
            document.getElementById('save-btn')?.click();
        }
    });
    
    aceEditors[cellId] = editor;
    return editor;
}

function syncAceToTextarea(cellId) {
    const editor = aceEditors[cellId];
    const textarea = document.getElementById(`source-${cellId}`);
    if (editor && textarea) {
        textarea.value = editor.getValue();
    }
}

function getAceContent(cellId) {
    const editor = aceEditors[cellId];
    return editor ? editor.getValue() : '';
}

function destroyAceEditor(cellId) {
    if (aceEditors[cellId]) {
        aceEditors[cellId].destroy();
        delete aceEditors[cellId];
    }
}

// ==================== Focused Cell Tracking ====================
let focusedCellId = null;
let lastKeyTime = 0;
let lastKey = '';

function setFocusedCell(cellId) {
    document.querySelectorAll('.cell.focused').forEach(c => c.classList.remove('focused'));
    focusedCellId = cellId;
    if (cellId) {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell) cell.classList.add('focused');
    }
}

function focusNextCell(cellId) {
    // Focus a cell and optionally its editor
    setFocusedCell(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    // Scroll cell into view
    cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // If it's a code cell with Ace editor, focus the editor
    if (cell.dataset.type === 'code') {
        const editor = aceEditors[cellId];
        if (editor) {
            editor.focus();
        }
    } else if (cell.dataset.type === 'prompt') {
        // For prompt cells, focus the prompt textarea if visible
        const promptTextarea = cell.querySelector('.prompt-content[name="prompt_source"]');
        if (promptTextarea && promptTextarea.style.display !== 'none') {
            promptTextarea.focus();
        } else {
            // If prompt has been run (has preview), just keep cell selected
            // User can double-click to edit
        }
    } else if (cell.dataset.type === 'note') {
        // For note cells, just ensure the cell is focused/selected
        // Don't auto-open edit mode
    }
}

function getFocusedCellId() {
    const active = document.activeElement;
    if (active) {
        const cell = active.closest('.cell');
        if (cell) return cell.id.replace('cell-', '');
    }
    for (const [cellId, editor] of Object.entries(aceEditors)) {
        if (editor.isFocused()) return cellId;
    }
    return focusedCellId;
}

function moveToNextCell(currentCell) {
    // Find the next cell in DOM order
    // Cells are siblings within #cells, separated by .add-row divs
    let sibling = currentCell.nextElementSibling;
    while (sibling) {
        if (sibling.classList.contains('cell')) {
            const nextCellId = sibling.id.replace('cell-', '');
            focusNextCell(nextCellId);
            return;
        }
        sibling = sibling.nextElementSibling;
    }
    // No next cell found - we're at the last cell
    // Could optionally create a new cell here, but for now just stay on current
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', e => {
    const target = e.target;
    const mod = e.ctrlKey || e.metaKey;
    const inAce = target.closest('.ace_editor');
    const inInput = target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.isContentEditable;
    
    let currentCellId = getFocusedCellId();
    if (!currentCellId && target.closest('.cell')) {
        currentCellId = target.closest('.cell').id.replace('cell-', '');
    }
    
    // ===== D D to delete cell (Jupyter style) =====
    if (e.key === 'd' || e.key === 'D') {
        if (!inInput && !inAce) {
            const now = Date.now();
            if (lastKey === 'd' && (now - lastKeyTime) < 500) {
                // Double D pressed
                if (currentCellId) {
                    e.preventDefault();
                    const cell = document.getElementById(`cell-${currentCellId}`);
                    if (cell) {
                        const deleteBtn = cell.querySelector('button[hx-delete]');
                        if (deleteBtn) deleteBtn.click();
                    }
                }
                lastKey = '';
                lastKeyTime = 0;
                return;
            }
            lastKey = 'd';
            lastKeyTime = now;
        }
    } else {
        // Reset D D sequence on any other key
        lastKey = '';
        lastKeyTime = 0;
    }
    
    // ===== Shift+Enter - Run current cell (Jupyter style) =====
    if (e.shiftKey && e.key === 'Enter' && !inAce) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncAceToTextarea(cellId);
                // Also sync prompt textarea
                syncPromptContent(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) {
                    // Code or Prompt cell - run it (focus will move via server response)
                    btn.click();
                } else {
                    // Note cell - no run button, just move to next cell
                    moveToNextCell(cell);
                }
            }
        }
    }

    // ===== Ctrl/Cmd+Enter - Run current cell =====
    if (mod && e.key === 'Enter' && !inAce) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncAceToTextarea(cellId);
                syncPromptContent(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) {
                    btn.click();
                } else {
                    // Note cell - just move to next cell
                    moveToNextCell(cell);
                }
            }
        }
    }
    
    // ===== Ctrl/Cmd+S - Save notebook =====
    if (mod && e.key === 's' && !inAce) {
        e.preventDefault();
        document.getElementById('save-btn')?.click();
    }
    
    // ===== Escape - Exit edit mode =====
    if (e.key === 'Escape') {
        if (document.activeElement) {
            document.activeElement.blur();
        }
        Object.values(aceEditors).forEach(ed => ed.blur());
    }
    
    // ===== Ctrl/Cmd+Shift+D or Ctrl/Cmd+Backspace - Delete cell =====
    if (mod && (e.key === 'Backspace' || (e.shiftKey && e.key === 'D'))) {
        if (currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const deleteBtn = cell.querySelector('button[hx-delete]');
                if (deleteBtn) deleteBtn.click();
            }
        }
    }
    
    // ===== Alt+Up or Ctrl/Cmd+Shift+Up - Move cell up =====
    if ((e.altKey && e.key === 'ArrowUp') || (mod && e.shiftKey && e.key === 'ArrowUp')) {
        if (currentCellId && !inAce) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const moveBtn = cell.querySelector('button[title="Move up"]');
                if (moveBtn) moveBtn.click();
            }
        }
    }
    
    // ===== Alt+Down or Ctrl/Cmd+Shift+Down - Move cell down =====
    if ((e.altKey && e.key === 'ArrowDown') || (mod && e.shiftKey && e.key === 'ArrowDown')) {
        if (currentCellId && !inAce) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const moveBtn = cell.querySelector('button[title="Move down"]');
                if (moveBtn) moveBtn.click();
            }
        }
    }
    
    // ===== Add cell shortcuts (not in input) =====
    if (!inInput && !inAce) {
        if (mod && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=code', {target: '#cells'});
        }
        if (mod && e.shiftKey && e.key === 'N') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=note', {target: '#cells'});
        }
        if (mod && e.shiftKey && e.key === 'P') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=prompt', {target: '#cells'});
        }
    }
});

// Sync prompt content before running
function syncPromptContent(cellId) {
    // Try finding by ID first (newer format), then by name (for compatibility)
    let promptTextarea = document.getElementById(`prompt-${cellId}`);
    if (!promptTextarea) {
        promptTextarea = document.querySelector(`#cell-${cellId} .prompt-content[name="prompt_source"]`);
    }
    const hiddenSource = document.getElementById(`source-${cellId}`);
    if (promptTextarea && hiddenSource) {
        hiddenSource.value = promptTextarea.value;
    }
}

// Also sync when Ace editor content needs to go to hidden field
function syncAllContent(cellId) {
    syncAceToTextarea(cellId);
    syncPromptContent(cellId);
}

// ==================== Markdown Rendering ====================
function renderMarkdown(text) {
    if (!text) return '<p style="color: var(--text-muted);">Click to edit...</p>';
    
    // Process code blocks first and store them
    const codeBlocks = [];
    text = text.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, (match, lang, code) => {
        const idx = codeBlocks.length;
        const escaped = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        codeBlocks.push(`<pre data-lang="${lang || 'text'}"><code class="language-${lang || 'text'}">${escaped}</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></pre>`);
        return `__CODE_BLOCK_${idx}__`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    // Headers
    text = text.replace(/^### (.*)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.*)$/gm, '<h1>$1</h1>');
    // Lists
    text = text.replace(/^- (.*)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\\/li>)/gs, '<ul>$1</ul>');
    // Numbered lists
    text = text.replace(/^\\d+\\. (.*)$/gm, '<li>$1</li>');
    // Paragraphs
    text = text.replace(/\\n\\n/g, '</p><p>');
    text = text.replace(/\\n/g, '<br>');
    
    // Restore code blocks
    codeBlocks.forEach((block, idx) => {
        text = text.replace(`__CODE_BLOCK_${idx}__`, block);
    });
    
    return '<p>' + text + '</p>';
}

// Copy code to clipboard
function copyCode(btn) {
    const pre = btn.closest('pre');
    const code = pre.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });
}

// ==================== Preview/Edit Toggle ====================
function setupPreviewEditing() {
    // Double-click on preview to edit (note cells, AI response, and user prompt after run)
    document.querySelectorAll('.md-preview, .ai-preview, .prompt-preview').forEach(preview => {
        // Avoid adding duplicate listeners
        if (preview.dataset.hasEditListener) return;
        preview.dataset.hasEditListener = 'true';

        preview.addEventListener('dblclick', function(e) {
            if (e.target.closest('.copy-btn')) return; // Don't trigger on copy button
            const cellId = this.dataset.cellId;
            const field = this.dataset.field;
            switchToEdit(cellId, field);
        });
    });
}

function switchToEdit(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        preview.style.display = 'none';
        textarea.style.display = 'block';
        textarea.focus();
    }
}

function switchToPreview(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        // Update preview content
        preview.innerHTML = renderMarkdown(textarea.value);
        preview.style.display = 'block';
        textarea.style.display = 'none';
    }
}

// Update preview when content changes
function updatePreview(cellId, field) {
    const textarea = document.getElementById(`${field}-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    if (textarea && preview && preview.style.display !== 'none') {
        preview.innerHTML = renderMarkdown(textarea.value);
    }
}

// ==================== Initialization ====================
function initCell(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    
    // Initialize Ace editor for code cells
    if (cell.dataset.type === 'code') {
        initAceEditor(cellId);
    }
    
    // Setup preview for note cells
    const notePreview = document.getElementById(`preview-${cellId}`);
    const noteSource = document.getElementById(`source-${cellId}`);
    if (notePreview && noteSource && cell.dataset.type === 'note') {
        notePreview.innerHTML = renderMarkdown(noteSource.value);
    }
    
    // Setup AI response preview for prompt cells
    const aiPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    const aiTextarea = document.getElementById(`output-${cellId}`);
    if (aiPreview && aiTextarea) {
        const content = aiTextarea.value;
        if (content && content.trim()) {
            aiPreview.innerHTML = renderMarkdown(content);
        } else {
            aiPreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Click ▶ to generate response...</p>';
        }
    }

    // Setup user prompt preview for prompt cells (after they've been run)
    const promptPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="prompt"]`);
    const promptTextarea = document.getElementById(`prompt-${cellId}`);
    if (promptPreview && promptTextarea) {
        const content = promptTextarea.value;
        if (content && content.trim()) {
            promptPreview.innerHTML = renderMarkdown(content);
        } else {
            promptPreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">No prompt...</p>';
        }
    }
    
    // Track focus - both focusin (for editors/inputs) and click (for cell background)
    cell.addEventListener('focusin', () => setFocusedCell(cellId));
    cell.addEventListener('click', (e) => {
        // Only set focus if clicking directly on cell or its non-interactive children
        // This allows clicking anywhere on the cell to select it
        setFocusedCell(cellId);
    });
}

// Cleanup before HTMX swaps
document.addEventListener('htmx:beforeSwap', (e) => {
    // Only destroy Ace editors within the swap target (efficient)
    const target = e.detail.target;
    if (target) {
        // If target itself is a cell with an ace-container
        if (target.classList && target.classList.contains('cell')) {
            const container = target.querySelector('.ace-container');
            if (container) {
                const cellId = container.id.replace('ace-', '');
                destroyAceEditor(cellId);
            }
        } else {
            // If target contains ace-containers
            target.querySelectorAll('.ace-container').forEach(container => {
                const cellId = container.id.replace('ace-', '');
                destroyAceEditor(cellId);
            });
        }
    }
});

// After HTMX settles (fires after all HTMX processing is complete)
document.addEventListener('htmx:afterSettle', (e) => {
    // Small delay to ensure DOM is fully ready and Ace can initialize properly
    setTimeout(() => {
        // Only initialize cells within the swap target (efficient - not ALL cells)
        const target = e.detail.target || e.detail.elt;
        if (target) {
            // If target is a cell, initialize just that cell
            if (target.classList && target.classList.contains('cell')) {
                const cellId = target.id.replace('cell-', '');
                initCell(cellId);
            } else {
                // If target contains cells (e.g., OOB swap to #cells), initialize those
                target.querySelectorAll('.cell').forEach(cell => {
                    const cellId = cell.id.replace('cell-', '');
                    initCell(cellId);
                });
            }
        }
        setupPreviewEditing();
    }, 20);
});

// On page load
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.cell').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        initCell(cellId);
    });
    setupPreviewEditing();
});

// Auto-resize textareas
document.addEventListener('input', e => {
    if (e.target.tagName === 'TEXTAREA') {
        e.target.style.height = 'auto';
        e.target.style.height = Math.max(60, e.target.scrollHeight) + 'px';
    }
});

// ==================== WebSocket for Streaming ====================
let ws = null;
let streamingCellId = null;

function connectWebSocket(notebookId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'stream_chunk') {
            appendToResponse(data.cell_id, data.chunk);
        } else if (data.type === 'stream_end') {
            finishStreaming(data.cell_id);
        }
    };
    
    ws.onclose = function() {
        setTimeout(() => connectWebSocket(notebookId), 3000);
    };
}

function appendToResponse(cellId, chunk) {
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        if (textarea.value === 'Generating...' || textarea.value === 'Click ▶ to generate response...') {
            textarea.value = '';
        }
        textarea.value += chunk;
        if (preview) {
            preview.innerHTML = renderMarkdown(textarea.value);
        }
    }
}

function finishStreaming(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.remove('streaming');
    }
    streamingCellId = null;
}

function startStreaming(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.add('streaming');
    }
    streamingCellId = cellId;
    const textarea = document.getElementById(`output-${cellId}`);
    if (textarea) {
        textarea.value = 'Generating...';
    }
}
"""

# ============================================================================
# FastHTML App with WebSocket
# ============================================================================

app, rt = fast_app(
    pico=False,
    exts='ws',
    hdrs=(
        Style(css), 
        # Ace Editor
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/ace.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/mode-python.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/theme-monokai.min.js"),
        # Highlight.js for markdown code blocks
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"),
        # App JS (after libraries)
        Script(js),
    )
)

# ============================================================================
# Components
# ============================================================================

def TypeSelect(cell_id: str, current: str, nb_id: str):
    return Select(
        Option("code", value="code", selected=current=="code"),
        Option("note", value="note", selected=current=="note"),
        Option("prompt", value="prompt", selected=current=="prompt"),
        cls="type-select", name="cell_type",
        hx_post=f"/notebook/{nb_id}/cell/{cell_id}/type",
        hx_target=f"#cell-{cell_id}",
        hx_swap="outerHTML"
    )

def CellView(cell: Cell, notebook_id: str):
    """Render a single cell"""
    meta_info = []
    if cell.execution_count:
        meta_info.append(Span(f"[{cell.execution_count}]"))
    if cell.time_run:
        meta_info.append(Span(cell.time_run))
    
    header = Div(
        Div(
            Span(cell.cell_type.upper(), cls=f"cell-badge {cell.cell_type}"),
            Span(*meta_info, cls="cell-meta") if meta_info else None,
        ),
        Div(
            TypeSelect(cell.id, cell.cell_type, notebook_id),
            Button("▶", cls="btn btn-sm btn-run",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/run",
                   hx_target=f"#cell-{cell.id}",
                   hx_swap="outerHTML",
                   hx_vals=f"js:{{source: document.getElementById('source-{cell.id}')?.value || ''}}",
                   onclick=f"syncAceToTextarea('{cell.id}'); syncPromptContent('{cell.id}');",
                   title="Run (Shift+Enter)") if cell.cell_type != "note" else None,
            Button("↑", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/up",
                   hx_target="#cells", hx_swap="outerHTML", title="Move up"),
            Button("↓", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/down",
                   hx_target="#cells", hx_swap="outerHTML", title="Move down"),
            Button("×", cls="btn btn-sm btn-icon",
                   hx_delete=f"/notebook/{notebook_id}/cell/{cell.id}",
                   hx_target="#cells",
                   hx_swap="outerHTML", title="Delete (D D)"),
            cls="cell-actions"
        ),
        cls="cell-header"
    )
    
    if cell.cell_type == "code":
        body = Div(
            # Hidden textarea for form submission - Ace reads from this
            Textarea(cell.source, name="source", id=f"source-{cell.id}",
                    style="display: none;",
                    hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                    hx_trigger="blur changed", hx_swap="none"),
            # Ace Editor container - empty, content comes from textarea
            Div(id=f"ace-{cell.id}", cls="ace-container"),
            Div(cell.output,
                cls=f"output{' error' if cell.output and ('Error' in cell.output or 'Traceback' in cell.output) else ''}")
                if cell.output else None,
            Script(f"setTimeout(() => initAceEditor('{cell.id}'), 0);"),
            cls="cell-body"
        )
    elif cell.cell_type == "note":
        body = Div(
            Textarea(cell.source, cls="source", name="source", id=f"source-{cell.id}",
                    placeholder="# Markdown notes...",
                    hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                    hx_trigger="blur changed", hx_swap="none",
                    style="display: none;",
                    onblur=f"switchToPreview('{cell.id}', 'source')"),
            Div(id=f"preview-{cell.id}", cls="md-preview",
                data_cell_id=cell.id, data_field="source"),
            Div("Double-click to edit • Escape to finish", cls="edit-hint"),
            cls="cell-body"
        )
    else:  # Prompt
        # Determine if prompt has been run (has output)
        has_output = bool(cell.output and cell.output.strip())

        # User prompt section - show preview if has output, else show textarea
        if has_output:
            # After run: show markdown preview (double-click to edit)
            user_prompt_section = Div(
                Div(Span("👤"), " Your Prompt ",
                    Span("(double-click to edit)", style="font-weight: normal; opacity: 0.6;"),
                    cls="prompt-label user"),
                Textarea(cell.source, cls="prompt-content", name="prompt_source",
                        id=f"prompt-{cell.id}",
                        placeholder="Ask the AI anything...",
                        hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                        hx_trigger="blur changed", hx_swap="none",
                        style="display: none;",
                        oninput=f"document.getElementById('source-{cell.id}').value = this.value",
                        onblur=f"switchToPreview('{cell.id}', 'prompt')"),
                Div(cls="md-preview prompt-preview", data_cell_id=cell.id, data_field="prompt"),
                cls="prompt-section"
            )
        else:
            # Before run: show editable textarea
            user_prompt_section = Div(
                Div(Span("👤"), " Your Prompt", cls="prompt-label user"),
                Textarea(cell.source, cls="prompt-content", name="prompt_source",
                        id=f"prompt-{cell.id}",
                        placeholder="Ask the AI anything...",
                        hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                        hx_trigger="blur changed", hx_swap="none",
                        oninput=f"document.getElementById('source-{cell.id}').value = this.value"),
                cls="prompt-section"
            )

        body = Div(
            # Hidden source field for hx-vals
            Input(type="hidden", id=f"source-{cell.id}", name="source", value=cell.source),
            user_prompt_section,
            # AI response section with preview/edit toggle
            Div(
                Div(Span("🤖"), " AI Response ",
                    Span("(double-click to edit)", style="font-weight: normal; opacity: 0.6;"),
                    cls="prompt-label ai"),
                Textarea(cell.output if cell.output else "",
                        cls="prompt-content", name="output", id=f"output-{cell.id}",
                        placeholder="Click ▶ to generate response...",
                        hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/output",
                        hx_trigger="blur changed", hx_swap="none",
                        style="display: none; min-height: 80px;",
                        onblur=f"switchToPreview('{cell.id}', 'output')"),
                Div(cls="ai-preview", data_cell_id=cell.id, data_field="output"),
                cls="prompt-section"
            ),
            cls="cell-body"
        )
    
    return Div(header, body, id=f"cell-{cell.id}", cls="cell", data_type=cell.cell_type)

def AddButtons(pos: int, nb_id: str):
    return Div(
        Button("+ Code", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=code",
               hx_target="#cells", hx_swap="outerHTML"),
        Button("+ Note", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=note",
               hx_target="#cells", hx_swap="outerHTML"),
        Button("+ Prompt", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=prompt",
               hx_target="#cells", hx_swap="outerHTML"),
        cls="add-row"
    )

def AllCellsContent(nb: Notebook):
    """Returns just the cell content (for innerHTML swaps)"""
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i+1, nb.id)])
    # Return items wrapped in a fragment-like container
    # Using Div without id so HTMX swaps the inner content correctly
    return Div(*items)

def AllCells(nb: Notebook):
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i+1, nb.id)])
    return Div(*items, id="cells")

def NotebookPage(nb: Notebook, notebook_list: List[str]):
    return Titled(
        f"{nb.title} - LLM Notebook",
        Div(
            Div(
                Div(Span("📓", cls="title-icon"), Span(nb.title, cls="title")),
                Div(
                    Select(
                        Option("Learning", value="learning", selected=nb.dialog_mode=="learning"),
                        Option("Concise", value="concise", selected=nb.dialog_mode=="concise"),
                        Option("Standard", value="standard", selected=nb.dialog_mode=="standard"),
                        cls="mode-select", name="mode",
                        hx_post=f"/notebook/{nb.id}/mode", hx_swap="none", title="AI Mode"
                    ),
                    Button("🔄 Restart", cls="btn btn-sm",
                           hx_post="/kernel/restart", hx_target="#status", title="Restart kernel"),
                    Button("💾 Save", cls="btn btn-sm btn-save", id="save-btn",
                           hx_post=f"/notebook/{nb.id}/save", hx_target="#status", title="Save (Ctrl+S)"),
                    Button("📥 Export", cls="btn btn-sm",
                           hx_get=f"/notebook/{nb.id}/export", title="Download .ipynb"),
                    cls="toolbar"
                ),
                cls="header"
            ),
            Div(id="status"),
            Div(
                *[A(name, href=f"/notebook/{name}", 
                    cls=f"file-item{' active' if name == nb.id else ''}") 
                  for name in notebook_list],
                A("+ New", href="/notebook/new", cls="file-item"),
                cls="file-list"
            ) if notebook_list else None,
            AllCells(nb),
            Script(f"document.addEventListener('DOMContentLoaded', () => connectWebSocket('{nb.id}'));"),
            cls="container"
        )
    )

# ============================================================================
# Routes
# ============================================================================

@rt("/")
def get():
    return RedirectResponse("/notebook/default", status_code=302)

@rt("/notebook/new")
def get():
    new_id = uuid.uuid4().hex[:8]
    nb = Notebook(id=new_id, title=f"notebook_{new_id}")
    nb.cells = [Cell(cell_type="note", source="# New Notebook\n\nStart writing here...")]
    notebooks[new_id] = nb
    save_notebook(new_id)
    return RedirectResponse(f"/notebook/{new_id}", status_code=302)

@rt("/notebook/{nb_id}")
def get(nb_id: str):
    nb = get_notebook(nb_id)
    nb_list = list_notebooks() or [nb_id]
    return NotebookPage(nb, nb_list)

@rt("/notebook/{nb_id}/save")
def post(nb_id: str):
    save_notebook(nb_id)
    return Div("✓ Saved", cls="status success")

@rt("/notebook/{nb_id}/mode")
def post(nb_id: str, mode: str):
    nb = get_notebook(nb_id)
    nb.dialog_mode = mode
    return ""

@rt("/notebook/{nb_id}/export")
def get(nb_id: str):
    nb = get_notebook(nb_id)
    content = json.dumps(nb.to_ipynb(), indent=2)
    return Response(content=content, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{nb_id}.ipynb"'})

# Cell operations - now include notebook ID in path
@rt("/notebook/{nb_id}/cell/add")
def post(nb_id: str, pos: int = -1, type: str = "code"):
    nb = get_notebook(nb_id)
    if pos < 0:
        pos = len(nb.cells)
    nb.cells.insert(pos, Cell(cell_type=type))
    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}")
def delete(nb_id: str, cid: str):
    nb = get_notebook(nb_id)
    nb.cells = [c for c in nb.cells if c.id != cid]
    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}/source")
def post(nb_id: str, cid: str, source: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.source = source
            break
    return ""

@rt("/notebook/{nb_id}/cell/{cid}/output")
def post(nb_id: str, cid: str, output: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.output = output
            break
    return ""

@rt("/notebook/{nb_id}/cell/{cid}/type")
def post(nb_id: str, cid: str, cell_type: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.cell_type = cell_type
            c.output = ""
            c.execution_count = None
            return CellView(c, nb.id)
    return ""

@rt("/notebook/{nb_id}/cell/{cid}/move/{direction}")
def post(nb_id: str, cid: str, direction: str):
    nb = get_notebook(nb_id)
    for i, c in enumerate(nb.cells):
        if c.id == cid:
            if direction == "up" and i > 0:
                nb.cells[i], nb.cells[i-1] = nb.cells[i-1], nb.cells[i]
            elif direction == "down" and i < len(nb.cells) - 1:
                nb.cells[i], nb.cells[i+1] = nb.cells[i+1], nb.cells[i]
            break
    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}/run")
async def post(nb_id: str, cid: str, source: str = None):
    nb = get_notebook(nb_id)
    cell_index = None
    target_cell = None

    for i, c in enumerate(nb.cells):
        if c.id == cid:
            cell_index = i
            target_cell = c
            break

    if target_cell is None:
        return ""

    c = target_cell

    # Update source if provided (from Ace editor via hx-vals)
    if source is not None:
        c.source = source

    if c.cell_type == "code":
        stdout, stderr, success, exec_count = kernel.execute(c.source)
        c.output = stdout + stderr
        c.execution_count = exec_count
        c.time_run = datetime.now().strftime("%H:%M:%S")

    elif c.cell_type == "prompt":
        # Build context
        context_parts = []
        for prev in nb.cells:
            if prev.id == cid: break
            if prev.cell_type == "code":
                context_parts.append(f"```python\n{prev.source}\n```")
                if prev.output:
                    context_parts.append(f"Output:\n```\n{prev.output}\n```")
            elif prev.cell_type == "note":
                context_parts.append(prev.source)
            elif prev.cell_type == "prompt" and prev.output:
                context_parts.append(f"User: {prev.source}\n\nAssistant: {prev.output}")
        context = "\n\n".join(context_parts)

        # Stream via WebSocket if available, otherwise collect
        response_parts = []
        async for chunk in mock_llm_stream(c.source, context):
            response_parts.append(chunk)
            # Send via WebSocket
            if nb_id in ws_connections:
                msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": chunk})
                for send in ws_connections[nb_id].values():
                    try:
                        await send(msg)
                    except: pass

        c.output = "".join(response_parts)
        c.time_run = datetime.now().strftime("%H:%M:%S")

        # Send end signal
        if nb_id in ws_connections:
            msg = json.dumps({"type": "stream_end", "cell_id": cid})
            for send in ws_connections[nb_id].values():
                try:
                    await send(msg)
                except: pass

    # Determine next cell ID for auto-focus
    next_cell_id = None
    is_last_cell = cell_index == len(nb.cells) - 1

    if is_last_cell:
        # Add a new code cell using OOB swap
        new_cell = Cell(cell_type="code")
        nb.cells.append(new_cell)
        new_cell_index = len(nb.cells) - 1
        next_cell_id = new_cell.id

        # Return: updated cell (main) + new cell with AddButtons (OOB appended to #cells)
        # Use a wrapper div with hx-swap-oob to append the new elements
        return (
            CellView(c, nb.id),  # Main response - replaces the run cell
            Div(
                CellView(new_cell, nb.id),
                AddButtons(new_cell_index + 1, nb.id),
                hx_swap_oob="beforeend:#cells"  # Append to end of #cells
            ),
            # Script to focus the next cell after DOM settles
            Script(f"setTimeout(() => focusNextCell('{next_cell_id}'), 50);")
        )
    else:
        # Get the next cell's ID
        next_cell_id = nb.cells[cell_index + 1].id

    # Return updated cell + script to focus next cell
    return (
        CellView(c, nb.id),
        Script(f"setTimeout(() => focusNextCell('{next_cell_id}'), 50);")
    )

@rt("/kernel/restart")
def post():
    kernel.restart()
    return Div("✓ Kernel restarted", cls="status success")

# ============================================================================
# WebSocket for Streaming
# ============================================================================

def on_connect(ws, send, nb_id):
    if nb_id not in ws_connections:
        ws_connections[nb_id] = {}
    ws_connections[nb_id][id(ws)] = send

def on_disconnect(ws, nb_id):
    if nb_id in ws_connections:
        ws_connections[nb_id].pop(id(ws), None)

@app.ws('/ws/{nb_id}')
async def ws(ws, send, nb_id: str):
    on_connect(ws, send, nb_id)
    try:
        while True:
            msg = await ws.receive_text()
            # Handle any client messages if needed
    except:
        pass
    finally:
        on_disconnect(ws, nb_id)

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("🚀 LLM Notebook starting at http://localhost:8000")
    print("   Notebooks saved to: ./notebooks/")
    print("   Format: Solveit-compatible .ipynb")
    print("")
    print("   Keyboard shortcuts (Jupyter-style):")
    print("   • Shift+Enter       - Run cell")
    print("   • Ctrl/Cmd+S        - Save notebook")
    print("   • D D               - Delete cell (press D twice)")
    print("   • Ctrl/Cmd+Shift+C  - Add code cell")
    print("   • Ctrl/Cmd+Shift+N  - Add note cell")
    print("   • Ctrl/Cmd+Shift+P  - Add prompt cell")
    print("   • Alt+↑/↓           - Move cell up/down")
    print("   • Escape            - Exit edit mode")
    print("   • Double-click      - Edit markdown/response")
    serve(port=8000)
