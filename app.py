"""
Dialeng - Open source Solveit-like notebook with FastHTML

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

# New streaming kernel
from services.kernel import KernelService
from services.kernel.execution_queue import ExecutionQueue
from document.cell import CellState

# DialogHelper compatibility and LLM services
from services import (
    get_msg_idx, find_msgs, read_msg, cell_to_dict,
    build_context_messages, llm_service
)
from services.credential_service import (
    detect_credentials, get_available_modes, print_credential_status, CredentialStatus
)
from services.dialeng_config import (
    load_config, get_config, print_config_status
)

# UI Components (extracted to ui/ package)
from ui import (
    CellView, NotebookPage, AllCells, AllCellsContent,
    AllCellsOOB, CellViewOOB, AddButtons,
    TypeSelect, CollapseBtn, get_collapse_class
)

# ============================================================================
# Constants
# ============================================================================

SOLVEIT_VER = 2

# Load configuration (creates dialeng_config.json with defaults if it doesn't exist)
DIALENG_CONFIG = load_config()

# Detect credentials at startup
CREDENTIAL_STATUS = detect_credentials()
AVAILABLE_DIALOG_MODES = get_available_modes(CREDENTIAL_STATUS)

# Models from config
AVAILABLE_MODELS = DIALENG_CONFIG.get_model_choices()
DEFAULT_MODEL = DIALENG_CONFIG.get_default_model()

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

class CollapseLevel(int, Enum):
    EXPANDED = 0    # Fully visible
    SCROLLABLE = 1  # Limited height, scrollable
    SUMMARY = 2     # First line only with ellipsis

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
    collapsed: bool = False  # Legacy: kept for backwards compatibility
    input_collapse: int = 0  # CollapseLevel: 0=expanded, 1=scrollable, 2=summary
    output_collapse: int = 0  # CollapseLevel: 0=expanded, 1=scrollable, 2=summary
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
            if self.input_collapse: cell["metadata"]["input_collapse"] = self.input_collapse
            if self.output_collapse: cell["metadata"]["output_collapse"] = self.output_collapse
        elif self.cell_type == CellType.NOTE.value:
            cell = {
                "cell_type": "markdown",
                "id": self.id,
                "metadata": {},
                "source": self._to_source_lines(self.source)
            }
            if self.collapsed: cell["metadata"]["collapsed"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
            if self.input_collapse: cell["metadata"]["input_collapse"] = self.input_collapse
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
            if self.input_collapse: cell["metadata"]["input_collapse"] = self.input_collapse
            if self.output_collapse: cell["metadata"]["output_collapse"] = self.output_collapse
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
                pinned=metadata.get("pinned", False),
                input_collapse=metadata.get("input_collapse", 0),
                output_collapse=metadata.get("output_collapse", 1)  # Default to scrollable for code cells
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
                    pinned=metadata.get("pinned", False),
                    input_collapse=metadata.get("input_collapse", 0),
                    output_collapse=metadata.get("output_collapse", 0)
                )
            else:
                return cls(
                    id=cell_id, cell_type=CellType.NOTE.value, source=source,
                    collapsed=metadata.get("collapsed", False),
                    pinned=metadata.get("pinned", False),
                    input_collapse=metadata.get("input_collapse", 0)
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


# Default dialog mode based on credentials and config
DEFAULT_DIALOG_MODE = DIALENG_CONFIG.default_mode if CREDENTIAL_STATUS.available else "mock"

@dataclass
class Notebook:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "Untitled Notebook"
    cells: List[Cell] = field(default_factory=list)
    dialog_mode: str = DEFAULT_DIALOG_MODE
    model: str = DEFAULT_MODEL

    def to_ipynb(self) -> Dict[str, Any]:
        return {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11.0"},
                "solveit_dialog_mode": self.dialog_mode,
                "solveit_model": self.model,
                "solveit_ver": SOLVEIT_VER
            },
            "cells": [cell.to_jupyter_cell() for cell in self.cells]
        }
    
    @classmethod
    def from_ipynb(cls, data: Dict[str, Any], notebook_id: str = None) -> "Notebook":
        metadata = data.get("metadata", {})
        cells = [Cell.from_jupyter_cell(c) for c in data.get("cells", [])]
        # Get saved dialog mode, but override to "mock" if no credentials available
        saved_mode = metadata.get("solveit_dialog_mode", DEFAULT_DIALOG_MODE)
        # If no credentials available, force mock mode regardless of saved value
        effective_mode = "mock" if not CREDENTIAL_STATUS.available else saved_mode
        return cls(
            id=notebook_id or uuid.uuid4().hex[:8],
            title="Imported Notebook", cells=cells,
            dialog_mode=effective_mode,
            model=metadata.get("solveit_model", DEFAULT_MODEL)
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
# Python Kernel (Streaming Subprocess)
# ============================================================================

# KernelService manages subprocess kernels per notebook with:
# - Real-time streaming output (stdout/stderr as they happen)
# - Hard interrupt via SIGINT (can stop tight loops, C extensions)
# - Rich output support (images, plots, HTML)
# - Persistent namespace across cells

kernel_service = KernelService()

# ExecutionQueue instances per notebook (created lazily)
execution_queues: Dict[str, ExecutionQueue] = {}

def get_execution_queue(nb_id: str) -> ExecutionQueue:
    """Get or create execution queue for a notebook."""
    if nb_id not in execution_queues:
        queue = ExecutionQueue(kernel_service)
        execution_queues[nb_id] = queue
        # Register callbacks for WebSocket broadcasting
        queue.on_output(nb_id, _make_output_callback(nb_id))
        queue.on_state_change(nb_id, _make_state_callback(nb_id))
    return execution_queues[nb_id]

def _make_output_callback(nb_id: str):
    """Create output callback for streaming cell output via WebSocket."""
    async def callback(cell, output):
        await broadcast_cell_output(nb_id, cell.id, output)
    return callback

def _make_state_callback(nb_id: str):
    """Create state callback for broadcasting cell state changes."""
    async def callback(cell, state):
        if state == CellState.RUNNING:
            # Send code_stream_start when execution begins
            if nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "code_stream_start", "cell_id": cell.id})
                for send in list(ws_connections[nb_id]):
                    try:
                        await send(msg)
                    except:
                        pass

        await broadcast_cell_state(nb_id, cell.id, state)
        await broadcast_queue_state(nb_id)

        if state in (CellState.SUCCESS, CellState.ERROR):
            # Finalize cell output and send code_stream_end
            await finalize_cell_execution(nb_id, cell, state == CellState.ERROR)
    return callback


async def finalize_cell_execution(nb_id: str, cell, has_error: bool):
    """Convert cell outputs to HTML string and broadcast final state."""
    # Convert outputs to HTML string
    output_parts = []
    for output in cell.outputs:
        if output.output_type == 'stream':
            output_parts.append(ansi_to_html(output.content))
        elif output.output_type == 'execute_result':
            if output_parts and output_parts[-1] and not output_parts[-1].endswith('\n'):
                output_parts.append('\n')
            output_parts.append(ansi_to_html(output.content))
        elif output.output_type == 'error':
            tb_text = '\n'.join(output.traceback or [])
            output_parts.append(ansi_to_html(tb_text))
        elif output.output_type == 'display_data':
            html_content = render_mime_bundle(output.content, output.metadata)
            output_parts.append(html_content)

    cell.output = ''.join(output_parts)
    cell.time_run = datetime.now().strftime("%H:%M:%S")

    # Send code_stream_end signal
    if nb_id in ws_connections and ws_connections[nb_id]:
        msg = json.dumps({"type": "code_stream_end", "cell_id": cell.id, "has_error": has_error})
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except:
                pass

    # Broadcast final cell state via OOB swap
    await broadcast_to_notebook(nb_id, CellViewOOB(cell, nb_id))

# ============================================================================
# Mock LLM with Streaming
# ============================================================================

async def mock_llm_stream(prompt: str, context: str, use_thinking: bool = False):
    """Mock LLM for demo (replace with real API)"""
    # Simulate thinking phase if enabled
    if use_thinking:
        yield {"type": "thinking_start"}
        # Simulate thinking with 🧠 indicators
        for _ in range(3):
            yield {"type": "thinking", "content": "🧠 "}
            await asyncio.sleep(0.3)
        yield {"type": "thinking_end"}

    # Always echo the user's prompt first, then provide a response
    response = f"""You said:

> {prompt}

---

This is a **demo response**. In production, connect to Claude, OpenAI, or local models.

**Key features**:
- Both prompt AND response are editable
- Double-click to edit rendered markdown
- Press `Escape` to finish editing
- `Ctrl+Enter` runs cells
- Cancel generation with ⏹ button"""

    # Stream word by word
    words = response.split(' ')
    for i, word in enumerate(words):
        yield {"type": "chunk", "content": word + (' ' if i < len(words) - 1 else '')}
        await asyncio.sleep(0.02)

# ============================================================================
# Storage
# ============================================================================

notebooks: Dict[str, Notebook] = {}
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Track active WebSocket connections per notebook (list of send functions)
ws_connections: Dict[str, List[Any]] = {}

# Track cancelled cell generations
cancelled_cells: set = set()

# DialogHelper data queues for bidirectional browser communication
# Structure: {notebook_id: {data_id: asyncio.Queue}}
data_queues: Dict[str, Dict[str, asyncio.Queue]] = {}

def get_data_queue(dlg_name: str, data_id: str) -> asyncio.Queue:
    """Get or create a data queue for dialoghelper push/pop operations."""
    if dlg_name not in data_queues:
        data_queues[dlg_name] = {}
    if data_id not in data_queues[dlg_name]:
        data_queues[dlg_name][data_id] = asyncio.Queue()
    return data_queues[dlg_name][data_id]

def get_notebook(notebook_id: str) -> Notebook:
    """Get or create a notebook - ALWAYS requires notebook_id"""
    if notebook_id not in notebooks:
        path = NOTEBOOKS_DIR / f"{notebook_id}.ipynb"
        if path.exists():
            notebooks[notebook_id] = Notebook.load(str(path))
        else:
            nb = Notebook(id=notebook_id, title=notebook_id)
            nb.cells = [
                Cell(cell_type="note", source="# Welcome to Dialeng! 🚀\n\nAn open-source notebook with **prompt cells** for AI interaction.\n\n**Keyboard Shortcuts (Jupyter-style):**\n- `Shift+Enter` - Run cell (recommended)\n- `Ctrl/Cmd+Enter` - Run cell (alternative)\n- `Ctrl/Cmd+S` - Save notebook\n- `D D` - Delete cell (press D twice)\n- `Ctrl/Cmd+Shift+C` - Add code cell\n- `Ctrl/Cmd+Shift+N` - Add note cell\n- `Ctrl/Cmd+Shift+P` - Add prompt cell\n- `Alt+↑/↓` - Move cell up/down\n- `Escape` - Exit edit mode\n- Double-click - Edit markdown/response"),
                Cell(cell_type="code", source="# Try running some Python (Shift+Enter)\nx = [1, 2, 3, 4, 5]\nprint(f'Sum: {sum(x)}')\nprint(f'Average: {sum(x)/len(x)}')\nx", output_collapse=1),
                Cell(cell_type="note", source="## 🔄 Streaming Output Tests\n\nThe cells below demonstrate real-time streaming output. Run them to see output appear incrementally."),
                Cell(cell_type="code", source="# Test 1: Basic streaming with sleep\nfrom time import sleep\n\nfor i in range(5):\n    print(f\"Step {i + 1}/5: Processing...\")\n    sleep(1)\n\nprint(\"Done!\")", output_collapse=1),
                Cell(cell_type="code", source="# Test 2: Progress bar with tqdm\nfrom tqdm import tqdm\nfrom time import sleep\n\nfor i in tqdm(range(20), desc=\"Processing\"):\n    sleep(0.1)", output_collapse=1),
                Cell(cell_type="code", source="# Test 3: ANSI colors (if supported)\nprint(\"\\033[31mRed text\\033[0m\")\nprint(\"\\033[32mGreen text\\033[0m\")\nprint(\"\\033[33mYellow text\\033[0m\")\nprint(\"\\033[34mBlue text\\033[0m\")\nprint(\"\\033[1mBold text\\033[0m\")", output_collapse=1),
                Cell(cell_type="note", source="## 📊 Rich Output Tests\n\nThese cells test display of images, HTML, and other rich content."),
                Cell(cell_type="code", source="# Test 4: HTML display\nfrom IPython.display import HTML, display\n\ndisplay(HTML(\"\"\"\n<div style=\"padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;\">\n    <h3>🎨 Rich HTML Output</h3>\n    <p>This is rendered HTML with styling!</p>\n    <button style=\"padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;\">Click me (won't do anything)</button>\n</div>\n\"\"\"))", output_collapse=1),
                Cell(cell_type="code", source="# Test 5: Matplotlib plot\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\nplt.figure(figsize=(8, 4))\nplt.plot(x, np.sin(x), label='sin(x)')\nplt.plot(x, np.cos(x), label='cos(x)')\nplt.legend()\nplt.title('Trigonometric Functions')\nplt.xlabel('x')\nplt.ylabel('y')\nplt.grid(True, alpha=0.3)\nplt.show()", output_collapse=1),
                Cell(cell_type="code", source="# Test 6: Error handling\n# This will raise an error - the traceback should display properly\nresult = 1 / 0", output_collapse=1),
                Cell(cell_type="note", source="## 💬 Prompt Cell\n\nUse the prompt cell below to chat with the AI assistant."),
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

def render_mime_bundle(data: dict, metadata: dict = None) -> str:
    """
    Convert Jupyter MIME bundle to HTML.
    Priority: text/html > image/svg+xml > image/png > image/jpeg > text/markdown > text/plain

    Args:
        data: Dict with MIME types as keys and content as values
        metadata: Optional dict with rendering hints (width, height, etc.)

    Returns:
        HTML string for rendering
    """
    import html as html_module
    metadata = metadata or {}

    # HTML - render directly (trusted user code, matches Jupyter behavior)
    if 'text/html' in data:
        return f'<div class="mime-html">{data["text/html"]}</div>'

    # SVG - render inline
    if 'image/svg+xml' in data:
        return f'<div class="mime-svg">{data["image/svg+xml"]}</div>'

    # PNG image - base64 data URL
    if 'image/png' in data:
        width = metadata.get('width', '')
        height = metadata.get('height', '')
        style_parts = []
        if width:
            style_parts.append(f'width:{width}px')
        if height:
            style_parts.append(f'height:{height}px')
        style = ';'.join(style_parts)
        style_attr = f' style="{style}"' if style else ''
        return f'<img class="mime-image" src="data:image/png;base64,{data["image/png"]}"{style_attr} />'

    # JPEG image
    if 'image/jpeg' in data:
        return f'<img class="mime-image" src="data:image/jpeg;base64,{data["image/jpeg"]}" />'

    # GIF image
    if 'image/gif' in data:
        return f'<img class="mime-image" src="data:image/gif;base64,{data["image/gif"]}" />'

    # Markdown - wrap for potential rendering
    if 'text/markdown' in data:
        return f'<div class="mime-markdown">{data["text/markdown"]}</div>'

    # LaTeX - wrap for MathJax/KaTeX processing
    if 'text/latex' in data:
        return f'<div class="mime-latex">{html_module.escape(data["text/latex"])}</div>'

    # JSON - pretty print
    if 'application/json' in data:
        json_str = json.dumps(data['application/json'], indent=2)
        return f'<pre class="mime-json">{html_module.escape(json_str)}</pre>'

    # Plain text fallback
    if 'text/plain' in data:
        return f'<pre class="mime-text">{html_module.escape(data["text/plain"])}</pre>'

    # Unknown format - show raw
    return f'<pre class="mime-unknown">{html_module.escape(str(data))}</pre>'


def ansi_to_html(text: str) -> str:
    """
    Convert ANSI escape codes to HTML spans with inline styles.

    Handles common ANSI codes for colors (30-37, 90-97), backgrounds (40-47),
    bold (1), and reset (0).
    """
    import re
    import html as html_module

    ANSI_COLORS = {
        '30': '#000', '31': '#c00', '32': '#0a0', '33': '#a50',
        '34': '#00a', '35': '#a0a', '36': '#0aa', '37': '#aaa',
        '90': '#555', '91': '#f55', '92': '#5f5', '93': '#ff5',
        '94': '#55f', '95': '#f5f', '96': '#5ff', '97': '#fff',
    }
    ANSI_BG_COLORS = {
        '40': '#000', '41': '#c00', '42': '#0a0', '43': '#a50',
        '44': '#00a', '45': '#a0a', '46': '#0aa', '47': '#aaa',
    }

    result = []
    open_spans = 0

    # Split by ANSI escape sequences
    parts = re.split(r'(\x1b\[[0-9;]*m)', text)

    for part in parts:
        match = re.match(r'\x1b\[([0-9;]*)m', part)
        if match:
            codes = match.group(1).split(';')
            for code in codes:
                if code == '0' or code == '':
                    # Reset all styles
                    while open_spans > 0:
                        result.append('</span>')
                        open_spans -= 1
                elif code == '1':
                    result.append('<span style="font-weight:bold">')
                    open_spans += 1
                elif code in ANSI_COLORS:
                    result.append(f'<span style="color:{ANSI_COLORS[code]}">')
                    open_spans += 1
                elif code in ANSI_BG_COLORS:
                    result.append(f'<span style="background:{ANSI_BG_COLORS[code]}">')
                    open_spans += 1
        else:
            result.append(html_module.escape(part))

    # Close any remaining open spans
    while open_spans > 0:
        result.append('</span>')
        open_spans -= 1

    return ''.join(result)


# ============================================================================
# Collaborative WebSocket Broadcasting
# ============================================================================

async def broadcast_to_notebook(nb_id: str, component, exclude_send: Any = None):
    """Broadcast an HTML component to all WebSocket connections for a notebook.

    This sends HTML components directly via WebSocket. The JavaScript client
    processes hx-swap-oob attributes to update the DOM.

    Args:
        nb_id: The notebook ID to broadcast to
        component: FastHTML component to send
        exclude_send: Optional send function to exclude (e.g., the sender)
    """
    if nb_id not in ws_connections or not ws_connections[nb_id]:
        print(f"[BROADCAST] No connections for notebook {nb_id}")
        return

    connections = ws_connections[nb_id]
    print(f"[BROADCAST] Sending to {len(connections)} connections for {nb_id}")

    # Convert component to HTML string using to_xml
    # FastHTML's str() on components returns the ID, not HTML
    html_str = to_xml(component)
    print(f"[BROADCAST] HTML length: {len(html_str)} chars, starts with: {html_str[:100]}")

    # Track which connections are still alive
    alive = []
    sent_count = 0

    for i, send in enumerate(connections):
        if send is exclude_send:
            alive.append(send)  # Keep but don't send
            continue
        try:
            # Send the HTML string directly
            print(f"[BROADCAST] Sending to connection {i}, send function: {send}", flush=True)
            await send(html_str)
            print(f"[BROADCAST] Successfully sent to connection {i}", flush=True)
            alive.append(send)
            sent_count += 1
        except Exception as e:
            print(f"[BROADCAST] Failed to send to connection {i} (removing dead connection): {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Don't add to alive - this removes the dead connection

    # Replace with only alive connections
    ws_connections[nb_id] = alive
    print(f"[BROADCAST] Sent to {sent_count} clients, {len(alive)} connections remain")


async def broadcast_queue_state(nb_id: str):
    """Broadcast current queue state to all clients."""
    queue = get_execution_queue(nb_id)
    status = queue.get_status(nb_id)

    msg = json.dumps({
        "type": "queue_update",
        "running_cell_id": status.current_cell_id,
        "queued_cell_ids": status.queued_cell_ids
    })

    if nb_id in ws_connections and ws_connections[nb_id]:
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except:
                pass


async def broadcast_cell_state(nb_id: str, cell_id: str, state: CellState):
    """Broadcast cell state change to all clients."""
    msg = json.dumps({
        "type": "cell_state_change",
        "cell_id": cell_id,
        "state": state.value
    })

    if nb_id in ws_connections and ws_connections[nb_id]:
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except:
                pass


async def broadcast_cell_output(nb_id: str, cell_id: str, output):
    """Broadcast cell output chunk to all clients."""
    if output.output_type == 'stream':
        msg = json.dumps({
            "type": "code_stream_chunk",
            "cell_id": cell_id,
            "chunk": output.content,
            "stream": output.stream_name
        })
    elif output.output_type == 'execute_result':
        msg = json.dumps({
            "type": "code_stream_chunk",
            "cell_id": cell_id,
            "chunk": output.content,
            "stream": "stdout"
        })
    elif output.output_type == 'error':
        tb_text = '\n'.join(output.traceback or [])
        msg = json.dumps({
            "type": "code_stream_chunk",
            "cell_id": cell_id,
            "chunk": tb_text,
            "stream": "stderr"
        })
    elif output.output_type == 'display_data':
        html_content = render_mime_bundle(output.content, output.metadata)
        msg = json.dumps({
            "type": "code_display_data",
            "cell_id": cell_id,
            "html": html_content
        })
    else:
        return  # Unknown type, skip

    if nb_id in ws_connections and ws_connections[nb_id]:
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except:
                pass


# NOTE: CSS and JavaScript have been extracted to static/ directory:
#   - static/css/themes.css    (theme color variables)
#   - static/css/base.css      (reset, typography, layout)
#   - static/css/components.css (cells, buttons, badges)
#   - static/css/editor.css    (Ace editor styles)
#   - static/js/app.js         (all client-side logic)
#
# See docs/how_it_works/07_code_organization.md for details.


# ============================================================================
# FastHTML App with WebSocket
# ============================================================================

app, rt = fast_app(
    pico=False,
    exts='ws',
    hdrs=(
        # CSS - External stylesheets (order matters: themes first for variables)
        Link(rel="stylesheet", href="/static/css/themes.css"),
        Link(rel="stylesheet", href="/static/css/base.css"),
        Link(rel="stylesheet", href="/static/css/components.css"),
        Link(rel="stylesheet", href="/static/css/editor.css"),
        # Ace Editor
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/ace.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/mode-python.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/theme-monokai.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/theme-chrome.min.js"),
        # Highlight.js for markdown code blocks
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"),
        # App JS - External JavaScript (after libraries)
        Script(src="/static/js/app.js"),
    )
)

# Static file serving
@rt("/static/{path:path}")
async def get(path: str):
    """Serve static files from the static/ directory."""
    from starlette.responses import FileResponse
    file_path = Path("static") / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return "Not found", 404


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
    return NotebookPage(nb, nb_list, AVAILABLE_DIALOG_MODES, AVAILABLE_MODELS)

@rt("/notebook/{nb_id}/save")
def post(nb_id: str):
    save_notebook(nb_id)
    return Div("✓ Saved", cls="status success")

@rt("/notebook/{nb_id}/mode")
def post(nb_id: str, mode: str):
    nb = get_notebook(nb_id)
    nb.dialog_mode = mode
    return ""

@rt("/notebook/{nb_id}/model")
def post(nb_id: str, model: str):
    nb = get_notebook(nb_id)
    nb.model = model
    return ""

@rt("/notebook/{nb_id}/export")
def get(nb_id: str):
    nb = get_notebook(nb_id)
    content = json.dumps(nb.to_ipynb(), indent=2)
    return Response(content=content, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{nb_id}.ipynb"'})

# Cell operations - now include notebook ID in path
@rt("/notebook/{nb_id}/cell/add")
async def post(nb_id: str, pos: int = -1, type: str = "code"):
    nb = get_notebook(nb_id)
    if pos < 0:
        pos = len(nb.cells)
    # Code cells default to scrollable output for better screen space usage
    if type == "code":
        nb.cells.insert(pos, Cell(cell_type=type, output_collapse=1))
    else:
        nb.cells.insert(pos, Cell(cell_type=type))

    # Broadcast to collaborators - send HTML with OOB swap
    await broadcast_to_notebook(nb_id, AllCellsOOB(nb))

    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}")
async def delete(nb_id: str, cid: str):
    nb = get_notebook(nb_id)

    # Remove from execution queue if queued
    queue = get_execution_queue(nb_id)
    queue.cancel_queued(nb_id, cid)

    nb.cells = [c for c in nb.cells if c.id != cid]

    # Broadcast queue state update
    await broadcast_queue_state(nb_id)

    # Broadcast to collaborators - send HTML with OOB swap
    await broadcast_to_notebook(nb_id, AllCellsOOB(nb))

    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}/source")
def post(nb_id: str, cid: str, source: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            old_source = c.source
            c.source = source
            # CRITICAL: Clear output when source changes to prevent stale data in context
            # This ensures that when context is built for subsequent cells,
            # we don't include an old assistant response that doesn't match the new source.
            if old_source != source:
                c.clear_outputs()
                logger.info(f"Cell {cid}: Source changed, cleared outputs to prevent stale context")
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
async def post(nb_id: str, cid: str, cell_type: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.cell_type = cell_type
            c.output = ""
            c.execution_count = None

            # Broadcast cell type change to collaborators using OOB swap
            await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))

            return CellView(c, nb.id)
    return ""

@rt("/notebook/{nb_id}/cell/{cid}/move/{direction}")
async def post(nb_id: str, cid: str, direction: str):
    nb = get_notebook(nb_id)
    for i, c in enumerate(nb.cells):
        if c.id == cid:
            if direction == "up" and i > 0:
                nb.cells[i], nb.cells[i-1] = nb.cells[i-1], nb.cells[i]
            elif direction == "down" and i < len(nb.cells) - 1:
                nb.cells[i], nb.cells[i+1] = nb.cells[i+1], nb.cells[i]
            break

    # Broadcast to collaborators using OOB swap
    await broadcast_to_notebook(nb_id, AllCellsOOB(nb))

    return AllCells(nb)

@rt("/notebook/{nb_id}/cell/{cid}/collapse")
async def post(nb_id: str, cid: str, collapsed: str):
    nb = get_notebook(nb_id)
    cell = None
    for c in nb.cells:
        if c.id == cid:
            cell = c
            c.collapsed = collapsed.lower() == "true"
            break

    # Broadcast full collapse state change to collaborators using OOB swap
    if cell:
        await broadcast_to_notebook(nb_id, CellViewOOB(cell, nb_id))

    return ""

@rt("/notebook/{nb_id}/cell/{cid}/collapse-section")
async def post(nb_id: str, cid: str, section: str, level: int):
    """Update collapse level for input or output section"""
    nb = get_notebook(nb_id)
    cell = None
    for c in nb.cells:
        if c.id == cid:
            cell = c
            if section == "input":
                c.input_collapse = level
            elif section == "output":
                c.output_collapse = level
            elif section == "both":
                c.input_collapse = level
                c.output_collapse = level
            break

    # Broadcast collapse state change to collaborators using OOB swap
    if cell:
        await broadcast_to_notebook(nb_id, CellViewOOB(cell, nb_id))

    return ""

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
        queue = get_execution_queue(nb_id)

        # Check if already queued or running - ignore duplicate requests
        if queue.is_cell_queued(nb_id, cid):
            print(f"[CODE RUN] Cell {cid} already queued/running, ignoring duplicate", flush=True)
            return ""

        print(f"[CODE RUN] Queueing cell {cid} in notebook {nb_id}", flush=True)

        # Queue the cell - returns immediately, execution happens in background
        queue.queue_cell(nb_id, c)

        # Broadcast queue state to all clients
        await broadcast_queue_state(nb_id)

        return ""

    elif c.cell_type == "prompt":
        # Remove from cancelled set if it was there
        cancelled_cells.discard(cid)

        # Choose stream source based on dialog mode
        if nb.dialog_mode == "mock":
            # Build simple context string for mock (backwards compatibility)
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
            stream_func = mock_llm_stream(c.source, context, c.use_thinking)
        else:
            # Use real LLM via claudette-agent with dialoghelper context building
            context_messages = build_context_messages(nb, cid)
            stream_func = llm_service.stream_response(
                c.source, context_messages, nb.dialog_mode, nb.model, c.use_thinking
            )

        # Stream via WebSocket to all connected clients
        # Collaborators will receive the final cell state via OOB broadcast after completion
        response_parts = []
        try:
            async for item in stream_func:
                # Check if cancelled
                if cid in cancelled_cells:
                    cancelled_cells.discard(cid)
                    break

                # Handle errors from LLM service
                if item["type"] == "error":
                    response_parts.append(f"\n\n**Error:** {item['content']}")
                    # Send error to WebSocket
                    if nb_id in ws_connections and ws_connections[nb_id]:
                        msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": f"\n\n**Error:** {item['content']}"})
                        for send in ws_connections[nb_id]:
                            try:
                                await send(msg)
                            except:
                                pass
                    break

                # Collect response chunks
                if item["type"] == "chunk":
                    response_parts.append(item["content"])

                # Send streaming updates via WebSocket
                if nb_id in ws_connections and ws_connections[nb_id]:
                    if item["type"] == "thinking_start":
                        msg = json.dumps({"type": "thinking_start", "cell_id": cid})
                    elif item["type"] == "thinking_end":
                        msg = json.dumps({"type": "thinking_end", "cell_id": cid})
                    elif item["type"] == "thinking":
                        msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": item["content"], "thinking": True})
                    else:  # chunk
                        msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": item["content"]})

                    # Iterate over list (not dict.values())
                    for send in ws_connections[nb_id]:
                        try:
                            await send(msg)
                        except:
                            pass
        except Exception as e:
            # Catch any unexpected errors during streaming
            error_msg = f"\n\n**Error:** Streaming error: {str(e)}"
            response_parts.append(error_msg)
            if nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": error_msg})
                for send in ws_connections[nb_id]:
                    try:
                        await send(msg)
                    except:
                        pass
        finally:
            # Always send stream_end to ensure UI is not left frozen
            c.output = "".join(response_parts)
            c.time_run = datetime.now().strftime("%H:%M:%S")

            # Send end signal to all clients
            if nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "stream_end", "cell_id": cid})
                for send in ws_connections[nb_id]:
                    try:
                        await send(msg)
                    except:
                        pass

        # Broadcast final prompt cell state to collaborators using OOB swap
        await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))

    # Determine next cell ID for auto-focus
    next_cell_id = None
    is_last_cell = cell_index == len(nb.cells) - 1

    if is_last_cell:
        # Add a new code cell using OOB swap (with scrollable output default)
        new_cell = Cell(cell_type="code", output_collapse=1)
        nb.cells.append(new_cell)
        new_cell_index = len(nb.cells) - 1
        next_cell_id = new_cell.id

        # Broadcast new cell addition to collaborators using OOB swap
        await broadcast_to_notebook(nb_id, AllCellsOOB(nb))

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

@rt("/notebook/{nb_id}/kernel/restart")
def post(nb_id: str):
    """Restart the kernel for a specific notebook."""
    kernel_service.restart(nb_id)
    return Div("✓ Kernel restarted", cls="status success")

@rt("/notebook/{nb_id}/kernel/interrupt")
def post(nb_id: str):
    """Interrupt currently running code in the notebook's kernel."""
    success = kernel_service.interrupt(nb_id)
    if success:
        return Div("✓ Execution interrupted", cls="status success")
    else:
        return Div("No kernel to interrupt", cls="status warning")

@rt("/notebook/{nb_id}/queue/cancel_all")
async def post(nb_id: str):
    """Cancel running cell AND clear entire queue."""
    print(f"[CANCEL_ALL] Received cancel_all request for notebook {nb_id}")
    queue = get_execution_queue(nb_id)

    # Get current queue status before cancelling
    status_before = queue.get_status(nb_id)
    print(f"[CANCEL_ALL] Queue status before: running={status_before.current_cell_id}, queued={status_before.queued_cell_ids}")

    # First interrupt the running cell
    kernel_service.interrupt(nb_id)
    print(f"[CANCEL_ALL] Kernel interrupt sent")

    # Then clear the queue
    queue.cancel_all(nb_id)
    print(f"[CANCEL_ALL] Queue cancel_all called")

    # Get status after cancelling
    status_after = queue.get_status(nb_id)
    print(f"[CANCEL_ALL] Queue status after: running={status_after.current_cell_id}, queued={status_after.queued_cell_ids}")

    # Broadcast updated queue state
    await broadcast_queue_state(nb_id)

    return {"status": "ok"}

# ============================================================================
# DialogHelper Compatibility Endpoints
# ============================================================================
# These endpoints implement the server-side API that dialoghelper's call_endp()
# uses to programmatically manipulate cells. They leverage the shared logic in
# services/dialoghelper_service.py

@rt("/curr_dialog_")
def post(dlg_name: str, with_messages: bool = False):
    """Get current dialog info."""
    nb = get_notebook(dlg_name)
    result = {"name": nb.id, "mode": nb.dialog_mode}
    if with_messages:
        result["messages"] = [cell_to_dict(c) for c in nb.cells]
    return result

@rt("/msg_idx_")
def post(dlg_name: str, id_: str):
    """Get message index by ID - uses shared get_msg_idx()."""
    nb = get_notebook(dlg_name)
    # dialoghelper library expects {"idx": idx} (accesses result['idx'])
    return {"idx": get_msg_idx(nb, id_)}

@rt("/find_msgs_")
def post(dlg_name: str, re_pattern: str = "", msg_type: str = "", limit: int = 100):
    """Search messages - uses shared find_msgs()."""
    nb = get_notebook(dlg_name)
    results = find_msgs(nb, re_pattern=re_pattern, msg_type=msg_type, limit=limit)
    # dialoghelper expects {"msgs": [...]} - use 'content' for consistency with read_msg
    msgs = [{
        "idx": i,
        "id": c.id,
        "type": c.cell_type,
        "content": c.source,  # Use 'content' to match dialoghelper's read_msg format
        "output": c.output,
        "pinned": c.pinned,
        "skipped": c.skipped
    } for i, c in results]
    return {"msgs": msgs}

@rt("/read_msg_")
def post(dlg_name: str, n: int = 0, relative: bool = True, id_: str = "",
         view_range: str = "", nums: bool = False, current_idx: int = 0):
    """Read message content - uses shared read_msg()."""
    nb = get_notebook(dlg_name)
    return read_msg(nb, n=n, relative=relative, msgid=id_,
                    current_idx=current_idx, view_range=view_range, nums=nums)

@rt("/add_relative_")
async def post(dlg_name: str, content: str, placement: str = "after", id_: str = "",
         msg_type: str = "code", output: str = "", time_run: str = "",
         is_exported: str = "", skipped: str = "",
         i_collapsed: str = "0", o_collapsed: str = "0",
         heading_collapsed: str = "", pinned: str = "", run: str = ""):
    """Add message relative to another - uses shared get_msg_idx().

    Boolean params use str type because HTTP form data sends 'True'/'False' strings.
    """
    print(f"[ADD_RELATIVE] dlg_name={dlg_name}, ws_connections keys={list(ws_connections.keys())}", flush=True)
    nb = get_notebook(dlg_name)
    new_cell = Cell(
        cell_type=msg_type, source=content, output=output,
        skipped=_str_to_bool(skipped), pinned=_str_to_bool(pinned),
        input_collapse=int(i_collapsed) if i_collapsed else 0,
        output_collapse=int(o_collapsed) if o_collapsed else 0,
        is_exported=_str_to_bool(is_exported), time_run=time_run
    )
    # Find insertion point using shared function
    if id_:
        ref_idx = get_msg_idx(nb, id_)
        if ref_idx == -1:
            return {"error": f"Message {id_} not found"}
        insert_idx = ref_idx + 1 if placement == "after" else ref_idx
    else:
        insert_idx = len(nb.cells)  # Append to end

    nb.cells.insert(insert_idx, new_cell)

    # Broadcast to all connected clients so they see the new cell immediately
    try:
        await broadcast_to_notebook(dlg_name, AllCellsOOB(nb))
        print(f"[ADD_RELATIVE] Broadcast completed for {dlg_name}", flush=True)
    except Exception as e:
        print(f"[ADD_RELATIVE] Broadcast error: {e}", flush=True)

    # Optionally trigger execution
    if _str_to_bool(run):
        # Queue for execution (implementation depends on kernel service)
        pass

    # dialoghelper expects just the cell ID as plain text (not JSON)
    # This is used to set __msg_id for relative operations
    return new_cell.id

@rt("/rm_msg_")
async def post(dlg_name: str, msid: str):
    """Remove message by ID."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msid)
    if idx >= 0:
        nb.cells.pop(idx)
        # Broadcast to all connected clients so they see the cell removed immediately
        await broadcast_to_notebook(dlg_name, AllCellsOOB(nb))
    return {"status": "ok"}

def _str_to_bool(val: str) -> bool:
    """Convert string form value to boolean. Handles 'True', 'true', '1', etc."""
    if val is None:
        return None
    return str(val).lower() in ('true', '1', 'yes', 'on')


@rt("/update_msg_")
async def post(dlg_name: str, id_: str,
               content: str = None, msg_type: str = None, output: str = None,
               time_run: str = None, is_exported: str = None, skipped: str = None,
               i_collapsed: str = None, o_collapsed: str = None,
               heading_collapsed: str = None, pinned: str = None):
    """Update message properties.

    FastHTML requires explicit params (no **kwargs).
    Boolean params use str type because HTTP form data sends 'True'/'False' strings,
    and FastHTML can't convert 'True' to int.
    """
    print(f"[UPDATE_MSG] dlg_name={dlg_name}, id_={id_}, pinned={pinned}", flush=True)
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        print(f"[UPDATE_MSG] Cell {cell.id} before: pinned={cell.pinned}", flush=True)

        # Map and apply each field if provided (not None)
        if content is not None:
            cell.source = content
        if msg_type is not None:
            cell.cell_type = msg_type
        if output is not None:
            cell.output = output
        if time_run is not None:
            cell.time_run = time_run
        if is_exported is not None:
            cell.is_exported = _str_to_bool(is_exported)
        if skipped is not None:
            cell.skipped = _str_to_bool(skipped)
        if i_collapsed is not None:
            cell.input_collapse = int(i_collapsed) if i_collapsed else 0
        if o_collapsed is not None:
            cell.output_collapse = int(o_collapsed) if o_collapsed else 0
        if heading_collapsed is not None:
            cell.heading_collapsed = _str_to_bool(heading_collapsed)
        if pinned is not None:
            cell.pinned = _str_to_bool(pinned)
            print(f"[UPDATE_MSG] Set pinned={cell.pinned}", flush=True)

        print(f"[UPDATE_MSG] Cell {cell.id} after: pinned={cell.pinned}", flush=True)
        # Broadcast the updated cell to all connected clients
        await broadcast_to_notebook(dlg_name, CellViewOOB(cell, dlg_name))
    # Return the cell ID - dialoghelper uses this to set __msg_id for relative operations
    return id_

@rt("/add_runq_")
async def post(dlg_name: str, ids: str, api: str = ""):
    """Add message to execution queue."""
    nb = get_notebook(dlg_name)
    # ids can be comma-separated, but for now we just handle the first one
    msgid = ids.split(',')[0] if ids else ""
    idx = get_msg_idx(nb, msgid)
    if idx < 0:
        return {"error": f"Message {msgid} not found"}

    cell = nb.cells[idx]
    if cell.cell_type != "code":
        return {"error": "Only code cells can be executed"}

    # Get execution queue and queue the cell
    queue = get_execution_queue(dlg_name)
    queue.queue_cell(dlg_name, cell)

    # Broadcast queue state to all clients
    await broadcast_queue_state(dlg_name)

    return {"status": "ok", "cell_id": cell.id}

@rt("/msg_insert_line_")
def post(dlg_name: str, id_: str, insert_line: int, new_str: str):
    """Insert line at position in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines.insert(insert_line, new_str)
        cell.source = '\n'.join(lines)
    return {"status": "ok"}

@rt("/msg_str_replace_")
def post(dlg_name: str, id_: str, old_str: str, new_str: str):
    """Replace string in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        nb.cells[idx].source = nb.cells[idx].source.replace(old_str, new_str, 1)
    return {"status": "ok"}

@rt("/msg_strs_replace_")
def post(dlg_name: str, id_: str, old_strs: str, new_strs: str):
    """Replace multiple strings (JSON arrays)."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        old_list = json.loads(old_strs)
        new_list = json.loads(new_strs)
        for old, new in zip(old_list, new_list):
            cell.source = cell.source.replace(old, new, 1)
    return {"status": "ok"}

@rt("/msg_replace_lines_")
def post(dlg_name: str, id_: str, start_line: int, end_line: int, new_content: str):
    """Replace line range in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines[start_line:end_line] = [new_content]
        cell.source = '\n'.join(lines)
    return {"status": "ok"}

@rt("/add_html_")
async def post(dlg_name: str, content: str):
    """Add HTML content (for OOB swaps) - broadcasts via WebSocket."""
    await broadcast_to_notebook(dlg_name, Safe(content))
    return {"status": "ok"}

@rt("/push_data_blocking_")
async def post(dlg_name: str, data_id: str, data: str = ""):
    """Push data to a queue for pop_data_blocking_ to consume."""
    queue = get_data_queue(dlg_name, data_id)
    try:
        parsed_data = json.loads(data) if data else {}
    except json.JSONDecodeError:
        parsed_data = {"raw": data}
    await queue.put(parsed_data)
    return {"status": "ok"}

@rt("/pop_data_blocking_")
async def post(dlg_name: str, data_id: str, timeout: int = 15):
    """Pop blocking data (for events) - async with timeout."""
    queue = get_data_queue(dlg_name, data_id)
    try:
        data = await asyncio.wait_for(queue.get(), timeout=timeout)
        return data
    except asyncio.TimeoutError:
        return {"error": "timeout"}

# ============================================================================
# WebSocket for Streaming
# ============================================================================

# Use FastHTML's ws decorator with a simpler pattern
# The key insight: we register on connect, not on first message

async def ws_on_connect(send, scope):
    """Called when WebSocket connection is established."""
    # Extract notebook ID from scope path
    path = scope.get('path', '')
    # Path is like /ws/notebook_id
    parts = path.strip('/').split('/')
    nb_id = parts[1] if len(parts) > 1 else 'default'

    if nb_id not in ws_connections:
        ws_connections[nb_id] = []
    ws_connections[nb_id].append(send)
    print(f"[WS] Client connected to {nb_id}. Total: {len(ws_connections[nb_id])}", flush=True)

async def ws_on_disconnect(send, scope):
    """Called when WebSocket connection is closed."""
    path = scope.get('path', '')
    parts = path.strip('/').split('/')
    nb_id = parts[1] if len(parts) > 1 else 'default'

    if nb_id in ws_connections and send in ws_connections[nb_id]:
        ws_connections[nb_id].remove(send)
        print(f"[WS] Client disconnected from {nb_id}. Total: {len(ws_connections[nb_id])}", flush=True)

@app.ws('/ws/{nb_id}', conn=ws_on_connect, disconn=ws_on_disconnect)
async def ws(msg, send, nb_id: str):
    """Handle incoming WebSocket messages."""
    # FastHTML may pass _empty or None for empty/initial messages - ignore them
    if msg is None or not isinstance(msg, str) or not msg:
        return

    print(f"[WS] Message from {nb_id}: {msg[:50]}", flush=True)

    try:
        data = json.loads(msg)
        if data.get("type") == "cancel":
            cell_id = data.get("cell_id")
            if cell_id:
                cancelled_cells.add(cell_id)
                print(f"[WS] Cancelled cell {cell_id}", flush=True)
    except json.JSONDecodeError:
        pass

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("🚀 Dialeng starting at http://localhost:8000")
    print("   Notebooks saved to: ./notebooks/")
    print("   Format: Solveit-compatible .ipynb")
    print("")
    # Print credential status
    print_credential_status(CREDENTIAL_STATUS)
    print("")
    # Print config status
    print_config_status(DIALENG_CONFIG)
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
