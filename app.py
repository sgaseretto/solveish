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


# ============================================================================
# CSS
# ============================================================================

css = """
/* Dark theme (default) */
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
    --accent-yellow: #f0c000;
    --accent-red: #f85149;
    --border: #30363d;
}

/* Light theme */
[data-theme="light"] {
    --bg-primary: #ffffff;
    --bg-secondary: #f6f8fa;
    --bg-cell: #eaeef2;
    --bg-input: #ffffff;
    --text-primary: #24292f;
    --text-muted: #57606a;
    --accent-blue: #0969da;
    --accent-green: #1a7f37;
    --accent-purple: #8250df;
    --accent-orange: #bf8700;
    --accent-yellow: #d4a000;
    --accent-red: #cf222e;
    --border: #d0d7de;
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
.btn-cancel-all { background: var(--accent-red); border-color: var(--accent-red); color: white; }
.btn-cancel-all:hover { background: #c62828; }
.btn-save { background: var(--accent-blue); border-color: var(--accent-blue); }

.mode-select, .model-select {
    padding: 4px 8px; background: var(--bg-cell); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text-primary); font-size: 0.8rem;
}

.model-select {
    margin-left: 4px;
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
.cell.queued { border-color: var(--accent-yellow, #f0c000); }

.cell.queued .btn-run {
    background: var(--accent-yellow, #f0c000);
    color: #000;
    cursor: not-allowed;
}

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

/* Code cell output container */
.cell-output {
    background: var(--bg-input);
    border-top: 1px solid var(--border);
}
.cell-output:empty { display: none; }
.cell-output.error { }

/* Stream output (print, tqdm, etc.) */
.cell-output .stream-output {
    margin: 0;
    padding: 12px;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 400px;
    overflow-y: auto;
    background: transparent;
}
.cell-output .stream-output.stderr,
.cell-output.error .stream-output {
    color: var(--accent-red);
}

/* Display data (images, HTML, plots, etc.) */
.cell-output .display-data {
    padding: 12px;
}
.cell-output .mime-image {
    max-width: 100%;
    height: auto;
    display: block;
}
.cell-output .mime-svg svg {
    max-width: 100%;
    height: auto;
}
.cell-output .mime-text,
.cell-output .mime-json {
    margin: 0;
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
}
.cell-output .mime-html {
    /* Allow full HTML rendering - trusted user code */
}
.cell-output pre {
    margin: 0;
    padding: 12px;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Legacy output class for backwards compatibility */
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

/* Thinking indicator */
.thinking-indicator {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.85rem; color: var(--accent-purple);
    animation: thinking-pulse 1.5s ease-in-out infinite;
}
@keyframes thinking-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Cell collapse/fold - Legacy full collapse */
.cell.collapsed .cell-body { display: none; }
.cell.collapsed { opacity: 0.7; }
.collapse-btn {
    background: none; border: none; cursor: pointer;
    color: var(--text-muted); font-size: 0.85rem;
    padding: 2px 4px; transition: transform 0.2s;
}
.collapse-btn:hover { color: var(--text-primary); }
.cell.collapsed .collapse-btn { transform: rotate(-90deg); }

/* Multi-level collapse: Scrollable mode - limited height with scroll */
.collapse-scrollable {
    max-height: 168px !important;
    overflow-y: auto !important;
    min-height: 40px !important;
}
.collapse-scrollable::-webkit-scrollbar {
    width: 6px;
}
.collapse-scrollable::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}
.collapse-scrollable::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
.collapse-scrollable::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Multi-level collapse: Summary mode - first line only with ellipsis */
.collapse-summary {
    max-height: 2.25em !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    padding: 6px 12px !important;
}
.collapse-summary * {
    display: inline !important;
    white-space: nowrap !important;
}

/* Collapse section buttons */
.section-collapse-btn {
    background: none; border: none; cursor: pointer;
    color: var(--text-muted); font-size: 0.7rem;
    padding: 2px 6px; margin-left: 8px;
    border-radius: 3px; transition: all 0.15s;
}
.section-collapse-btn:hover {
    background: var(--bg-cell);
    color: var(--text-primary);
}
/* Collapse level indicators */
.section-collapse-btn[data-level="0"]::after { content: "▼"; }
.section-collapse-btn[data-level="1"]::after { content: "◐"; }
.section-collapse-btn[data-level="2"]::after { content: "▬"; }

/* Cell header collapse controls */
.collapse-controls {
    display: flex; align-items: center; gap: 4px;
    font-size: 0.7rem; color: var(--text-muted);
}
.collapse-controls .label {
    font-size: 0.6rem; text-transform: uppercase; opacity: 0.7;
}

/* Cancel button */
.btn-cancel {
    background: var(--accent-red); border-color: var(--accent-red); color: #fff;
}
.btn-cancel:hover { background: #d73a49; }

/* Theme toggle */
.theme-toggle {
    background: var(--bg-cell); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; cursor: pointer;
    font-size: 1rem; transition: all 0.15s;
}
.theme-toggle:hover { border-color: var(--accent-blue); }

/* Mobile responsive */
@media (max-width: 768px) {
    .container { padding: 10px; }
    .header { flex-direction: column; gap: 12px; align-items: stretch; }
    .title { font-size: 1.1rem; justify-content: center; }
    .toolbar { justify-content: center; flex-wrap: wrap; }
    .cell-header { flex-direction: column; gap: 8px; padding: 8px; }
    .cell-actions { justify-content: center; flex-wrap: wrap; }
    .cell-badge { align-self: flex-start; }
    .add-row { flex-direction: column; gap: 4px; }
    .add-row .btn { width: 100%; justify-content: center; }
    .file-list { flex-direction: column; }
    .file-item { text-align: center; }
    .source, .prompt-content { font-size: 0.85rem; }
    .ace-container { min-height: 100px; }
    .btn { padding: 8px 12px; }
    .btn-sm { padding: 6px 10px; }
    .mode-select, .model-select { width: 100%; margin-left: 0; }
}

@media (max-width: 480px) {
    .container { padding: 8px; }
    .header { padding: 8px 0; margin-bottom: 12px; }
    .title { font-size: 1rem; }
    .toolbar { gap: 4px; }
    .btn { font-size: 0.8rem; padding: 6px 8px; }
    .cell { margin-bottom: 8px; border-radius: 6px; }
    .cell-header { padding: 6px 8px; }
    .cell-badge { font-size: 0.6rem; padding: 2px 6px; }
    .cell-body { padding: 0; }
    .source, .prompt-content { padding: 10px; min-height: 50px; }
    .output, .cell-output pre, .cell-output .stream-output { padding: 10px; font-size: 0.8rem; }
    .md-preview, .ai-preview { padding: 10px; }
    .edit-hint { font-size: 0.65rem; padding: 3px 10px; }
}

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
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px var(--accent-blue) !important;
}

/* Ensure focused state overrides other states */
.cell.focused.streaming {
    border-color: var(--accent-blue) !important;
}
.cell.focused.queued {
    border-color: var(--accent-blue) !important;
}
"""

# ============================================================================
# JavaScript
# ============================================================================

js = """
// ==================== Global Cell Selection (Event Delegation) ====================
// Use event delegation on document to ensure cell selection works even when clicking buttons
document.addEventListener('mousedown', (e) => {
    const cell = e.target.closest('.cell');
    if (cell) {
        const cellId = cell.id.replace('cell-', '');
        if (cellId && typeof setFocusedCell === 'function') {
            setFocusedCell(cellId);
        }
    }
}, true);  // Use capture phase to get the event before it's stopped

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
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    editor.setTheme(currentTheme === 'light' ? 'ace/theme/chrome' : 'ace/theme/monokai');
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
    
    // When Ace editor gets focus, also set cell as focused
    editor.on('focus', () => {
        setFocusedCell(cellId);
    });

    // Shift+Enter to run AND move to next cell (Jupyter style)
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
                // Move to next cell immediately (Jupyter behavior)
                moveToNextCell(cell);
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

    // Double-Escape to cancel all (handled by global keydown listener)
    // No need for Ace-specific binding since Escape blurs the editor first

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
    
    // ===== Shift+Enter - Run current cell AND move to next (Jupyter style) =====
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
                    // Click run button first
                    btn.click();
                    // Move focus to next cell IMMEDIATELY (don't wait for server)
                    // This is Jupyter behavior - Shift+Enter runs AND moves
                    moveToNextCell(cell);
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
    
    // ===== Escape - Exit edit mode OR Cancel All (double Escape) =====
    if (e.key === 'Escape') {
        const now = Date.now();
        // Check for double-Escape (like Jupyter's I I for interrupt)
        if (lastKey === 'Escape' && (now - lastKeyTime) < 500) {
            // Double Escape pressed - cancel all execution
            e.preventDefault();
            cancelAllExecution();
            lastKey = '';
            lastKeyTime = 0;
            return;
        }
        // Single Escape - exit edit mode
        if (document.activeElement) {
            document.activeElement.blur();
        }
        Object.values(aceEditors).forEach(ed => ed.blur());
        lastKey = 'Escape';
        lastKeyTime = now;
    }

    // ===== Z - Collapse shortcuts =====
    // Z: cycle input collapse, Shift+Z: cycle output collapse, Alt+Z: cycle both
    if ((e.key === 'z' || e.key === 'Z') && !inInput && !inAce) {
        if (currentCellId) {
            e.preventDefault();
            if (e.altKey) {
                // Alt+Z: cycle both
                cycleCollapseLevel(currentCellId, 'both');
            } else if (e.shiftKey) {
                // Shift+Z: cycle output
                cycleCollapseLevel(currentCellId, 'output');
            } else {
                // Z: cycle input
                cycleCollapseLevel(currentCellId, 'input');
            }
        }
    }

    // ===== 0-3: Set specific collapse level =====
    // 0-3 for input, Shift+0-3 for output
    if (['0', '1', '2', '3'].includes(e.key) && !inInput && !inAce && !mod) {
        if (currentCellId) {
            const level = parseInt(e.key);
            if (e.shiftKey) {
                e.preventDefault();
                setCollapseLevel(currentCellId, 'output', level);
                // Also save to server
                fetch(`${window.location.pathname}/cell/${currentCellId}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=output&level=${level}`
                });
            } else if (e.altKey) {
                e.preventDefault();
                // Alt+number: set both to same level
                setCollapseLevel(currentCellId, 'input', level);
                setCollapseLevel(currentCellId, 'output', level);
                fetch(`${window.location.pathname}/cell/${currentCellId}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=both&level=${level}`
                });
            }
        }
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
                // Reset streaming state for this cell (HTMX swap means request completed)
                if (streamingCellId === cellId) {
                    finishStreaming(cellId);
                }
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

// Handle HTMX errors - ensure streaming state is reset for both prompt and code cells
function resetCellOnError(e, errorMsg) {
    // Check if this is a cell-related request
    const target = e.detail?.target;
    if (target && target.id && target.id.startsWith('cell-')) {
        const cellId = target.id.replace('cell-', '');
        const cell = document.getElementById(`cell-${cellId}`);

        if (cell && cell.classList.contains('streaming')) {
            // Determine cell type and reset appropriately
            const isCodeCell = cell.querySelector('.ace-container') !== null;
            const isPromptCell = cell.querySelector('.prompt-source') !== null;

            if (isPromptCell && streamingCellId === cellId) {
                finishStreaming(cellId);
                const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
                if (preview) {
                    preview.innerHTML = `<p style="color: var(--error);">${errorMsg}</p>`;
                }
            } else if (isCodeCell) {
                finishCodeStreaming(cellId, true);
                const outputEl = document.getElementById(`output-${cellId}`);
                if (outputEl) {
                    outputEl.innerHTML = `<pre class="stream-output" style="color: var(--error);">${errorMsg}</pre>`;
                }
            }
        }
    }

    // Fallback: reset prompt streaming if we have a streaming cell
    if (streamingCellId) {
        finishStreaming(streamingCellId);
    }
}

document.addEventListener('htmx:responseError', (e) => {
    console.error('[HTMX] Response error:', e.detail);
    resetCellOnError(e, 'Request failed. Please try again.');
});

document.addEventListener('htmx:sendError', (e) => {
    console.error('[HTMX] Send error:', e.detail);
    resetCellOnError(e, 'Network error. Please check your connection.');
});

document.addEventListener('htmx:timeout', (e) => {
    console.error('[HTMX] Timeout:', e.detail);
    resetCellOnError(e, 'Request timed out. Please try again.');
});

// On page load
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
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

// ==================== Theme Toggle ====================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update Ace editor themes
    const aceTheme = newTheme === 'light' ? 'ace/theme/chrome' : 'ace/theme/monokai';
    Object.values(aceEditors).forEach(editor => {
        editor.setTheme(aceTheme);
    });

    // Update toggle button
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = newTheme === 'light' ? '🌙' : '☀️';
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = savedTheme === 'light' ? '🌙' : '☀️';
}

// ==================== Model Select Toggle ====================
function toggleModelSelect(mode) {
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        // Show model dropdown only for non-mock modes
        modelSelect.style.display = mode === 'mock' ? 'none' : '';
    }
}

// ==================== Cell Collapse ====================
// Collapse levels: 0=expanded, 1=scrollable, 2=summary
const COLLAPSE_LEVELS = ['', 'collapse-scrollable', 'collapse-summary'];
const COLLAPSE_LABELS = ['Expanded', 'Scrollable', 'Summary'];

function toggleCollapse(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.toggle('collapsed');
        // Send update to server
        const isCollapsed = cell.classList.contains('collapsed');
        fetch(`${window.location.pathname}/cell/${cellId}/collapse`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `collapsed=${isCollapsed}`
        });
    }
}

function cycleCollapseLevel(cellId, section) {
    // section can be 'input', 'output', or 'both'
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    if (section === 'both') {
        cycleCollapseLevel(cellId, 'input');
        cycleCollapseLevel(cellId, 'output');
        return;
    }

    // Find the section element
    const sectionEl = cell.querySelector(`[data-collapse-section="${section}"]`);
    const btn = cell.querySelector(`[data-collapse-btn="${section}"]`);
    if (!sectionEl) return;

    // Get current level
    let currentLevel = 0;
    for (let i = COLLAPSE_LEVELS.length - 1; i > 0; i--) {
        if (COLLAPSE_LEVELS[i] && sectionEl.classList.contains(COLLAPSE_LEVELS[i])) {
            currentLevel = i;
            break;
        }
    }

    // Cycle to next level (0 -> 1 -> 2 -> 3 -> 0)
    const nextLevel = (currentLevel + 1) % COLLAPSE_LEVELS.length;

    // Remove all collapse classes
    COLLAPSE_LEVELS.forEach(cls => {
        if (cls) sectionEl.classList.remove(cls);
    });

    // Add new collapse class if not expanded
    if (COLLAPSE_LEVELS[nextLevel]) {
        sectionEl.classList.add(COLLAPSE_LEVELS[nextLevel]);
    }

    // Update button indicator
    if (btn) {
        btn.setAttribute('data-level', nextLevel);
        btn.title = `${section === 'input' ? 'Input' : 'Output'}: ${COLLAPSE_LABELS[nextLevel]} (click to cycle)`;
    }

    // Send update to server
    fetch(`${window.location.pathname}/cell/${cellId}/collapse-section`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `section=${section}&level=${nextLevel}`
    });
}

function setCollapseLevel(cellId, section, level) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    const sectionEl = cell.querySelector(`[data-collapse-section="${section}"]`);
    const btn = cell.querySelector(`[data-collapse-btn="${section}"]`);
    if (!sectionEl) return;

    // Remove all collapse classes
    COLLAPSE_LEVELS.forEach(cls => {
        if (cls) sectionEl.classList.remove(cls);
    });

    // Add new collapse class if not expanded
    if (COLLAPSE_LEVELS[level]) {
        sectionEl.classList.add(COLLAPSE_LEVELS[level]);
    }

    // Update button indicator
    if (btn) {
        btn.setAttribute('data-level', level);
        btn.title = `${section === 'input' ? 'Input' : 'Output'}: ${COLLAPSE_LABELS[level]} (click to cycle)`;
    }
}

// ==================== Cancel Streaming ====================
let cancelledCells = new Set();

function cancelStreaming(cellId) {
    cancelledCells.add(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.remove('streaming');
        // Hide cancel button, show run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }
    // Send cancel message via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: 'cancel', cell_id: cellId}));
    }
    streamingCellId = null;
}

// ==================== WebSocket for Streaming ====================
let ws = null;
let streamingCellId = null;
let currentNotebookId = null;  // Global notebook ID for use in cancelAllExecution, etc.

function connectWebSocket(notebookId) {
    currentNotebookId = notebookId;  // Store globally for other functions to use
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);

    ws.onopen = function() {
        console.log('[WS] Connected to notebook:', notebookId);
        // Send join message to register this connection with the notebook
        ws.send(JSON.stringify({type: 'join', notebook_id: notebookId}));
    };

    ws.onmessage = function(event) {
        const msg = event.data;

        // Debug: Log every message received
        console.log('[WS] RAW message received, length:', msg?.length, 'type:', typeof msg, 'starts:', msg?.substring?.(0, 50));

        // Check if message is HTML (OOB swap from collaborator) or JSON (streaming)
        if (msg && typeof msg === 'string' && msg.startsWith('<')) {
            // HTML with hx-swap-oob - process as OOB swap
            console.log('[WS] Received OOB HTML swap, length:', msg.length);
            processOOBSwap(msg);
            return;
        }

        // JSON message for streaming, thinking indicators, etc.
        let data;
        try {
            data = JSON.parse(msg);
        } catch (e) {
            console.error('[WS] Failed to parse JSON message:', msg?.substring?.(0, 100), e);
            return;
        }
        console.log('[WS] Received message:', data.type, 'cell_id:', data.cell_id || 'none');

        if (data.type === 'stream_chunk') {
            // Skip if cancelled
            if (cancelledCells.has(data.cell_id)) return;
            appendToResponse(data.cell_id, data.chunk, data.thinking);
            // Reset streaming timeout on activity
            resetStreamingTimeout();
        } else if (data.type === 'stream_end') {
            console.log('[WS] stream_end received for cell:', data.cell_id);
            cancelledCells.delete(data.cell_id);
            finishStreaming(data.cell_id);
        } else if (data.type === 'thinking_start') {
            showThinkingIndicator(data.cell_id);
            resetStreamingTimeout();
        } else if (data.type === 'thinking_end') {
            hideThinkingIndicator(data.cell_id);
        } else if (data.type === 'code_stream_start') {
            // Code cell execution started - show streaming indicator
            console.log('[WS] code_stream_start received for cell:', data.cell_id);
            startCodeStreaming(data.cell_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_stream_chunk') {
            // Append output chunk to code cell
            console.log('[WS] code_stream_chunk received for cell:', data.cell_id, 'stream:', data.stream, 'length:', data.chunk?.length || 0);
            appendCodeOutput(data.cell_id, data.chunk, data.stream);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_stream_end') {
            // Code cell execution finished
            console.log('[WS] code_stream_end received for cell:', data.cell_id, 'has_error:', data.has_error);
            finishCodeStreaming(data.cell_id, data.has_error);
        } else if (data.type === 'code_display_data') {
            // Rich output (image, HTML, plot, etc.)
            console.log('[WS] code_display_data received for cell:', data.cell_id);
            appendDisplayData(data.cell_id, data.html);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'queue_update') {
            // Queue state update from server
            console.log('[WS] queue_update received:', data);
            handleQueueUpdate(data);
        } else if (data.type === 'cell_state_change') {
            // Cell state change (queued, running, idle)
            console.log('[WS] cell_state_change received:', data.cell_id, data.state);
            // State changes are now handled via queue_update for consistency
        }
    };

    ws.onclose = function() {
        console.log('[WS] Disconnected, reconnecting in 3s...');
        setTimeout(() => connectWebSocket(notebookId), 3000);
    };

    ws.onerror = function(error) {
        console.error('[WS] Error:', error);
    };
}

function appendToResponse(cellId, chunk, isThinking) {
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        if (textarea.value === 'Generating...' || textarea.value === 'Click ▶ to generate response...' || textarea.value.startsWith('🧠')) {
            textarea.value = '';
        }
        textarea.value += chunk;
        if (preview) {
            preview.innerHTML = renderMarkdown(textarea.value);
        }
    }
}

function showThinkingIndicator(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (preview) {
        preview.innerHTML = '<div class="thinking-indicator"><span>🧠</span> Thinking...</div>';
    }
    if (cell) {
        const header = cell.querySelector('.cell-header');
        if (header && !header.querySelector('.thinking-indicator')) {
            const indicator = document.createElement('span');
            indicator.className = 'thinking-indicator';
            indicator.innerHTML = '🧠 Thinking...';
            indicator.id = `thinking-${cellId}`;
            header.querySelector('.cell-actions')?.prepend(indicator);
        }
    }
}

function hideThinkingIndicator(cellId) {
    const indicator = document.getElementById(`thinking-${cellId}`);
    if (indicator) indicator.remove();
}

function finishStreaming(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.remove('streaming');
        // Hide cancel button, show run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }
    hideThinkingIndicator(cellId);
    streamingCellId = null;
    // Clear safety timeout
    if (streamingTimeoutId) {
        clearTimeout(streamingTimeoutId);
        streamingTimeoutId = null;
    }
}

let streamingTimeoutId = null;
const STREAMING_TIMEOUT_MS = 120000; // 2 minutes safety timeout

function startStreaming(cellId, useThinking) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.add('streaming');
        // Show cancel button, hide run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = '';
        if (runBtn) runBtn.style.display = 'none';
    }
    streamingCellId = cellId;
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        textarea.value = useThinking ? '🧠 Thinking...' : 'Generating...';
    }
    if (preview && useThinking) {
        preview.innerHTML = '<div class="thinking-indicator"><span>🧠</span> Thinking...</div>';
    }

    // Set safety timeout to reset streaming state
    if (streamingTimeoutId) clearTimeout(streamingTimeoutId);
    streamingTimeoutId = setTimeout(() => {
        if (streamingCellId === cellId) {
            console.warn('[Streaming] Safety timeout reached, resetting streaming state');
            finishStreaming(cellId);
        }
    }, STREAMING_TIMEOUT_MS);
}

function resetStreamingTimeout() {
    // Call this when we receive streaming activity to reset the timeout
    if (streamingTimeoutId && streamingCellId) {
        clearTimeout(streamingTimeoutId);
        const cellId = streamingCellId;
        streamingTimeoutId = setTimeout(() => {
            if (streamingCellId === cellId) {
                console.warn('[Streaming] Safety timeout reached, resetting streaming state');
                finishStreaming(cellId);
            }
        }, STREAMING_TIMEOUT_MS);
    }
}

// ==================== Code Cell Streaming Functions ====================

// ANSI color code mapping
const ANSI_COLORS = {
    '30': '#000', '31': '#c00', '32': '#0a0', '33': '#a50',
    '34': '#00a', '35': '#a0a', '36': '#0aa', '37': '#aaa',
    '90': '#555', '91': '#f55', '92': '#5f5', '93': '#ff5',
    '94': '#55f', '95': '#f5f', '96': '#5ff', '97': '#fff',
    '40': 'background:#000', '41': 'background:#c00',
    '42': 'background:#0a0', '43': 'background:#a50',
    '44': 'background:#00a', '45': 'background:#a0a',
    '46': 'background:#0aa', '47': 'background:#aaa'
};

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function ansiToHtml(text) {
    let result = '';
    let openSpans = 0;

    const parts = text.split(/(\\x1b\\[[0-9;]*m)/);
    for (const part of parts) {
        const match = part.match(/\\x1b\\[([0-9;]*)m/);
        if (match) {
            const codes = match[1].split(';');
            for (const code of codes) {
                if (code === '0' || code === '') {
                    // Reset all styles
                    while (openSpans > 0) {
                        result += '</span>';
                        openSpans--;
                    }
                } else if (code === '1') {
                    result += '<span style="font-weight:bold">';
                    openSpans++;
                } else if (ANSI_COLORS[code]) {
                    const style = ANSI_COLORS[code].includes(':')
                        ? ANSI_COLORS[code]
                        : `color:${ANSI_COLORS[code]}`;
                    result += `<span style="${style}">`;
                    openSpans++;
                }
            }
        } else {
            result += escapeHtml(part);
        }
    }

    // Close any remaining open spans
    while (openSpans > 0) {
        result += '</span>';
        openSpans--;
    }

    return result;
}

// Track raw text content for carriage return handling
const streamTextContent = new Map();

function startCodeStreaming(cellId) {
    console.log('[Code] startCodeStreaming called for cell:', cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (!cell) {
        console.error('[Code] Cell element not found:', `cell-${cellId}`);
        return;
    }

    if (!outputEl) {
        console.error('[Code] Output element not found:', `output-${cellId}`);
    }

    cell.classList.add('streaming');
    const cancelBtn = cell.querySelector('.btn-cancel');
    const runBtn = cell.querySelector('.btn-run');
    if (cancelBtn) cancelBtn.style.display = '';
    if (runBtn) runBtn.style.display = 'none';

    if (outputEl) {
        outputEl.innerHTML = '';  // Clear for fresh output
        outputEl.classList.remove('error');
    }

    // Reset text content tracker for this cell
    streamTextContent.set(cellId, '');

    console.log('[Code] Started streaming for cell:', cellId, 'cell found:', !!cell, 'output found:', !!outputEl);
}

function appendCodeOutput(cellId, chunk, streamName) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Get or create stream output container
    let streamEl = outputEl.querySelector('.stream-output');
    if (!streamEl) {
        streamEl = document.createElement('pre');
        streamEl.className = 'stream-output';
        outputEl.appendChild(streamEl);
    }

    if (streamName === 'stderr') {
        outputEl.classList.add('error');
    }

    // Get current raw text and apply chunk
    let currentText = streamTextContent.get(cellId) || '';

    // Handle carriage return for progress bars (tqdm)
    if (chunk.includes('\\r')) {
        const lines = currentText.split('\\n');
        const parts = chunk.split('\\r');

        for (let i = 0; i < parts.length; i++) {
            if (i === 0) {
                // First part appends to current line
                lines[lines.length - 1] += parts[i];
            } else {
                // After \\r, replace current line content
                lines[lines.length - 1] = parts[i];
            }
        }
        currentText = lines.join('\\n');
    } else {
        currentText += chunk;
    }

    // Store updated raw text
    streamTextContent.set(cellId, currentText);

    // Render with ANSI color conversion
    streamEl.innerHTML = ansiToHtml(currentText);
    streamEl.scrollTop = streamEl.scrollHeight;
}

function appendDisplayData(cellId, html) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Create display data container
    const displayEl = document.createElement('div');
    displayEl.className = 'display-data';
    displayEl.innerHTML = html;
    outputEl.appendChild(displayEl);

    // Execute any scripts in the HTML (for interactive widgets)
    displayEl.querySelectorAll('script').forEach(script => {
        const newScript = document.createElement('script');
        newScript.textContent = script.textContent;
        script.parentNode.replaceChild(newScript, script);
    });
}

function finishCodeStreaming(cellId, hasError) {
    console.log('[Code] finishCodeStreaming called for cell:', cellId, 'hasError:', hasError);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (cell) {
        cell.classList.remove('streaming');
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }

    if (outputEl && hasError) {
        outputEl.classList.add('error');
    }

    // Clean up text content tracker
    streamTextContent.delete(cellId);

    // Clear the streaming timeout
    clearCodeStreamingTimeout(cellId);

    console.log('[Code] Finished streaming for cell:', cellId, hasError ? '(with errors)' : '');
}

// ============================================================================
// Queue State Management
// ============================================================================

// Track queue state for cells
const cellQueueState = new Map(); // cellId -> {state: 'queued'|'running'|'idle', position: number}

function handleQueueUpdate(data) {
    // Clear all previous queue states
    cellQueueState.forEach((_, cellId) => {
        clearCellQueueState(cellId);
    });
    cellQueueState.clear();

    // Set running cell
    if (data.running_cell_id) {
        updateCellQueueState(data.running_cell_id, 'running', null);
    }

    // Set queued cells with positions
    if (data.queued_cell_ids) {
        data.queued_cell_ids.forEach((cellId, index) => {
            updateCellQueueState(cellId, 'queued', index + 1);
        });
    }

    // Show/hide Cancel All button based on queue state
    const hasQueuedOrRunning = data.running_cell_id || (data.queued_cell_ids && data.queued_cell_ids.length > 0);
    updateCancelAllButton(hasQueuedOrRunning);
}

function updateCellQueueState(cellId, state, position) {
    cellQueueState.set(cellId, { state, position });
    updateCellVisualState(cellId, state, position);
}

function clearCellQueueState(cellId) {
    cellQueueState.delete(cellId);
    updateCellVisualState(cellId, 'idle', null);
}

function updateCellVisualState(cellId, state, queuePosition) {
    const cell = document.getElementById(`cell-${cellId}`);
    const runBtn = cell?.querySelector('.btn-run');
    const outputEl = document.getElementById(`output-${cellId}`);

    if (!cell) return;

    // Remove queued class only - streaming is managed by startCodeStreaming/stopCodeStreaming
    cell.classList.remove('queued');

    switch(state) {
        case 'queued':
            cell.classList.add('queued');
            if (runBtn) {
                runBtn.style.display = '';
                runBtn.innerHTML = '⏳';
                runBtn.disabled = true;
            }
            if (outputEl) {
                outputEl.innerHTML = `<pre class="stream-output" style="color: var(--accent-yellow);">Queued (position ${queuePosition})...</pre>`;
                outputEl.classList.remove('error');
            }
            break;

        case 'running':
            // Running state is handled by startCodeStreaming
            // Just ensure queued class is removed
            break;

        case 'idle':
        default:
            if (runBtn) {
                runBtn.style.display = '';
                runBtn.innerHTML = '▶';
                runBtn.disabled = false;
            }
            break;
    }
}

function updateCancelAllButton(show) {
    const cancelAllBtn = document.querySelector('.btn-cancel-all');
    if (cancelAllBtn) {
        cancelAllBtn.style.display = show ? '' : 'none';
    }
}

async function cancelAllExecution() {
    if (!currentNotebookId) {
        console.error('[Queue] Cannot cancel: no notebook ID set');
        return;
    }
    try {
        console.log('[Queue] Cancelling all execution for notebook:', currentNotebookId);
        await fetch(`/notebook/${currentNotebookId}/queue/cancel_all`, { method: 'POST' });
    } catch (e) {
        console.error('[Queue] Failed to cancel all:', e);
    }
}

// Code cell streaming timeout mechanism
let codeStreamingTimeouts = new Map();  // Track timeouts per cell
const CODE_STREAMING_TIMEOUT_MS = 30000; // 30 seconds safety timeout (reduced for better UX)

// Called immediately when user clicks run on a code cell
// Provides visual feedback before WebSocket code_stream_start arrives
function prepareCodeRun(cellId) {
    console.log('[Code] prepareCodeRun called for cell:', cellId);

    // Skip if cell is already queued or running
    const queueState = cellQueueState.get(cellId);
    if (queueState && (queueState.state === 'queued' || queueState.state === 'running')) {
        console.log('[Code] Cell already queued/running, skipping prepareCodeRun');
        return;
    }

    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (cell) {
        cell.classList.add('streaming');
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = '';
        if (runBtn) runBtn.style.display = 'none';
    }

    // Clear output and show "Queuing..." indicator (queue_update will update to "Queued (position N)...")
    if (outputEl) {
        outputEl.innerHTML = '<pre class="stream-output" style="color: var(--text-muted);">Queuing...</pre>';
        outputEl.classList.remove('error');
    }

    // Reset text content tracker
    streamTextContent.set(cellId, '');

    // Set safety timeout to reset streaming state if server doesn't respond
    clearCodeStreamingTimeout(cellId);
    const timeoutId = setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.warn('[Code] Safety timeout reached for cell:', cellId);
            finishCodeStreaming(cellId, true);
            const outputEl = document.getElementById(`output-${cellId}`);
            if (outputEl) {
                const currentOutput = outputEl.textContent?.trim();
                if (!currentOutput || currentOutput === 'Running...') {
                    outputEl.innerHTML = '<pre class="stream-output" style="color: var(--error);">Execution timed out. Please try again.</pre>';
                }
            }
        }
    }, CODE_STREAMING_TIMEOUT_MS);
    codeStreamingTimeouts.set(cellId, timeoutId);

    console.log('[Code] Preparing to run cell:', cellId);
}

function clearCodeStreamingTimeout(cellId) {
    const timeoutId = codeStreamingTimeouts.get(cellId);
    if (timeoutId) {
        clearTimeout(timeoutId);
        codeStreamingTimeouts.delete(cellId);
    }
}

function resetCodeStreamingTimeout(cellId) {
    // Call this when we receive streaming activity to reset the timeout
    clearCodeStreamingTimeout(cellId);
    const timeoutId = setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.warn('[Code] Safety timeout reached for cell:', cellId);
            finishCodeStreaming(cellId, true);
        }
    }, CODE_STREAMING_TIMEOUT_MS);
    codeStreamingTimeouts.set(cellId, timeoutId);
}

function interruptCodeCell(notebookId, cellId) {
    console.log('[Code] interruptCodeCell called for cell:', cellId);

    // Use cancelAllExecution to stop running cell AND clear the queue
    // This ensures we don't just interrupt one cell and continue with others
    cancelAllExecution();

    // Clear the streaming timeout for this cell
    clearCodeStreamingTimeout(cellId);

    // Wait a bit for server to finish, then check if cell is still stuck
    setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.log('[Code] Cell still streaming after interrupt, forcing reset');
            finishCodeStreaming(cellId, true);
            const outputEl = document.getElementById(`output-${cellId}`);
            if (outputEl) {
                const currentOutput = outputEl.textContent?.trim();
                if (!currentOutput || currentOutput === 'Running...' || currentOutput === 'Stopping...') {
                    outputEl.innerHTML = '<pre class="stream-output" style="color: var(--warning);">Execution interrupted</pre>';
                }
            }
        }
    }, 2000); // Wait 2 seconds for server to respond

    // Immediately update UI to show stopping state
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        const outputEl = document.getElementById(`output-${cellId}`);
        if (outputEl) {
            const currentOutput = outputEl.textContent?.trim();
            if (currentOutput === 'Running...' || currentOutput.startsWith('Queued')) {
                outputEl.innerHTML = '<pre class="stream-output" style="color: var(--text-muted);">Stopping...</pre>';
            }
        }
    }
}

// ==================== Collaborative WebSocket OOB Swap Handler ====================

function processOOBSwap(html) {
    // Process HTML with hx-swap-oob attributes from WebSocket
    // This handles both full cells container updates and single cell updates
    console.log('[OOB] processOOBSwap called, HTML length:', html.length);

    // Parse the HTML to extract the element(s)
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    const elements = template.content.children;
    console.log('[OOB] Parsed elements count:', elements.length);

    for (const element of elements) {
        const oobAttr = element.getAttribute('hx-swap-oob');
        console.log('[OOB] Element tag:', element.tagName, 'id:', element.id, 'oobAttr:', oobAttr);

        // Handle swap strategies like "beforeend:#js-script" for script injection
        if (oobAttr && oobAttr.includes(':')) {
            const [swapStrategy, targetSelector] = oobAttr.split(':');
            const target = document.querySelector(targetSelector);
            console.log('[OOB] Swap strategy:', swapStrategy, 'target:', targetSelector, 'found:', !!target);

            if (target) {
                element.removeAttribute('hx-swap-oob');

                // For script injection, we need to manually execute the scripts
                // innerHTML doesn't auto-execute scripts for security reasons
                const scripts = element.querySelectorAll('script');
                if (scripts.length > 0) {
                    console.log('[OOB] Found', scripts.length, 'script(s) to execute');
                    scripts.forEach(script => {
                        const newScript = document.createElement('script');
                        // Copy all attributes
                        Array.from(script.attributes).forEach(attr => {
                            newScript.setAttribute(attr.name, attr.value);
                        });
                        newScript.textContent = script.textContent;

                        if (swapStrategy === 'beforeend') {
                            target.appendChild(newScript);
                        } else if (swapStrategy === 'afterbegin') {
                            target.insertBefore(newScript, target.firstChild);
                        } else {
                            target.appendChild(newScript);
                        }
                        console.log('[OOB] Script executed');
                    });
                } else {
                    // Regular content, use innerHTML based on swap strategy
                    if (swapStrategy === 'beforeend') {
                        target.insertAdjacentHTML('beforeend', element.innerHTML);
                    } else if (swapStrategy === 'afterbegin') {
                        target.insertAdjacentHTML('afterbegin', element.innerHTML);
                    } else if (swapStrategy === 'innerHTML') {
                        target.innerHTML = element.innerHTML;
                    }
                }
            }
            continue;
        }

        if (oobAttr !== 'true') continue;

        // Handle script elements specially - they need to be manually executed
        if (element.tagName === 'SCRIPT') {
            console.log('[OOB] Executing script element with id:', element.id);
            const newScript = document.createElement('script');
            // Copy all attributes except hx-swap-oob
            Array.from(element.attributes).forEach(attr => {
                if (attr.name !== 'hx-swap-oob') {
                    newScript.setAttribute(attr.name, attr.value);
                }
            });
            newScript.textContent = element.textContent;

            // If a script with this ID exists, replace it; otherwise append to body
            const existingScript = element.id ? document.getElementById(element.id) : null;
            if (existingScript) {
                existingScript.replaceWith(newScript);
            } else {
                document.body.appendChild(newScript);
            }
            console.log('[OOB] Script executed successfully');
            continue;
        }

        const targetId = element.id;
        if (!targetId) {
            console.log('[OOB] Skipping - no targetId');
            continue;
        }

        const target = document.getElementById(targetId);
        if (!target) {
            console.log('[OOB] Skipping - target not found for id:', targetId);
            continue;
        }
        console.log('[OOB] Found target element:', targetId);

        // Check if this is a cell update
        if (targetId.startsWith('cell-')) {
            const cellId = targetId.replace('cell-', '');
            const isEditing = target.contains(document.activeElement);
            const isStreaming = target.classList.contains('streaming');

            // Skip update if user is editing this cell or it's streaming
            if (isEditing || isStreaming) {
                console.log('[WS] Skipping OOB swap for cell being edited/streamed:', cellId);
                continue;
            }

            // Replace the cell
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);

            // CRITICAL: Reinitialize HTMX bindings on the new element
            // Without this, hx-post/hx-get attributes won't work!
            const newCell = document.getElementById(targetId);
            if (newCell) {
                htmx.process(newCell);

                // Reinitialize Ace editor if it's a code cell
                if (newCell.dataset.type === 'code') {
                    setTimeout(() => initAceEditor(cellId), 0);
                }
            }

            // Re-render previews for this cell
            renderCellPreviews(cellId);
        }
        else if (targetId === 'cells') {
            // Full cells container update (e.g., from dialoghelper add_msg)
            console.log('[OOB] Processing cells container update');
            // Save currently focused cell ID before update
            const focusedCell = document.activeElement?.closest('.cell');
            const focusedCellId = focusedCell?.id?.replace('cell-', '');

            // Only skip if user is actively typing AND no cell is currently streaming
            // If ANY cell is streaming (executing), we need to allow updates for add_msg() to work
            // The Ace editor's hidden textarea keeps focus during execution, but that's not "real typing"
            const isInInput = document.activeElement?.matches('input, textarea, .ace_text-input');
            const anyCellStreaming = document.querySelector('.cell.streaming') !== null;
            const shouldSkip = isInInput && !anyCellStreaming;
            console.log('[OOB] isInInput:', isInInput, 'anyCellStreaming:', anyCellStreaming, 'shouldSkip:', shouldSkip);

            if (shouldSkip) {
                console.log('[OOB] Skipping cells container update - user is typing and no cell is streaming');
                continue;
            }

            // Replace the cells container
            console.log('[OOB] Replacing cells container');
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            console.log('[OOB] Cells container replaced successfully');

            // CRITICAL: Reinitialize HTMX bindings on the new cells container
            // Without this, hx-post/hx-get attributes won't work!
            const newCells = document.getElementById('cells');
            if (newCells) {
                htmx.process(newCells);
            }

            // Reinitialize Ace editors for all code cells
            reinitializeAceEditors();

            // Re-render all markdown previews
            renderAllPreviews();

            // Restore focus if possible
            if (focusedCellId) {
                const restoredCell = document.getElementById(`cell-${focusedCellId}`);
                if (restoredCell) {
                    setFocusedCell(focusedCellId);
                }
            }
        }
    }
}

function reinitializeAceEditors() {
    // Destroy all existing Ace editors
    for (const cellId of Object.keys(aceEditors)) {
        destroyAceEditor(cellId);
    }

    // Find all code cells and initialize their editors
    document.querySelectorAll('.cell[data-type="code"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => initAceEditor(cellId), 0);
    });
}

function renderAllPreviews() {
    // Re-render all markdown previews after a collaborative update
    document.querySelectorAll('.md-preview, .ai-preview, .prompt-preview').forEach(preview => {
        const cellId = preview.dataset.cellId;
        const field = preview.dataset.field;
        if (cellId && field) {
            renderCellPreviews(cellId);
        }
    });
}

function renderCellPreviews(cellId) {
    // Render markdown preview for a specific cell
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    // Handle note cells
    const notePreview = document.getElementById(`preview-${cellId}`);
    if (notePreview) {
        const textarea = document.getElementById(`source-${cellId}`);
        if (textarea) {
            notePreview.innerHTML = renderMarkdown(textarea.value);
        }
    }

    // Handle prompt cells - render both prompt and output previews
    const promptPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="prompt"]`);
    if (promptPreview) {
        const promptTextarea = document.getElementById(`prompt-${cellId}`);
        if (promptTextarea) {
            promptPreview.innerHTML = renderMarkdown(promptTextarea.value);
        }
    }

    const outputPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (outputPreview) {
        const outputTextarea = document.getElementById(`output-${cellId}`);
        if (outputTextarea && outputTextarea.value) {
            outputPreview.innerHTML = renderMarkdown(outputTextarea.value);
        }
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
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.6/theme-chrome.min.js"),
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

def get_collapse_class(level: int) -> str:
    """Get CSS class for collapse level"""
    if level == 1: return "collapse-scrollable"
    if level == 2: return "collapse-summary"
    return ""

def CollapseBtn(cell_id: str, section: str, level: int) -> Button:
    """Create a collapse button for input/output section"""
    labels = {0: "▼", 1: "◐", 2: "▬"}
    tooltips = {0: "Expanded", 1: "Scrollable", 2: "Summary"}
    return Button(
        labels.get(level, "▼"),
        cls="section-collapse-btn",
        data_collapse_btn=section,
        data_level=str(level),
        onclick=f"cycleCollapseLevel('{cell_id}', '{section}')",
        title=f"{section.capitalize()}: {tooltips.get(level, 'Expanded')} (click to cycle)"
    )

def CellView(cell: Cell, notebook_id: str):
    """Render a single cell"""
    meta_info = []
    if cell.execution_count:
        meta_info.append(Span(f"[{cell.execution_count}]"))
    if cell.time_run:
        meta_info.append(Span(cell.time_run))

    # Collapse controls in header
    collapse_controls = []
    if cell.cell_type == "code":
        collapse_controls = [
            Span("In:", cls="label"),
            CollapseBtn(cell.id, "input", cell.input_collapse),
            Span("Out:", cls="label"),
            CollapseBtn(cell.id, "output", cell.output_collapse),
        ]
    elif cell.cell_type == "note":
        collapse_controls = [
            CollapseBtn(cell.id, "input", cell.input_collapse),
        ]
    else:  # prompt
        collapse_controls = [
            Span("In:", cls="label"),
            CollapseBtn(cell.id, "input", cell.input_collapse),
            Span("Out:", cls="label"),
            CollapseBtn(cell.id, "output", cell.output_collapse),
        ]

    header = Div(
        Div(
            Button("▼", cls="collapse-btn", onclick=f"toggleCollapse('{cell.id}')", title="Collapse/Expand (full)"),
            Span(cell.cell_type.upper(), cls=f"cell-badge {cell.cell_type}"),
            Span(*meta_info, cls="cell-meta") if meta_info else None,
            Div(*collapse_controls, cls="collapse-controls") if collapse_controls else None,
        ),
        Div(
            TypeSelect(cell.id, cell.cell_type, notebook_id),
            Button("▶", cls="btn btn-sm btn-run",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/run",
                   hx_target=f"#cell-{cell.id}",
                   hx_swap="none" if cell.cell_type == "code" else "outerHTML",  # Code cells update via WebSocket
                   hx_vals=f"js:{{source: document.getElementById('source-{cell.id}')?.value || ''}}",
                   hx_timeout="120s",  # Extended timeout for long-running cells (matplotlib, etc.)
                   onclick=(f"syncAceToTextarea('{cell.id}'); syncPromptContent('{cell.id}'); startStreaming('{cell.id}', {str(cell.use_thinking).lower()});"
                            if cell.cell_type == "prompt"
                            else f"syncAceToTextarea('{cell.id}'); prepareCodeRun('{cell.id}');"),
                   title="Run (Shift+Enter)") if cell.cell_type != "note" else None,
            Button("⏹", cls="btn btn-sm btn-cancel",
                   onclick=f"cancelStreaming('{cell.id}')" if cell.cell_type == "prompt" else f"interruptCodeCell('{notebook_id}', '{cell.id}')",
                   title="Cancel generation" if cell.cell_type == "prompt" else "Interrupt execution (Ctrl+C)",
                   style="display: none;") if cell.cell_type in ("prompt", "code") else None,
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
        input_collapse_cls = get_collapse_class(cell.input_collapse)
        output_collapse_cls = get_collapse_class(cell.output_collapse)
        body = Div(
            # Hidden textarea for form submission - Ace reads from this
            Textarea(cell.source, name="source", id=f"source-{cell.id}",
                    style="display: none;",
                    hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                    hx_trigger="blur changed", hx_swap="none"),
            # Ace Editor container - with collapse support
            Div(
                Div(id=f"ace-{cell.id}", cls="ace-container"),
                cls=f"cell-input {input_collapse_cls}".strip(),
                data_collapse_section="input"
            ),
            Div(
                Pre(NotStr(cell.output), cls="stream-output") if cell.output else "",
                id=f"output-{cell.id}",
                cls=f"cell-output{' error' if cell.output and ('Error' in cell.output or 'Traceback' in cell.output) else ''} {output_collapse_cls}".strip(),
                data_collapse_section="output"
            ),
            Script(f"setTimeout(() => initAceEditor('{cell.id}'), 0);"),
            cls="cell-body"
        )
    elif cell.cell_type == "note":
        input_collapse_cls = get_collapse_class(cell.input_collapse)
        body = Div(
            Textarea(cell.source, cls="source", name="source", id=f"source-{cell.id}",
                    placeholder="# Markdown notes...",
                    hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                    hx_trigger="blur changed", hx_swap="none",
                    style="display: none;",
                    onblur=f"switchToPreview('{cell.id}', 'source')"),
            Div(
                Div(id=f"preview-{cell.id}", cls="md-preview",
                    data_cell_id=cell.id, data_field="source"),
                cls=f"cell-input {input_collapse_cls}".strip(),
                data_collapse_section="input"
            ),
            Div("Double-click to edit • Escape to finish • Z to cycle collapse", cls="edit-hint"),
            cls="cell-body"
        )
    else:  # Prompt
        # Determine if prompt has been run (has output)
        has_output = bool(cell.output and cell.output.strip())
        input_collapse_cls = get_collapse_class(cell.input_collapse)
        output_collapse_cls = get_collapse_class(cell.output_collapse)

        # User prompt section - show preview if has output, else show textarea
        if has_output:
            # After run: show markdown preview (double-click to edit)
            user_prompt_section = Div(
                Div(Span("👤"), " Your Prompt ",
                    Span("(double-click to edit)", style="font-weight: normal; opacity: 0.6;"),
                    cls="prompt-label user"),
                Div(
                    Textarea(cell.source, cls="prompt-content", name="prompt_source",
                            id=f"prompt-{cell.id}",
                            placeholder="Ask the AI anything...",
                            hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                            hx_trigger="blur changed", hx_swap="none",
                            style="display: none;",
                            oninput=f"document.getElementById('source-{cell.id}').value = this.value",
                            onblur=f"switchToPreview('{cell.id}', 'prompt')"),
                    Div(cls="md-preview prompt-preview", data_cell_id=cell.id, data_field="prompt"),
                    cls=f"prompt-input {input_collapse_cls}".strip(),
                    data_collapse_section="input"
                ),
                cls="prompt-section"
            )
        else:
            # Before run: show editable textarea
            user_prompt_section = Div(
                Div(Span("👤"), " Your Prompt", cls="prompt-label user"),
                Div(
                    Textarea(cell.source, cls="prompt-content", name="prompt_source",
                            id=f"prompt-{cell.id}",
                            placeholder="Ask the AI anything...",
                            hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                            hx_trigger="blur changed", hx_swap="none",
                            oninput=f"document.getElementById('source-{cell.id}').value = this.value"),
                    cls=f"prompt-input {input_collapse_cls}".strip(),
                    data_collapse_section="input"
                ),
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
                Div(
                    Textarea(cell.output if cell.output else "",
                            cls="prompt-content", name="output", id=f"output-{cell.id}",
                            placeholder="Click ▶ to generate response...",
                            hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/output",
                            hx_trigger="blur changed", hx_swap="none",
                            style="display: none; min-height: 80px;",
                            onblur=f"switchToPreview('{cell.id}', 'output')"),
                    Div(cls="ai-preview", data_cell_id=cell.id, data_field="output"),
                    cls=f"prompt-output {output_collapse_cls}".strip(),
                    data_collapse_section="output"
                ),
                cls="prompt-section"
            ),
            cls="cell-body"
        )
    
    collapsed_cls = " collapsed" if cell.collapsed else ""
    return Div(header, body, id=f"cell-{cell.id}", cls=f"cell{collapsed_cls}", data_type=cell.cell_type)

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

def AllCellsOOB(nb: Notebook):
    """Returns AllCells with hx-swap-oob for WebSocket broadcasting.

    HTMX will automatically swap this element by ID when received via WebSocket.
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i+1, nb.id)])
    return Div(*items, id="cells", hx_swap_oob="true")

def CellViewOOB(cell: Cell, notebook_id: str):
    """Returns CellView with hx-swap-oob for WebSocket broadcasting.

    HTMX will automatically swap this element by ID when received via WebSocket.
    """
    # Get the regular cell view
    cell_div = CellView(cell, notebook_id)
    # The cell already has id=f"cell-{cell.id}", just need to add OOB attribute
    # We need to recreate with the OOB attribute since CellView returns a complete Div
    # Read the cell ID from the component and recreate with OOB
    return Div(
        *cell_div.children,
        id=f"cell-{cell.id}",
        cls=cell_div.attrs.get('class', ''),
        hx_swap_oob="true",
        **{k: v for k, v in cell_div.attrs.items() if k not in ('id', 'class')}
    )

def NotebookPage(nb: Notebook, notebook_list: List[str]):
    return Titled(
        f"{nb.title} - Dialeng",
        Div(
            Div(
                Div(Span("📓", cls="title-icon"), Span(nb.title, cls="title")),
                Div(
                    Button("☀️", cls="theme-toggle", id="theme-toggle",
                           onclick="toggleTheme()", title="Toggle light/dark theme"),
                    Select(
                        *[Option(label, value=mode_id, selected=nb.dialog_mode==mode_id)
                          for mode_id, label in AVAILABLE_DIALOG_MODES],
                        cls="mode-select", name="mode", id="mode-select",
                        hx_post=f"/notebook/{nb.id}/mode", hx_swap="none", title="AI Mode",
                        onchange="toggleModelSelect(this.value)"
                    ),
                    Select(
                        *[Option(label, value=model_id, selected=nb.model==model_id)
                          for model_id, label in AVAILABLE_MODELS],
                        cls="model-select", name="model", id="model-select",
                        hx_post=f"/notebook/{nb.id}/model", hx_swap="none", title="Model",
                        style="display: none;" if nb.dialog_mode == "mock" else ""
                    ),
                    Button("🔄 Restart", cls="btn btn-sm",
                           hx_post=f"/notebook/{nb.id}/kernel/restart", hx_target="#status", title="Restart kernel"),
                    Button("⏹ Cancel All", cls="btn btn-sm btn-cancel-all", id="cancel-all-btn",
                           onclick=f"cancelAllExecution()", title="Cancel running cell and clear queue (Esc Esc)",
                           style="display: none;"),
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
            Script(f"window.NOTEBOOK_ID = '{nb.id}';"),  # Expose notebook ID for dialoghelper
            Script(f"document.addEventListener('DOMContentLoaded', () => connectWebSocket('{nb.id}'));"),
            Div(id="js-script"),  # Container for dialoghelper script injection via HTMX OOB
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
def post(dlg_name: str, msgid: str):
    """Get message index by ID - uses shared get_msg_idx()."""
    nb = get_notebook(dlg_name)
    # dialoghelper expects {"msgid": idx} not {"idx": idx}
    return {"msgid": get_msg_idx(nb, msgid)}

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
def post(dlg_name: str, n: int = 0, relative: bool = True, msgid: str = "",
         view_range: str = "", nums: bool = False, current_idx: int = 0):
    """Read message content - uses shared read_msg()."""
    nb = get_notebook(dlg_name)
    return read_msg(nb, n=n, relative=relative, msgid=msgid,
                    current_idx=current_idx, view_range=view_range, nums=nums)

@rt("/add_relative_")
async def post(dlg_name: str, content: str, placement: str = "after", msgid: str = "",
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
    if msgid:
        ref_idx = get_msg_idx(nb, msgid)
        if ref_idx == -1:
            return {"error": f"Message {msgid} not found"}
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
async def post(dlg_name: str, msgid: str,
               content: str = None, msg_type: str = None, output: str = None,
               time_run: str = None, is_exported: str = None, skipped: str = None,
               i_collapsed: str = None, o_collapsed: str = None,
               heading_collapsed: str = None, pinned: str = None):
    """Update message properties.

    FastHTML requires explicit params (no **kwargs).
    Boolean params use str type because HTTP form data sends 'True'/'False' strings,
    and FastHTML can't convert 'True' to int.
    """
    print(f"[UPDATE_MSG] dlg_name={dlg_name}, msgid={msgid}, pinned={pinned}", flush=True)
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msgid)
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
    return msgid

@rt("/add_runq_")
async def post(dlg_name: str, msgid: str, api: str = ""):
    """Add message to execution queue."""
    nb = get_notebook(dlg_name)
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
def post(dlg_name: str, msgid: str, insert_line: int, new_str: str):
    """Insert line at position in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msgid)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines.insert(insert_line, new_str)
        cell.source = '\n'.join(lines)
    return {"status": "ok"}

@rt("/msg_str_replace_")
def post(dlg_name: str, msgid: str, old_str: str, new_str: str):
    """Replace string in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msgid)
    if idx >= 0:
        nb.cells[idx].source = nb.cells[idx].source.replace(old_str, new_str, 1)
    return {"status": "ok"}

@rt("/msg_strs_replace_")
def post(dlg_name: str, msgid: str, old_strs: str, new_strs: str):
    """Replace multiple strings (JSON arrays)."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msgid)
    if idx >= 0:
        cell = nb.cells[idx]
        old_list = json.loads(old_strs)
        new_list = json.loads(new_strs)
        for old, new in zip(old_list, new_list):
            cell.source = cell.source.replace(old, new, 1)
    return {"status": "ok"}

@rt("/msg_replace_lines_")
def post(dlg_name: str, msgid: str, start_line: int, end_line: int, new_content: str):
    """Replace line range in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msgid)
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
