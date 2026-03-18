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
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from enum import Enum
from pathlib import Path

# New streaming kernel
from dialeng.services.kernel import KernelService
from dialeng.services.kernel.execution_queue import ExecutionQueue
from dialeng.document.cell import Cell, CellType, CellState, CellOutput, CollapseLevel

# DialogHelper compatibility and LLM services
from dialeng.services import (
    get_msg_idx, find_msgs, read_msg, cell_to_dict,
    build_context_messages, llm_service
)
from dialeng.services.dialoghelper_service import (
    format_msgs_as_xml, format_msgs_as_json,
    clipboard_copy, clipboard_paste,
    log_change, get_change_log,
)
from dialeng.services.credential_service import (
    detect_credentials, get_available_modes, print_credential_status, CredentialStatus
)
from dialeng.services.dialeng_config import (
    load_config, get_config, print_config_status, update_config, get_config_dict
)
from dialeng.services.shell_service import (
    SHFMT_AVAILABLE, warn_missing_shfmt, print_shfmt_status
)

# UI Components (extracted to ui/ package)
from dialeng.ui import (
    CellView, NotebookPage, AllCells, AllCellsContent,
    AllCellsOOB, CellViewOOB, CellOutputOOB, CellHeaderOOB, AddButtons,
    TypeSelect, CollapseBtn, get_collapse_class, get_cell_state_classes
)
from dialeng.ui.mime import render_mime_bundle

# Extension system
from dialeng.core.extensions import load_extensions
from dialeng.core.registry import registry
from dialeng.services.autorun_service import process_autorun

# ============================================================================
# Constants
# ============================================================================

SOLVEIT_VER = 2

# Load configuration — use in-memory defaults at import time.
# The real config file (inside the notebooks dir) is loaded when set_root_dir() is called.
DIALENG_CONFIG = load_config(create_if_missing=False)

# Detect credentials at startup
CREDENTIAL_STATUS = detect_credentials()
AVAILABLE_DIALOG_MODES = get_available_modes(CREDENTIAL_STATUS)

# Models from config - default model depends on detected provider
AVAILABLE_MODELS = DIALENG_CONFIG.get_model_choices()
AVAILABLE_MODEL_IDS = [model_id for model_id, _ in AVAILABLE_MODELS]
DEFAULT_MODEL = DIALENG_CONFIG.get_default_model(CREDENTIAL_STATUS.backend)


def validate_model_id(model_id: str) -> str:
    """Validate a model ID and return a valid one.

    Model selection follows this priority:
    1. If the given model_id exists in available models, use it
    2. Otherwise, fall back to the provider-specific default

    This ensures per-notebook model selection is remembered (when valid)
    while gracefully handling cases where saved model IDs become invalid
    (e.g., config changed, model removed, notebook from different setup).

    Args:
        model_id: The model ID to validate (e.g., from notebook metadata)

    Returns:
        A valid model ID - either the original if valid, or the default
    """
    if model_id in AVAILABLE_MODEL_IDS:
        return model_id
    # Model ID not found - use provider default
    return DEFAULT_MODEL

# Load extensions (cell types, callbacks, services)
# Extensions are Python files in the extensions/ directory
_loaded_extensions = load_extensions(silent=True)
if _loaded_extensions:
    print(f"Loaded {len(_loaded_extensions)} extension(s): {', '.join(_loaded_extensions)}")

from dialeng.document.prompt_utils import (
    SEPARATOR_PREFIX, SEPARATOR_SUFFIX, SEPARATOR_PATTERN,
    make_separator, split_prompt_content, join_prompt_content
)

# ============================================================================
# Data Models
# ============================================================================

# Cell, CellType, CellState, CellOutput, CollapseLevel imported from document.cell
from dialeng.document.notebook import Notebook

# Default dialog mode based on credentials and config
DEFAULT_DIALOG_MODE = DIALENG_CONFIG.default_mode if CREDENTIAL_STATUS.available else "mock"

# ============================================================================
# Python Kernel (Streaming Subprocess)
# ============================================================================

# KernelService manages subprocess kernels per notebook with:
# - Real-time streaming output (stdout/stderr as they happen)
# - Hard interrupt via SIGINT (can stop tight loops, C extensions)
# - Rich output support (images, plots, HTML)
# - Persistent namespace across cells

kernel_service = KernelService()

# Colab session manager — initialized lazily by _init_colab() after the real config is loaded
colab_session_manager = None
colab_auth_service = None


def _init_colab():
    """Initialize Colab services if colab is enabled in the (real) config.

    Called from set_root_dir() after the config file is loaded, NOT at module
    import time — because the module-level load_config() uses in-memory defaults
    where colab.enabled is always False.
    """
    global colab_auth_service, colab_session_manager
    if not DIALENG_CONFIG.colab_enabled:
        return
    if colab_auth_service is not None:
        return  # already initialized

    import asyncio
    import concurrent.futures
    from dialeng.services.colab import ColabAuthService, ColabSessionManager
    from dialeng.services.colab.colab_auth import resolve_oauth_credentials, print_colab_credential_status

    # Resolve and validate OAuth credentials (validates defaults, auto-extracts from VSIX if needed)
    # Use a thread to avoid "cannot be called from a running event loop" in uvicorn workers
    def _resolve_creds():
        return asyncio.run(resolve_oauth_credentials())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
        _colab_creds = _pool.submit(_resolve_creds).result()

    colab_auth_service = ColabAuthService(credentials=_colab_creds)
    colab_session_manager = ColabSessionManager(colab_auth_service)
    kernel_service.set_colab_session_manager(colab_session_manager)
    print_colab_credential_status(_colab_creds)
    print(f"   Colab: enabled (authenticated={colab_auth_service.is_authenticated})")

# ExecutionQueue instances per notebook (created lazily)
execution_queues: Dict[str, ExecutionQueue] = {}

def get_execution_queue(nb_id: str) -> ExecutionQueue:
    """Get or create execution queue for a notebook."""
    if nb_id not in execution_queues:
        # Get callback handler from registry (includes registered 2-way callbacks)
        callback_handler = registry.get_callback_handler()
        queue = ExecutionQueue(kernel_service, callback_handler=callback_handler)
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

            # After first successful execution, broadcast kernel-connected so status bar updates
            if state == CellState.SUCCESS and nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "kernel_connected"})
                for send in list(ws_connections[nb_id]):
                    try:
                        await send(msg)
                    except:
                        pass
    return callback


async def finalize_cell_execution(nb_id: str, cell, has_error: bool):
    """Broadcast final cell state via OOB swaps after execution completes.

    Note: We preserve cell.outputs as-is from kernel streaming rather than
    flattening to a string via cell.output setter, which would destroy
    structured outputs (display_data, MIME bundles, etc.).
    """
    cell.time_run = datetime.now().strftime("%H:%M:%S")

    # Send code_stream_end signal
    if nb_id in ws_connections and ws_connections[nb_id]:
        msg = json.dumps({"type": "code_stream_end", "cell_id": cell.id, "has_error": has_error})
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except:
                pass

    # Broadcast output + header via targeted OOB swaps (preserves Monaco editor DOM)
    await broadcast_to_notebook(nb_id, CellOutputOOB(cell))
    await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))

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
NOTEBOOKS_DIR = Path(os.environ.get("DIALENG_NOTEBOOKS_DIR", "notebooks"))
NOTEBOOKS_DIR.mkdir(exist_ok=True)

def set_root_dir(root: Path):
    """Set the notebooks root directory. Called by CLI before main().

    Also reloads dialeng_config.json from the new root so the config
    lives alongside the notebooks (not in whatever CWD the command ran from).
    """
    global NOTEBOOKS_DIR, DIALENG_CONFIG
    NOTEBOOKS_DIR = root
    NOTEBOOKS_DIR.mkdir(exist_ok=True)

    # Reload config from the project directory
    DIALENG_CONFIG = load_config(config_path=NOTEBOOKS_DIR / "dialeng_config.json", force_reload=True)

    # Initialize Colab now that we have the real config
    _init_colab()

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

def _load_notebook(path: str) -> Notebook:
    """Load a notebook with app-level credential/model validation."""
    nb = Notebook.load(str(path),
                       default_dialog_mode=DEFAULT_DIALOG_MODE,
                       model_validator=validate_model_id)
    # Override mode to mock if no credentials available
    if not CREDENTIAL_STATUS.available:
        nb.dialog_mode = "mock"
    # Ensure model is always set (old notebooks may not have it saved)
    if not nb.model:
        nb.model = DEFAULT_MODEL
    return nb


def _nb_id_encode_part(part: str) -> str:
    """Escape tildes in a single path component: ~ → ~~"""
    return part.replace("~", "~~")

def _nb_id_from_path(path: Path) -> str:
    """Derive a collision-proof, URL-safe notebook ID from a file path.

    Uses ~ as path separator with ~~ escaping for literal tildes in names.
    The encoding is fully reversible via _nb_id_to_relpath().

    Examples: notebooks/test.ipynb → 'test'
              notebooks/subfolder/test.ipynb → 'subfolder~test'
              notebooks/a/b/test.ipynb → 'a~b~test'
              notebooks/my_project/analysis.ipynb → 'my_project~analysis'
              notebooks/has~tilde.ipynb → 'has~~tilde'
    """
    try:
        rel = path.resolve().relative_to(NOTEBOOKS_DIR.resolve())
    except ValueError:
        return _nb_id_encode_part(path.stem)
    parts = list(rel.parts)
    parts[-1] = rel.stem  # Remove .ipynb extension
    encoded = [_nb_id_encode_part(p) for p in parts]
    return "~".join(encoded) if len(encoded) > 1 else encoded[0]

def _nb_id_to_relpath(notebook_id: str) -> Path:
    """Reverse a notebook ID back to a relative path (without .ipynb extension).

    Decodes ~ separator and ~~ escape to reconstruct the original path.
    This is the inverse of _nb_id_from_path().

    Examples: 'test' → Path('test')
              'subfolder~test' → Path('subfolder/test')
              'a~b~test' → Path('a/b/test')
              'has~~tilde' → Path('has~tilde')
              'a~~b~c' → Path('a~b/c')
    """
    # Replace ~~ with a placeholder, split on ~, restore placeholder to ~
    placeholder = "\x00"
    safe = notebook_id.replace("~~", placeholder)
    parts = safe.split("~")
    parts = [p.replace(placeholder, "~") for p in parts]
    return Path(*parts) if len(parts) > 1 else Path(parts[0])

def _find_notebook_path(notebook_id: str) -> Optional[Path]:
    """Find a notebook file by ID using direct path reconstruction."""
    # Decode the ID to a relative path and look up directly
    rel = _nb_id_to_relpath(notebook_id)
    direct_path = NOTEBOOKS_DIR / f"{rel}.ipynb"
    if direct_path.exists():
        return direct_path
    # Check if an in-memory notebook has a path set
    if notebook_id in notebooks and notebooks[notebook_id].path:
        p = Path(notebooks[notebook_id].path)
        if p.exists():
            return p
    return None

def _find_notebook_by_name(name: str) -> Optional[str]:
    """Find a notebook by its relative path name (e.g., 'subfolder/test').

    Returns the notebook ID if found (loading it if necessary), or None.
    """
    target = NOTEBOOKS_DIR / f"{name}.ipynb"
    if not target.exists() or not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        return None
    nb_id = _nb_id_from_path(target)
    # Load into memory if not already
    if nb_id not in notebooks:
        nb = _load_notebook(target)
        nb.id = nb_id  # Ensure ID matches the URL-safe key, not just the filename stem
        notebooks[nb_id] = nb
    return nb_id

def get_notebook(notebook_id: str) -> Notebook:
    """Get or create a notebook - ALWAYS requires notebook_id"""
    if notebook_id not in notebooks:
        path = _find_notebook_path(notebook_id)
        if path:
            nb = _load_notebook(path)
            nb.id = notebook_id  # Ensure ID matches the URL-safe key
            notebooks[notebook_id] = nb
        else:
            nb = Notebook(id=notebook_id, title=notebook_id,
                         dialog_mode=DEFAULT_DIALOG_MODE, model=DEFAULT_MODEL)
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
        nb = notebooks[notebook_id]
        # Use existing path if set (preserves subdirectory location), else reconstruct from ID
        path = nb.path if nb.path else NOTEBOOKS_DIR / f"{_nb_id_to_relpath(notebook_id)}.ipynb"
        nb.save(str(path))
        # Auto-extract #| export cells to lib dir if notebook has #| default_exp
        try:
            from dialeng.services.lib_export_service import maybe_extract
            result = maybe_extract(Path(path), root_dir=NOTEBOOKS_DIR)
            if result:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"lib export: {result['module']} ({result['cells_exported']} cells)")
                # Re-upload to Colab if this notebook has an active Colab kernel
                import asyncio
                asyncio.create_task(_upload_lib_to_colab(notebook_id))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"lib export failed for {notebook_id}: {e}")

def list_notebooks() -> List[str]:
    return [p.stem for p in NOTEBOOKS_DIR.glob("*.ipynb")]


def _deduplicate_response_text(text: str) -> str:
    """
    Detect and remove duplicated content in response text.

    Sometimes the LLM produces duplicated output, especially when:
    - Tool results are processed without proper context
    - Multiple tool loop iterations cause confusion

    This function detects patterns like:
    - "ResponseABC...ResponseABC" (exact duplication)
    - "ResponseABC...fragmentABC" (partial duplication with fragment)

    Args:
        text: The response text that may contain duplication

    Returns:
        Cleaned text with duplications removed

    ## Bug Fix History (2026-01-25)

    ### Issue: False Positive Truncation

    **Symptom:** Legitimate responses were being truncated. For example, a 150-character
    response "Based on the calculations:\n\n**Statistics for [10, 20, 30, 40]:**..."
    was cut off after only 50 characters.

    **Root Cause:** The partial overlap detection loop (lines ~659-663) was checking if
    ANY suffix of `first_end` appeared in `second_sample`, including single characters:

        # OLD CODE (buggy):
        for i in range(min(50, len(first_end))):
            if first_end[i:] in second_sample:  # At i=49, first_end[49:] = ","
                return text[:split_point].strip()  # FALSE POSITIVE!

    When i reached high values (e.g., 49), `first_end[49:]` would be just a single
    character like `","`. Common punctuation trivially appears in most text, causing
    false positive "duplication" detection.

    **Fix:** Added minimum overlap length requirement (20 characters). Short matches
    like single punctuation marks are now ignored:

        # NEW CODE (fixed):
        min_overlap_len = 20
        for i in range(min(50, len(first_end) - min_overlap_len)):
            overlap = first_end[i:]
            if len(overlap) >= min_overlap_len and overlap in second_sample:
                return text[:split_point].strip()

    ### Future Improvement Suggestions

    1. **Smarter duplication detection:** Instead of substring matching, consider
       using sequence alignment algorithms (like difflib.SequenceMatcher) with a
       similarity threshold (e.g., 80% match = duplication).

    2. **LLM-specific patterns:** Track known LLM duplication patterns (e.g., when
       tool results cause the model to repeat its analysis verbatim).

    3. **Confidence scoring:** Return both the cleaned text and a confidence score
       indicating how certain we are that duplication was detected.

    4. **Configurable threshold:** Make min_overlap_len configurable via dialeng_config.json
       so users can tune sensitivity based on their use case.

    5. **Unit tests:** Add comprehensive tests with edge cases to prevent regressions.
    """
    if not text or len(text) < 100:
        return text

    # Try to find duplicated content by looking for repeated substrings
    # Start from the middle and work backwards to find the longest match
    text_len = len(text)

    # Check if the second half is largely a repeat of the first half
    for split_point in range(text_len // 3, 2 * text_len // 3):
        first_part = text[:split_point].strip()
        second_part = text[split_point:].strip()

        if not first_part or not second_part:
            continue

        # Check if second_part starts similarly to first_part (within first 100 chars)
        first_start = first_part[:100] if len(first_part) > 100 else first_part
        if first_start in second_part[:200]:
            # Found duplication - return just the first part
            return first_part

        # Check if second_part ends similarly to first_part's ending
        # IMPORTANT: Require minimum 20 characters for overlap to avoid false positives
        # from short common strings like ",", ":", "-", etc.
        first_end = first_part[-100:] if len(first_part) > 100 else first_part
        if len(second_part) > 50:
            second_sample = second_part[:150]
            # Look for a significant overlap (must be at least 20 chars to be meaningful)
            min_overlap_len = 20
            for i in range(min(50, len(first_end) - min_overlap_len)):
                overlap = first_end[i:]
                if len(overlap) >= min_overlap_len and overlap in second_sample:
                    # Found partial duplication - return text up to where duplication starts
                    return text[:split_point].strip()

    return text


def _format_tool_steps_markdown(tool_events: dict) -> str:
    """
    Format tool events into collapsible HTML/markdown for persisting in output.

    Uses HTML <details>/<summary> tags for native collapsibility in markdown renderers.
    Shows chronological trace of: variable substitutions, tool calls, AI reasoning.
    Tool inputs and outputs use nested <details> for individual expandability.

    Args:
        tool_events: Dict with "var_substitutions", "tool_calls", and "steps" lists

    Returns:
        Markdown string with collapsible tool steps, or empty string if no events
    """
    import html as html_module

    steps = tool_events.get("steps", [])
    var_subs = tool_events.get("var_substitutions", [])
    tool_calls = tool_events.get("tool_calls", [])

    # No events to show
    if not steps and not var_subs and not tool_calls:
        return ""

    # Count total steps for summary
    total_steps = len(steps) if steps else (len(var_subs) + len(tool_calls))
    summary_text = f"LLM Steps ({total_steps})"

    parts = []
    parts.append('<details class="tool-steps-container">')
    parts.append(f'<summary class="tool-steps-summary">🔧 {summary_text}</summary>')
    parts.append('<div class="tool-steps-content">')

    def format_tool_step(name: str, status: str, tool_input: dict, result: dict) -> str:
        """Format a single tool call step with collapsible input/output."""
        status_icon = "✅" if status == "success" else "❌"
        input_json = json.dumps(tool_input, indent=2)

        # Format result content
        if isinstance(result, dict):
            result_content = result.get("result", {})
            if isinstance(result_content, dict):
                result_text = result_content.get("content", str(result_content))
            else:
                result_text = str(result_content)
        else:
            result_text = str(result)

        # Truncate long results
        if len(result_text) > 500:
            result_text = result_text[:500] + "..."

        escaped_name = html_module.escape(name)
        escaped_input = html_module.escape(input_json)
        escaped_result = html_module.escape(result_text)

        return f'''<div class="step step-tool">
<div class="step-header"><span class="step-icon">{status_icon}</span><strong>{escaped_name}</strong></div>
<details class="step-input-details">
<summary class="step-toggle">📥 Input</summary>
<pre class="step-pre">{escaped_input}</pre>
</details>
<details class="step-output-details">
<summary class="step-toggle">📤 Output</summary>
<pre class="step-pre">{escaped_result}</pre>
</details>
</div>'''

    # Use chronological steps if available, otherwise fall back to grouped view
    if steps:
        for i, step in enumerate(steps):
            step_type = step.get("type", "")

            if step_type == "var":
                name = html_module.escape(step.get("name", ""))
                value = html_module.escape(str(step.get("value", ""))[:100])
                if len(str(step.get("value", ""))) > 100:
                    value += "..."
                parts.append(f'<div class="step step-var"><span class="step-icon">📝</span><code>${name}</code> → <code>{value}</code></div>')

            elif step_type == "tool":
                name = step.get("name", "")
                status = step.get("status", "success")
                tool_input = step.get("input", {})
                result = step.get("result", {})
                parts.append(format_tool_step(name, status, tool_input, result))

            elif step_type == "reasoning":
                content = html_module.escape(step.get("content", ""))
                if content:
                    # Truncate long reasoning (configurable, default 500 chars, 0 = no limit)
                    truncate_limit = get_config().reasoning_truncate_chars
                    if truncate_limit > 0 and len(content) > truncate_limit:
                        content = content[:truncate_limit] + "..."
                    parts.append(f'<div class="step step-reasoning"><span class="step-icon">💭</span><span class="step-text">{content}</span></div>')

    else:
        # Fallback: grouped view (variables first, then tool calls)
        if var_subs:
            for sub in var_subs:
                name = html_module.escape(sub.get("name", ""))
                value = html_module.escape(str(sub.get("value", ""))[:100])
                if len(str(sub.get("value", ""))) > 100:
                    value += "..."
                parts.append(f'<div class="step step-var"><span class="step-icon">📝</span><code>${name}</code> → <code>{value}</code></div>')

        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "")
                status = tc.get("status", "success")
                tool_input = tc.get("input", {})
                result = tc.get("result", {})
                parts.append(format_tool_step(name, status, tool_input, result))

    parts.append('</div>')
    parts.append('</details>')
    parts.append('')  # Single empty line before response

    return '\n'.join(parts)


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
        return

    connections = ws_connections[nb_id]

    # Convert component to HTML string using to_xml
    # FastHTML's str() on components returns the ID, not HTML
    html_str = to_xml(component)

    # Track which connections are still alive
    alive = []

    for send in connections:
        if send is exclude_send:
            alive.append(send)  # Keep but don't send
            continue
        try:
            await send(html_str)
            alive.append(send)
        except Exception:
            # Connection closed/dead - silently remove it
            # This is expected when browser tabs close or refresh
            pass

    # Replace with only alive connections
    ws_connections[nb_id] = alive


async def broadcast_json(nb_id: str, data: dict, exclude_send=None):
    """Broadcast a JSON message to all WebSocket connections for a notebook.

    Used for lightweight updates (source changes, class updates) that don't
    need full HTML OOB swaps. The client-side WebSocket handler processes
    these by type.
    """
    if nb_id not in ws_connections or not ws_connections[nb_id]:
        return
    msg = json.dumps(data)
    alive = []
    for send in ws_connections[nb_id]:
        if send is exclude_send:
            alive.append(send)
            continue
        try:
            await send(msg)
            alive.append(send)
        except Exception:
            pass
    ws_connections[nb_id] = alive


async def broadcast_all_json(data: dict):
    """Broadcast a JSON message to ALL WebSocket connections across all notebooks."""
    for nb_id in list(ws_connections.keys()):
        await broadcast_json(nb_id, data)


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


async def broadcast_kernel_status(nb_id: str, status: str):
    """Broadcast kernel status change (connected, busy, error, restarting) to all clients."""
    msg = json.dumps({"type": f"kernel_{status}"})
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
        msg_data = {
            "type": "code_display_data",
            "cell_id": cell_id,
            "html": html_content
        }
        if output.display_id:
            msg_data["display_id"] = output.display_id
        msg = json.dumps(msg_data)
    elif output.output_type == 'update_display_data':
        html_content = render_mime_bundle(output.content, output.metadata)
        msg = json.dumps({
            "type": "code_update_display",
            "cell_id": cell_id,
            "html": html_content,
            "display_id": output.display_id
        })
    elif output.output_type == 'clear_output':
        msg = json.dumps({
            "type": "code_clear_output",
            "cell_id": cell_id,
            "wait": output.content  # bool: wait for next output before clearing
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
#   - static/css/editor.css    (Monaco editor styles)
#   - static/js/app.js         (all client-side logic)
#
# See docs/how_it_works/07_code_organization.md for details.


# ============================================================================
# FastHTML App with WebSocket
# ============================================================================

app, rt = fast_app(
    pico=False,
    exts='ws',
    static_path=str(Path(__file__).parent),
    hdrs=(
        # CSS - External stylesheets (order matters: themes first for variables)
        Link(rel="stylesheet", href="/static/css/themes.css"),
        Link(rel="stylesheet", href="/static/css/base.css"),
        Link(rel="stylesheet", href="/static/css/components.css"),
        Link(rel="stylesheet", href="/static/css/editor.css"),
        # Monaco Editor (AMD loader - Monaco modules loaded in app.js via require())
        Script(src="https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/loader.min.js"),
        # Highlight.js for markdown code blocks
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"),
        # App JS - External JavaScript (after libraries)
        Script(src="/static/js/app.js"),
    )
)

# AUTORUN processing on startup (async, needs event loop)
@app.on_event("startup")
async def _autorun_startup():
    # Reload config from the correct path — the Uvicorn reloader spawns a worker
    # process that re-imports this module, but set_root_dir() only runs in main().
    # Without this, DIALENG_CONFIG has DEFAULT_CONFIG (no extension settings).
    global DIALENG_CONFIG
    DIALENG_CONFIG = load_config(config_path=NOTEBOOKS_DIR / "dialeng_config.json", force_reload=True)
    # Initialize Colab in the worker process (set_root_dir() only runs in the main process)
    _init_colab()
    await process_autorun(kernel_service, notebooks_dir=NOTEBOOKS_DIR)

# ============================================================================
# Extension Development Endpoints
# ============================================================================

@rt("/dialeng/reload-extensions")
async def post():
    """Re-extract and reload all AUTORUN extensions, then refresh all clients.

    Used during extension development to pick up changes to #| export cells
    without restarting the server. Call from a notebook:
        from dialeng.dev import reload_extensions
        reload_extensions()
    """
    from dialeng.services.autorun_service import reload_autorun_extensions
    result = reload_autorun_extensions()
    await broadcast_all_json({"type": "extensions_reloaded"})
    return result

@rt("/dialeng/{nb_id}/ext/{action_name}")
async def post(nb_id: str, action_name: str, request):
    """Execute a registered extension action.

    Extensions register actions via @register_action("name") in #| export cells.
    Toolbar buttons or notebook code can POST here to trigger server-side logic.
    """
    import asyncio
    handler = registry.actions.get(action_name)
    if not handler:
        return {"error": f"Unknown action: {action_name}"}, 404
    try:
        # Extract form data from request (FastHTML ignores **kwargs)
        form = await request.form()
        kwargs = dict(form)
        if asyncio.iscoroutinefunction(handler):
            result = await handler(nb_id=nb_id, **kwargs)
        else:
            result = handler(nb_id=nb_id, **kwargs)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        print(f"[EXT] Action '{action_name}' failed: {e}", flush=True)
        return {"error": str(e)}, 500

# Static file serving
@rt("/static/{path:path}")
async def get(path: str):
    """Serve static files from the static/ directory with no-cache headers for dev."""
    from starlette.responses import FileResponse
    file_path = Path(__file__).parent / "static" / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, headers={"Cache-Control": "no-cache"})
    return "Not found", 404


# ============================================================================
# Routes
# ============================================================================

@rt("/")
def get():
    return RedirectResponse("/dialeng/default", status_code=302)

@rt("/dialeng/")
async def get(name: str = ""):
    """Handle /dialeng/ and /notebook?name=path/to/notebook.

    With name param: resolve the path to a notebook ID and render directly
    (keeps the clean URL in the browser). Supports solveit-style URLs.
    Without: redirect to default notebook.
    """
    if name:
        nb_id = _find_notebook_by_name(name)
        if not nb_id:
            # Notebook not found at that path — create it
            dir_part = str(Path(name).parent) if '/' in name else ""
            name_part = Path(name).name
            return RedirectResponse(f"/dialeng/new?dir={dir_part}&name={name_part}", status_code=302)
        # Render the notebook page directly (same as /dialeng/{nb_id})
        return await _render_notebook_page(nb_id)
    return RedirectResponse("/dialeng/default", status_code=302)

@rt("/dialeng/new")
def get(dir: str = "", name: str = ""):
    display_name = name.strip() if name.strip() else uuid.uuid4().hex[:8]
    target_dir = NOTEBOOKS_DIR / dir if dir else NOTEBOOKS_DIR
    file_path = target_dir / f"{display_name}.ipynb"

    # Derive URL-safe ID from path: "subfolder/test.ipynb" → "subfolder_test"
    new_id = _nb_id_from_path(file_path)

    # If this notebook already exists in memory (e.g., re-creation after delete),
    # remove the stale version so we get fresh template cells
    notebooks.pop(new_id, None)

    nb = Notebook(id=new_id, title=display_name,
                  dialog_mode=DEFAULT_DIALOG_MODE, model=DEFAULT_MODEL)

    # Load cells from TEMPLATE.ipynb (parent-first hierarchy)
    from dialeng.services.template_service import find_templates, load_template_cells
    template_paths = find_templates(target_dir, NOTEBOOKS_DIR)
    if template_paths:
        template_cells = load_template_cells(template_paths)
        if template_cells:
            nb.cells = template_cells
            print(f"[TEMPLATE] Applied {len(template_cells)} cells from {len(template_paths)} template(s) to {new_id}", flush=True)
            for tp in template_paths:
                print(f"[TEMPLATE]   - {tp}", flush=True)
        else:
            nb.cells = [Cell(cell_type="note", source="# New Notebook\n\nStart writing here...")]
    else:
        nb.cells = [Cell(cell_type="note", source="# New Notebook\n\nStart writing here...")]

    # Set path so save_notebook writes to the correct subdirectory
    nb.path = file_path
    notebooks[new_id] = nb
    save_notebook(new_id)
    # Redirect to clean name-based URL
    nb_name = f"{dir}/{display_name}" if dir else display_name
    return RedirectResponse(f"/dialeng/?name={nb_name}", status_code=302)

# ============================================================================
# File Explorer Routes
# ============================================================================

@rt("/files")
def get(path: str = ""):
    """Get file list content for HTMX swap."""
    from dialeng.ui.file_explorer import FileListContent
    target = NOTEBOOKS_DIR / path if path else NOTEBOOKS_DIR
    # Validate path is within NOTEBOOKS_DIR
    if not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        target = NOTEBOOKS_DIR
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return FileListContent(target, NOTEBOOKS_DIR, "", kernel_notebooks=kernel_nbs)

@rt("/files/new-folder")
def post(path: str = "", name: str = ""):
    """Create a new folder and return updated file list."""
    from dialeng.ui.file_explorer import FileListContent
    if not name or '/' in name or name.startswith('.'):
        return Div("Invalid folder name", cls="status error")
    target = NOTEBOOKS_DIR / path if path else NOTEBOOKS_DIR
    if not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        target = NOTEBOOKS_DIR
    new_folder = target / name
    new_folder.mkdir(parents=True, exist_ok=True)
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return FileListContent(target, NOTEBOOKS_DIR, "", kernel_notebooks=kernel_nbs)

@rt("/files/delete")
def post(path: str = ""):
    """Delete a notebook file and return updated file list."""
    from dialeng.ui.file_explorer import FileListContent
    if not path or '..' in path:
        return Div("Invalid path", cls="status error")
    target = NOTEBOOKS_DIR / f"{path}.ipynb"
    if not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        return Div("Invalid path", cls="status error")
    if target.exists() and target.is_file():
        target.unlink()
        # Remove from in-memory notebooks if loaded
        nb_id = _nb_id_from_path(target)
        if nb_id in notebooks:
            del notebooks[nb_id]
    parent = target.parent
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return FileListContent(parent, NOTEBOOKS_DIR, "", kernel_notebooks=kernel_nbs)

# ============================================================================
# Notebook Routes
# ============================================================================

async def _render_notebook_page(nb_id: str):
    """Render a notebook page by ID (shared by /dialeng/{nb_id} and /dialeng/?name=...)."""
    nb = get_notebook(nb_id)

    # Determine notebook path for CRAFT discovery: use nb.path, or find on disk
    nb_path = nb.path
    if not nb_path:
        found = _find_notebook_path(nb_id)
        nb_path = found if found else NOTEBOOKS_DIR / f"{nb_id}.ipynb"
        if found:
            nb.path = found  # Cache for future use

    # Check for unexecuted CRAFT code cells (but don't execute yet — wait for kernel selection)
    has_craft_code = False
    if nb_path and Path(nb_path).exists():
        from dialeng.services.craft_service import find_craft_files, get_craft_code_cells, _executed_craft
        craft_paths = find_craft_files(nb_path, NOTEBOOKS_DIR)
        nb_path_resolved = Path(nb_path).resolve()
        craft_paths = [cp for cp in craft_paths if Path(cp).resolve() != nb_path_resolved]
        if craft_paths:
            craft_cells = get_craft_code_cells(craft_paths)
            executed = _executed_craft.get(nb_id, set())
            unexecuted = [(cid, src) for cid, src in craft_cells if cid not in executed]
            has_craft_code = bool(unexecuted)

    nb_list = list_notebooks() or [nb_id]
    config = get_config()
    nb_kernel_alive = kernel_service.kernel_is_alive(nb_id)
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return NotebookPage(nb, nb_list, AVAILABLE_DIALOG_MODES, AVAILABLE_MODELS, config,
                        shfmt_available=SHFMT_AVAILABLE,
                        colab_enabled=colab_auth_service is not None,
                        colab_authenticated=colab_auth_service.is_authenticated if colab_auth_service else False,
                        notebooks_dir=NOTEBOOKS_DIR,
                        kernel_alive=nb_kernel_alive,
                        kernel_notebooks=kernel_nbs,
                        has_craft_code=has_craft_code)

@rt("/dialeng/{nb_id}")
async def get(nb_id: str):
    if not nb_id or not nb_id.strip():
        return RedirectResponse("/dialeng/default", status_code=302)
    return await _render_notebook_page(nb_id)

@rt("/dialeng/{nb_id}/save")
def post(nb_id: str):
    save_notebook(nb_id)
    return Div("✓ Saved", cls="status success")

@rt("/dialeng/{nb_id}/mode")
def post(nb_id: str, mode: str):
    nb = get_notebook(nb_id)
    nb.dialog_mode = mode
    return ""

@rt("/dialeng/{nb_id}/model")
def post(nb_id: str, model: str):
    nb = get_notebook(nb_id)
    nb.model = model
    return ""

@rt("/dialeng/{nb_id}/safe_mode")
def post(nb_id: str, safe_mode: str = "false"):
    """Toggle safe mode for shell commands in this notebook."""
    nb = get_notebook(nb_id)
    # Convert string to boolean (checkbox sends "true" or "false")
    nb.safe_mode = safe_mode.lower() in ("true", "on", "1", "yes")
    return ""

# ============================================================================
# Kernel Helpers
# ============================================================================

async def _inject_lib_syspath(nb_id: str):
    """If lib dir exists in NOTEBOOKS_DIR, add NOTEBOOKS_DIR to the kernel's sys.path."""
    from dialeng.services.lib_export_service import get_lib_name
    lib_name = get_lib_name(NOTEBOOKS_DIR)
    lib_dir = NOTEBOOKS_DIR / lib_name
    if not lib_dir.exists():
        return
    notebooks_path = str(NOTEBOOKS_DIR.resolve())
    setup_code = f"import sys; sys.path.insert(0, {notebooks_path!r}) if {notebooks_path!r} not in sys.path else None"
    cell = Cell(id="_lib_syspath_setup", cell_type=CellType.CODE, source=setup_code)
    try:
        async for _ in kernel_service.execute_cell(nb_id, cell):
            pass
        print(f"[LIB] Injected sys.path for {lib_name}/ into kernel {nb_id}", flush=True)
    except Exception as e:
        print(f"[LIB] Failed to inject sys.path for {nb_id}: {e}", flush=True)


async def _upload_lib_to_colab(nb_id: str):
    """If kernel is Colab and lib dir exists, upload module files to remote VM."""
    from dialeng.services.lib_export_service import get_lib_name

    # Check if this notebook uses a Colab kernel
    nb = notebooks.get(nb_id)
    if not nb or getattr(nb, 'kernel_type', 'local') != 'colab':
        return

    if not kernel_service.has_kernel(nb_id):
        return

    lib_name = get_lib_name(NOTEBOOKS_DIR)
    lib_dir = NOTEBOOKS_DIR / lib_name
    if not lib_dir.exists():
        return

    # Collect all .py files in the lib directory
    py_files = list(lib_dir.rglob("*.py"))
    if not py_files:
        return

    # Build code that recreates the directory structure and writes files on the VM
    import base64
    code_parts = ["import os, base64"]
    dirs_created = set()
    for py_file in py_files:
        rel_path = py_file.relative_to(NOTEBOOKS_DIR)
        content = py_file.read_text(encoding="utf-8")
        # Ensure parent dirs exist (deduplicate makedirs calls)
        parent = str(rel_path.parent)
        if parent not in dirs_created:
            code_parts.append(f"os.makedirs({parent!r}, exist_ok=True)")
            dirs_created.add(parent)
        # Use base64 encoding to avoid string escaping issues
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        code_parts.append(
            f"open({str(rel_path)!r}, 'w').write(base64.b64decode({b64!r}).decode('utf-8'))"
        )

    setup_code = "\n".join(code_parts)
    cell = Cell(id="_lib_colab_upload", cell_type=CellType.CODE, source=setup_code)
    try:
        async for _ in kernel_service.execute_cell(nb_id, cell):
            pass
        print(f"[LIB] Uploaded {len(py_files)} module files to Colab kernel {nb_id}", flush=True)
    except Exception as e:
        print(f"[LIB] Failed to upload lib files to Colab: {e}", flush=True)


# ============================================================================
# Kernel Management Routes
# ============================================================================

@rt("/dialeng/{nb_id}/kernel/type")
async def post(nb_id: str, kernel_type: str):
    """Change the kernel type for a notebook."""
    nb = get_notebook(nb_id)
    reg = registry.kernels.get(kernel_type)
    if not reg:
        return Div("Invalid kernel type", cls="status error")
    if reg.requires_auth:
        if not colab_auth_service:
            return Div("Colab not configured. Enable it in Settings.", cls="status error")
        if not colab_auth_service.is_authenticated:
            return Div("Not authenticated with Google. Click 'Connect Colab' first.", cls="status error")

    nb.kernel_type = kernel_type
    runtime_type = nb.colab_runtime_type
    await kernel_service.set_kernel_type(nb_id, kernel_type, runtime_type=runtime_type)

    # New kernel needs fresh CRAFT initialization (sys.path, uploads, CRAFT cells)
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)

    # Broadcast kernel type change to all clients
    msg = json.dumps({"type": "kernel_type_changed", "kernel_type": kernel_type})
    if nb_id in ws_connections and ws_connections[nb_id]:
        for send in list(ws_connections[nb_id]):
            try:
                await send(msg)
            except Exception:
                pass

    return Div(f"Kernel: {reg.label}", cls="status success")

@rt("/dialeng/{nb_id}/kernel/runtime")
async def post(nb_id: str, runtime_type: str):
    """Change the Colab runtime type (cpu/gpu/tpu) for a notebook."""
    nb = get_notebook(nb_id)
    if runtime_type not in ("cpu", "gpu", "tpu"):
        return Div("Invalid runtime type", cls="status error")
    if nb.kernel_type != 'colab':
        return Div("Runtime type only applies to Colab kernels", cls="status error")

    nb.colab_runtime_type = runtime_type

    # New kernel needs fresh CRAFT initialization
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)

    # Switch the runtime type - this creates a new kernel object
    if colab_session_manager:
        new_kernel = await colab_session_manager.set_runtime_type(nb_id, runtime_type)
        # Update kernel_service to point to the new kernel (old reference is stale)
        kernel_service._kernels[nb_id] = new_kernel

    labels = {"cpu": "CPU", "gpu": "GPU (T4)", "tpu": "TPU"}
    return Div(f"Runtime: {labels.get(runtime_type, runtime_type)}", cls="status success")

@rt("/dialeng/{nb_id}/kernel/status")
def get(nb_id: str):
    """Get current kernel status for the notebook."""
    if kernel_service.has_kernel(nb_id):
        kernel = kernel_service.get_kernel(nb_id)
        status = kernel.get_status()
        return status.__dict__
    nb = get_notebook(nb_id)
    return {"is_alive": False, "kernel_type": nb.kernel_type}

@rt("/dialeng/{nb_id}/kernel/info")
def get(nb_id: str):
    """Get kernel toolbar button with connection info (for HTMX swap)."""
    from dialeng.ui.kernel_modal import KernelToolbarButton
    nb = get_notebook(nb_id)
    kernel_info = None
    if kernel_service.has_kernel(nb_id):
        kernel = kernel_service.get_kernel(nb_id)
        status = kernel.get_status()
        if status.is_alive:
            kernel_info = {
                'language': 'Python',
                'version': getattr(status, 'python_version', ''),
                'display_name': getattr(status, 'display_name', 'Python'),
            }
    return KernelToolbarButton(nb, kernel_info)

@rt("/dialeng/{nb_id}/kernel/modal")
def get(nb_id: str):
    """Get the kernel selection modal (for HTMX swap)."""
    from dialeng.ui.kernel_modal import KernelModal
    nb = get_notebook(nb_id)
    colab_auth = colab_auth_service.is_authenticated if colab_auth_service else False
    return KernelModal(nb.id, nb.kernel_type, colab_auth, nb.colab_runtime_type)

@rt("/dialeng/{nb_id}/kernel/craft-init")
async def post(nb_id: str):
    """Execute kernel setup (_lib/ sys.path) and CRAFT code cells after kernel selection.

    Called by the client after the user selects a kernel, so setup
    code runs before any manual cell execution.
    """
    nb = get_notebook(nb_id)
    nb_path = nb.path
    if not nb_path:
        found = _find_notebook_path(nb_id)
        nb_path = found if found else NOTEBOOKS_DIR / f"{nb_id}.ipynb"
        if found:
            nb.path = found

    # Collect CRAFT cells to execute
    unexecuted = []
    if nb_path and Path(nb_path).exists():
        from dialeng.services.craft_service import find_craft_files, get_craft_code_cells, _executed_craft
        craft_paths = find_craft_files(nb_path, NOTEBOOKS_DIR)
        nb_path_resolved = Path(nb_path).resolve()
        craft_paths = [cp for cp in craft_paths if Path(cp).resolve() != nb_path_resolved]
        if craft_paths:
            craft_cells = get_craft_code_cells(craft_paths)
            executed = _executed_craft.get(nb_id, set())
            unexecuted = [(cid, src) for cid, src in craft_cells if cid not in executed]
            if unexecuted:
                # Mark all cells as executed synchronously to prevent double execution
                _executed_craft.setdefault(nb_id, set()).update(cid for cid, _ in unexecuted)
                print(f"[CRAFT] Executing {len(unexecuted)} code cells for {nb_id} from {len(craft_paths)} CRAFT file(s)", flush=True)
                for cp in craft_paths:
                    print(f"[CRAFT]   - {cp}", flush=True)

    # Run lib init and CRAFT cells in a single sequential task to avoid
    # racing on Colab's lazy kernel connection (assign_and_connect)
    async def _init_and_craft():
        try:
            await _inject_lib_syspath(nb_id)
            await _upload_lib_to_colab(nb_id)
            for cid, src in unexecuted:
                try:
                    cell = Cell(id=cid, cell_type=CellType.CODE, source=src)
                    async for _ in kernel_service.execute_cell(nb_id, cell):
                        pass
                    print(f"[CRAFT] Executed: {cid}", flush=True)
                except Exception as e:
                    print(f"[CRAFT] Error executing cell {cid}: {e}", flush=True)
            if unexecuted:
                print(f"[CRAFT] All cells executed for {nb_id}", flush=True)
            # Signal frontend that kernel setup is complete → dot turns green
            await broadcast_kernel_status(nb_id, "connected")
        except Exception as e:
            print(f"[CRAFT] Init failed for {nb_id}: {e}", flush=True)
            await broadcast_kernel_status(nb_id, "error")
    asyncio.create_task(_init_and_craft())
    return ""

# ============================================================================
# Google OAuth Routes (for Colab integration)
# ============================================================================

@rt("/auth/google")
def get(request):
    """Initiate Google OAuth2 flow for Colab access."""
    if not colab_auth_service:
        return Div("Colab integration not configured", cls="status error")
    import secrets
    state = secrets.token_urlsafe(32)
    # Derive redirect URI from the actual request so it works on any port
    redirect_uri = f"{request.url.scheme}://{request.url.netloc}/auth/google/callback"
    auth_url = colab_auth_service.get_auth_url(state=state, redirect_uri=redirect_uri)
    return RedirectResponse(auth_url)

@rt("/auth/google/callback")
async def get(request, code: str = "", error: str = "", state: str = ""):
    """Handle Google OAuth2 callback."""
    if error:
        return Titled("Authentication Error", Div(f"Google auth failed: {error}", cls="status error"),
                       A("← Back to Dialeng", href="/"))
    if not code:
        return Titled("Authentication Error", Div("No authorization code received", cls="status error"),
                       A("← Back to Dialeng", href="/"))
    if not colab_auth_service:
        return Titled("Error", Div("Colab integration not configured", cls="status error"))
    try:
        # Redirect URI must match what was used in get_auth_url()
        redirect_uri = f"{request.url.scheme}://{request.url.netloc}/auth/google/callback"
        await colab_auth_service.handle_callback(code, redirect_uri=redirect_uri)
        # Notify the parent window and close the popup
        from starlette.responses import HTMLResponse
        return HTMLResponse("""<!DOCTYPE html><html><body>
<script>
// Signal auth success via both postMessage and localStorage (localStorage
// fires a 'storage' event on other same-origin tabs even when window.opener
// is null due to cross-origin navigation through accounts.google.com)
try { localStorage.setItem('colab-auth-event', Date.now().toString()); } catch(e) {}
if (window.opener) { window.opener.postMessage('colab-authenticated', '*'); }
window.close();
</script>
<p>Authenticated! You can close this window.</p>
</body></html>""")
    except Exception as e:
        return Titled("Authentication Error",
                       Div(f"Failed to exchange token: {e}", cls="status error"),
                       A("← Back to Dialeng", href="/"))

@rt("/auth/google/logout")
async def post():
    """Disconnect from Google / clear Colab tokens."""
    if colab_auth_service:
        colab_auth_service.logout()
    return Div(
        Button("Connect Colab", cls="btn btn-sm btn-colab", id="colab-auth-btn",
               onclick="window.open('/auth/google', '_blank', 'width=500,height=700')",
               title="Sign in with Google for Colab access"),
        id="colab-auth-container",
    )

@rt("/auth/google/status")
def get():
    """Check Colab authentication status (JSON)."""
    if not colab_auth_service:
        return {"authenticated": False, "enabled": False}
    return {"authenticated": colab_auth_service.is_authenticated, "enabled": True}

@rt("/dialeng/{nb_id}/export")
def get(nb_id: str):
    nb = get_notebook(nb_id)
    content = json.dumps(nb.to_ipynb(), indent=2)
    return Response(content=content, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{nb_id}.ipynb"'})

# ============================================================================
# Code Completion Endpoint
# ============================================================================

@rt("/api/complete/{nb_id}")
async def post(nb_id: str, code: str, cursor_pos: int):
    """Code completion endpoint."""
    code_to_cursor = code[:cursor_pos] if cursor_pos <= len(code) else code
    matches = await kernel_service.complete(nb_id, code_to_cursor)
    return {"matches": matches}


# Outline Sidebar Endpoints
# ============================================================================

@rt("/dialeng/{nb_id}/outline")
async def get(nb_id: str):
    """Get notebook outline for the sidebar.

    Returns the OutlineSidebar component with:
    - Headings extracted from note cells
    - Variables from kernel namespace
    - Functions from kernel namespace
    """
    from dialeng.ui.outline import OutlineSidebar, extract_headings_from_markdown

    nb = get_notebook(nb_id)

    # Extract headings from note cells
    headings = []
    for cell in nb.cells:
        if cell.cell_type == "note":
            cell_headings = extract_headings_from_markdown(cell.source)
            for h in cell_headings:
                headings.append({
                    'text': h['text'],
                    'cell_id': cell.id,
                    'level': h['level']
                })

    # Get variables and functions from kernel namespace
    variables = []
    functions = []

    # Only get namespace info if the kernel is alive for this notebook
    if kernel_service.has_kernel(nb_id) and kernel_service.kernel_is_alive(nb_id):
        try:
            ns_info = await kernel_service.get_namespace_info(nb_id)
            if ns_info:
                variables = ns_info.get('variables', [])
                functions = ns_info.get('functions', [])
        except Exception as e:
            print(f"Error getting kernel namespace: {e}")

    return OutlineSidebar(nb_id, headings, variables, functions, is_open=True)

# ============================================================================
# Settings Endpoints
# ============================================================================

@rt("/settings")
def get():
    """Get current settings as JSON for API use."""
    config_dict = get_config_dict()
    return Response(content=json.dumps(config_dict, indent=2),
                    media_type="application/json")

@rt("/settings")
async def post(request):
    """Update settings from the settings form.

    Parses form data and updates the config file.
    Returns a status message for the settings sidebar.
    """
    form_data = await request.form()

    # Build updates dict from form data
    # Form field names use dot notation: "aws.region", "modes.default", etc.
    updates = {}

    for field_name, value in form_data.multi_items():
        # Parse the dotted path into nested dict
        keys = field_name.split('.')

        # Handle toggle values: hidden input sends "off", checkbox sends "on"
        # When checkbox is checked, both "off" and "on" are sent — "on" wins
        # (last value for same name). When unchecked, only "off" is sent.
        if value == 'on':
            value = True
        elif value == 'off':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            # Try to parse as float
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass  # Keep as string

        # Build nested dict for this path
        current = updates
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    try:
        # Apply updates to config (explicitly use project dir path)
        config_path = NOTEBOOKS_DIR / "dialeng_config.json"
        update_config(updates, config_path=config_path)

        # Reload the global config
        global DIALENG_CONFIG, colab_auth_service, colab_session_manager
        DIALENG_CONFIG = load_config(config_path=config_path, force_reload=True)

        # Lazily initialize or tear down Colab services based on new config
        colab_changed = False
        if DIALENG_CONFIG.colab_enabled and colab_auth_service is None:
            from dialeng.services.colab import ColabAuthService, ColabSessionManager
            from dialeng.services.colab.colab_auth import resolve_oauth_credentials
            _creds = await resolve_oauth_credentials()
            colab_auth_service = ColabAuthService(credentials=_creds)
            colab_session_manager = ColabSessionManager(colab_auth_service)
            kernel_service.set_colab_session_manager(colab_session_manager)
            logger.info(f"Colab services initialized from settings (OAuth source: {_creds.source})")
            colab_changed = True
        elif not DIALENG_CONFIG.colab_enabled and colab_auth_service is not None:
            if colab_session_manager:
                colab_session_manager.shutdown_all()
            colab_auth_service = None
            colab_session_manager = None
            kernel_service.set_colab_session_manager(None)
            logger.info("Colab services disabled from settings")
            colab_changed = True

        # If Colab state changed, reload the page so toolbar updates
        if colab_changed:
            return Div(
                "Settings saved! Reloading...",
                Script("setTimeout(() => window.location.reload(), 500);"),
                cls="settings-status success"
            )

        return Div(
            "Settings saved successfully!",
            cls="settings-status success"
        )
    except Exception as e:
        return Div(
            f"Error saving settings: {str(e)}",
            cls="settings-status error"
        )

# Cell operations - now include notebook ID in path
@rt("/dialeng/{nb_id}/cell/add")
async def post(nb_id: str, pos: int = -1, type: str = "code"):
    nb = get_notebook(nb_id)
    if pos < 0:
        pos = len(nb.cells)
    # Code cells default to scrollable output for better screen space usage
    if type == "code":
        nb.cells.insert(pos, Cell(cell_type=type, output_collapse=1))
    else:
        nb.cells.insert(pos, Cell(cell_type=type))

    # FOUST fix (FOUST = Flash of Unstyled Text — Monaco renders code white first,
    # then tokenizes async via web worker; destroying/recreating the editor causes a
    # visible flash before syntax highlighting reappears).
    # Broadcast granular cell_add JSON instead of AllCellsOOB. AllCellsOOB replaced
    # the entire #cells container, destroying every Monaco editor. cell_add inserts
    # a single cell via insertAdjacentHTML — existing editors are untouched.
    # Note: the initiating tab also gets the HTMX response (AllCells). The client
    # skips cell_add if the cell already exists in DOM to avoid duplicates.
    new_cell = nb.cells[pos]
    cell_html = to_xml(CellView(new_cell, nb_id))
    add_html = to_xml(AddButtons(pos + 1, nb_id))
    await broadcast_json(nb_id, {
        "type": "cell_add",
        "cell_id": new_cell.id,
        "pos": pos,
        "html": cell_html + add_html
    })

    return AllCells(nb)

@rt("/dialeng/{nb_id}/cell/{cid}")
async def delete(nb_id: str, cid: str):
    nb = get_notebook(nb_id)

    # Remove from execution queue if queued
    queue = get_execution_queue(nb_id)
    queue.cancel_queued(nb_id, cid)

    nb.cells = [c for c in nb.cells if c.id != cid]

    # Broadcast queue state update
    await broadcast_queue_state(nb_id)

    # FOUST fix: broadcast granular cell_delete JSON instead of AllCellsOOB.
    # The client removes just this cell + its adjacent add-row from DOM.
    # Other cells' Monaco editors are completely untouched.
    await broadcast_json(nb_id, {"type": "cell_delete", "cell_id": cid})

    return AllCells(nb)

@rt("/dialeng/{nb_id}/cell/{cid}/source")
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
                print(f"[SOURCE UPDATE] Cell {cid}: Source changed, cleared outputs to prevent stale context")
            break
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/output")
def post(nb_id: str, cid: str, output: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.output = output
            break
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/type")
async def post(nb_id: str, cid: str, cell_type: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.cell_type = cell_type
            c.output = ""
            c.execution_count = None

            # Cell type change is the ONE case where CellViewOOB is correct:
            # the input section fundamentally changes (Monaco editor ↔ textarea),
            # so full DOM replacement is unavoidable and expected.
            await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))

            return CellView(c, nb.id)
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/move/{direction}")
async def post(nb_id: str, cid: str, direction: str):
    nb = get_notebook(nb_id)
    for i, c in enumerate(nb.cells):
        if c.id == cid:
            if direction == "up" and i > 0:
                nb.cells[i], nb.cells[i-1] = nb.cells[i-1], nb.cells[i]
            elif direction == "down" and i < len(nb.cells) - 1:
                nb.cells[i], nb.cells[i+1] = nb.cells[i+1], nb.cells[i]
            break

    # FOUST fix: broadcast granular cell_move JSON instead of AllCellsOOB.
    # The client uses DOM insertBefore() to swap adjacent cells. insertBefore
    # MOVES nodes (doesn't copy), so Monaco editors survive with full state.
    await broadcast_json(nb_id, {"type": "cell_move", "cell_id": cid, "direction": direction})

    return AllCells(nb)

@rt("/dialeng/{nb_id}/cell/{cid}/collapse")
async def post(nb_id: str, cid: str, collapsed: str):
    nb = get_notebook(nb_id)
    cell = None
    for c in nb.cells:
        if c.id == cid:
            cell = c
            c.collapsed = collapsed.lower() == "true"
            break

    # Broadcast collapse state via targeted header OOB + class update
    if cell:
        await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))
        await broadcast_json(nb_id, {"type": "cell_class_update", "cell_id": cell.id, "cls": get_cell_state_classes(cell)})

    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/collapse-section")
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

    # FOUST fix: broadcast collapse via JSON instead of CellViewOOB.
    # The client reuses the existing setCollapseLevel() function to update
    # CSS classes in-place — no DOM replacement needed.
    if cell:
        await broadcast_json(nb_id, {
            "type": "cell_collapse_update",
            "cell_id": cell.id,
            "section": section,
            "input_collapse": cell.input_collapse,
            "output_collapse": cell.output_collapse
        })

    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/run")
async def post(nb_id: str, cid: str, source: str = None):
    # Guard: require explicit kernel selection before any cell execution
    if not kernel_service.has_kernel(nb_id):
        from starlette.responses import Response
        resp = Response("", status_code=200, headers={
            "HX-Trigger": json.dumps({"kernel-required": {"cellId": cid}})
        })
        return resp

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

    # Update source if provided (from Monaco editor via hx-vals)
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
            # Use real LLM with dialoghelper context building
            context_messages = build_context_messages(nb, cid)

            # Check if prompt contains special syntax ($`var` or &`func`)
            from dialeng.services.prompt_parser import has_special_syntax
            from dialeng.services.dialeng_config import get_config

            config = get_config()
            max_steps = config.tool_max_steps
            include_builtins = config.tool_builtin_enabled

            if has_special_syntax(c.source) or include_builtins:
                # Use tool-enabled streaming
                kernel = kernel_service.get_kernel(nb_id)
                stream_func = llm_service.stream_response_with_tools(
                    c.source, context_messages, nb.dialog_mode, nb.model, c.use_thinking,
                    kernel=kernel, notebook_id=nb_id, max_steps=max_steps,
                    include_builtins=include_builtins
                )
            else:
                # Use regular streaming (no tools)
                stream_func = llm_service.stream_response(
                    c.source, context_messages, nb.dialog_mode, nb.model, c.use_thinking
                )

        # Stream via WebSocket to all connected clients
        # Collaborators will receive the final cell state via OOB broadcast after completion
        #
        # Text chunk handling for tool loops:
        # - Text BEFORE first tool_call → reasoning (inside LLM Steps)
        # - Text BETWEEN tool_result and next tool_call → reasoning (inside LLM Steps)
        # - Text AFTER last tool_result (no more tool_calls) → final response (outside LLM Steps)
        #
        # Track tool events for persisting in output
        tool_events = {
            "var_substitutions": [],  # List of {"name": ..., "value": ...}
            "tool_calls": [],  # List of {"id": ..., "name": ..., "input": ..., "result": ..., "status": ...}
            "steps": []  # Chronological list of all steps: {"type": "var"|"tool"|"reasoning", ...}
        }
        current_tool_call = {}  # Track in-progress tool call to pair with result
        has_active_tool_call = False  # Track if we're waiting for a tool_result
        had_any_tools = False  # Track if any tool calls occurred in this stream

        # Separate text buffers for proper placement
        pre_tool_text = []  # Text before first tool call
        post_tool_text = []  # Text after last tool_result (potential final response or more reasoning)

        # Notify clients that this prompt cell is now generating.
        # This is needed for cells created programmatically (e.g., via _add_msg_unsafe
        # with run_mode='run') where no UI button click triggers startStreaming().
        if nb_id in ws_connections and ws_connections[nb_id]:
            msg = json.dumps({"type": "prompt_stream_start", "cell_id": cid})
            for send in list(ws_connections[nb_id]):
                try:
                    await send(msg)
                except:
                    pass

        try:
            async for item in stream_func:
                # Check if cancelled
                if cid in cancelled_cells:
                    cancelled_cells.discard(cid)
                    break

                # Handle errors from LLM service
                if item["type"] == "error":
                    post_tool_text.append(f"\n\n**Error:** {item['content']}")
                    # Send error to WebSocket
                    if nb_id in ws_connections and ws_connections[nb_id]:
                        msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": f"\n\n**Error:** {item['content']}"})
                        for send in ws_connections[nb_id]:
                            try:
                                await send(msg)
                            except:
                                pass
                    break

                # Collect response chunks with proper categorization
                if item["type"] == "chunk":
                    if had_any_tools:
                        # We've had at least one tool call
                        # Text after tool_result could be more reasoning or final response
                        # We won't know until the stream ends or another tool_call comes
                        post_tool_text.append(item["content"])
                    else:
                        # No tools yet - this could be reasoning before first tool
                        pre_tool_text.append(item["content"])

                # Track tool events for persistence
                if item["type"] == "var_substituted":
                    tool_events["var_substitutions"].append({
                        "name": item.get("name", ""),
                        "value": item.get("value", "")
                    })
                    tool_events["steps"].append({
                        "type": "var",
                        "name": item.get("name", ""),
                        "value": item.get("value", "")
                    })
                elif item["type"] == "tool_call":
                    # Save any accumulated text as reasoning before this tool call
                    if pre_tool_text:
                        reasoning_text = "".join(pre_tool_text).strip()
                        if reasoning_text:
                            tool_events["steps"].append({
                                "type": "reasoning",
                                "content": reasoning_text
                            })
                        pre_tool_text = []

                    if post_tool_text:
                        # Text between previous tool_result and this tool_call is reasoning
                        reasoning_text = "".join(post_tool_text).strip()
                        if reasoning_text:
                            tool_events["steps"].append({
                                "type": "reasoning",
                                "content": reasoning_text
                            })
                        post_tool_text = []

                    had_any_tools = True
                    has_active_tool_call = True
                    tool_id = item.get("id", "")
                    current_tool_call[tool_id] = {
                        "id": tool_id,
                        "name": item.get("name", ""),
                        "input": item.get("input", {}),
                        "result": None,
                        "status": "pending"
                    }
                elif item["type"] == "tool_result":
                    tool_id = item.get("id", "")
                    if tool_id in current_tool_call:
                        current_tool_call[tool_id]["result"] = item.get("result", {})
                        current_tool_call[tool_id]["status"] = item.get("result", {}).get("status", "success")
                        tool_events["tool_calls"].append(current_tool_call[tool_id])
                        tool_events["steps"].append({
                            "type": "tool",
                            **current_tool_call[tool_id]
                        })
                        del current_tool_call[tool_id]
                    has_active_tool_call = False

                # Send streaming updates via WebSocket
                if nb_id in ws_connections and ws_connections[nb_id]:
                    if item["type"] == "thinking_start":
                        msg = json.dumps({"type": "thinking_start", "cell_id": cid})
                    elif item["type"] == "thinking_end":
                        msg = json.dumps({"type": "thinking_end", "cell_id": cid})
                    elif item["type"] == "thinking":
                        msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": item["content"], "thinking": True})
                    elif item["type"] == "var_substituted":
                        # Variable was substituted - notify client
                        msg = json.dumps({
                            "type": "var_substituted",
                            "cell_id": cid,
                            "var_name": item.get("name", ""),
                            "var_value": item.get("value", "")
                        })
                    elif item["type"] == "tool_available":
                        # Tool became available - notify client
                        msg = json.dumps({
                            "type": "tool_available",
                            "cell_id": cid,
                            "tool_name": item.get("name", ""),
                            "tool_type": item.get("tool_type", "dynamic")
                        })
                    elif item["type"] == "tool_call":
                        # AI is calling a tool - notify client
                        msg = json.dumps({
                            "type": "tool_call",
                            "cell_id": cid,
                            "tool_id": item.get("id", ""),
                            "tool_name": item.get("name", ""),
                            "tool_input": item.get("input", {})
                        })
                    elif item["type"] == "tool_result":
                        # Tool returned a result - notify client
                        msg = json.dumps({
                            "type": "tool_result",
                            "cell_id": cid,
                            "tool_id": item.get("id", ""),
                            "tool_name": item.get("name", ""),
                            "result": item.get("result", {}),
                            "status": item.get("status", "success")
                        })
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
            post_tool_text.append(error_msg)
            if nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "stream_chunk", "cell_id": cid, "chunk": error_msg})
                for send in ws_connections[nb_id]:
                    try:
                        await send(msg)
                    except:
                        pass
        finally:
            # Determine the final response text based on whether tools were used
            if had_any_tools:
                # Tools were used: post_tool_text is the final response to user
                # All text after the last tool_result goes OUTSIDE the LLM Steps
                # Pre-tool and inter-tool reasoning is already captured as steps
                response_text = "".join(post_tool_text)
            else:
                # No tools: pre_tool_text is the entire response
                response_text = "".join(pre_tool_text)

            # Deduplicate response text to handle LLM output issues
            response_text = _deduplicate_response_text(response_text)

            # Build tool steps markdown if there were any tool events
            tool_steps_md = _format_tool_steps_markdown(tool_events)

            # Prepend LLM steps to response if any
            c.output = tool_steps_md + response_text if tool_steps_md else response_text
            c.time_run = datetime.now().strftime("%H:%M:%S")

            # Send end signal to all clients
            if nb_id in ws_connections and ws_connections[nb_id]:
                msg = json.dumps({"type": "stream_end", "cell_id": cid})
                for send in ws_connections[nb_id]:
                    try:
                        await send(msg)
                    except:
                        pass

        # FOUST fix: use targeted updates instead of CellViewOOB for prompt completion.
        # The output was already streamed to the client via stream_chunk/stream_end,
        # so only the header (execution count, time) and cell classes (state badges)
        # need updating. No full cell replacement, no DOM churn.
        await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
        await broadcast_json(nb_id, {"type": "cell_class_update", "cell_id": c.id, "cls": get_cell_state_classes(c)})

    # Determine next cell ID for auto-focus
    next_cell_id = None
    is_last_cell = cell_index == len(nb.cells) - 1

    if is_last_cell:
        # Add a new code cell using OOB swap (with scrollable output default)
        new_cell = Cell(cell_type="code", output_collapse=1)
        nb.cells.append(new_cell)
        new_cell_index = len(nb.cells) - 1
        next_cell_id = new_cell.id

        # FOUST fix: broadcast granular cell_add instead of AllCellsOOB.
        # IMPORTANT: The HTMX response below ALSO appends the new cell via
        # hx_swap_oob="beforeend:#cells". Both the WS message and HTMX response
        # reach the initiating tab. The client's cell_add handler has a duplicate
        # guard (checks if cell already exists in DOM) to prevent creating two
        # copies of the same cell — without this guard, the cell appears twice
        # and the duplicate has no HTMX bindings or Monaco editor, making it
        # unselectable and uneditable.
        cell_html = to_xml(CellView(new_cell, nb_id))
        add_html = to_xml(AddButtons(new_cell_index + 1, nb_id))
        await broadcast_json(nb_id, {
            "type": "cell_add",
            "cell_id": new_cell.id,
            "pos": new_cell_index,
            "html": cell_html + add_html
        })

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

@rt("/dialeng/{nb_id}/kernel/restart")
async def post(nb_id: str):
    """Restart the kernel for a specific notebook."""
    await broadcast_kernel_status(nb_id, "restarting")
    try:
        await kernel_service.restart_async(nb_id)
    except Exception as e:
        print(f"[KERNEL] Restart failed for {nb_id}: {e}", flush=True)
        await broadcast_kernel_status(nb_id, "error")
        return Div(f"Kernel restart failed: {e}", cls="status error")

    # Clear CRAFT execution tracking so CRAFT cells re-execute on next page load
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)
    print(f"[CRAFT] Reset execution tracking for {nb_id} (kernel restart)", flush=True)

    # Inject _lib/ sys.path and re-run CRAFT code cells in the fresh kernel
    nb = notebooks.get(nb_id)
    craft_cells = []
    if nb:
        nb_path = nb.path
        if not nb_path:
            found = _find_notebook_path(nb_id)
            nb_path = found if found else NOTEBOOKS_DIR / f"{nb_id}.ipynb"
        if nb_path and Path(nb_path).exists():
            from dialeng.services.craft_service import find_craft_files, get_craft_code_cells, _executed_craft
            craft_paths = find_craft_files(nb_path, NOTEBOOKS_DIR)
            # Exclude self if this notebook is a CRAFT file
            nb_path_resolved = Path(nb_path).resolve()
            craft_paths = [cp for cp in craft_paths if Path(cp).resolve() != nb_path_resolved]
            if craft_paths:
                craft_cells = get_craft_code_cells(craft_paths)
                if craft_cells:
                    # Mark synchronously to prevent race conditions
                    _executed_craft.setdefault(nb_id, set()).update(cid for cid, _ in craft_cells)
                    print(f"[CRAFT] Re-executing {len(craft_cells)} code cells after kernel restart", flush=True)

    async def _restart_setup():
        try:
            # Inject lib sys.path and re-upload to Colab before running CRAFT cells
            await _inject_lib_syspath(nb_id)
            await _upload_lib_to_colab(nb_id)
            for cid, src in craft_cells:
                try:
                    cell = Cell(id=cid, cell_type=CellType.CODE, source=src)
                    async for _ in kernel_service.execute_cell(nb_id, cell):
                        pass
                    print(f"[CRAFT] Executed: {cid}", flush=True)
                except Exception as e:
                    print(f"[CRAFT] Error executing cell {cid}: {e}", flush=True)
            if craft_cells:
                print(f"[CRAFT] All cells re-executed for {nb_id}", flush=True)
            # Signal frontend that restart setup is complete → dot turns green
            await broadcast_kernel_status(nb_id, "connected")
        except Exception as e:
            print(f"[CRAFT] Restart setup failed for {nb_id}: {e}", flush=True)
            await broadcast_kernel_status(nb_id, "error")
    asyncio.create_task(_restart_setup())

    return Div(cls="status")

@rt("/dialeng/{nb_id}/kernel/interrupt")
async def post(nb_id: str):
    """Interrupt currently running code in the notebook's kernel."""
    success = await kernel_service.interrupt_async(nb_id)
    if success:
        return Div("✓ Execution interrupted", cls="status success")
    else:
        return Div("No kernel to interrupt", cls="status warning")

@rt("/dialeng/{nb_id}/queue/cancel_all")
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

def _resolve_current_idx(dlg_name: str, nb) -> int:
    """Resolve the index of the currently executing cell for a notebook.
    Used by endpoints with relative addressing (e.g. read_msg_) when
    dialoghelper doesn't send current_idx explicitly."""
    if dlg_name in execution_queues:
        queue = execution_queues[dlg_name]
        status = queue.get_status(dlg_name)
        if status.current_cell_id:
            idx = get_msg_idx(nb, status.current_cell_id)
            if idx >= 0:
                return idx
    return 0

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
def post(dlg_name: str, re_pattern: str = "", msg_type: str = "", limit: int = 100,
         use_case: str = "", use_regex: str = "True",
         only_err: str = "", only_exp: str = "", only_chg: str = "",
         ids: str = "", include_output: str = "True", include_meta: str = "",
         as_xml: str = "", nums: str = "",
         trunc_out: str = "True", trunc_in: str = "",
         headers_only: str = "", header_section: str = ""):
    """Search messages - uses shared find_msgs() with XML/JSON output."""
    nb = get_notebook(dlg_name)
    results = find_msgs(
        nb, re_pattern=re_pattern, msg_type=msg_type, limit=limit,
        use_case=_str_to_bool(use_case),
        use_regex=_str_to_bool(use_regex) if use_regex else True,
        only_err=_str_to_bool(only_err),
        only_exp=_str_to_bool(only_exp),
        only_chg=_str_to_bool(only_chg),
        ids=ids,
        include_output=_str_to_bool(include_output) if include_output else True,
    )

    fmt_kwargs = dict(
        include_output=_str_to_bool(include_output) if include_output else True,
        include_meta=_str_to_bool(include_meta),
        nums=_str_to_bool(nums),
        trunc_out=_str_to_bool(trunc_out) if trunc_out else True,
        trunc_in=_str_to_bool(trunc_in),
        headers_only=_str_to_bool(headers_only),
        header_section=header_section,
    )

    if _str_to_bool(as_xml):
        return format_msgs_as_xml(results, **fmt_kwargs)

    msgs = format_msgs_as_json(results, **fmt_kwargs)
    return {"msgs": msgs}

@rt("/read_msg_")
def post(dlg_name: str, n: int = 0, relative: bool = True, id_: str = "",
         view_range: str = "", nums: bool = False, current_idx: int = -1):
    """Read message content - uses shared read_msg()."""
    nb = get_notebook(dlg_name)
    # Resolve current_idx from execution queue if not explicitly provided
    if current_idx < 0 and relative and not id_:
        current_idx = _resolve_current_idx(dlg_name, nb)
    elif current_idx < 0:
        current_idx = 0
    return read_msg(nb, n=n, relative=relative, msgid=id_,
                    current_idx=current_idx, view_range=view_range, nums=nums)

@rt("/add_relative_")
async def post(dlg_name: str, content: str, placement: str = "add_after", id_: str = "",
         msg_type: str = "code", output: str = "", time_run: str = "",
         is_exported: str = "", skipped: str = "",
         i_collapsed: str = "0", o_collapsed: str = "0",
         heading_collapsed: str = "", pinned: str = "",
         run: str = "", run_mode: str = ""):
    """Add message relative to another - uses shared get_msg_idx().

    Supports both old placement values (after/before) and new ones
    (add_after/add_before/at_start/at_end).
    Boolean params use str type because HTTP form data sends 'True'/'False' strings.
    """
    print(f"[ADD_RELATIVE] dlg_name={dlg_name}, ws_connections keys={list(ws_connections.keys())}", flush=True)
    nb = get_notebook(dlg_name)
    new_cell = Cell(
        cell_type=msg_type, source=content,
        skipped=_str_to_bool(skipped), pinned=_str_to_bool(pinned),
        input_collapse=int(i_collapsed) if i_collapsed else 0,
        output_collapse=int(o_collapsed) if o_collapsed else 0,
        heading_collapsed=_str_to_bool(heading_collapsed),
        is_exported=_str_to_bool(is_exported), time_run=time_run
    )
    if output: new_cell.output = output

    # Normalize placement values (support both old and new)
    placement_map = {"after": "add_after", "before": "add_before"}
    placement = placement_map.get(placement, placement)

    # Find insertion point using shared function
    if placement == "at_start":
        insert_idx = 0
    elif placement == "at_end":
        insert_idx = len(nb.cells)
    elif id_:
        ref_idx = get_msg_idx(nb, id_)
        if ref_idx == -1:
            return {"error": f"Message {id_} not found"}
        insert_idx = ref_idx + 1 if placement == "add_after" else ref_idx
    else:
        # No explicit id_ — resolve from currently executing cell
        ref_idx = _resolve_current_idx(dlg_name, nb)
        insert_idx = ref_idx + 1 if placement == "add_after" else ref_idx

    nb.cells.insert(insert_idx, new_cell)

    # FOUST fix: broadcast granular cell_add JSON instead of AllCellsOOB.
    # This is the dialoghelper version of the add route — same pattern as /cell/add.
    try:
        cell_html = to_xml(CellView(new_cell, dlg_name))
        add_html = to_xml(AddButtons(insert_idx + 1, dlg_name))
        await broadcast_json(dlg_name, {
            "type": "cell_add",
            "cell_id": new_cell.id,
            "pos": insert_idx,
            "html": cell_html + add_html
        })
        print(f"[ADD_RELATIVE] Broadcast completed for {dlg_name}", flush=True)
    except Exception as e:
        print(f"[ADD_RELATIVE] Broadcast error: {e}", flush=True)

    # Optionally trigger execution (support both old 'run' and new 'run_mode')
    should_run = _str_to_bool(run) or run_mode
    if should_run:
        if new_cell.cell_type == "prompt":
            # Prompt cells need LLM execution, not Python kernel.
            # Fire-and-forget via background task so /add_relative_ returns immediately
            # (otherwise the caller times out waiting for the LLM to finish streaming).
            async def _run_prompt():
                import httpx
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            f'http://localhost:8000/dialeng/{dlg_name}/cell/{new_cell.id}/run',
                            timeout=300.0
                        )
                    except Exception as e:
                        print(f"[ADD_RELATIVE] Prompt run failed: {e}", flush=True)
            asyncio.create_task(_run_prompt())
        else:
            queue = get_execution_queue(dlg_name)
            queue.queue_cell(dlg_name, new_cell)

    # New dialoghelper expects JSON with 'id' key (res['id'])
    return {"id": new_cell.id}

@rt("/rm_msg_")
async def post(dlg_name: str, msid: str, log_changed: str = ""):
    """Remove message by ID. Optionally logs the deleted content."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, msid)
    if idx >= 0:
        cell = nb.cells[idx]
        if _str_to_bool(log_changed):
            log_change(dlg_name, "delete", msid, {
                "source": cell.source,
                "type": cell.cell_type,
                "output": cell.output[:200] if cell.output else "",
            })
        nb.cells.pop(idx)
        # FOUST fix: granular cell_delete instead of AllCellsOOB (dialoghelper route)
        await broadcast_json(dlg_name, {"type": "cell_delete", "cell_id": msid})
    return {"status": "ok", "id": msid}

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
               heading_collapsed: str = None, pinned: str = None,
               log_changed: str = ""):
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

        # Track changes for logging
        changes = {}

        # Map and apply each field if provided (not None)
        if content is not None:
            if _str_to_bool(log_changed) and cell.source != content:
                changes["content"] = {"old": cell.source[:200], "new": content[:200]}
            cell.source = content
        if msg_type is not None:
            if _str_to_bool(log_changed) and cell.cell_type != msg_type:
                changes["msg_type"] = {"old": cell.cell_type, "new": msg_type}
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

        # Log changes if requested
        if _str_to_bool(log_changed) and changes:
            log_change(dlg_name, "update", id_, changes)

        print(f"[UPDATE_MSG] Cell {cell.id} after: pinned={cell.pinned}", flush=True)
        # Smart broadcast: use targeted updates to preserve Monaco editor DOM
        if msg_type is not None:
            # Type change requires full cell re-render
            await broadcast_to_notebook(dlg_name, CellViewOOB(cell, dlg_name))
        else:
            # Source change → JSON update (no FOUST)
            if content is not None:
                await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
            # Output change → targeted OOB
            if output is not None:
                await broadcast_to_notebook(dlg_name, CellOutputOOB(cell))
            # Header/state changes → targeted header OOB + class update
            if any(v is not None for v in [time_run, is_exported, skipped, pinned, i_collapsed, o_collapsed, heading_collapsed]):
                await broadcast_to_notebook(dlg_name, CellHeaderOOB(cell, dlg_name))
                await broadcast_json(dlg_name, {"type": "cell_class_update", "cell_id": cell.id, "cls": get_cell_state_classes(cell)})
    # New dialoghelper expects JSON with 'id' key (res['id'])
    return {"id": id_}

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
async def post(dlg_name: str, id_: str, insert_line: int, new_str: str):
    """Insert line at position in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines.insert(insert_line, new_str)
        cell.source = '\n'.join(lines)
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
    return {"status": "ok"}

@rt("/msg_str_replace_")
async def post(dlg_name: str, id_: str, old_str: str, new_str: str,
         start_line: int = 0, end_line: int = 0,
         n_matches: int = 1, re_filter: str = "", invert_filter: str = ""):
    """Replace string in message with optional line range and regex filtering."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    source = cell.source

    # Apply line range restriction if specified
    if start_line or end_line:
        lines = source.split('\n')
        s = (start_line - 1) if start_line > 0 else 0
        e = end_line if end_line > 0 else len(lines)
        # Only replace within the specified line range
        before = '\n'.join(lines[:s])
        target = '\n'.join(lines[s:e])
        after_lines = '\n'.join(lines[e:])

        # Apply regex filter if specified
        if re_filter:
            inv = _str_to_bool(invert_filter)
            filtered_lines = []
            for line in target.split('\n'):
                matches = bool(re.search(re_filter, line))
                if (matches and not inv) or (not matches and inv):
                    line = line.replace(old_str, new_str, n_matches if n_matches else 0) if n_matches != 0 else line.replace(old_str, new_str)
                filtered_lines.append(line)
            target = '\n'.join(filtered_lines)
        else:
            count = n_matches if n_matches else 0  # 0 means replace all
            target = target.replace(old_str, new_str, count) if count else target.replace(old_str, new_str)

        parts = [p for p in [before, target, after_lines] if p]
        cell.source = '\n'.join(parts)
    else:
        count = n_matches if n_matches else 0
        cell.source = source.replace(old_str, new_str, count) if count else source.replace(old_str, new_str)

    # JSON source update — preserves Monaco editor DOM (no FOUST)
    await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
    return {"status": "ok"}

@rt("/msg_strs_replace_")
async def post(dlg_name: str, id_: str, old_strs: str, new_strs: str):
    """Replace multiple strings (JSON arrays)."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        old_list = json.loads(old_strs)
        new_list = json.loads(new_strs)
        for old, new in zip(old_list, new_list):
            cell.source = cell.source.replace(old, new, 1)
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
    return {"status": "ok"}

@rt("/msg_replace_lines_")
async def post(dlg_name: str, id_: str, start_line: int, end_line: int, new_content: str):
    """Replace line range in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines[start_line:end_line] = [new_content]
        cell.source = '\n'.join(lines)
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
    return {"status": "ok"}

@rt("/msg_del_lines_")
async def post(dlg_name: str, id_: str, start_line: int, end_line: int,
         re_filter: str = "", invert_filter: str = ""):
    """Delete line range in message, optionally filtered by regex."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    lines = cell.source.split('\n')
    s = (start_line - 1) if start_line > 0 else 0
    e = end_line if end_line > 0 else len(lines)

    if re_filter:
        inv = _str_to_bool(invert_filter)
        # Only delete lines matching (or not matching if inverted) the filter
        target_lines = lines[s:e]
        kept = []
        for line in target_lines:
            matches = bool(re.search(re_filter, line))
            if (matches and not inv) or (not matches and inv):
                continue  # Delete this line
            kept.append(line)
        lines[s:e] = kept
    else:
        del lines[s:e]

    cell.source = '\n'.join(lines)
    # JSON source update — preserves Monaco editor DOM (no FOUST)
    await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
    return {"status": "ok"}

@rt("/msg_pyrun_")
async def post(dlg_name: str, id_: str, code: str):
    """Execute Python code against message text.

    The cell source is available as `text` in the code's namespace.
    The code should modify `text` to update the cell.
    """
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    namespace = {"text": cell.source}
    try:
        exec(code, namespace)
        if "text" in namespace and namespace["text"] != cell.source:
            cell.source = namespace["text"]
            # JSON source update — preserves Monaco editor DOM (no FOUST)
            await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Clipboard Operations
# ============================================================================

@rt("/msg_clipboard_")
async def post(dlg_name: str, ids: str = "", id_: str = "", cmd: str = "copy"):
    """Copy or cut messages to clipboard."""
    nb = get_notebook(dlg_name)
    # Build list of cell IDs
    cell_ids = [s.strip() for s in ids.split(',') if s.strip()] if ids else []
    if id_ and id_ not in cell_ids:
        cell_ids.append(id_)

    if not cell_ids:
        return {"error": "No cell IDs provided"}

    result = clipboard_copy(nb, dlg_name, cell_ids, cut=(cmd == "cut"))

    if cmd == "cut":
        # FOUST fix: granular cell_delete per cell instead of AllCellsOOB
        for cid in cell_ids:
            await broadcast_json(dlg_name, {"type": "cell_delete", "cell_id": cid})

    return result

@rt("/msg_paste_")
async def post(dlg_name: str, id_: str = "", after: str = "True"):
    """Paste messages from clipboard."""
    nb = get_notebook(dlg_name)
    new_ids = clipboard_paste(nb, dlg_name, ref_id=id_, after=_str_to_bool(after))

    if new_ids:
        # FOUST fix: granular cell_add per pasted cell instead of AllCellsOOB
        for new_id in new_ids:
            for idx, c in enumerate(nb.cells):
                if c.id == new_id:
                    cell_html = to_xml(CellView(c, dlg_name))
                    add_html = to_xml(AddButtons(idx + 1, dlg_name))
                    await broadcast_json(dlg_name, {
                        "type": "cell_add",
                        "cell_id": new_id,
                        "pos": idx,
                        "html": cell_html + add_html
                    })
                    break

    return {"status": "ok", "ids": new_ids}

# ============================================================================
# UI Toggle Operations
# ============================================================================

@rt("/toggle_header_collapse_")
async def post(dlg_name: str, id_: str):
    """Toggle heading_collapsed state on a cell."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    cell.heading_collapsed = not cell.heading_collapsed
    # Targeted header OOB (preserves editor DOM)
    await broadcast_to_notebook(dlg_name, CellHeaderOOB(cell, dlg_name))
    return {"status": "ok", "heading_collapsed": cell.heading_collapsed}

@rt("/dialeng/{nb_id}/cell/{cell_id}/toggle/{prop}")
async def post(nb_id: str, cell_id: str, prop: str):
    """Toggle a boolean cell property (skipped, pinned, is_exported)."""
    allowed = {'skipped', 'pinned', 'is_exported'}
    if prop not in allowed:
        return {"error": f"Cannot toggle '{prop}'"}
    nb = get_notebook(nb_id)
    idx = get_msg_idx(nb, cell_id)
    if idx < 0:
        return {"error": f"Cell {cell_id} not found"}
    cell = nb.cells[idx]
    setattr(cell, prop, not getattr(cell, prop))
    # Sync #| export directive with is_exported flag
    if prop == 'is_exported':
        cell.sync_export_directive()
    # Targeted header OOB + class update (preserves editor DOM)
    await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))
    await broadcast_json(nb_id, {"type": "cell_class_update", "cell_id": cell.id, "cls": get_cell_state_classes(cell)})
    return {"status": "ok", prop: getattr(cell, prop)}

@rt("/bookmark_")
async def post(dlg_name: str, id_: str, n: int):
    """Toggle numbered bookmark (1-9) on a cell."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    # Toggle: if already bookmarked with same number, remove it
    if cell.bookmark == n:
        cell.bookmark = 0
    else:
        # Clear any existing bookmark with this number from other cells
        for c in nb.cells:
            if c.bookmark == n:
                c.bookmark = 0
        cell.bookmark = n

    # Targeted header OOB (preserves editor DOM)
    await broadcast_to_notebook(dlg_name, CellHeaderOOB(cell, dlg_name))
    return {"status": "ok", "bookmark": cell.bookmark}

@rt("/toggle_comment_")
async def post(dlg_name: str, ids: str = "", id_: str = ""):
    """Toggle line comments on code cells."""
    nb = get_notebook(dlg_name)
    cell_ids = [s.strip() for s in ids.split(',') if s.strip()] if ids else []
    if id_ and id_ not in cell_ids:
        cell_ids.append(id_)

    for cid in cell_ids:
        idx = get_msg_idx(nb, cid)
        if idx < 0:
            continue
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        # Determine if we should comment or uncomment (majority rules)
        commented = sum(1 for l in lines if l.lstrip().startswith('#') and l.strip() != '#')
        if commented > len(lines) / 2:
            # Uncomment: remove leading '# ' or '#'
            new_lines = []
            for l in lines:
                stripped = l.lstrip()
                indent = l[:len(l) - len(stripped)]
                if stripped.startswith('# '):
                    new_lines.append(indent + stripped[2:])
                elif stripped.startswith('#'):
                    new_lines.append(indent + stripped[1:])
                else:
                    new_lines.append(l)
            cell.source = '\n'.join(new_lines)
        else:
            # Comment: add '# ' prefix
            cell.source = '\n'.join(
                (l[:len(l) - len(l.lstrip())] + '# ' + l.lstrip()) if l.strip() else l
                for l in lines
            )
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, {"type": "cell_source_update", "cell_id": cell.id, "source": cell.source})

    return {"status": "ok"}

# ============================================================================
# Dialog Management Endpoints
# ============================================================================

@rt("/create_dialog_")
async def post(name: str):
    """Create a new dialog/notebook or ensure it's loaded.

    If the notebook exists on disk, it will be loaded.
    If not, a new empty notebook is created.
    The kernel is shared across all notebooks.
    """
    try:
        nb = get_notebook(name)  # get_notebook handles creation
        return {"status": "ok", "name": name, "action": "loaded" if nb.cells else "created"}
    except Exception as e:
        return {"error": str(e)}

@rt("/stop_kernel_")
async def post(name: str):
    """Stop the execution queue for a dialog/notebook."""
    try:
        if name in execution_queues:
            queue = execution_queues[name]
            queue.cancel_all()
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"error": str(e)}

@rt("/rm_dialog_")
async def post(name: str):
    """Delete a dialog/notebook from memory and optionally disk."""
    try:
        # Cancel any running executions
        if name in execution_queues:
            execution_queues[name].cancel_all()
            del execution_queues[name]
        # Remove from in-memory registry
        if name in notebooks:
            del notebooks[name]
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"error": str(e)}

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
# Markdown Rendering API
# ============================================================================

@rt("/render-markdown")
async def post(text: str):
    """Render markdown to HTML using mistlefoot's ExtendedHtmlRenderer."""
    try:
        from mistletoe import markdown as md_render
        from mistlefoot import ExtendedHtmlRenderer
        html = md_render(text, ExtendedHtmlRenderer)
        return {"html": html}
    except ImportError:
        return {"html": text}

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
async def ws(msg: str, send, nb_id: str):
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

def main(root_dir: Path = None, port: int = 8000):
    """CLI entry point for dialeng."""
    if root_dir is not None:
        set_root_dir(root_dir)
    print(f"🚀 Dialeng starting at http://localhost:{port}")
    print(f"   Root directory: {NOTEBOOKS_DIR.resolve()}")
    print("   Format: Solveit-compatible .ipynb")
    print("")
    # Print credential status
    print_credential_status(CREDENTIAL_STATUS)
    print("")
    # Print config status (pass detected backend to show active default)
    print_config_status(DIALENG_CONFIG, CREDENTIAL_STATUS.backend)
    print("")
    # Print shell/safecmd status
    print("   Shell Execution:")
    print_shfmt_status()
    if not SHFMT_AVAILABLE:
        warn_missing_shfmt()
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
    serve(appname="dialeng.app", port=port, reload_excludes=[".autorun_modules/*"])


if __name__ == "__main__":
    main()
