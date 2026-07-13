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
import uuid, json, os, sys, io, traceback, asyncio, re, ast, logging, time
import inspect
from functools import wraps
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr
from enum import Enum
from pathlib import Path
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from dialeng.notebook_id import (
    nb_id_encode_part as _shared_nb_id_encode_part,
    nb_id_from_path as _shared_nb_id_from_path,
    nb_id_to_relpath as _shared_nb_id_to_relpath,
)

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
    get_context_inspection,
)
from dialeng.services.credential_service import (
    detect_credentials, detect_credential_inventory, refresh_credential_inventory,
    refresh_credential_inventory_async,
    build_initial_credential_inventory, credential_status_from_inventory,
    get_available_modes, print_credential_status, print_credential_inventory,
    CredentialInventory, CredentialStatus,
)
from dialeng.services.dialeng_config import (
    load_config, get_config, print_config_status
)
from dialeng.services.settings_service import (
    SettingsConflictError,
    SettingsValidationError,
    apply_modes_for,
    commit_settings,
    get_setting_definitions,
    settings_snapshot,
)
from dialeng.services.shell_service import (
    SHFMT_AVAILABLE, warn_missing_shfmt, print_shfmt_status
)
from dialeng.services.broadcast_service import BroadcastManager
from dialeng.services.file_editor_service import FileEditorService
from dialeng.services.operation_log import elapsed_ms, log_operation, new_operation_id
from dialeng.services.performance_service import (
    RequestMetricsMiddleware,
    performance_monitor,
    monitor_event_loop_lag,
)
from dialeng.services.runtime_registry import (
    LifecycleState,
    NotebookRuntimeRegistry,
    RuntimeRevisionConflict,
)
from dialeng.services.persistence_service import (
    AutosaveManager,
    AutosaveSettings,
    ExternalFileChangedError,
    atomic_write_json,
    discard_recovery,
    find_newer_recovery,
)
from dialeng.services.file_listing_cache import directory_listing_cache
from dialeng.services.outline_cache import outline_cache
from dialeng.services.notebook_mutation_service import notebook_mutation_service
from dialeng.services.openai_codex_auth import get_openai_codex_auth_service
from dialeng.services.skill_service import set_skill_root
from dialeng.routes import (
    register_diagnostics_routes,
    register_e2e_routes,
    register_extension_routes,
    register_skill_routes,
    register_websocket_routes,
)

# UI Components (extracted to ui/ package)
from dialeng.ui import (
    CellView, NotebookPage, AllCells, AllCellsContent,
    AllCellsOOB, CellViewOOB, CellOutputOOB, CellHeaderOOB, AddButtons,
    TypeSelect, CollapseBtn, get_collapse_class, get_cell_state_classes,
    ModelSelect, WorkspacePage,
)
from dialeng.ui.file_editor import FileEditorPage, FileEditorPane
from dialeng.ui.settings import SettingsSidebar
from dialeng.ui.mime import render_mime_bundle
from dialeng.logging_config import build_log_config

# Extension system
from dialeng.core.extensions import load_extensions, get_builtin_contribution_store
from dialeng.core.registry import registry
from dialeng.services.autorun_service import process_autorun
from dialeng.extensions.manager import ExtensionManager
from dialeng.extensions.events import event_bus

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

SOLVEIT_VER = 2

# Load configuration — use in-memory defaults at import time.
# The real config file (inside the notebooks dir) is loaded when set_root_dir() is called.
DIALENG_CONFIG = load_config(create_if_missing=False)

# Build a local-only inventory at import time. Network and subprocess probes run
# after the server is responsive.
CREDENTIAL_INVENTORY = build_initial_credential_inventory()
CREDENTIAL_STATUS = credential_status_from_inventory(CREDENTIAL_INVENTORY)
AVAILABLE_DIALOG_MODES = get_available_modes(CREDENTIAL_STATUS)

# New notebooks use the current default provider's configured model.
DEFAULT_MODEL = DIALENG_CONFIG.get_default_model(CREDENTIAL_STATUS.backend)
DEFAULT_LLM_PROVIDER = CREDENTIAL_INVENTORY.default_provider()
openai_codex_auth_service = get_openai_codex_auth_service()


def _apply_llm_inventory(inventory: CredentialInventory) -> None:
    """Apply a completed provider inventory on the application event loop."""
    global CREDENTIAL_STATUS, CREDENTIAL_INVENTORY, AVAILABLE_DIALOG_MODES
    global DEFAULT_MODEL, DEFAULT_LLM_PROVIDER
    global DEFAULT_DIALOG_MODE
    CREDENTIAL_INVENTORY = inventory
    selected = CREDENTIAL_INVENTORY.get(CREDENTIAL_INVENTORY.default_provider())
    if selected and selected.id != "mock_only" and selected.authenticated:
        CREDENTIAL_STATUS = CredentialStatus(
            available=True,
            provider=selected.provider,
            backend=selected.backend,
            source=selected.source,
            details=selected.details,
        )
    else:
        CREDENTIAL_STATUS = CredentialStatus(False, "mock_only", "none", "none", "No LLM provider authenticated")
    AVAILABLE_DIALOG_MODES = get_available_modes(CREDENTIAL_STATUS)
    DEFAULT_LLM_PROVIDER = CREDENTIAL_INVENTORY.default_provider()
    DEFAULT_MODEL = DIALENG_CONFIG.get_default_model(DEFAULT_LLM_PROVIDER)
    DEFAULT_DIALOG_MODE = DIALENG_CONFIG.default_mode if CREDENTIAL_INVENTORY.available else "mock"
    llm_service.invalidate_providers()


def _refresh_llm_runtime(*, probe_all: bool = True) -> None:
    """Synchronously refresh provider/model availability for explicit actions."""
    if probe_all:
        inventory = refresh_credential_inventory(NOTEBOOKS_DIR / ".env")
    else:
        inventory = detect_credential_inventory(NOTEBOOKS_DIR / ".env", force=True)
    _apply_llm_inventory(inventory)


def _load_initial_llm_runtime() -> None:
    """Load local provider hints without running remote probes."""
    inventory = build_initial_credential_inventory(NOTEBOOKS_DIR / ".env")
    _apply_llm_inventory(inventory)
    if "remote_service_state" in globals():
        remote_service_state["providers"] = (
            "validating"
            if any(status.validation_state == "validating" for status in inventory.providers.values())
            else "ready"
        )
        remote_service_state["colab"] = "validating" if DIALENG_CONFIG.colab_enabled else "disabled"

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
runtime_registry = NotebookRuntimeRegistry()

# Colab session manager is initialized asynchronously after the server is ready.
colab_session_manager = None
colab_auth_service = None
_settings_update_lock = asyncio.Lock()
_settings_apply_lock = asyncio.Lock()
_settings_activation: dict[str, Any] = {"status": "ready"}
_settings_activation_tasks: set[asyncio.Task] = set()


async def _init_colab_async() -> None:
    """Initialize Colab without blocking application startup or the event loop."""
    global colab_auth_service, colab_session_manager
    if not DIALENG_CONFIG.colab_enabled:
        return
    if colab_auth_service is not None:
        return  # already initialized

    from dialeng.services.colab import ColabAuthService, ColabSessionManager
    from dialeng.services.colab.colab_auth import resolve_oauth_credentials, print_colab_credential_status

    with performance_monitor.measure("colab.credentials.resolve"):
        colab_creds = await resolve_oauth_credentials(DIALENG_CONFIG.colab_auth_provider)

    colab_auth_service = ColabAuthService(
        credentials=colab_creds,
        auth_provider=DIALENG_CONFIG.colab_auth_provider,
    )
    with performance_monitor.measure("colab.session.validate"):
        validated = await colab_auth_service.validate_session()
    colab_session_manager = ColabSessionManager(
        colab_auth_service,
        default_accelerator=DIALENG_CONFIG.colab_default_accelerator,
    )
    if validated:
        try:
            with performance_monitor.measure("colab.metadata.prune"):
                await colab_session_manager.prune_stale_metadata()
        except Exception as e:
            logger.warning("Failed to prune Colab session metadata during startup: %s", e)
    kernel_service.set_colab_session_manager(colab_session_manager)
    print_colab_credential_status(colab_creds)
    logger.info(
        "Colab services initialized (oauth_source=%s, auth_provider=%s, authenticated=%s, email=%s, validation_error=%s)",
        colab_creds.source,
        DIALENG_CONFIG.colab_auth_provider,
        validated,
        colab_auth_service.account_email,
        colab_auth_service.session_error,
    )
    print(f"   Colab: enabled (authenticated={colab_auth_service.is_authenticated})")

# ExecutionQueue instances per notebook (created lazily)
execution_queues = runtime_registry.field_map("execution_queue")

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
            await broadcast_json(nb_id, {"type": "code_stream_start", "cell_id": cell.id})

        await broadcast_cell_state(nb_id, cell.id, state)
        await broadcast_queue_state(nb_id)

        if state in (CellState.SUCCESS, CellState.ERROR):
            # Finalize cell output and send code_stream_end
            await finalize_cell_execution(nb_id, cell, state == CellState.ERROR)
    return callback


async def finalize_cell_execution(nb_id: str, cell, has_error: bool):
    """Broadcast final cell state via OOB swaps after execution completes.

    Note: We preserve cell.outputs as-is from kernel streaming rather than
    flattening to a string via cell.output setter, which would destroy
    structured outputs (display_data, MIME bundles, etc.).
    """
    async with runtime_registry.mutation(nb_id):
        nb = notebooks.get(nb_id)
        if nb is None or nb.get_cell(cell.id) is None:
            logger.info(
                "Ignoring completion for a cell removed during execution (notebook=%s, cell=%s)",
                nb_id,
                cell.id,
            )
            return
        cell.time_run = datetime.now().strftime("%H:%M:%S")
        nb.modified = True

        # Send code_stream_end signal
        await broadcast_json(nb_id, {"type": "code_stream_end", "cell_id": cell.id, "has_error": has_error})

        # Broadcast output + header via targeted OOB swaps (preserves Monaco editor DOM)
        await broadcast_to_notebook(nb_id, CellOutputOOB(cell))
        await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))

        cell_type = str(getattr(cell, "cell_type", ""))
        if cell_type in {"code", "shell"}:
            await _mark_outline_dirty(nb_id, reason=f"{cell_type}_execution_complete")

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

notebooks = runtime_registry.field_map("notebook")
# Deleted notebook IDs remain terminal for already-open browser tabs. Explicit
# creation clears the tombstone, while stale requests receive HTTP 410.
deleting_notebooks: set[str] = set()
deleted_notebooks: Dict[str, Dict[str, Any]] = {}
NOTEBOOKS_DIR = Path(os.environ.get("DIALENG_NOTEBOOKS_DIR", "notebooks"))
NOTEBOOKS_DIR.mkdir(exist_ok=True)
file_editor_service = FileEditorService(NOTEBOOKS_DIR)
extension_manager = ExtensionManager(
    NOTEBOOKS_DIR,
    contribution_store=get_builtin_contribution_store(),
    development_paths=tuple(
        Path(path).expanduser().resolve()
        for path in os.environ.get("DIALENG_EXTENSION_PATHS", "").split(os.pathsep)
        if path
    ),
)
from dialeng.services.autorun_service import set_autorun_contribution_store
set_autorun_contribution_store(extension_manager.store)

def set_root_dir(root: Path):
    """Set the notebooks root directory. Called by CLI before main().

    Also reloads dialeng_config.json from the new root so the config
    lives alongside the notebooks (not in whatever CWD the command ran from).
    """
    global NOTEBOOKS_DIR, DIALENG_CONFIG
    if root.resolve() != NOTEBOOKS_DIR.resolve():
        deleting_notebooks.clear()
        deleted_notebooks.clear()
    NOTEBOOKS_DIR = root
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    file_editor_service.set_root(NOTEBOOKS_DIR)
    extension_manager.set_project_root(NOTEBOOKS_DIR)

    # Reload config from the project directory
    DIALENG_CONFIG = load_config(config_path=NOTEBOOKS_DIR / "dialeng_config.json", force_reload=True)
    set_skill_root(NOTEBOOKS_DIR)
    _load_initial_llm_runtime()

# Track active WebSocket connections per notebook (list of send functions)
ws_connections: Dict[str, List[Any]] = {}
broadcast_manager = BroadcastManager(
    ws_connections,
    monitor=performance_monitor,
    on_client_count_changed=kernel_service.set_client_count,
)

# Track cancelled cell generations
cancelled_cells: set = set()
prompt_cancel_events: Dict[str, asyncio.Event] = {}

# Track non-queue kernel setup/sync work per notebook so the UI can reflect
# attach/init/upload phases instead of inferring state from cell streaming alone.
kernel_setup_state = runtime_registry.field_map("setup_state")
kernel_setup_tasks = runtime_registry.field_map("setup_task")
kernel_sync_tasks = runtime_registry.field_map("sync_task")
kernel_generations = runtime_registry.field_map("kernel_generation")
notebook_revisions = runtime_registry.field_map("notebook_revision")
structure_revisions = runtime_registry.field_map("structure_revision")
outline_revisions = runtime_registry.field_map("outline_revision")
event_stream_revisions = runtime_registry.field_map("stream_revisions")
server_shutdown_started = False
event_loop_monitor_task: Optional[asyncio.Task] = None
remote_service_init_task: Optional[asyncio.Task] = None
project_file_watch_task: Optional[asyncio.Task] = None
remote_service_state = {"providers": "idle", "colab": "idle"}


def _active_prompt_cells(notebook_id: str) -> list[str]:
    prefix = f"{notebook_id}:"
    return [key[len(prefix):] for key in prompt_cancel_events if key.startswith(prefix)]


def _prompt_operation_conflict(notebook_id: str, operation: str) -> JSONResponse:
    active_cells = _active_prompt_cells(notebook_id)
    return JSONResponse({
        "status": "conflict",
        "type": "prompt_operation_conflict",
        "operation": operation,
        "notebook_id": notebook_id,
        "active_cell_ids": active_cells,
        "message": "A prompt is still running. Wait for it to finish or cancel it before changing the notebook.",
        "actions": ["wait", "cancel"],
    }, status_code=409)


def _cancel_active_prompts(notebook_id: str) -> list[str]:
    active_cells = _active_prompt_cells(notebook_id)
    for cell_id in active_cells:
        cancelled_cells.add(cell_id)
        event = prompt_cancel_events.get(f"{notebook_id}:{cell_id}")
        if event:
            event.set()
    return active_cells


def _normalized_llm_usage(usage: Optional[dict]) -> dict[str, int]:
    usage = usage or {}
    details = usage.get("input_tokens_details") or {}
    cached = usage.get("cached_tokens", details.get("cached_tokens", 0))
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "cached_tokens": int(cached or 0),
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def _merge_llm_usage(total: dict[str, int], usage: Optional[dict]) -> None:
    normalized = _normalized_llm_usage(usage)
    for key, value in normalized.items():
        total[key] = int(total.get(key) or 0) + value


def _notebook_llm_usage(notebook) -> dict[str, int]:
    total = _normalized_llm_usage({})
    for cell in notebook.cells:
        execution = getattr(cell, "llm_execution", None) or {}
        _merge_llm_usage(total, execution.get("usage"))
    return total


def _classify_llm_failure(message: str) -> dict[str, Any]:
    lowered = message.lower()
    if any(token in lowered for token in ("401", "authentication", "authenticate", "expired", "unauthorized")):
        return {"category": "authentication", "action": "connect", "label": "Connect OpenAI", "retryable": False}
    if (
        any(token in lowered for token in ("entitlement", "model_not_found", "does not have access", "not available for provider"))
        or ("model" in lowered and any(token in lowered for token in ("not found", "no access", "unsupported")))
    ):
        return {"category": "model_entitlement", "action": "choose_model", "label": "Choose Model", "retryable": False}
    if any(token in lowered for token in ("429", "rate limit", "quota")):
        return {"category": "rate_limit", "action": "retry", "label": "Retry", "retryable": True}
    if any(token in lowered for token in ("malformed", "invalid tool", "tool arguments", "json decode")):
        return {"category": "malformed_tool_call", "action": "retry", "label": "Retry", "retryable": True}
    if any(token in lowered for token in ("timeout", "timed out", "network", "connection", "socket", "transport")):
        return {"category": "network", "action": "retry", "label": "Retry", "retryable": True}
    if any(token in lowered for token in ("500", "502", "503", "504", "server error")):
        return {"category": "server", "action": "retry", "label": "Retry", "retryable": True}
    return {"category": "provider", "action": "retry", "label": "Retry", "retryable": True}


def serialized_notebook_mutation(handler):
    """Serialize a route mutation and make supplied operation IDs idempotent."""
    signature = inspect.signature(handler)

    @wraps(handler)
    async def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        notebook_id = (
            bound.arguments.get("nb_id")
            or bound.arguments.get("dlg_name")
            or bound.arguments.get("name")
        )
        expected_revision = bound.arguments.get("expected_revision", -1)
        expected_structure_revision = bound.arguments.get("expected_structure_revision", -1)
        operation_id = bound.arguments.get("operation_id") or None
        if not notebook_id:
            return await handler(*args, **kwargs)
        if _active_prompt_cells(notebook_id):
            return _prompt_operation_conflict(notebook_id, handler.__name__)
        try:
            async with runtime_registry.mutation(
                notebook_id,
                expected_revision=expected_revision,
                expected_structure_revision=expected_structure_revision,
                operation_id=operation_id,
            ) as lease:
                if lease.duplicate:
                    performance_monitor.increment("notebook.mutations.duplicate")
                    return lease.previous_result if lease.previous_result is not None else ""
                result = await handler(*args, **kwargs)
                lease.result = result
                return result
        except RuntimeRevisionConflict as conflict:
            performance_monitor.increment("notebook.mutations.revision_conflict")
            return JSONResponse({
                "status": "conflict",
                "type": "notebook_revision_conflict",
                "revision_domain": conflict.domain,
                "notebook_id": conflict.notebook_id,
                "expected_revision": conflict.expected,
                "actual_revision": conflict.actual,
            }, status_code=409)

    return wrapped

# DialogHelper data queues for bidirectional browser communication
# Structure: {notebook_id: {data_id: asyncio.Queue}}
data_queues = runtime_registry.field_map("data_queues")


def _notebook_is_idle(notebook_id: str) -> bool:
    queue = execution_queues.get(notebook_id)
    queue_idle = True
    if queue is not None:
        status = queue.get_status(notebook_id)
        queue_idle = not status.current_cell_id and not status.queued_cell_ids
    prompt_active = any(key.startswith(f"{notebook_id}:") for key in prompt_cancel_events)
    return queue_idle and not prompt_active and notebook_id not in kernel_setup_state


async def _on_notebook_autosaved(notebook_id: str, notebook: Notebook, path: Path) -> None:
    try:
        from dialeng.services.lib_export_service import maybe_extract
        result = maybe_extract(path, root_dir=NOTEBOOKS_DIR)
        if result:
            _schedule_colab_lib_sync(notebook_id, reason="autosave")
    except Exception:
        logger.exception("Autosave lib export failed (notebook=%s)", notebook_id)
    await broadcast_json(notebook_id, {"type": "notebook_saved", "autosave": True})


autosave_manager = AutosaveManager(
    lambda: NOTEBOOKS_DIR,
    _notebook_is_idle,
    monitor=performance_monitor,
    on_saved=_on_notebook_autosaved,
)


def _autosave_settings() -> AutosaveSettings:
    return AutosaveSettings(
        enabled=DIALENG_CONFIG.autosave_enabled,
        debounce_ms=DIALENG_CONFIG.autosave_debounce_ms,
        idle_only=DIALENG_CONFIG.autosave_idle_only,
        recovery_snapshots=DIALENG_CONFIG.autosave_recovery_snapshots,
        max_recovery_files=DIALENG_CONFIG.autosave_max_recovery_files,
    )

def get_data_queue(dlg_name: str, data_id: str) -> asyncio.Queue:
    """Get or create a data queue for dialoghelper push/pop operations."""
    if dlg_name not in data_queues:
        data_queues[dlg_name] = {}
    if data_id not in data_queues[dlg_name]:
        data_queues[dlg_name][data_id] = asyncio.Queue()
    return data_queues[dlg_name][data_id]


def _current_kernel_generation(nb_id: str) -> int:
    """Return the current background-work generation for a notebook."""
    return kernel_generations.get(nb_id, 0)


def _kernel_runtime_context(nb_id: str) -> tuple[str, Optional[str]]:
    """Return the kernel type and runtime id for logging."""
    if not kernel_service.has_kernel(nb_id):
        nb = notebooks.get(nb_id)
        return getattr(nb, "kernel_type", "local"), None
    kernel = kernel_service.get_kernel(nb_id)
    status = kernel.get_status()
    return status.kernel_type, getattr(status, "runtime_id", None)


def _cancel_task_if_running(task_map: Dict[str, asyncio.Task], nb_id: str, label: str, reason: str) -> None:
    """Cancel a running background task for a notebook."""
    task = task_map.pop(nb_id, None)
    if task and not task.done():
        logger.info("Cancelling %s task (notebook=%s, reason=%s, task=%s)", label, nb_id, reason, task.get_name())
        task.cancel()


async def _invalidate_kernel_background_work(
    nb_id: str,
    reason: str,
    *,
    broadcast_snapshot: bool = True,
) -> int:
    """Bump notebook generation, cancel stale setup/sync tasks, and clear state."""
    generation = _current_kernel_generation(nb_id) + 1
    kernel_generations[nb_id] = generation
    _cancel_task_if_running(kernel_setup_tasks, nb_id, "kernel_setup", reason)
    _cancel_task_if_running(kernel_sync_tasks, nb_id, "kernel_sync", reason)
    if kernel_setup_state.pop(nb_id, None) is not None:
        logger.info("Cleared kernel setup state after invalidation (notebook=%s, reason=%s, generation=%s)", nb_id, reason, generation)
    if broadcast_snapshot:
        await broadcast_kernel_snapshot(nb_id)
    return generation


async def _teardown_notebook_runtime(
    nb_id: str,
    reason: str,
    *,
    broadcast_snapshot: bool = True,
    drop_connections: bool = True,
) -> None:
    """Cancel notebook work, release kernel resources, and clear ephemeral state."""
    generation = await _invalidate_kernel_background_work(
        nb_id,
        reason=f"teardown:{reason}",
        broadcast_snapshot=broadcast_snapshot,
    )
    queue = execution_queues.pop(nb_id, None)
    if queue:
        try:
            queue.cancel_all(nb_id)
        except Exception:
            logger.exception("Failed to cancel execution queue during notebook teardown (notebook=%s)", nb_id)
    if kernel_service.has_kernel(nb_id):
        await kernel_service.shutdown_async(nb_id)
        await event_bus.emit("kernel.stopped", {
            "notebook_id": nb_id,
            "reason": reason,
        })
    lifecycle = LifecycleState.DELETED if "delete" in reason else LifecycleState.STOPPED
    runtime_registry.clear_runtime(nb_id, lifecycle=lifecycle)
    outline_cache.invalidate_notebook(nb_id)
    if drop_connections:
        broadcast_manager.drop_notebook(nb_id)
    logger.info(
        "Notebook runtime torn down (notebook=%s, reason=%s, invalidated_generation=%s)",
        nb_id,
        reason,
        generation,
    )


async def _shutdown_server_runtime(reason: str = "server_shutdown") -> None:
    """Cancel notebook work and shutdown all kernels during server exit."""
    global server_shutdown_started
    if server_shutdown_started:
        return
    server_shutdown_started = True

    logger.info("Starting Dialeng shutdown cleanup (reason=%s)", reason)

    pending_tasks: list[asyncio.Task] = []
    for task_map, label in (
        (kernel_setup_tasks, "kernel_setup"),
        (kernel_sync_tasks, "kernel_sync"),
    ):
        for nb_id, task in list(task_map.items()):
            if task and not task.done():
                logger.info(
                    "Cancelling %s task during shutdown (notebook=%s, task=%s)",
                    label,
                    nb_id,
                    task.get_name(),
                )
                task.cancel()
                pending_tasks.append(task)
        task_map.clear()

    if pending_tasks:
        results = await asyncio.gather(*pending_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.warning("Background task raised during shutdown cleanup: %s", result)

    for nb_id, queue in list(execution_queues.items()):
        try:
            queue.cancel_all(nb_id)
        except Exception as e:
            logger.warning("Failed to cancel execution queue during shutdown (notebook=%s): %s", nb_id, e)
    execution_queues.clear()

    try:
        await kernel_service.shutdown_all_async()
    finally:
        kernel_setup_state.clear()
        kernel_generations.clear()
        notebook_revisions.clear()
        structure_revisions.clear()
        outline_revisions.clear()
        event_stream_revisions.clear()
        data_queues.clear()
        logger.info("Dialeng shutdown cleanup complete")


def _assert_kernel_generation_current(nb_id: str, generation: int, source: str) -> None:
    """Abort stale setup work once a newer kernel generation exists."""
    current = _current_kernel_generation(nb_id)
    if generation != current:
        raise asyncio.CancelledError(
            f"Stale kernel setup work ignored for notebook {nb_id}: source={source}, generation={generation}, current={current}"
        )


def _get_auth_snapshot() -> dict:
    """Return the current Colab auth status in a JSON-safe structure."""
    if not colab_auth_service:
        enabled = bool(getattr(DIALENG_CONFIG, "colab_enabled", False))
        return {
            "enabled": enabled,
            "authenticated": False,
            "has_tokens": False,
            "email": None,
            "validation_error": None,
            "validation_state": remote_service_state["colab"],
            "auth_provider": getattr(DIALENG_CONFIG, "colab_auth_provider", "oauth_local"),
        }
    status = colab_auth_service.get_status()
    status["validation_state"] = remote_service_state["colab"]
    return status


def _get_llm_status_snapshot(nb: Optional[Notebook] = None) -> dict:
    """Return provider/model availability without exposing credentials."""
    selected_provider = getattr(nb, "llm_provider", None) or DEFAULT_LLM_PROVIDER
    selected_model = getattr(nb, "model", None) or DIALENG_CONFIG.get_default_model(selected_provider)
    providers = []
    for status in CREDENTIAL_INVENTORY.providers.values():
        if status.id == "mock_only":
            continue
        if (
            status.authenticated
            or status.validation_state == "validating"
            or status.id == selected_provider
        ):
            providers.append(status.to_dict())
    selected = CREDENTIAL_INVENTORY.get(selected_provider)
    return {
        "available": CREDENTIAL_INVENTORY.available,
        "default_provider": CREDENTIAL_INVENTORY.default_provider(),
        "selected_provider": selected_provider,
        "selected_model": selected_model,
        "selection_available": bool(selected and selected.authenticated),
        "validation_state": remote_service_state["providers"],
        "providers": providers,
        "openai_auth": openai_codex_auth_service.status(),
    }


def _get_queue_payload(nb_id: str) -> dict:
    """Return queue status without creating a queue object unnecessarily."""
    if nb_id in execution_queues:
        status = execution_queues[nb_id].get_status(nb_id)
        return {
            "running_cell_id": status.current_cell_id,
            "queued_cell_ids": list(status.queued_cell_ids),
            "queued_count": status.queued_count,
            "is_processing": status.is_processing,
        }
    return {
        "running_cell_id": None,
        "queued_cell_ids": [],
        "queued_count": 0,
        "is_processing": False,
    }


def _build_notebook_snapshot(nb_id: str) -> dict:
    """Return notebook-level state that should stay consistent across tabs."""
    nb = get_notebook(nb_id)
    return {
        "title": nb.title,
        "revision": _get_notebook_revision(nb_id),
        "structure_revision": _get_structure_revision(nb_id),
        "ordered_cell_ids": _ordered_cell_ids(nb),
        "cell_versions": {cell.id: cell.version for cell in nb.cells},
        "dialog_mode": nb.dialog_mode,
        "model": nb.model,
        "llm_provider": nb.llm_provider,
        "safe_mode": nb.safe_mode,
        "kernel_type": getattr(nb, "kernel_type", "local"),
        "colab_runtime_type": getattr(nb, "colab_runtime_type", "cpu"),
    }


def _get_notebook_revision(nb_id: str) -> int:
    """Return the current runtime notebook revision."""
    return notebook_revisions.get(nb_id, 0)


def _get_structure_revision(nb_id: str) -> int:
    """Return the current token for ordered cell structure mutations."""
    return structure_revisions.get(nb_id, 0)


def _touch_notebook_revision(nb_id: str, reason: str) -> int:
    """Bump runtime revision for backend-authoritative notebook mutations."""
    revision = runtime_registry.next_notebook_revision(nb_id)
    logger.info(
        "Notebook revision advanced (notebook=%s, revision=%s, reason=%s)",
        nb_id,
        revision,
        reason,
    )
    notebook = notebooks.get(nb_id)
    if notebook is not None:
        autosave_manager.schedule(nb_id, notebook, _autosave_settings())
    return revision


def _touch_structure_revision(nb_id: str, reason: str) -> int:
    """Advance cell-order state and the durable notebook event stream together."""
    structure_revision = runtime_registry.next_structure_revision(nb_id)
    _touch_notebook_revision(nb_id, reason)
    logger.info(
        "Notebook structure revision advanced (notebook=%s, revision=%s, reason=%s)",
        nb_id,
        structure_revision,
        reason,
    )
    return structure_revision


def _expected_version_conflicts(cell, expected_version: int = -1) -> bool:
    return expected_version is not None and expected_version >= 0 and expected_version != cell.version


def _get_outline_revision(nb_id: str) -> int:
    """Return the current outline revision for a notebook."""
    return outline_revisions.get(nb_id, 0)


def _get_stream_revision(nb_id: str, stream: str) -> int:
    if stream == "outline":
        return _get_outline_revision(nb_id)
    return event_stream_revisions.get(nb_id, {}).get(stream, 0)


def _next_stream_revision(nb_id: str, stream: str) -> int:
    return runtime_registry.next_stream_revision(nb_id, stream)


def _event_stream(event_type: str) -> str:
    if event_type == "notebook_saved":
        return "persistence"
    if event_type == "llm_status":
        return "llm"
    if event_type.startswith("cell_") or event_type == "prompt_output_update":
        return "notebook"
    if event_type in {
        "stream_chunk", "stream_end", "prompt_stream_start", "prompt_run_terminal", "prompt_execution_update", "thinking_start", "thinking_end",
        "code_stream_start", "code_stream_chunk", "code_stream_end", "code_display_data",
        "code_update_display", "code_clear_output", "var_substituted", "tool_available",
        "tool_call", "tool_result", "llm_usage", "tool_confirmation_request",
    }:
        return "output"
    return "kernel"


def _protocol_event(nb_id: str, data: dict) -> dict:
    """Attach a monotonic stream revision and deduplication ID to an event."""
    payload = dict(data)
    event_type = str(payload.get("type") or "event")
    requested_stream = payload.get("stream")
    if event_type == "code_stream_chunk" and requested_stream in {"stdout", "stderr"}:
        # Jupyter's stdout/stderr channel is payload data, not a protocol
        # revision stream. Keeping both in `stream` caused false gaps after a
        # page reload because clients never snapshot a `stdout` revision.
        payload["output_stream"] = requested_stream
        stream = "output"
    else:
        stream = str(requested_stream or _event_stream(event_type))
    payload["stream"] = stream
    payload.setdefault("notebook_id", nb_id)
    payload.setdefault("stream", stream)
    payload.setdefault("event_id", uuid.uuid4().hex)
    if stream == "notebook":
        payload.setdefault("notebook_revision", _get_notebook_revision(nb_id))
    if "revision" not in payload:
        if stream == "outline":
            payload["revision"] = _get_outline_revision(nb_id)
        else:
            payload["revision"] = _next_stream_revision(nb_id, stream)
    if event_type in {"kernel_snapshot", "queue_update", "llm_status"}:
        payload["authoritative"] = True
    return payload


def _parse_kernel_connection_state(raw_state: Optional[str]) -> tuple[str, Optional[str]]:
    """Split a kernel connection_state into machine state + detail."""
    if not raw_state:
        return "disconnected", None
    if raw_state.startswith("initializing:"):
        return "initializing", raw_state.split(":", 1)[1].strip()
    return raw_state, None


def _get_revision_snapshot(nb_id: str) -> dict[str, int]:
    """Return the current revision baseline for every client event stream."""
    return {
        "notebook": _get_stream_revision(nb_id, "notebook"),
        "kernel": _get_stream_revision(nb_id, "kernel"),
        "outline": _get_outline_revision(nb_id),
        "llm": _get_stream_revision(nb_id, "llm"),
        "output": _get_stream_revision(nb_id, "output"),
        "persistence": _get_stream_revision(nb_id, "persistence"),
    }


def _build_kernel_snapshot(nb_id: str) -> dict:
    """Build the backend-authoritative notebook/kernel snapshot for the UI."""
    nb = get_notebook(nb_id)
    queue_payload = _get_queue_payload(nb_id)
    auth_payload = _get_auth_snapshot()
    setup_payload = kernel_setup_state.get(nb_id, {})

    kernel = kernel_service._kernels.get(nb_id)
    kernel_status = kernel.get_status() if kernel else None
    connection_state, connection_detail = _parse_kernel_connection_state(
        kernel_status.connection_state if kernel_status else None
    )

    selected_kernel_type = getattr(nb, "kernel_type", "local")
    kernel_exists = kernel is not None
    kernel_is_alive = bool(kernel_status.is_alive) if kernel_status else False
    kernel_is_busy = bool(kernel_status.is_busy) if kernel_status else False
    setup_active = bool(setup_payload.get("is_active"))
    auth_required = selected_kernel_type == "colab" and not auth_payload["authenticated"]
    has_queue_work = bool(queue_payload["running_cell_id"] or queue_payload["queued_cell_ids"])

    if auth_required:
        display_state = "auth_required"
    elif setup_active:
        display_state = "restarting" if setup_payload.get("source") == "kernel_restart" else "initializing"
    elif has_queue_work or kernel_is_busy:
        display_state = "busy"
    elif connection_state == "initializing":
        display_state = "initializing"
    elif connection_state == "connecting":
        display_state = "connecting"
    elif connection_state == "degraded":
        display_state = "degraded"
    elif kernel_exists and kernel_is_alive:
        display_state = "connected"
    else:
        display_state = "disconnected"

    can_run = (
        kernel_exists
        and not auth_required
        and not setup_active
        and kernel_is_alive
        and connection_state not in {"connecting", "initializing"}
    )

    revisions = _get_revision_snapshot(nb_id)
    return {
        "type": "kernel_snapshot",
        "notebook_id": nb_id,
        "stream": "kernel",
        "revision": revisions["kernel"],
        "event_id": uuid.uuid4().hex,
        "authoritative": True,
        "revisions": revisions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notebook": _build_notebook_snapshot(nb_id),
        "kernel": {
            "selected_type": selected_kernel_type,
            "exists": kernel_exists,
            "is_alive": kernel_is_alive,
            "is_busy": kernel_is_busy,
            "execution_count": kernel_status.execution_count if kernel_status else 0,
            "runtime_id": kernel_status.runtime_id if kernel_status else None,
            "kernel_id": kernel_status.kernel_id if kernel_status else None,
            "jupyter_session_id": kernel_status.jupyter_session_id if kernel_status else None,
            "token_expires_at": kernel_status.token_expires_at if kernel_status else None,
            "last_activity_at": kernel_status.last_activity_at if kernel_status else None,
            "connection_state": connection_state,
            "connection_detail": connection_detail,
            "display_state": display_state,
            "can_run": can_run,
            "auth_required": auth_required,
        },
        "queue": queue_payload,
        "setup": {
            "is_active": setup_active,
            "source": setup_payload.get("source"),
            "phase": setup_payload.get("phase"),
            "detail": setup_payload.get("detail"),
        },
        "outline": {
            "revision": _get_outline_revision(nb_id),
        },
        "auth": auth_payload,
        "llm": _get_llm_status_snapshot(nb),
    }


async def send_kernel_snapshot(nb_id: str, send) -> None:
    """Send the latest notebook/kernel snapshot to a single WebSocket client."""
    await send(json.dumps(_build_kernel_snapshot(nb_id)))


async def broadcast_kernel_snapshot(nb_id: str):
    """Broadcast the latest notebook/kernel snapshot to all clients."""
    _next_stream_revision(nb_id, "kernel")
    await broadcast_manager.broadcast_json(nb_id, _build_kernel_snapshot(nb_id))


async def broadcast_all_kernel_snapshots():
    """Broadcast refreshed kernel snapshots for every notebook with clients."""
    for nb_id in broadcast_manager.notebook_ids():
        await broadcast_kernel_snapshot(nb_id)


async def broadcast_llm_status(nb_id: str) -> None:
    """Broadcast provider/model inventory independently of kernel state."""
    payload = {"type": "llm_status", **_get_llm_status_snapshot(get_notebook(nb_id))}
    await broadcast_manager.broadcast_json(nb_id, _protocol_event(nb_id, payload))


async def broadcast_all_llm_statuses() -> None:
    for nb_id in broadcast_manager.notebook_ids():
        await broadcast_llm_status(nb_id)


async def _mark_outline_dirty(nb_id: str, reason: str, *, broadcast_snapshot: bool = True) -> None:
    """Bump the outline revision when server state changes can affect it."""
    revision = runtime_registry.next_outline_revision(nb_id)
    logger.info(
        "Outline marked dirty (notebook=%s, revision=%s, reason=%s)",
        nb_id,
        revision,
        reason,
    )
    if broadcast_snapshot:
        await broadcast_kernel_snapshot(nb_id)


async def _mark_all_outlines_dirty(reason: str) -> None:
    """Bump outline revisions for all active notebooks, then broadcast snapshots once."""
    notebook_ids = set(notebooks.keys()) | set(broadcast_manager.notebook_ids()) | set(kernel_service._kernels.keys())
    for nb_id in notebook_ids:
        revision = runtime_registry.next_outline_revision(nb_id)
        logger.info(
            "Outline marked dirty (notebook=%s, revision=%s, reason=%s)",
            nb_id,
            revision,
            reason,
        )
    if notebook_ids:
        await broadcast_all_kernel_snapshots()


async def _set_kernel_setup_state(
    nb_id: str,
    *,
    source: str,
    phase: str,
    detail: str = "",
    is_active: bool = True,
    generation: Optional[int] = None,
) -> None:
    """Update tracked notebook setup state and push a fresh kernel snapshot."""
    kernel_setup_state[nb_id] = {
        "source": source,
        "phase": phase,
        "detail": detail,
        "is_active": is_active,
        "generation": generation if generation is not None else _current_kernel_generation(nb_id),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Kernel setup state updated (notebook=%s, source=%s, phase=%s, active=%s, generation=%s, detail=%s)",
        nb_id, source, phase, is_active, kernel_setup_state[nb_id]["generation"], detail,
    )
    await broadcast_kernel_snapshot(nb_id)


async def _clear_kernel_setup_state(nb_id: str, *, expected_generation: Optional[int] = None) -> None:
    """Clear tracked notebook setup state and broadcast the fresh snapshot."""
    current = kernel_setup_state.get(nb_id)
    if current and expected_generation is not None and current.get("generation") != expected_generation:
        logger.info(
            "Skipping kernel setup state clear for stale generation (notebook=%s, expected_generation=%s, current_generation=%s)",
            nb_id, expected_generation, current.get("generation"),
        )
        return
    if kernel_setup_state.pop(nb_id, None) is not None:
        logger.info("Kernel setup state cleared for notebook %s", nb_id)
        await broadcast_kernel_snapshot(nb_id)

def _load_notebook(path: str) -> Notebook:
    """Load a notebook with app-level credential/model validation."""
    start = time.perf_counter()
    try:
        nb = Notebook.load(str(path),
                           default_dialog_mode=DEFAULT_DIALOG_MODE)
    except Exception as exc:
        performance_monitor.observe_duration(
            "notebook.load",
            elapsed_ms(start),
            labels={"outcome": "error"},
            error_type=type(exc).__name__,
        )
        raise
    # Notebooks without provider metadata use the current authenticated default.
    if not nb.llm_provider:
        nb.llm_provider = CREDENTIAL_INVENTORY.default_provider()
        nb.modified = True
    selected = CREDENTIAL_INVENTORY.get(nb.llm_provider)
    if nb.llm_provider == "mock_only" and CREDENTIAL_INVENTORY.available:
        nb.llm_provider = CREDENTIAL_INVENTORY.default_provider()
        selected = CREDENTIAL_INVENTORY.get(nb.llm_provider)
        nb.modified = True
    # Only force mock when there is no authenticated real provider at all.
    # Persisted unavailable selections remain visible and are rejected at run.
    if not CREDENTIAL_INVENTORY.available:
        nb.dialog_mode = "mock"
    if selected and (selected.authenticated or not nb.model):
        validated_model = DIALENG_CONFIG.validate_model(nb.llm_provider, nb.model)
        if validated_model != nb.model:
            nb.model = validated_model
            nb.modified = True
    performance_monitor.observe_duration(
        "notebook.load",
        elapsed_ms(start),
        labels={"outcome": "success"},
        cell_count=len(nb.cells),
        file_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
    )
    return nb


def _nb_id_encode_part(part: str) -> str:
    """Escape tildes in a single path component: ~ → ~~"""
    return _shared_nb_id_encode_part(part)

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
    return _shared_nb_id_from_path(path, NOTEBOOKS_DIR)

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
    return _shared_nb_id_to_relpath(notebook_id)

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
    if nb_id in deleting_notebooks or nb_id in deleted_notebooks:
        return None
    # Load into memory if not already
    if nb_id not in notebooks:
        nb = _load_notebook(target)
        nb.id = nb_id  # Ensure ID matches the URL-safe key, not just the filename stem
        notebooks[nb_id] = nb
    return nb_id

def _assert_notebook_available(notebook_id: str) -> None:
    """Reject stale requests for a deleting or deleted notebook."""
    if notebook_id in deleting_notebooks or notebook_id in deleted_notebooks:
        raise HTTPException(
            status_code=410,
            detail=f"Notebook '{notebook_id}' was deleted.",
        )


def get_notebook(notebook_id: str) -> Notebook:
    """Get or create a notebook - ALWAYS requires notebook_id"""
    _assert_notebook_available(notebook_id)
    if notebook_id not in notebooks:
        path = _find_notebook_path(notebook_id)
        if path:
            nb = _load_notebook(path)
            nb.id = notebook_id  # Ensure ID matches the URL-safe key
            notebooks[notebook_id] = nb
        else:
            nb = Notebook(id=notebook_id, title=notebook_id,
                         dialog_mode=DEFAULT_DIALOG_MODE, model=DEFAULT_MODEL,
                         llm_provider=DEFAULT_LLM_PROVIDER,
                         colab_runtime_type=DIALENG_CONFIG.colab_default_runtime)
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

def save_notebook(notebook_id: str, *, overwrite_external: bool = False):
    _assert_notebook_available(notebook_id)
    if notebook_id in notebooks:
        nb = notebooks[notebook_id]
        # Use existing path if set (preserves subdirectory location), else reconstruct from ID
        path = nb.path if nb.path else NOTEBOOKS_DIR / f"{_nb_id_to_relpath(notebook_id)}.ipynb"
        start = time.perf_counter()
        autosave_manager.cancel_pending(notebook_id)
        try:
            nb.save(
                str(path),
                atomic=DIALENG_CONFIG.persistence_atomic_explicit_save,
                overwrite_external=overwrite_external,
            )
        except Exception as exc:
            performance_monitor.observe_duration(
                "notebook.save",
                elapsed_ms(start),
                labels={"mode": "explicit", "outcome": "error"},
                cell_count=len(nb.cells),
                error_type=type(exc).__name__,
            )
            raise
        performance_monitor.observe_duration(
            "notebook.save",
            elapsed_ms(start),
            labels={"mode": "explicit", "outcome": "success"},
            cell_count=len(nb.cells),
            file_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
        )
        # Auto-extract #| export cells to lib dir if notebook has #| default_exp
        try:
            from dialeng.services.lib_export_service import maybe_extract
            result = maybe_extract(Path(path), root_dir=NOTEBOOKS_DIR)
            if result:
                logger.info(
                    "lib export completed (notebook=%s, module=%s, cells_exported=%s)",
                    notebook_id, result['module'], result['cells_exported'],
                )
                _schedule_colab_lib_sync(notebook_id, reason="save")
        except Exception as e:
            logger.warning("lib export failed for %s: %s", notebook_id, e)

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
    # Convert component to HTML string using to_xml
    # FastHTML's str() on components returns the ID, not HTML
    html_str = to_xml(component)
    await broadcast_manager.broadcast_text(nb_id, html_str, exclude_send=exclude_send)


async def broadcast_json(nb_id: str, data: dict, exclude_send=None):
    """Broadcast a JSON message to all WebSocket connections for a notebook.

    Used for lightweight updates (source changes, class updates) that don't
    need full HTML OOB swaps. The client-side WebSocket handler processes
    these by type.
    """
    await broadcast_manager.broadcast_json(
        nb_id,
        _protocol_event(nb_id, data),
        exclude_send=exclude_send,
    )


async def broadcast_all_json(data: dict):
    """Broadcast a JSON message to ALL WebSocket connections across all notebooks."""
    await broadcast_manager.broadcast_all_json(data)


async def _extension_notify(message: str, *, tone: str = "info", notebook_id: str = "") -> None:
    payload = {"type": "status_message", "message": message, "tone": tone}
    if notebook_id:
        await broadcast_json(notebook_id, payload)
    else:
        await broadcast_all_json(payload)


extension_manager.context_services["notify"] = _extension_notify


def _ordered_cell_ids(nb) -> list[str]:
    """Return the notebook's canonical cell order."""
    return [cell.id for cell in nb.cells]


def _cell_add_payload(nb, nb_id: str, cell_id: str) -> dict:
    """Build a structural add payload with canonical notebook order."""
    cell = nb.get_cell(cell_id)
    idx = nb.get_cell_index(cell_id)
    if cell is None or idx < 0:
        raise ValueError(f"Cell {cell_id} not found in notebook {nb_id}")
    cell_html = to_xml(CellView(cell, nb_id))
    add_html = to_xml(AddButtons(idx + 1, nb_id))
    return {
        "type": "cell_add",
        "cell_id": cell.id,
        "version": cell.version,
        "pos": idx,
        "html": cell_html + add_html,
        "ordered_cell_ids": _ordered_cell_ids(nb),
        "notebook_revision": _get_notebook_revision(nb_id),
        "structure_revision": _get_structure_revision(nb_id),
        "cell_versions": {c.id: c.version for c in nb.cells},
    }


def _cell_delete_payload(nb, nb_id: str, cell_id: str) -> dict:
    """Build a structural delete payload with canonical notebook order."""
    return {
        "type": "cell_delete",
        "cell_id": cell_id,
        "ordered_cell_ids": _ordered_cell_ids(nb),
        "notebook_revision": _get_notebook_revision(nb_id),
        "structure_revision": _get_structure_revision(nb_id),
        "cell_versions": {c.id: c.version for c in nb.cells},
    }


def _cell_move_payload(nb, nb_id: str, cell_id: str) -> dict:
    """Build a structural move payload with canonical notebook order."""
    return {
        "type": "cell_move",
        "cell_id": cell_id,
        "ordered_cell_ids": _ordered_cell_ids(nb),
        "notebook_revision": _get_notebook_revision(nb_id),
        "structure_revision": _get_structure_revision(nb_id),
        "cell_versions": {c.id: c.version for c in nb.cells},
    }


def _cell_type_payload(nb, nb_id: str, cell_id: str) -> dict:
    """Build a revisioned full-cell replacement after changing cell type."""
    cell = nb.get_cell(cell_id)
    if cell is None:
        raise ValueError(f"Cell {cell_id} not found in notebook {nb_id}")
    return {
        "type": "cell_type_change",
        "cell_id": cell.id,
        "cell_type": str(cell.cell_type),
        "version": cell.version,
        "html": to_xml(CellView(cell, nb_id)),
        "notebook_revision": _get_notebook_revision(nb_id),
    }


def _cell_source_payload(nb_id: str, cell) -> dict:
    """Build a source sync payload with cell and notebook revisions."""
    return {
        "type": "cell_source_update",
        "cell_id": cell.id,
        "cell_type": str(cell.cell_type),
        "source": cell.source,
        "output": cell.output if str(cell.cell_type) == "prompt" else None,
        "execution_count": cell.execution_count,
        "time_run": cell.time_run,
        "version": cell.version,
        "notebook_revision": _get_notebook_revision(nb_id),
    }


def _cell_output_commit_payload(nb_id: str, cell) -> dict:
    """Advance browser revisions for persisted output-only mutations."""
    return {
        "type": "cell_output_commit",
        "cell_id": cell.id,
        "version": cell.version,
        "notebook_revision": _get_notebook_revision(nb_id),
    }


async def _broadcast_cell_version_conflict(
    nb_id: str,
    cell,
    expected_version: int,
    *,
    field: str = "source",
) -> None:
    await broadcast_json(nb_id, {
        "type": "cell_conflict",
        "cell_id": cell.id,
        "conflict_field": field,
        "expected_version": expected_version,
        "actual_version": cell.version,
        "source": cell.source,
        "output": cell.output,
        "version": cell.version,
        "notebook_revision": _get_notebook_revision(nb_id),
    })


def _commit_cell_source(nb_id: str, nb, cell, source: str, *, reason: str) -> bool:
    changed = cell.update_source(source)
    if changed:
        nb.modified = True
        _touch_notebook_revision(nb_id, reason)
    return changed


def _commit_cell_output(nb_id: str, nb, cell, output: str, *, reason: str) -> bool:
    changed = cell.update_output(output)
    if changed:
        nb.modified = True
        _touch_notebook_revision(nb_id, reason)
    return changed


def _prompt_output_payload(nb_id: str, cell) -> dict:
    """Build a canonical prompt output sync payload."""
    return {
        "type": "prompt_output_update",
        "cell_id": cell.id,
        "output": cell.output,
        "version": cell.version,
        "notebook_revision": _get_notebook_revision(nb_id),
        "time_run": cell.time_run,
    }


async def _extension_change_cell_type(nb_id: str, cell_id: str, cell_type: str) -> dict:
    """Public extension mutation adapter with normal revisions and broadcasts."""
    allowed = {value for value, _label in registry.get_cell_type_choices()}
    if cell_type not in allowed:
        raise ValueError(f"Unknown cell type: {cell_type}")
    async with runtime_registry.mutation(nb_id):
        nb = get_notebook(nb_id)
        cell = nb.get_cell(cell_id)
        if cell is None:
            raise KeyError(f"Unknown cell: {cell_id}")
        old_type = str(cell.cell_type)
        if old_type == cell_type:
            return {"ok": True, "changed": False, "cell_id": cell_id, "cell_type": cell_type}
        cell.cell_type = cell_type
        cell.clear_outputs()
        cell.version += 1
        cell.last_modified = datetime.now()
        nb.modified = True
        _touch_notebook_revision(nb_id, "extension_cell_type_change")
        await event_bus.emit("cell.type_changed", {
            "notebook_id": nb_id,
            "cell_id": cell_id,
            "old_type": old_type,
            "new_type": cell_type,
            "source": "extension_api",
        })
        await broadcast_json(nb_id, _cell_type_payload(nb, nb_id, cell_id))
        await _mark_outline_dirty(nb_id, reason="extension_cell_type_change")
        return {"ok": True, "changed": True, "cell_id": cell_id, "cell_type": cell_type}


extension_manager.context_services["change_cell_type"] = _extension_change_cell_type


async def broadcast_queue_state(nb_id: str):
    """Broadcast current queue state to all clients."""
    status = _get_queue_payload(nb_id)

    await broadcast_json(nb_id, {
        "type": "queue_update",
        "running_cell_id": status["running_cell_id"],
        "queued_cell_ids": status["queued_cell_ids"],
    })
    await broadcast_kernel_snapshot(nb_id)


async def broadcast_cell_state(nb_id: str, cell_id: str, state: CellState):
    """Broadcast cell state change to all clients."""
    await broadcast_json(nb_id, {
        "type": "cell_state_change",
        "cell_id": cell_id,
        "state": state.value,
    })


async def broadcast_kernel_status(nb_id: str, status: str):
    """Broadcast kernel status change (connected, busy, error, restarting) to all clients."""
    await broadcast_json(nb_id, {"type": f"kernel_{status}"})
    await broadcast_kernel_snapshot(nb_id)


async def broadcast_cell_output(nb_id: str, cell_id: str, output):
    """Broadcast cell output chunk to all clients."""
    if output.output_type == 'stream':
        msg_data = {
            "type": "code_stream_chunk",
            "cell_id": cell_id,
            "chunk": output.content,
            "stream": output.stream_name
        }
    elif output.output_type == 'execute_result':
        html_content = render_mime_bundle(
            output.content if isinstance(output.content, dict) else {"text/plain": str(output.content)},
            output.metadata,
        )
        msg_data = {
            "type": "code_display_data",
            "cell_id": cell_id,
            "html": html_content,
            "output_kind": "execute_result",
        }
    elif output.output_type == 'error':
        tb_text = '\n'.join(output.traceback or [])
        if not tb_text:
            tb_text = f"{output.ename or 'Error'}: {output.evalue or ''}".rstrip()
        msg_data = {
            "type": "code_stream_chunk",
            "cell_id": cell_id,
            "chunk": tb_text,
            "stream": "stderr"
        }
    elif output.output_type == 'display_data':
        html_content = render_mime_bundle(output.content, output.metadata)
        msg_data = {
            "type": "code_display_data",
            "cell_id": cell_id,
            "html": html_content,
            "output_kind": "display_data",
        }
        if output.display_id:
            msg_data["display_id"] = output.display_id
    elif output.output_type == 'update_display_data':
        html_content = render_mime_bundle(output.content, output.metadata)
        msg_data = {
            "type": "code_update_display",
            "cell_id": cell_id,
            "html": html_content,
            "display_id": output.display_id
        }
    elif output.output_type == 'clear_output':
        wait = output.content.get("wait", False) if isinstance(output.content, dict) else output.content
        msg_data = {
            "type": "code_clear_output",
            "cell_id": cell_id,
            "wait": bool(wait),
        }
    else:
        return  # Unknown type, skip

    await broadcast_json(nb_id, msg_data)


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


def _static_asset_url(relative_path: str) -> str:
    """Return a package static URL that changes when the asset changes."""
    asset = Path(__file__).parent / "static" / relative_path
    try:
        version = f"{asset.stat().st_mtime_ns:x}"
    except OSError:
        version = "missing"
    return f"/static/{relative_path}?v={version}"


app, rt = fast_app(
    pico=False,
    # Cell and notebook mutations must not depend on a public CDN being
    # reachable. FastHTML still supplies its other default browser helpers.
    htmx=False,
    exts='ws',
    static_path=str(Path(__file__).parent),
    hdrs=(
        Script(src=_static_asset_url("vendor/htmx-2.0.7.min.js")),
        # CSS - External stylesheets (order matters: themes first for variables)
        Link(rel="stylesheet", href=_static_asset_url("css/themes.css")),
        Link(rel="stylesheet", href=_static_asset_url("css/base.css")),
        Link(rel="stylesheet", href=_static_asset_url("css/components.css")),
        Link(rel="stylesheet", href=_static_asset_url("css/editor.css")),
        # Highlight.js for markdown code blocks
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"),
        Script(src=_static_asset_url("js/event_dispatcher.js")),
        Script(src=_static_asset_url("js/notebook_client.js")),
        Script(src=_static_asset_url("js/prompt_response_editor.js")),
        Script(src=_static_asset_url("js/structure_controller.js")),
        Script(src=_static_asset_url("js/execution_ui.js")),
        Script(src=_static_asset_url("js/notifications.js")),
        Script(src=_static_asset_url("js/browser_diagnostics.js")),
        # App JS - External JavaScript (after libraries)
        Script(src=_static_asset_url("js/app.js")),
    )
)
app.add_middleware(RequestMetricsMiddleware, monitor=performance_monitor)


async def _initialize_remote_services() -> None:
    """Validate providers and Colab in the background, then notify open tabs."""
    remote_service_state["providers"] = "validating"
    remote_service_state["colab"] = "validating" if DIALENG_CONFIG.colab_enabled else "disabled"
    performance_monitor.record_event("remote_services.begin")
    provider_task = asyncio.create_task(
        refresh_credential_inventory_async(NOTEBOOKS_DIR / ".env"),
        name="dialeng-provider-inventory",
    )
    colab_task = asyncio.create_task(_init_colab_async(), name="dialeng-colab-init")
    provider_result, colab_result = await asyncio.gather(
        provider_task,
        colab_task,
        return_exceptions=True,
    )

    if isinstance(provider_result, BaseException):
        remote_service_state["providers"] = "error"
        performance_monitor.record_event(
            "remote_services.provider_failed",
            error_type=type(provider_result).__name__,
        )
        logger.exception(
            "Provider inventory validation failed",
            exc_info=(type(provider_result), provider_result, provider_result.__traceback__),
        )
    else:
        _apply_llm_inventory(provider_result)
        remote_service_state["providers"] = "ready"

    if isinstance(colab_result, BaseException):
        remote_service_state["colab"] = "error"
        performance_monitor.record_event(
            "remote_services.colab_failed",
            error_type=type(colab_result).__name__,
        )
        logger.exception(
            "Colab background initialization failed",
            exc_info=(type(colab_result), colab_result, colab_result.__traceback__),
        )
    elif DIALENG_CONFIG.colab_enabled:
        remote_service_state["colab"] = "ready"

    await broadcast_all_llm_statuses()
    await broadcast_all_kernel_snapshots()
    performance_monitor.record_event(
        "remote_services.complete",
        provider_available=CREDENTIAL_INVENTORY.available,
        colab_ready=colab_auth_service is not None,
    )


async def _watch_project_files() -> None:
    """Invalidate affected explorer directories and notify open tabs."""
    try:
        from watchfiles import awatch
    except ImportError:
        logger.warning("watchfiles is unavailable; external explorer changes require refresh")
        return
    async for changes in awatch(NOTEBOOKS_DIR):
        affected: set[str] = set()
        for _change, raw_path in changes:
            changed = Path(raw_path)
            try:
                relative = changed.resolve().relative_to(NOTEBOOKS_DIR.resolve())
            except ValueError:
                continue
            if any(part in {".git", "__pycache__", ".ipynb_checkpoints", "recovery"} for part in relative.parts):
                continue
            parent = changed.parent
            directory_listing_cache.invalidate(parent)
            try:
                parent_relative = parent.resolve().relative_to(NOTEBOOKS_DIR.resolve())
                affected.add("" if str(parent_relative) == "." else parent_relative.as_posix())
            except ValueError:
                continue
        if affected:
            await event_bus.emit("project.files_changed", {"paths": sorted(affected)})
            await broadcast_all_json({
                "type": "file_explorer_changed",
                "paths": sorted(affected),
            })


# AUTORUN processing on startup (async, needs event loop)
async def _autorun_startup():
    # Reload config from the correct path — the Uvicorn reloader spawns a worker
    # process that re-imports this module, but set_root_dir() only runs in main().
    # Without this, DIALENG_CONFIG has DEFAULT_CONFIG (no extension settings).
    global DIALENG_CONFIG, event_loop_monitor_task, remote_service_init_task, project_file_watch_task
    performance_monitor.record_event("startup.begin")
    await event_bus.emit("app.starting", {"project_root": str(NOTEBOOKS_DIR)})
    with performance_monitor.measure("startup.config"):
        DIALENG_CONFIG = load_config(config_path=NOTEBOOKS_DIR / "dialeng_config.json", force_reload=True)
        set_skill_root(NOTEBOOKS_DIR)
    with performance_monitor.measure("startup.provider_inventory.local"):
        _load_initial_llm_runtime()
    with performance_monitor.measure("startup.extensions"):
        extension_manager.refresh()
        extension_errors = await extension_manager.activate_enabled()
        for extension_id, error in extension_errors.items():
            logger.error("Could not restore extension %s: %s", extension_id, error)
    if remote_service_init_task is None or remote_service_init_task.done():
        remote_service_init_task = asyncio.create_task(
            _initialize_remote_services(),
            name="dialeng-remote-services",
        )
    with performance_monitor.measure("startup.autorun"):
        await process_autorun(kernel_service, notebooks_dir=NOTEBOOKS_DIR)
    if event_loop_monitor_task is None or event_loop_monitor_task.done():
        event_loop_monitor_task = asyncio.create_task(
            monitor_event_loop_lag(performance_monitor),
            name="dialeng-event-loop-monitor",
        )
    if project_file_watch_task is None or project_file_watch_task.done():
        project_file_watch_task = asyncio.create_task(
            _watch_project_files(),
            name="dialeng-project-file-watch",
        )
    performance_monitor.record_event("startup.complete")
    await event_bus.emit("project.opened", {"project_root": str(NOTEBOOKS_DIR)})
    await event_bus.emit("app.ready", {"project_root": str(NOTEBOOKS_DIR)})


async def _app_shutdown():
    """Release notebook kernels and background tasks on server shutdown."""
    global event_loop_monitor_task, remote_service_init_task, project_file_watch_task
    await event_bus.emit("app.stopping", {"project_root": str(NOTEBOOKS_DIR)})
    if remote_service_init_task and not remote_service_init_task.done():
        remote_service_init_task.cancel()
        await asyncio.gather(remote_service_init_task, return_exceptions=True)
    remote_service_init_task = None
    if event_loop_monitor_task and not event_loop_monitor_task.done():
        event_loop_monitor_task.cancel()
        await asyncio.gather(event_loop_monitor_task, return_exceptions=True)
    event_loop_monitor_task = None
    if project_file_watch_task and not project_file_watch_task.done():
        project_file_watch_task.cancel()
        await asyncio.gather(project_file_watch_task, return_exceptions=True)
    project_file_watch_task = None
    for task in tuple(_settings_activation_tasks):
        if not task.done():
            task.cancel()
    if _settings_activation_tasks:
        await asyncio.gather(*tuple(_settings_activation_tasks), return_exceptions=True)
        _settings_activation_tasks.clear()
    await extension_manager.shutdown()
    await autosave_manager.flush_all(timeout_seconds=3.0)
    await _shutdown_server_runtime()


@asynccontextmanager
async def _app_lifespan(_app):
    """Own all application background work within one ASGI lifecycle."""
    await _autorun_startup()
    try:
        yield
    finally:
        await _app_shutdown()


app.router.lifespan_context = _app_lifespan


register_diagnostics_routes(
    rt,
    performance_monitor=performance_monitor,
    runtime_registry=runtime_registry,
    broadcast_manager=broadcast_manager,
    autosave_manager=autosave_manager,
    build_kernel_snapshot=_build_kernel_snapshot,
    runtime_counts=lambda: {
        "notebooks_loaded": len(notebooks),
        "kernels_attached": len(kernel_service._kernels),
        "execution_queues": len(execution_queues),
        "setup_tasks": len(kernel_setup_tasks),
        "sync_tasks": len(kernel_sync_tasks),
    },
)

register_skill_routes(
    rt,
    project_root=lambda: NOTEBOOKS_DIR,
    config_provider=lambda: DIALENG_CONFIG,
)


def _extension_command_available(expression: str, notebook_id: str) -> tuple[bool, str]:
    """Evaluate the intentionally small public command-availability grammar."""
    predicates = {
        "always": True,
        "workspace": True,
        "notebook.open": notebook_id in notebooks and notebook_id not in deleted_notebooks,
        "kernel.optional": True,
        "kernel.required": kernel_service.kernel_is_alive(notebook_id),
    }
    terms = [term.strip() for term in str(expression or "always").split("&&")]
    available = bool(terms) and all(predicates.get(term, False) for term in terms)
    if available:
        return True, ""
    if "notebook.open" in terms and not predicates["notebook.open"]:
        return False, "Open a notebook to use this command."
    if "kernel.required" in terms and not predicates["kernel.required"]:
        return False, "Attach a running kernel to use this command."
    return False, f"The command requires: {expression}."


register_extension_routes(
    rt,
    manager_provider=lambda: extension_manager,
    broadcast_all_json=broadcast_all_json,
    command_available=_extension_command_available,
)

if os.environ.get("DIALENG_E2E") == "1":
    register_e2e_routes(
        rt,
        kernel_service=kernel_service,
        broadcast_json=broadcast_json,
        broadcast_kernel_snapshot=broadcast_kernel_snapshot,
    )

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
    return RedirectResponse("/dialeng/workspace", status_code=302)


@rt("/dialeng/workspace")
def get(deleted: str = ""):
    """Render the project shell without implicitly creating a notebook."""
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return WorkspacePage(
        NOTEBOOKS_DIR,
        config=get_config(),
        kernel_notebooks=kernel_nbs,
        notebook_deleted=deleted.lower() in {"1", "true", "yes"},
    )

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
            return RedirectResponse("/dialeng/workspace", status_code=302)
        # Render the notebook page directly (same as /dialeng/{nb_id})
        return await _render_notebook_page(nb_id)
    return RedirectResponse("/dialeng/workspace", status_code=302)

@rt("/dialeng/new")
def get(dir: str = "", name: str = ""):
    display_name = name.strip() if name.strip() else uuid.uuid4().hex[:8]
    target_dir = NOTEBOOKS_DIR / dir if dir else NOTEBOOKS_DIR
    file_path = target_dir / f"{display_name}.ipynb"

    # Derive URL-safe ID from path: "subfolder/test.ipynb" → "subfolder_test"
    new_id = _nb_id_from_path(file_path)
    deleted_notebooks.pop(new_id, None)
    deleting_notebooks.discard(new_id)

    # If this notebook already exists in memory (e.g., re-creation after delete),
    # remove the stale version so we get fresh template cells
    notebooks.pop(new_id, None)

    nb = Notebook(id=new_id, title=display_name,
                  dialog_mode=DEFAULT_DIALOG_MODE, model=DEFAULT_MODEL,
                  llm_provider=DEFAULT_LLM_PROVIDER,
                  colab_runtime_type=DIALENG_CONFIG.colab_default_runtime)

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
def get(path: str = "", active_notebook_id: str = "", active_file_relpath: str = ""):
    """Get file list content for HTMX swap."""
    from dialeng.ui.file_explorer import FileListContent
    target = NOTEBOOKS_DIR / path if path else NOTEBOOKS_DIR
    # Validate path is within NOTEBOOKS_DIR
    if not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        target = NOTEBOOKS_DIR
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    return FileListContent(
        target,
        NOTEBOOKS_DIR,
        active_notebook_id,
        active_file_relpath=active_file_relpath,
        kernel_notebooks=kernel_nbs,
    )

@rt("/files/new-folder")
def post(path: str = "", name: str = "", active_notebook_id: str = "", active_file_relpath: str = ""):
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
    return FileListContent(
        target,
        NOTEBOOKS_DIR,
        active_notebook_id,
        active_file_relpath=active_file_relpath,
        kernel_notebooks=kernel_nbs,
    )

@rt("/files/delete")
async def post(path: str = "", active_notebook_id: str = "", active_file_relpath: str = ""):
    """Delete a file and return explorer HTML plus deletion metadata."""
    from dialeng.ui.file_explorer import FileListContent
    if not path or '..' in path:
        return JSONResponse({"ok": False, "message": "Invalid path"}, status_code=400)
    target = NOTEBOOKS_DIR / path
    if not target.resolve().is_relative_to(NOTEBOOKS_DIR.resolve()):
        return JSONResponse({"ok": False, "message": "Invalid path"}, status_code=400)
    deleted_nb_id = None
    if target.exists() and target.is_file():
        if target.suffix == ".ipynb":
            nb_id = _nb_id_from_path(target)
            before_delete = await event_bus.emit("notebook.before_delete", {
                "notebook_id": nb_id,
                "path": path,
            })
            if before_delete.blocked:
                return JSONResponse({
                    "ok": False,
                    "type": "extension_guard",
                    "message": before_delete.reason or "Notebook deletion blocked by extension",
                }, status_code=409)
            cancelled_prompts = _cancel_active_prompts(nb_id)
            if cancelled_prompts:
                logger.info(
                    "Cancelling active prompts before notebook deletion (notebook=%s, cells=%s)",
                    nb_id,
                    ",".join(cancelled_prompts),
                )
            async with runtime_registry.get_or_create(nb_id).mutation_lock:
                deleting_notebooks.add(nb_id)
                runtime_registry.set_lifecycle(nb_id, LifecycleState.DELETING)
                try:
                    await _teardown_notebook_runtime(
                        nb_id,
                        reason="file_delete",
                        broadcast_snapshot=False,
                        drop_connections=False,
                    )
                    target.unlink()
                    notebooks.pop(nb_id, None)
                    deleted_notebooks[nb_id] = {
                        "path": path,
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    deleted_nb_id = nb_id
                    await event_bus.emit("notebook.deleted", {
                        "notebook_id": nb_id,
                        "path": path,
                    })
                    await broadcast_json(nb_id, {
                        "type": "notebook_deleted",
                        "notebook_id": nb_id,
                        "path": path,
                        "workspace_url": "/dialeng/workspace?deleted=1",
                        "message": "Notebook deleted. Its kernel and pending work were stopped.",
                    })
                    await broadcast_manager.drain_notebook(nb_id)
                    broadcast_manager.drop_notebook(nb_id)
                except Exception:
                    logger.exception("Notebook deletion failed (notebook=%s, path=%s)", nb_id, path)
                    deleting_notebooks.discard(nb_id)
                    runtime_registry.set_lifecycle(nb_id, LifecycleState.ACTIVE)
                    await broadcast_kernel_snapshot(nb_id)
                    raise
                finally:
                    deleting_notebooks.discard(nb_id)
        else:
            target.unlink()
    parent = target.parent
    kernel_nbs = {nid for nid in kernel_service._kernels if kernel_service.kernel_is_alive(nid)}
    next_active_file = "" if active_file_relpath == path else active_file_relpath
    explorer = FileListContent(
        parent,
        NOTEBOOKS_DIR,
        "" if deleted_nb_id == active_notebook_id else active_notebook_id,
        active_file_relpath=next_active_file,
        kernel_notebooks=kernel_nbs,
    )
    return JSONResponse({
        "ok": True,
        "html": to_xml(explorer),
        "deleted_notebook_id": deleted_nb_id,
        "workspace_url": "/dialeng/workspace?deleted=1" if deleted_nb_id else None,
    })


@rt("/dialeng/file")
def get(path: str = ""):
    """Render a standalone text-file editor shell."""
    if not path:
        return RedirectResponse("/dialeng/workspace", status_code=302)
    target = file_editor_service.resolve_rel_path(path)
    if target is None or not target.exists() or not target.is_file():
        return Titled("File not found - Dialeng", Div("File not found.", cls="status error"))
    if target.suffix == ".ipynb":
        rel = target.resolve().relative_to(NOTEBOOKS_DIR.resolve())
        notebook_name = str(rel.with_suffix(""))
        return RedirectResponse(f"/dialeng/?name={notebook_name}", status_code=302)
    return FileEditorPage(rel_path=file_editor_service.rel_path_for(target), root_dir=NOTEBOOKS_DIR, config=get_config())


@rt("/dialeng/file/view")
def get(path: str = "", client_id: str = ""):
    """Return the backend-authoritative editor fragment for a file."""
    if not path:
        return HTMLResponse(to_xml(FileEditorPane(file_editor_service.open_file("", client_id or ""))))
    if not client_id:
        from dialeng.services.file_editor_service import FileOpenResult
        return HTMLResponse(to_xml(FileEditorPane(
            FileOpenResult(
                status="locked",
                rel_path=path,
                reason="Dialeng could not establish a file-edit session for this tab.",
            )
        )))
    return HTMLResponse(to_xml(FileEditorPane(file_editor_service.open_file(path, client_id))))


@rt("/dialeng/file/save")
def post(
    path: str = "",
    client_id: str = "",
    content: str = "",
    expected_mtime_ns: int = -1,
    expected_size: int = -1,
):
    """Save an open file if the current client owns its edit lease."""
    start = time.perf_counter()
    op_id = log_operation(
        logger,
        "file_save_start",
        path=path,
        client_id=client_id,
        expected_mtime_ns=expected_mtime_ns,
        expected_size=expected_size,
        bytes=len(content.encode("utf-8", errors="replace")),
    )
    result = file_editor_service.save_file(
        path,
        client_id,
        content,
        expected_mtime_ns=expected_mtime_ns,
        expected_size=expected_size,
    )
    log_operation(
        logger,
        "file_save_complete" if result.status == "editable" else "file_save_rejected",
        op_id=op_id,
        path=result.rel_path or path,
        client_id=client_id,
        status=result.status,
        mtime_ns=result.mtime_ns,
        size=result.size,
        duration_ms=elapsed_ms(start),
    )
    if result.status == "editable":
        return {
            "ok": True,
            "message": "File saved.",
            "language": result.language,
            "mtime_ns": result.mtime_ns,
            "size": result.size,
        }
    return {"ok": False, "message": result.reason or "File could not be saved.", "status": result.status}


@rt("/dialeng/file/heartbeat")
def post(path: str = "", client_id: str = ""):
    """Refresh the edit lease for a file."""
    ok = file_editor_service.heartbeat(path, client_id)
    return {"ok": ok}


@rt("/dialeng/file/release")
def post(path: str = "", client_id: str = ""):
    """Release a file-edit lease for the current client."""
    file_editor_service.release(path, client_id)
    return {"ok": True}

# ============================================================================
# Notebook Routes
# ============================================================================

async def _render_notebook_page(nb_id: str):
    """Render a notebook page by ID (shared by /dialeng/{nb_id} and /dialeng/?name=...)."""
    nb = get_notebook(nb_id)
    await event_bus.emit("notebook.opened", {
        "notebook_id": nb_id,
        "path": str(nb.path) if nb.path else "",
    })

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
    llm_snapshot = _get_llm_status_snapshot(nb)
    available_models = DIALENG_CONFIG.get_model_choices(nb.llm_provider)
    return NotebookPage(nb, nb_list, AVAILABLE_DIALOG_MODES, available_models,
                        available_providers=llm_snapshot["providers"], config=config,
                        shfmt_available=SHFMT_AVAILABLE,
                        colab_enabled=colab_auth_service is not None,
                        colab_authenticated=colab_auth_service.is_authenticated if colab_auth_service else False,
                        colab_account_email=colab_auth_service.account_email if colab_auth_service else None,
                        notebooks_dir=NOTEBOOKS_DIR,
                        kernel_alive=nb_kernel_alive,
                        kernel_notebooks=kernel_nbs,
                        has_craft_code=has_craft_code,
                        initial_revisions=_get_revision_snapshot(nb_id))

@rt("/dialeng/{nb_id}")
async def get(nb_id: str):
    if not nb_id or not nb_id.strip():
        return RedirectResponse("/dialeng/workspace", status_code=302)
    if nb_id in deleting_notebooks or nb_id in deleted_notebooks:
        return RedirectResponse("/dialeng/workspace?deleted=1", status_code=302)
    if nb_id not in notebooks and _find_notebook_path(nb_id) is None:
        return RedirectResponse("/dialeng/workspace", status_code=302)
    return await _render_notebook_page(nb_id)

@rt("/dialeng/{nb_id}/save")
async def post(nb_id: str, overwrite: bool = False):
    if _active_prompt_cells(nb_id):
        return _prompt_operation_conflict(nb_id, "save")
    before_save = await event_bus.emit("notebook.before_save", {
        "notebook_id": nb_id,
        "overwrite": overwrite,
    })
    if before_save.blocked:
        return JSONResponse({
            "status": "blocked",
            "type": "extension_guard",
            "message": before_save.reason or "Notebook save blocked by extension",
        }, status_code=409)
    try:
        save_notebook(nb_id, overwrite_external=overwrite)
    except ExternalFileChangedError as exc:
        return JSONResponse({
            "status": "conflict",
            "type": "external_file_changed",
            "message": str(exc),
            "reload_url": f"/dialeng/{nb_id}",
            "overwrite_url": f"/dialeng/{nb_id}/save?overwrite=true",
        }, status_code=409)
    await broadcast_json(nb_id, {"type": "notebook_saved"})
    await event_bus.emit("notebook.saved", {"notebook_id": nb_id, "overwrite": overwrite})
    return Div("Saved", cls="status success")


@rt("/dialeng/{nb_id}/recovery/status")
def get(nb_id: str):
    if not DIALENG_CONFIG.autosave_enabled or not DIALENG_CONFIG.autosave_recovery_snapshots:
        return {"enabled": False, "available": False}
    nb = get_notebook(nb_id)
    canonical = Path(nb.path) if nb.path else NOTEBOOKS_DIR / f"{_nb_id_to_relpath(nb_id)}.ipynb"
    candidate = find_newer_recovery(NOTEBOOKS_DIR, nb_id, canonical)
    if candidate is None:
        return {"enabled": True, "available": False}
    return {
        "enabled": True,
        "available": True,
        "created_at": datetime.fromtimestamp(
            candidate.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "inspect_url": f"/dialeng/file?path={candidate.relative_to(NOTEBOOKS_DIR).as_posix()}",
    }


@rt("/dialeng/{nb_id}/recovery")
async def post(nb_id: str, action: str):
    nb = get_notebook(nb_id)
    canonical = Path(nb.path) if nb.path else NOTEBOOKS_DIR / f"{_nb_id_to_relpath(nb_id)}.ipynb"
    candidate = find_newer_recovery(NOTEBOOKS_DIR, nb_id, canonical)
    if candidate is None:
        return JSONResponse({"ok": False, "message": "No newer recovery snapshot."}, status_code=404)
    if action == "discard":
        discard_recovery(candidate, NOTEBOOKS_DIR)
        return {"ok": True, "action": "discarded"}
    if action != "restore":
        return JSONResponse({"ok": False, "message": "Invalid recovery action."}, status_code=400)

    try:
        recovered_data = json.loads(candidate.read_text(encoding="utf-8"))
        # Validate notebook structure before replacing the canonical file.
        Notebook.load(str(candidate), default_dialog_mode=DEFAULT_DIALOG_MODE)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "Recovery snapshot validation failed (notebook=%s, path=%s): %s",
            nb_id,
            candidate,
            exc,
        )
        return JSONResponse({
            "ok": False,
            "status": "invalid_recovery",
            "message": "The recovery snapshot is invalid and was not restored.",
        }, status_code=422)
    atomic_write_json(canonical, recovered_data)
    restored = Notebook.load(str(canonical), default_dialog_mode=DEFAULT_DIALOG_MODE)
    restored.id = nb_id
    notebooks[nb_id] = restored
    notebook_revisions.pop(nb_id, None)
    structure_revisions.pop(nb_id, None)
    event_stream_revisions.pop(nb_id, None)
    discard_recovery(candidate, NOTEBOOKS_DIR)
    await broadcast_json(nb_id, {"type": "notebook_reloaded"})
    return {"ok": True, "action": "restored"}

@rt("/dialeng/{nb_id}/mode")
async def post(nb_id: str, mode: str):
    if _active_prompt_cells(nb_id):
        return _prompt_operation_conflict(nb_id, "mode_change")
    nb = get_notebook(nb_id)
    nb.dialog_mode = mode
    nb.modified = True
    await broadcast_kernel_snapshot(nb_id)
    return ""

@rt("/dialeng/{nb_id}/model")
async def post(nb_id: str, model: str):
    if _active_prompt_cells(nb_id):
        return _prompt_operation_conflict(nb_id, "model_change")
    nb = get_notebook(nb_id)
    valid_model = DIALENG_CONFIG.validate_model(nb.llm_provider, model)
    if valid_model != model:
        await broadcast_json(nb_id, {
            "type": "status_message",
            "message": f"Model '{model}' is not available for {nb.llm_provider}.",
            "tone": "error",
        })
        return ""
    nb.model = model
    nb.modified = True
    await broadcast_llm_status(nb_id)
    await broadcast_kernel_snapshot(nb_id)
    return ""


@rt("/dialeng/{nb_id}/provider")
async def post(nb_id: str, provider: str):
    """Select an authenticated provider and its compatible backend default."""
    if _active_prompt_cells(nb_id):
        return _prompt_operation_conflict(nb_id, "provider_change")
    nb = get_notebook(nb_id)
    status = CREDENTIAL_INVENTORY.get(provider)
    if not status or not status.authenticated or provider == "mock_only":
        await broadcast_json(nb_id, {
            "type": "status_message",
            "message": "That LLM provider is not authenticated.",
            "tone": "error",
        })
        return ModelSelect(nb, DIALENG_CONFIG.get_model_choices(nb.llm_provider))
    nb.llm_provider = provider
    nb.model = DIALENG_CONFIG.validate_model(provider, None)
    nb.modified = True
    await broadcast_llm_status(nb_id)
    await broadcast_kernel_snapshot(nb_id)
    return ModelSelect(nb, DIALENG_CONFIG.get_model_choices(provider))

@rt("/dialeng/{nb_id}/safe_mode")
async def post(nb_id: str, safe_mode: str = "false"):
    """Toggle safe mode for shell commands in this notebook."""
    nb = get_notebook(nb_id)
    # Convert string to boolean (checkbox sends "true" or "false")
    nb.safe_mode = safe_mode.lower() in ("true", "on", "1", "yes")
    nb.modified = True
    await broadcast_kernel_snapshot(nb_id)
    return ""

# ============================================================================
# Kernel Helpers
# ============================================================================

def _schedule_colab_lib_sync(nb_id: str, reason: str) -> None:
    """Schedule a serialized Colab lib upload after exports change."""
    nb = notebooks.get(nb_id)
    if not nb or getattr(nb, 'kernel_type', 'local') != 'colab' or not kernel_service.has_kernel(nb_id):
        return
    generation = _current_kernel_generation(nb_id)
    if kernel_setup_state.get(nb_id, {}).get("is_active"):
        logger.info(
            "Skipping Colab lib sync while another setup is active (notebook=%s, reason=%s, generation=%s, active_source=%s)",
            nb_id, reason, generation, kernel_setup_state[nb_id].get("source"),
        )
        return

    existing = kernel_sync_tasks.get(nb_id)
    if existing and not existing.done():
        logger.info("Colab lib sync already pending for notebook %s (reason=%s, generation=%s)", nb_id, reason, generation)
        return

    async def _run_sync():
        current_task = asyncio.current_task()
        try:
            _assert_kernel_generation_current(nb_id, generation, f"colab_lib_sync:{reason}")
            await _set_kernel_setup_state(
                nb_id,
                source=f"colab_lib_sync:{reason}",
                phase="upload_lib",
                detail="Uploading exported module files to Colab",
                generation=generation,
            )
            await _sync_project_lib_to_kernel(nb_id, source="colab_lib_sync", reason=reason, generation=generation)
        except asyncio.CancelledError:
            logger.info("Cancelled Colab lib sync (notebook=%s, reason=%s, generation=%s)", nb_id, reason, generation)
        except Exception:
            logger.exception("Colab lib sync failed for notebook %s (reason=%s)", nb_id, reason)
            await broadcast_kernel_status(nb_id, "error")
        finally:
            if kernel_sync_tasks.get(nb_id) is current_task:
                kernel_sync_tasks.pop(nb_id, None)
            await _clear_kernel_setup_state(nb_id, expected_generation=generation)

    task = asyncio.create_task(_run_sync(), name=f"colab-lib-sync:{nb_id}:{reason}")
    kernel_sync_tasks[nb_id] = task
    logger.info("Scheduled Colab lib sync for notebook %s (reason=%s, generation=%s)", nb_id, reason, generation)


def _collect_project_lib_files() -> tuple[str, Path, list[tuple[str, str]], int]:
    """Collect exported project library files for kernel sync."""
    from dialeng.services.lib_export_service import get_lib_name

    lib_name = get_lib_name(NOTEBOOKS_DIR)
    lib_dir = NOTEBOOKS_DIR / lib_name
    if not lib_dir.exists():
        return lib_name, lib_dir, [], 0

    files: list[tuple[str, str]] = []
    total_bytes = 0
    for py_file in sorted(lib_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(NOTEBOOKS_DIR))
        content = py_file.read_text(encoding="utf-8")
        files.append((rel_path, content))
        total_bytes += len(content.encode("utf-8"))
    return lib_name, lib_dir, files, total_bytes


async def _inject_lib_syspath(nb_id: str, *, source: str, generation: int):
    """If lib dir exists in NOTEBOOKS_DIR, add NOTEBOOKS_DIR to the kernel's sys.path."""
    _assert_kernel_generation_current(nb_id, generation, source)

    lib_name, lib_dir, _, _ = _collect_project_lib_files()
    if not lib_dir.exists():
        return

    kernel_type, runtime_id = _kernel_runtime_context(nb_id)
    start = time.perf_counter()
    try:
        result = await kernel_service.ensure_project_path(
            nb_id,
            str(NOTEBOOKS_DIR.resolve()),
            remote_root=".",
        )
        logger.info(
            "LIB path ready (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, lib=%s, project_root=%s, remote_root=%s, duration_ms=%.1f)",
            nb_id,
            source,
            generation,
            kernel_type,
            runtime_id,
            lib_name,
            result.get("project_root"),
            result.get("remote_root"),
            (time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        logger.warning(
            "Failed to prepare LIB path (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, lib=%s): %s",
            nb_id, source, generation, kernel_type, runtime_id, lib_name, e,
        )


async def _sync_project_lib_to_kernel(nb_id: str, *, source: str, reason: str, generation: int):
    """Sync exported project library files into the selected kernel."""
    _assert_kernel_generation_current(nb_id, generation, source)

    nb = notebooks.get(nb_id)
    if not nb or not kernel_service.has_kernel(nb_id):
        return

    lib_name, lib_dir, files, total_bytes = _collect_project_lib_files()
    if not lib_dir.exists() or not files:
        logger.info(
            "No LIB files to sync (notebook=%s, source=%s, reason=%s, generation=%s, lib=%s)",
            nb_id, source, reason, generation, lib_name,
        )
        return

    kernel_type, runtime_id = _kernel_runtime_context(nb_id)
    start = time.perf_counter()
    sample_files = ", ".join(path for path, _ in files[:5])
    try:
        result = await kernel_service.sync_project_files(nb_id, files, remote_root=".")
        logger.info(
            "LIB sync completed (notebook=%s, source=%s, reason=%s, generation=%s, kernel_type=%s, runtime_id=%s, lib=%s, file_count=%s, total_bytes=%s, remote_root=%s, status=%s, duration_ms=%.1f, sample_files=%s)",
            nb_id,
            source,
            reason,
            generation,
            kernel_type,
            runtime_id,
            lib_name,
            result.get("file_count", len(files)),
            result.get("total_bytes", total_bytes),
            result.get("remote_root", "."),
            result.get("status", "unknown"),
            (time.perf_counter() - start) * 1000,
            sample_files,
        )
    except Exception as e:
        logger.warning(
            "LIB sync failed (notebook=%s, source=%s, reason=%s, generation=%s, kernel_type=%s, runtime_id=%s, lib=%s, file_count=%s, total_bytes=%s): %s",
            nb_id,
            source,
            reason,
            generation,
            kernel_type,
            runtime_id,
            lib_name,
            len(files),
            total_bytes,
            e,
        )


def _resolve_notebook_disk_path(nb_id: str) -> Path:
    """Resolve a notebook path for CRAFT discovery."""
    nb = get_notebook(nb_id)
    nb_path = nb.path
    if not nb_path:
        found = _find_notebook_path(nb_id)
        nb_path = found if found else NOTEBOOKS_DIR / f"{_nb_id_to_relpath(nb_id)}.ipynb"
        if found:
            nb.path = found
    return Path(nb_path)


def _collect_craft_cells_for_setup(nb_id: str, *, only_unexecuted: bool) -> tuple[list[tuple[str, str]], list[Path]]:
    """Collect CRAFT code cells for notebook setup."""
    nb_path = _resolve_notebook_disk_path(nb_id)
    if not nb_path.exists():
        logger.info(
            "Skipping CRAFT discovery because notebook path does not exist (notebook=%s, path=%s)",
            nb_id,
            nb_path,
        )
        return [], []

    from dialeng.services.craft_service import (
        find_craft_files,
        get_craft_code_cells,
        is_craft_executed,
    )

    craft_paths = find_craft_files(nb_path, NOTEBOOKS_DIR)
    nb_path_resolved = nb_path.resolve()
    craft_paths = [cp for cp in craft_paths if Path(cp).resolve() != nb_path_resolved]
    if not craft_paths:
        logger.info(
            "No CRAFT notebooks discovered for setup (notebook=%s, notebook_path=%s, only_unexecuted=%s)",
            nb_id,
            nb_path,
            only_unexecuted,
        )
        return [], []

    craft_cells = get_craft_code_cells(craft_paths)
    if only_unexecuted:
        craft_cells = [(cid, src) for cid, src in craft_cells if not is_craft_executed(nb_id, cid)]
    logger.info(
        "Discovered CRAFT setup inputs (notebook=%s, notebook_path=%s, craft_files=%s, craft_cells=%s, only_unexecuted=%s)",
        nb_id,
        nb_path,
        len(craft_paths),
        len(craft_cells),
        only_unexecuted,
    )
    return craft_cells, craft_paths


async def _run_notebook_kernel_setup(
    nb_id: str,
    *,
    source: str,
    craft_cells: list[tuple[str, str]],
    craft_paths: list[Path],
    generation: int,
) -> None:
    """Run serialized notebook setup against the selected kernel."""
    from dialeng.services.craft_service import mark_craft_executed

    current_task = asyncio.current_task()
    total_start = time.perf_counter()
    kernel_type, runtime_id = _kernel_runtime_context(nb_id)
    op_id = log_operation(
        logger,
        "kernel_setup_start",
        notebook=nb_id,
        source=source,
        generation=generation,
        kernel_type=kernel_type,
        runtime_id=runtime_id,
        craft_cells=len(craft_cells),
    )
    logger.info(
        "Starting notebook kernel setup (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, craft_cells=%s)",
        nb_id, source, generation, kernel_type, runtime_id, len(craft_cells),
    )
    is_colab_kernel = getattr(get_notebook(nb_id), "kernel_type", "local") == "colab"
    if craft_paths:
        for craft_path in craft_paths:
            logger.info(
                "Notebook setup uses CRAFT file (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, path=%s)",
                nb_id, source, generation, kernel_type, runtime_id, craft_path,
            )

    try:
        _assert_kernel_generation_current(nb_id, generation, source)
        await _set_kernel_setup_state(
            nb_id,
            source=source,
            phase="inject_lib",
            detail="Injecting project library path",
            generation=generation,
        )
        await _inject_lib_syspath(nb_id, source=source, generation=generation)

        _assert_kernel_generation_current(nb_id, generation, source)
        await _set_kernel_setup_state(
            nb_id,
            source=source,
            phase="upload_lib",
            detail="Uploading exported module files to Colab" if is_colab_kernel else "Refreshing exported module files",
            generation=generation,
        )
        await _sync_project_lib_to_kernel(nb_id, source=source, reason="kernel_setup", generation=generation)

        total_craft = len(craft_cells)
        for idx, (cid, src) in enumerate(craft_cells, start=1):
            _assert_kernel_generation_current(nb_id, generation, source)
            craft_start = time.perf_counter()
            await _set_kernel_setup_state(
                nb_id,
                source=source,
                phase="craft",
                detail=f"Executing CRAFT cell {idx}/{total_craft}",
                generation=generation,
            )
            cell = Cell(id=cid, cell_type=CellType.CODE, source=src)
            async for _ in kernel_service.execute_cell(nb_id, cell):
                pass
            mark_craft_executed(nb_id, cid)
            logger.info(
                "Executed CRAFT cell during setup (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, craft_cell=%s, index=%s/%s, duration_ms=%.1f)",
                nb_id, source, generation, kernel_type, runtime_id, cid, idx, total_craft, (time.perf_counter() - craft_start) * 1000,
            )

        await _clear_kernel_setup_state(nb_id, expected_generation=generation)
        log_operation(
            logger,
            "kernel_setup_complete",
            op_id=op_id,
            notebook=nb_id,
            source=source,
            generation=generation,
            kernel_type=kernel_type,
            runtime_id=runtime_id,
            duration_ms=elapsed_ms(total_start),
        )
        logger.info(
            "Notebook kernel setup completed (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s, duration_ms=%.1f)",
            nb_id, source, generation, kernel_type, runtime_id, (time.perf_counter() - total_start) * 1000,
        )
        await broadcast_kernel_status(nb_id, "connected")
        await event_bus.emit("kernel.ready", {
            "notebook_id": nb_id,
            "kernel_type": kernel_type,
            "runtime_id": runtime_id,
            "source": source,
        })
    except asyncio.CancelledError:
        await _clear_kernel_setup_state(nb_id, expected_generation=generation)
        log_operation(
            logger,
            "kernel_setup_cancelled",
            op_id=op_id,
            notebook=nb_id,
            source=source,
            generation=generation,
            kernel_type=kernel_type,
            runtime_id=runtime_id,
            duration_ms=elapsed_ms(total_start),
        )
        logger.info(
            "Notebook kernel setup cancelled (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s)",
            nb_id, source, generation, kernel_type, runtime_id,
        )
    except Exception:
        await _clear_kernel_setup_state(nb_id, expected_generation=generation)
        log_operation(
            logger,
            "kernel_setup_failed",
            level=logging.ERROR,
            op_id=op_id,
            notebook=nb_id,
            source=source,
            generation=generation,
            kernel_type=kernel_type,
            runtime_id=runtime_id,
            duration_ms=elapsed_ms(total_start),
        )
        logger.exception(
            "Notebook kernel setup failed (notebook=%s, source=%s, generation=%s, kernel_type=%s, runtime_id=%s)",
            nb_id, source, generation, kernel_type, runtime_id,
        )
        await broadcast_kernel_status(nb_id, "error")
        raise
    finally:
        if kernel_setup_tasks.get(nb_id) is current_task:
            kernel_setup_tasks.pop(nb_id, None)


def _schedule_notebook_kernel_setup(
    nb_id: str,
    *,
    source: str,
    craft_cells: list[tuple[str, str]],
    craft_paths: list[Path],
) -> None:
    """Schedule notebook setup for the current kernel generation."""
    generation = _current_kernel_generation(nb_id)
    existing = kernel_setup_tasks.get(nb_id)
    if existing and not existing.done():
        logger.info(
            "Kernel setup already pending (notebook=%s, source=%s, generation=%s, task=%s)",
            nb_id, source, generation, existing.get_name(),
        )
        return

    async def _runner():
        await _run_notebook_kernel_setup(
            nb_id,
            source=source,
            craft_cells=craft_cells,
            craft_paths=craft_paths,
            generation=generation,
        )

    task = asyncio.create_task(_runner(), name=f"kernel-setup:{nb_id}:{source}:g{generation}")
    kernel_setup_tasks[nb_id] = task
    logger.info(
        "Scheduled kernel setup (notebook=%s, source=%s, generation=%s, craft_cells=%s, task=%s)",
        nb_id, source, generation, len(craft_cells), task.get_name(),
    )


# ============================================================================
# Kernel Management Routes
# ============================================================================

@rt("/dialeng/{nb_id}/kernel/type")
@serialized_notebook_mutation
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

    logger.info("Switching kernel type (notebook=%s, kernel_type=%s)", nb_id, kernel_type)
    await event_bus.emit("kernel.attaching", {
        "notebook_id": nb_id,
        "kernel_type": kernel_type,
    })
    await _invalidate_kernel_background_work(nb_id, reason=f"kernel_type:{kernel_type}")
    nb.kernel_type = kernel_type
    if kernel_type == "colab" and not getattr(nb, "colab_runtime_type", ""):
        nb.colab_runtime_type = DIALENG_CONFIG.colab_default_runtime
    nb.modified = True
    _touch_notebook_revision(nb_id, "kernel_type_change")
    runtime_type = nb.colab_runtime_type
    await kernel_service.set_kernel_type(nb_id, kernel_type, runtime_type=runtime_type)

    # New kernel needs fresh CRAFT initialization (sys.path, uploads, CRAFT cells)
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)

    # Broadcast kernel type change to all clients
    await broadcast_json(nb_id, {"type": "kernel_type_changed", "kernel_type": kernel_type})

    await _mark_outline_dirty(nb_id, reason="kernel_type_change", broadcast_snapshot=False)
    await broadcast_kernel_snapshot(nb_id)

    return Div(f"Kernel: {reg.label}", cls="status success")

@rt("/dialeng/{nb_id}/kernel/runtime")
@serialized_notebook_mutation
async def post(nb_id: str, runtime_type: str):
    """Change the Colab runtime type (cpu/gpu/tpu) for a notebook."""
    nb = get_notebook(nb_id)
    if runtime_type not in ("cpu", "gpu", "tpu"):
        return Div("Invalid runtime type", cls="status error")
    if nb.kernel_type != 'colab':
        return Div("Runtime type only applies to Colab kernels", cls="status error")

    nb.colab_runtime_type = runtime_type
    nb.modified = True
    _touch_notebook_revision(nb_id, "kernel_runtime_change")
    await _invalidate_kernel_background_work(nb_id, reason=f"runtime_type:{runtime_type}")

    # New kernel needs fresh CRAFT initialization
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)

    # Switch the runtime type - this creates a new kernel object
    if colab_session_manager:
        new_kernel = await colab_session_manager.set_runtime_type(nb_id, runtime_type)
        # Update kernel_service to point to the new kernel (old reference is stale)
        kernel_service.set_kernel_instance(nb_id, new_kernel)

    logger.info("Updated Colab runtime type (notebook=%s, runtime_type=%s)", nb_id, runtime_type)
    await _mark_outline_dirty(nb_id, reason="kernel_runtime_change", broadcast_snapshot=False)
    await broadcast_kernel_snapshot(nb_id)

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


@rt("/dialeng/{nb_id}/kernel/snapshot")
def get(nb_id: str):
    """Get the backend-authoritative kernel snapshot for the notebook."""
    if nb_id in deleting_notebooks or nb_id in deleted_notebooks:
        return JSONResponse({
            "type": "notebook_deleted",
            "notebook_id": nb_id,
            "workspace_url": "/dialeng/workspace?deleted=1",
            "message": "Notebook deleted.",
        }, status_code=410)
    return _build_kernel_snapshot(nb_id)


@rt("/dialeng/{nb_id}/recovery")
def get(nb_id: str):
    """Return full rendered cells plus authoritative runtime revisions after a gap."""
    if nb_id in deleting_notebooks or nb_id in deleted_notebooks:
        return JSONResponse({
            "type": "notebook_deleted",
            "notebook_id": nb_id,
            "workspace_url": "/dialeng/workspace?deleted=1",
            "message": "Notebook deleted.",
        }, status_code=410)
    nb = get_notebook(nb_id)
    return {
        "type": "recovery_snapshot",
        "cells_html": to_xml(AllCells(nb)),
        "snapshot": _build_kernel_snapshot(nb_id),
    }

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
    colab_email = colab_auth_service.account_email if colab_auth_service else None
    return KernelModal(nb.id, nb.kernel_type, colab_auth, nb.colab_runtime_type, colab_email)

@rt("/dialeng/{nb_id}/kernel/craft-init")
async def post(nb_id: str):
    """Execute kernel setup (_lib/ sys.path) and CRAFT code cells after kernel selection.

    Called by the client after the user selects a kernel, so setup
    code runs before any manual cell execution.
    """
    if kernel_setup_state.get(nb_id, {}).get("is_active"):
        logger.info(
            "Skipping duplicate kernel setup request (notebook=%s, active_source=%s)",
            nb_id, kernel_setup_state[nb_id].get("source"),
        )
        return ""

    craft_cells, craft_paths = _collect_craft_cells_for_setup(nb_id, only_unexecuted=True)
    logger.info(
        "Scheduling kernel setup after kernel selection (notebook=%s, craft_cells=%s)",
        nb_id, len(craft_cells),
    )
    _schedule_notebook_kernel_setup(
        nb_id,
        source="kernel_select",
        craft_cells=craft_cells,
        craft_paths=craft_paths,
    )
    return ""

# ============================================================================
# Google OAuth Routes (for Colab integration)
# ============================================================================

@rt("/auth/google")
def get(request):
    """Initiate Google OAuth2 flow for Colab access."""
    if not colab_auth_service:
        return Div("Colab integration not configured", cls="status error")
    if colab_auth_service.auth_provider == "adc":
        return Titled(
            "Colab ADC Authentication",
            Div(
                "Colab is configured to use Application Default Credentials. Run "
                "`gcloud auth application-default login` with Colab scopes if it is not authenticated.",
                cls="status",
            ),
            A("← Back to Dialeng", href="/"),
        )
    if colab_auth_service.auth_provider == "oauth_remote":
        state = colab_auth_service.create_auth_state()
        auth_url = colab_auth_service.get_remote_auth_url(state=state)
        return Titled(
            "Connect Colab",
            Div(
                P("Open the authorization URL, approve access, then paste the code Google displays."),
                A("Open Google authorization", href=auth_url, target="_blank", cls="btn btn-primary"),
                Form(
                    Input(type="hidden", name="state", value=state),
                    Input(type="text", name="code", placeholder="Paste authorization code", cls="setting-text"),
                    Button("Connect Colab", type="submit", cls="btn btn-primary"),
                    method="post",
                    action="/auth/google/remote",
                    cls="remote-oauth-form",
                ),
                cls="remote-oauth-panel",
            ),
        )
    state = colab_auth_service.create_auth_state()
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
    if not colab_auth_service.validate_auth_state(state):
        return Titled("Authentication Error",
                       Div("Invalid or expired OAuth state. Please try signing in again.", cls="status error"),
                       A("← Back to Dialeng", href="/"))
    try:
        # Redirect URI must match what was used in get_auth_url()
        redirect_uri = f"{request.url.scheme}://{request.url.netloc}/auth/google/callback"
        await colab_auth_service.handle_callback(code, redirect_uri=redirect_uri)
        await _mark_all_outlines_dirty("colab_auth_success")
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

@rt("/auth/google/remote")
async def post(code: str = "", state: str = ""):
    """Handle copy-paste OAuth code flow."""
    if not colab_auth_service:
        return Titled("Error", Div("Colab integration not configured", cls="status error"))
    if colab_auth_service.auth_provider != "oauth_remote":
        return Titled("Authentication Error", Div("Remote OAuth is not enabled.", cls="status error"))
    if not colab_auth_service.validate_auth_state(state):
        return Titled("Authentication Error",
                       Div("Invalid or expired OAuth state. Please retry authentication.", cls="status error"))
    if not code.strip():
        return Titled("Authentication Error", Div("No authorization code provided.", cls="status error"))
    try:
        await colab_auth_service.handle_remote_code(code.strip())
        await _mark_all_outlines_dirty("colab_remote_auth_success")
        from starlette.responses import HTMLResponse
        return HTMLResponse("""<!DOCTYPE html><html><body>
<script>
try { localStorage.setItem('colab-auth-event', Date.now().toString()); } catch(e) {}
if (window.opener) { window.opener.postMessage('colab-authenticated', '*'); }
</script>
<p>Authenticated. You can close this window.</p>
</body></html>""")
    except Exception as e:
        return Titled("Authentication Error",
                       Div(f"Failed to exchange remote auth code: {e}", cls="status error"),
                       A("← Retry", href="/auth/google"))

@rt("/auth/google/logout")
async def post():
    """Disconnect from Google / clear Colab tokens."""
    if colab_auth_service:
        colab_auth_service.logout()
    await _mark_all_outlines_dirty("colab_logout")
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
    return colab_auth_service.get_status()

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
            cell_headings, cache_hit = outline_cache.headings(
                cell,
                extract_headings_from_markdown,
            )
            performance_monitor.increment(
                "outline.heading_cache_hit" if cache_hit else "outline.heading_cache_miss"
            )
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
            kernel = kernel_service.get_kernel(nb_id)
            status = kernel.get_status()
            generation = _current_kernel_generation(nb_id)
            execution_count = status.execution_count
            ns_info = outline_cache.get_namespace(nb_id, generation, execution_count)
            if ns_info is None:
                performance_monitor.increment("outline.namespace_cache_miss")
                ns_info = await kernel_service.get_namespace_info(nb_id)
                if ns_info:
                    outline_cache.set_namespace(nb_id, generation, execution_count, ns_info)
            else:
                performance_monitor.increment("outline.namespace_cache_hit")
            if ns_info:
                variables = ns_info.get('variables', [])
                functions = ns_info.get('functions', [])
        except Exception as e:
            logger.warning("Failed to build outline namespace info (notebook=%s): %s", nb_id, e)

    return OutlineSidebar(nb_id, headings, variables, functions, is_open=True)

# ============================================================================
# Settings Endpoints
# ============================================================================


async def _refresh_llm_after_auth() -> None:
    """Rebuild inventory/provider instances and notify every connected tab."""
    _refresh_llm_runtime()
    await broadcast_all_llm_statuses()
    await broadcast_all_kernel_snapshots()


@rt("/dialeng/{nb_id}/llm/status")
def get(nb_id: str):
    return {"type": "llm_status", **_get_llm_status_snapshot(get_notebook(nb_id))}


@rt("/auth/openai/start")
async def post():
    """Start or join the single server-global Codex OAuth flow."""
    try:
        return openai_codex_auth_service.start_login()
    except Exception as exc:
        logger.exception("Failed to start OpenAI Codex OAuth")
        return Response(
            content=json.dumps({"ok": False, "error": str(exc)}),
            status_code=400,
            media_type="application/json",
        )


@rt("/auth/openai/status")
async def get():
    """Return OAuth status and finish a loopback callback when one arrived."""
    try:
        completed = await openai_codex_auth_service.complete_loopback_if_ready()
        if completed:
            await _refresh_llm_after_auth()
        return openai_codex_auth_service.status()
    except Exception as exc:
        openai_codex_auth_service.last_error = str(exc)
        logger.exception("Failed to complete OpenAI Codex loopback OAuth")
        return openai_codex_auth_service.status()


@rt("/auth/openai/complete")
async def post(request):
    """Complete OAuth from a pasted redirect URL, code#state, or bare code."""
    try:
        data = await request.json()
        await openai_codex_auth_service.complete_authorization(
            str(data.get("code", "")), str(data.get("state", ""))
        )
        await _refresh_llm_after_auth()
        return openai_codex_auth_service.status()
    except Exception as exc:
        openai_codex_auth_service.last_error = str(exc)
        logger.exception("Failed to complete OpenAI Codex manual OAuth")
        return Response(
            content=json.dumps({"authenticated": False, "error": str(exc)}),
            status_code=400,
            media_type="application/json",
        )


@rt("/auth/openai/logout")
async def post():
    openai_codex_auth_service.logout()
    await _refresh_llm_after_auth()
    return openai_codex_auth_service.status()

@rt("/settings")
def get():
    """Return the current backend-authoritative settings snapshot."""
    return JSONResponse(settings_snapshot(
        NOTEBOOKS_DIR / "dialeng_config.json",
        activation=_settings_activation,
    ))


@rt("/settings/view")
def get():
    """Return the current server-rendered settings surface for targeted refresh."""
    return SettingsSidebar(DIALENG_CONFIG)


async def _broadcast_settings_changed(snapshot: dict[str, Any]) -> None:
    await broadcast_all_json({"type": "settings_changed", **snapshot})


async def _activate_settings(revision: str, changed_paths: list[str]) -> None:
    """Apply saved settings without holding the request or event loop."""
    global CREDENTIAL_INVENTORY, CREDENTIAL_STATUS
    global colab_auth_service, colab_session_manager, _settings_activation

    async with _settings_apply_lock:
        config_path = NOTEBOOKS_DIR / "dialeng_config.json"
        current = settings_snapshot(config_path)
        if current["revision"] != revision:
            return

        _settings_activation = {
            "status": "validating",
            "revision": revision,
            "message": "Validating providers and applying runtime settings...",
        }
        await _broadcast_settings_changed(settings_snapshot(
            config_path, activation=_settings_activation
        ))
        try:
            set_skill_root(NOTEBOOKS_DIR)
            inventory = await refresh_credential_inventory_async(NOTEBOOKS_DIR / ".env")
            if settings_snapshot(config_path)["revision"] != revision:
                return
            _apply_llm_inventory(inventory)

            colab_changed = False
            # Auth-provider changes are intentionally restart-bound. Existing
            # authenticated runtimes are not torn down as a side effect of Save.
            if "colab.auth_provider" not in changed_paths:
                if DIALENG_CONFIG.colab_enabled and colab_auth_service is None:
                    from dialeng.services.colab import ColabAuthService, ColabSessionManager
                    from dialeng.services.colab.colab_auth import resolve_oauth_credentials

                    creds = await resolve_oauth_credentials(DIALENG_CONFIG.colab_auth_provider)
                    colab_auth_service = ColabAuthService(
                        credentials=creds,
                        auth_provider=DIALENG_CONFIG.colab_auth_provider,
                    )
                    await colab_auth_service.validate_session()
                    colab_session_manager = ColabSessionManager(
                        colab_auth_service,
                        default_accelerator=DIALENG_CONFIG.colab_default_accelerator,
                    )
                    if colab_auth_service.is_authenticated:
                        try:
                            await colab_session_manager.prune_stale_metadata()
                        except Exception as exc:
                            logger.warning(
                                "Failed to prune Colab session metadata from settings: %s",
                                exc,
                            )
                    kernel_service.set_colab_session_manager(colab_session_manager)
                    colab_changed = True
                elif not DIALENG_CONFIG.colab_enabled and colab_auth_service is not None:
                    if colab_session_manager:
                        await colab_session_manager.shutdown_all()
                    colab_auth_service = None
                    colab_session_manager = None
                    kernel_service.set_colab_session_manager(None)
                    colab_changed = True

            if colab_session_manager is not None:
                colab_session_manager.default_accelerator = (
                    DIALENG_CONFIG.colab_default_accelerator
                )

            if colab_changed:
                await _mark_all_outlines_dirty("colab_settings_change")
            else:
                await broadcast_all_kernel_snapshots()
            await broadcast_all_llm_statuses()

            reload_paths = {
                path for path in changed_paths
                if get_setting_definitions(DIALENG_CONFIG).get(path)
                and get_setting_definitions(DIALENG_CONFIG)[path].apply_mode == "extension_reload"
            }
            if reload_paths:
                for extension_id, owner in extension_manager.store.owners.items():
                    owned_paths = {
                        contribution.value.path
                        for contribution in owner.contributions
                        if contribution.kind == "settings.definition"
                    }
                    if extension_id in extension_manager.active and owned_paths & reload_paths:
                        await extension_manager.activate(extension_id, require_trust=True)

            _settings_activation = {
                "status": "ready",
                "revision": revision,
                "message": "Settings are active.",
            }
            await event_bus.emit("settings.activation_completed", {
                "revision": revision,
                "changed_paths": changed_paths,
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Saved settings could not be fully activated")
            _settings_activation = {
                "status": "failed",
                "revision": revision,
                "message": "Settings were saved, but runtime activation failed.",
                "error": str(exc),
            }
            await event_bus.emit("settings.activation_failed", {
                "revision": revision,
                "changed_paths": changed_paths,
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            if settings_snapshot(config_path)["revision"] == revision:
                await _broadcast_settings_changed(settings_snapshot(
                    config_path, activation=_settings_activation
                ))


def _track_settings_activation(task: asyncio.Task) -> None:
    _settings_activation_tasks.add(task)
    task.add_done_callback(_settings_activation_tasks.discard)

@rt("/settings")
async def post(request):
    """Atomically apply a revision-checked partial settings update."""
    try:
        config_path = NOTEBOOKS_DIR / "dialeng_config.json"
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            changes = payload.get("changes", {})
            expected_revision = str(payload.get("revision", ""))
        else:
            form_data = await request.form()
            expected_revision = str(form_data.get("_revision", ""))
            changes = {
                key: value
                for key, value in form_data.multi_items()
                if not key.startswith("_")
            }
        if not isinstance(changes, dict):
            raise SettingsValidationError({"_form": "Changes must be an object."})

        async with _settings_update_lock:
            config, snapshot, changed_paths = await asyncio.to_thread(
                commit_settings,
                config_path,
                changes,
                expected_revision,
            )
            global DIALENG_CONFIG, _settings_activation
            DIALENG_CONFIG = config

        modes = apply_modes_for(changed_paths, config)
        if changed_paths:
            _settings_activation = {
                "status": "pending",
                "revision": snapshot["revision"],
                "message": "Settings saved. Runtime validation is queued.",
            }
            snapshot["activation"] = dict(_settings_activation)
            snapshot["changed_paths"] = changed_paths
            snapshot["apply_modes"] = modes
            await _broadcast_settings_changed(snapshot)
            await event_bus.emit("settings.changed", {
                "revision": snapshot["revision"],
                "changed_paths": changed_paths,
                "apply_modes": modes,
            })
            task = asyncio.create_task(
                _activate_settings(snapshot["revision"], changed_paths),
                name=f"settings-activation:{snapshot['revision'][:8]}",
            )
            _track_settings_activation(task)
            return JSONResponse(snapshot, status_code=202)

        snapshot["changed_paths"] = []
        snapshot["apply_modes"] = []
        return JSONResponse(snapshot)
    except SettingsConflictError as exc:
        snapshot = settings_snapshot(
            NOTEBOOKS_DIR / "dialeng_config.json",
            activation=_settings_activation,
        )
        return JSONResponse(
            {"error": "settings_conflict", "message": str(exc), "snapshot": snapshot},
            status_code=409,
        )
    except SettingsValidationError as exc:
        return JSONResponse(
            {"error": "validation_error", "message": str(exc), "errors": exc.errors},
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Failed to save settings")
        return JSONResponse(
            {"error": "settings_save_failed", "message": str(exc)},
            status_code=500,
        )

# Cell operations - now include notebook ID in path
@rt("/dialeng/{nb_id}/cell/add")
@serialized_notebook_mutation
async def post(
    nb_id: str,
    pos: int = -1,
    type: str = "code",
    expected_revision: int = -1,
    expected_structure_revision: int = -1,
    operation_id: str = "",
):
    nb = get_notebook(nb_id)
    if pos < 0:
        pos = len(nb.cells)
    new_cell = notebook_mutation_service.add(nb, position=pos, cell_type=type)
    _touch_structure_revision(nb_id, "cell_add")
    await event_bus.emit("cell.added", {
        "notebook_id": nb_id,
        "cell_id": new_cell.id,
        "cell_type": str(new_cell.cell_type),
        "position": nb.cells.index(new_cell),
    })

    # FOUST fix (FOUST = Flash of Unstyled Text — Monaco renders code white first,
    # then tokenizes async via web worker; destroying/recreating the editor causes a
    # visible flash before syntax highlighting reappears).
    # Broadcast a structural cell_add payload instead of replacing #cells. The
    # browser reconciles its DOM from the backend-authored ordered_cell_ids list,
    # so add/move/delete all share one structural source of truth.
    await broadcast_json(nb_id, _cell_add_payload(nb, nb_id, new_cell.id))
    await _mark_outline_dirty(nb_id, reason="cell_add")

    return ""

@rt("/dialeng/{nb_id}/cell/{cid}")
@serialized_notebook_mutation
async def delete(
    nb_id: str,
    cid: str,
    expected_revision: int = -1,
    expected_structure_revision: int = -1,
    operation_id: str = "",
):
    nb = get_notebook(nb_id)

    before_delete = await event_bus.emit("cell.before_delete", {
        "notebook_id": nb_id,
        "cell_id": cid,
    })
    if before_delete.blocked:
        return JSONResponse({
            "status": "blocked",
            "type": "extension_guard",
            "message": before_delete.reason or "Cell deletion blocked by extension",
        }, status_code=409)

    # Remove from execution queue if queued
    queue = get_execution_queue(nb_id)
    queue.cancel_queued(nb_id, cid)

    removed = notebook_mutation_service.delete(nb, cid)
    if removed is None:
        return ""

    # Broadcast queue state update
    await broadcast_queue_state(nb_id)

    # FOUST fix: broadcast granular cell_delete JSON instead of AllCellsOOB.
    # The client removes just this cell + its adjacent add-row from DOM.
    # Other cells' Monaco editors are completely untouched.
    _touch_structure_revision(nb_id, "cell_delete")
    await event_bus.emit("cell.deleted", {
        "notebook_id": nb_id,
        "cell_id": cid,
        "cell_type": str(removed.cell_type),
    })
    await broadcast_json(nb_id, _cell_delete_payload(nb, nb_id, cid))
    await _mark_outline_dirty(nb_id, reason="cell_delete")

    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/source")
@serialized_notebook_mutation
async def post(
    nb_id: str,
    cid: str,
    source: str,
    expected_version: int = -1,
    expected_revision: int = -1,
    operation_id: str = "",
):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            if _expected_version_conflicts(c, expected_version):
                # A blur request can race a type change that already committed
                # the exact same source. Treat that retry as idempotent instead
                # of surfacing a false conflict after the DOM replacement.
                if c.source == source:
                    return ""
                log_operation(
                    logger,
                    "cell_source_conflict",
                    level=logging.WARNING,
                    notebook=nb_id,
                    cell=cid,
                    expected_version=expected_version,
                    actual_version=c.version,
                )
                logger.warning(
                    "Rejected stale cell source commit (notebook=%s, cell=%s, expected_version=%s, actual_version=%s)",
                    nb_id, cid, expected_version, c.version,
                )
                await _broadcast_cell_version_conflict(nb_id, c, expected_version)
                return {"status": "conflict", "version": c.version}

            changed = _commit_cell_source(nb_id, nb, c, source, reason="cell_source_commit")
            if not changed:
                break

            cell_type = str(getattr(c, "cell_type", ""))
            log_operation(
                logger,
                "cell_source_commit",
                notebook=nb_id,
                cell=cid,
                cell_type=cell_type,
                version=c.version,
                notebook_revision=_get_notebook_revision(nb_id),
            )
            logger.info(
                "Cell source committed (notebook=%s, cell=%s, cell_type=%s, version=%s)",
                nb_id,
                cid,
                cell_type,
                c.version,
            )

            # Every persisted source mutation must advance the browser's
            # notebook revision, including note/prompt cells that also receive
            # an HTML OOB refresh for their rendered preview.
            await broadcast_json(nb_id, _cell_source_payload(nb_id, c))
            if cell_type in {"code", "shell"}:
                await broadcast_to_notebook(nb_id, CellOutputOOB(c))
                await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))

            if cell_type == "note":
                await _mark_outline_dirty(nb_id, reason="note_source_commit")
            break
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/output")
@serialized_notebook_mutation
async def post(
    nb_id: str,
    cid: str,
    output: str,
    expected_version: int = -1,
    expected_revision: int = -1,
    operation_id: str = "",
):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            if _expected_version_conflicts(c, expected_version):
                log_operation(
                    logger,
                    "cell_output_conflict",
                    level=logging.WARNING,
                    notebook=nb_id,
                    cell=cid,
                    expected_version=expected_version,
                    actual_version=c.version,
                )
                logger.warning(
                    "Rejected stale cell output commit (notebook=%s, cell=%s, expected_version=%s, actual_version=%s)",
                    nb_id, cid, expected_version, c.version,
                )
                await _broadcast_cell_version_conflict(
                    nb_id, c, expected_version, field="output",
                )
                return {
                    "status": "conflict",
                    "conflict_field": "output",
                    "version": c.version,
                    "output": c.output,
                }

            changed = _commit_cell_output(nb_id, nb, c, output, reason="cell_output_commit")
            if not changed:
                break

            cell_type = str(getattr(c, "cell_type", ""))
            log_operation(
                logger,
                "cell_output_commit",
                notebook=nb_id,
                cell=cid,
                cell_type=cell_type,
                version=c.version,
                notebook_revision=_get_notebook_revision(nb_id),
            )
            logger.info(
                "Cell output committed (notebook=%s, cell=%s, cell_type=%s, version=%s)",
                nb_id,
                cid,
                cell_type,
                c.version,
            )

            if cell_type in {"code", "shell"}:
                await broadcast_json(nb_id, _cell_output_commit_payload(nb_id, c))
                await broadcast_to_notebook(nb_id, CellOutputOOB(c))
                await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
            elif cell_type == "prompt":
                await broadcast_json(nb_id, _prompt_output_payload(nb_id, c))
            else:
                await broadcast_json(nb_id, _cell_output_commit_payload(nb_id, c))
                await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
            break
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/type")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str, cell_type: str,
    source: Optional[str] = None,
    expected_version: int = -1,
    expected_cell_type: str = "",
    expected_revision: int = -1, operation_id: str = "",
):
    nb = get_notebook(nb_id)
    allowed_types = {value for value, _label in registry.get_cell_type_choices()}
    if cell_type not in allowed_types:
        return JSONResponse({"status": "invalid_cell_type"}, status_code=400)

    for c in nb.cells:
        if c.id == cid:
            current_type = str(c.cell_type)
            if expected_cell_type and current_type != expected_cell_type:
                await _broadcast_cell_version_conflict(nb_id, c, expected_version)
                return JSONResponse({
                    "status": "conflict",
                    "actual_type": current_type,
                    "actual_version": c.version,
                }, status_code=409)

            # The note/prompt editor's blur save and the select change can be
            # dispatched almost simultaneously. A one-step version advance is
            # safe when it committed the exact source included in this request.
            if _expected_version_conflicts(c, expected_version):
                if source is None or source != c.source:
                    await _broadcast_cell_version_conflict(nb_id, c, expected_version)
                    return JSONResponse({
                        "status": "conflict",
                        "actual_type": current_type,
                        "actual_version": c.version,
                    }, status_code=409)

            source_changed = source is not None and source != c.source
            type_changed = current_type != cell_type
            if not source_changed and not type_changed:
                return ""

            if source_changed:
                c.source = source
            c.cell_type = cell_type
            c.clear_outputs()
            c.version += 1
            c.last_modified = datetime.now()
            nb.modified = True
            _touch_notebook_revision(nb_id, "cell_type_change")
            await event_bus.emit("cell.type_changed", {
                "notebook_id": nb_id,
                "cell_id": cid,
                "old_type": current_type,
                "new_type": cell_type,
            })

            # Type changes require a full cell replacement, but the event must
            # still participate in the notebook revision stream. Unrevisioned
            # OOB HTML left the initiating tab stale and broke later mutations.
            await broadcast_json(nb_id, _cell_type_payload(nb, nb_id, cid))
            await _mark_outline_dirty(nb_id, reason="cell_type_change")

            return ""
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/move/{direction}")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str, direction: str,
    expected_revision: int = -1, expected_structure_revision: int = -1,
    operation_id: str = "",
):
    nb = get_notebook(nb_id)
    direction_map = {"up": -1, "down": 1}
    move_delta = direction_map.get(direction)
    if move_delta is None:
        return ""

    moved = notebook_mutation_service.move(nb, cid, move_delta)
    if not moved:
        return ""

    # Broadcast the backend-authoritative cell order so every client reorders
    # the existing DOM nodes from the same source of truth.
    _touch_structure_revision(nb_id, "cell_move")
    await event_bus.emit("cell.moved", {
        "notebook_id": nb_id,
        "cell_id": cid,
        "direction": direction,
        "position": next((i for i, cell in enumerate(nb.cells) if cell.id == cid), -1),
    })
    await broadcast_json(nb_id, _cell_move_payload(nb, nb_id, cid))
    await _mark_outline_dirty(nb_id, reason="cell_move")

    # Move buttons use hx_swap="none"; the browser applies the canonical order
    # from the WebSocket message and keeping the HTTP response empty avoids
    # unnecessary HTMX lifecycle work on the initiating tab.
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/duplicate")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str,
    expected_revision: int = -1, expected_structure_revision: int = -1,
    operation_id: str = "",
):
    """Duplicate a cell, inserting the copy immediately after the original."""
    nb = get_notebook(nb_id)
    new_cell = notebook_mutation_service.duplicate(nb, cid)
    if new_cell is not None:
        _touch_structure_revision(nb_id, "cell_duplicate")
        await event_bus.emit("cell.duplicated", {
            "notebook_id": nb_id,
            "source_cell_id": cid,
            "cell_id": new_cell.id,
            "cell_type": str(new_cell.cell_type),
        })
        await broadcast_json(nb_id, _cell_add_payload(nb, nb_id, new_cell.id))
        await _mark_outline_dirty(nb_id, reason="cell_duplicate")
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/clear-output")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str,
    expected_revision: int = -1, operation_id: str = "",
):
    """Clear outputs from a cell."""
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.clear_outputs()
            c.version += 1
            c.last_modified = datetime.now()
            nb.modified = True
            _touch_notebook_revision(nb_id, "cell_clear_output")
            await broadcast_json(nb_id, _cell_output_commit_payload(nb_id, c))
            cell_type = str(getattr(c, "cell_type", ""))
            if cell_type in {"code", "shell"}:
                await broadcast_to_notebook(nb_id, CellOutputOOB(c))
                await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
            else:
                await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
            return ""
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/merge-below")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str,
    expected_revision: int = -1, expected_structure_revision: int = -1,
    operation_id: str = "",
):
    """Merge cell with the one below it (append source, delete below)."""
    nb = get_notebook(nb_id)
    result = notebook_mutation_service.merge_below(nb, cid)
    if result is not None:
        _touch_structure_revision(nb_id, "cell_merge_below")
        await event_bus.emit("cell.merged", {
            "notebook_id": nb_id,
            "cell_id": result.target.id,
            "removed_cell_id": result.removed.id,
            "cell_type": str(result.target.cell_type),
        })
        await broadcast_json(nb_id, _cell_delete_payload(nb, nb_id, result.removed.id))
        cell_type = str(getattr(result.target, "cell_type", ""))
        if cell_type in {"code", "shell"}:
            await broadcast_json(nb_id, _cell_source_payload(nb_id, result.target))
            await broadcast_to_notebook(nb_id, CellOutputOOB(result.target))
            await broadcast_to_notebook(nb_id, CellHeaderOOB(result.target, nb_id))
        else:
            await broadcast_to_notebook(nb_id, CellViewOOB(result.target, nb_id))
        await _mark_outline_dirty(nb_id, reason="cell_merge_below")
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/extract-code-blocks")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str,
    expected_revision: int = -1, expected_structure_revision: int = -1,
    operation_id: str = "",
):
    """Extract fenced code blocks from cell output, create new code cells below."""
    import re
    nb = get_notebook(nb_id)
    for i, c in enumerate(nb.cells):
        if c.id == cid:
            # Parse fenced code blocks from output
            text = c.output or ''
            blocks = re.findall(r'```(?:\w*)\n(.*?)```', text, re.DOTALL)
            if not blocks:
                return ""
            # Insert code cells after this cell
            for j, block in enumerate(blocks):
                new_cell = Cell(cell_type="code", source=block.strip())
                nb.cells.insert(i + 1 + j, new_cell)
                nb.modified = True
                _touch_structure_revision(nb_id, "extract_code_blocks")
                await broadcast_json(nb_id, _cell_add_payload(nb, nb_id, new_cell.id))
            await _mark_outline_dirty(nb_id, reason="extract_code_blocks")
            return ""
    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/collapse")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str, collapsed: str,
    expected_revision: int = -1, operation_id: str = "",
):
    nb = get_notebook(nb_id)
    cell = None
    for c in nb.cells:
        if c.id == cid:
            cell = c
            next_collapsed = collapsed.lower() == "true"
            if c.collapsed != next_collapsed:
                c.collapsed = next_collapsed
                nb.modified = True
                _touch_notebook_revision(nb_id, "cell_collapse")
            break

    # Broadcast collapse state via targeted header OOB + class update
    if cell:
        await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))
        await broadcast_json(nb_id, {"type": "cell_class_update", "cell_id": cell.id, "cls": get_cell_state_classes(cell)})

    return ""

@rt("/dialeng/{nb_id}/cell/{cid}/collapse-section")
@serialized_notebook_mutation
async def post(
    nb_id: str, cid: str, section: str, level: int,
    expected_revision: int = -1, operation_id: str = "",
):
    """Update collapse level for input or output section"""
    nb = get_notebook(nb_id)
    cell = None
    for c in nb.cells:
        if c.id == cid:
            cell = c
            previous = (c.input_collapse, c.output_collapse)
            if section == "input":
                c.input_collapse = level
            elif section == "output":
                c.output_collapse = level
            elif section == "both":
                c.input_collapse = level
                c.output_collapse = level
            if previous != (c.input_collapse, c.output_collapse):
                nb.modified = True
                _touch_notebook_revision(nb_id, "cell_collapse_section")
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


@rt("/dialeng/{nb_id}/cell/{cid}/context")
def get(nb_id: str, cid: str):
    """Inspect the normalized notebook context used for this prompt cell."""
    nb = get_notebook(nb_id)
    cell = nb.get_cell(cid)
    if cell is None or str(cell.cell_type) != "prompt":
        return JSONResponse({"error": "Prompt cell not found."}, status_code=404)
    report = get_context_inspection(nb_id, cid)
    if report is None:
        build_context_messages(nb, cid)
        report = get_context_inspection(nb_id, cid) or {}
    from dialeng.services.prompt_parser import parse_prompt
    expressions = []
    tool_targets = []
    current_idx = nb.get_cell_index(cid)
    for context_cell in nb.cells[:current_idx + 1]:
        cell_expressions, cell_tools = parse_prompt(context_cell.source or "")
        expressions.extend(cell_expressions)
        tool_targets.extend(cell_tools)
    report.update({
        "provider": nb.llm_provider,
        "model": nb.model,
        "current_prompt_chars": len(cell.source or ""),
        "dynamic_expressions": list(dict.fromkeys(expressions)),
        "declared_tools": list(dict.fromkeys(tool_targets)),
        "built_in_tools_enabled": DIALENG_CONFIG.tool_builtin_enabled,
        "skills_enabled": bool(
            nb.llm_provider == "openai_codex" and DIALENG_CONFIG.skills_enabled
        ),
    })
    return report

@rt("/dialeng/{nb_id}/cell/{cid}/run")
async def post(nb_id: str, cid: str, source: str = None, expected_version: int = -1):
    # Guard: require explicit kernel selection before any cell execution
    if not kernel_service.has_kernel(nb_id):
        from starlette.responses import Response
        resp = Response("", status_code=200, headers={
            "HX-Trigger": json.dumps({"kernel-required": {"cellId": cid}})
        })
        return resp

    active_prompts = _active_prompt_cells(nb_id)
    if active_prompts:
        return _prompt_operation_conflict(nb_id, "cell_run")

    # Shift+Enter dispatches the run and next-cell requests together. Source
    # persistence must use the same lock as structural mutations so notebook
    # revision events are emitted in commit order instead of creating false gaps.
    async with runtime_registry.mutation(nb_id):
        nb = get_notebook(nb_id)
        cell_index = None
        target_cell = None

        for i, candidate in enumerate(nb.cells):
            if candidate.id == cid:
                cell_index = i
                target_cell = candidate
                break

        if target_cell is None:
            return ""

        c = target_cell
        run_op_id = log_operation(
            logger,
            "cell_run_requested",
            notebook=nb_id,
            cell=cid,
            cell_type=str(getattr(c, "cell_type", "")),
            expected_version=expected_version,
            version=c.version,
        )

        # Running a cell from the editor can submit fresher source than the last
        # blur commit. Keep old output visible until queue/running is confirmed.
        if source is not None and source != c.source:
            if _expected_version_conflicts(c, expected_version):
                log_operation(
                    logger,
                    "cell_run_source_conflict",
                    level=logging.WARNING,
                    op_id=run_op_id,
                    notebook=nb_id,
                    cell=cid,
                    expected_version=expected_version,
                    actual_version=c.version,
                )
                logger.warning(
                    "Rejected stale cell source commit during run (notebook=%s, cell=%s, expected_version=%s, actual_version=%s)",
                    nb_id, cid, expected_version, c.version,
                )
                await _broadcast_cell_version_conflict(nb_id, c, expected_version)
                return ""

            _commit_cell_source(nb_id, nb, c, source, reason="cell_run_source_commit")
            cell_type = str(getattr(c, "cell_type", ""))
            logger.info(
                "Cell source committed during run (notebook=%s, cell=%s, cell_type=%s, version=%s)",
                nb_id,
                cid,
                cell_type,
                c.version,
            )
            if cell_type in {"code", "shell"}:
                await broadcast_json(nb_id, _cell_source_payload(nb_id, c))
            else:
                await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))

    if c.cell_type == "code":
        queue = get_execution_queue(nb_id)

        # Check if already queued or running - ignore duplicate requests
        if queue.is_cell_queued(nb_id, cid):
            logger.info("Ignoring duplicate code run request (notebook=%s, cell=%s)", nb_id, cid)
            return ""

        logger.info("Queueing code cell (notebook=%s, cell=%s)", nb_id, cid)
        log_operation(logger, "cell_run_queued", op_id=run_op_id, notebook=nb_id, cell=cid)

        # Queue the cell - returns immediately, execution happens in background
        queue.queue_cell(nb_id, c)

        # Broadcast queue state to all clients
        await broadcast_queue_state(nb_id)

        return ""

    elif c.cell_type == "prompt":
        log_operation(logger, "prompt_run_started", op_id=run_op_id, notebook=nb_id, cell=cid)
        # Remove from cancelled set if it was there
        cancelled_cells.discard(cid)
        cancel_key = f"{nb_id}:{cid}"
        cancel_event = asyncio.Event()
        prompt_cancel_events[cancel_key] = cancel_event
        previous_output = c.output
        execution_started = time.perf_counter()
        execution_diagnostics: dict[str, Any] = {
            "operation_id": run_op_id,
            "provider": nb.llm_provider,
            "model": nb.model,
            "transport": "mock" if nb.dialog_mode == "mock" else "provider",
            "ttft_ms": None,
            "duration_ms": None,
            "retry_count": 0,
            "request_ids": [],
            "response_ids": [],
            "tool_steps": 0,
            "usage": _normalized_llm_usage({}),
            "status": "running",
        }
        failure_info: Optional[dict[str, Any]] = None
        prompt_source = c.source

        if nb.dialog_mode != "mock":
            selected_provider = CREDENTIAL_INVENTORY.get(nb.llm_provider)
            if not selected_provider or not selected_provider.authenticated:
                prompt_cancel_events.pop(cancel_key, None)
                await broadcast_json(nb_id, {
                    "type": "status_message",
                    "message": "The selected LLM provider is unavailable. Connect it or choose another provider; existing output was preserved.",
                    "tone": "error",
                })
                return ""
            if DIALENG_CONFIG.validate_model(nb.llm_provider, nb.model) != nb.model:
                prompt_cancel_events.pop(cancel_key, None)
                await broadcast_json(nb_id, {
                    "type": "status_message",
                    "message": "The selected model is not available for this provider. Existing output was preserved.",
                    "tone": "error",
                })
                return ""

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
            context_event = await event_bus.emit("prompt.context_built", {
                "notebook_id": nb_id,
                "cell_id": cid,
                "prompt": prompt_source,
                "context": context,
                "provider": "mock",
                "model": nb.model,
            })
            request_event = await event_bus.emit("prompt.before_request", context_event.payload)
            if request_event.blocked:
                prompt_cancel_events.pop(cancel_key, None)
                return JSONResponse({
                    "status": "blocked",
                    "type": "extension_guard",
                    "message": request_event.reason or "Prompt blocked by extension",
                }, status_code=409)
            prompt_source = str(request_event.payload.get("prompt", prompt_source))
            context = str(request_event.payload.get("context", context))
            stream_func = mock_llm_stream(prompt_source, context, c.use_thinking)
        else:
            # Use real LLM with dialoghelper context building
            context_messages = build_context_messages(nb, cid)
            context_event = await event_bus.emit("prompt.context_built", {
                "notebook_id": nb_id,
                "cell_id": cid,
                "prompt": prompt_source,
                "context_messages": context_messages,
                "provider": nb.llm_provider,
                "model": nb.model,
            })
            request_event = await event_bus.emit("prompt.before_request", context_event.payload)
            if request_event.blocked:
                prompt_cancel_events.pop(cancel_key, None)
                return JSONResponse({
                    "status": "blocked",
                    "type": "extension_guard",
                    "message": request_event.reason or "Prompt blocked by extension",
                }, status_code=409)
            prompt_source = str(request_event.payload.get("prompt", prompt_source))
            context_messages = request_event.payload.get("context_messages", context_messages)
            if not isinstance(context_messages, list):
                raise TypeError("prompt context transform must return context_messages as a list")

            # Check if prompt contains special syntax ($`var` or &`func`)
            from dialeng.services.prompt_parser import has_special_syntax
            from dialeng.services.dialeng_config import get_config

            config = get_config()
            max_steps = config.tool_max_steps
            include_builtins = config.tool_builtin_enabled
            codex_skills_available = False
            if nb.llm_provider == "openai_codex" and config.skills_enabled:
                from dialeng.services.skill_service import get_skill_service
                codex_skills_available = bool(get_skill_service().model_skills)

            if has_special_syntax(prompt_source) or include_builtins or codex_skills_available:
                # Use tool-enabled streaming
                kernel = kernel_service.get_kernel(nb_id)
                stream_func = llm_service.stream_response_with_tools(
                    prompt_source, context_messages, nb.dialog_mode, nb.model, c.use_thinking,
                    kernel=kernel, notebook_id=nb_id, max_steps=max_steps,
                    include_builtins=include_builtins,
                    provider_id=nb.llm_provider, cancel_event=cancel_event,
                )
            else:
                # Use regular streaming (no tools)
                stream_func = llm_service.stream_response(
                    prompt_source, context_messages, nb.dialog_mode, nb.model, c.use_thinking,
                    provider_id=nb.llm_provider, cancel_event=cancel_event,
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
        terminal_status = "completed"
        try:
            await broadcast_json(nb_id, {"type": "prompt_stream_start", "cell_id": cid})
            async for item in stream_func:
                await event_bus.emit("prompt.stream_event", {
                    "notebook_id": nb_id,
                    "cell_id": cid,
                    "event": item,
                })
                if (
                    item.get("type") in {"chunk", "thinking", "tool_call"}
                    and execution_diagnostics["ttft_ms"] is None
                ):
                    execution_diagnostics["ttft_ms"] = round(
                        (time.perf_counter() - execution_started) * 1000, 1,
                    )
                # Check if cancelled
                if cid in cancelled_cells:
                    cancelled_cells.discard(cid)
                    terminal_status = "cancelled"
                    break

                # Handle errors from LLM service
                if item["type"] == "error":
                    terminal_status = "error"
                    failure_info = {
                        **_classify_llm_failure(item["content"]),
                        "message": item["content"],
                    }
                    await broadcast_json(nb_id, {
                        "type": "status_message",
                        "message": item["content"],
                        "tone": "error",
                        "action": failure_info["action"],
                        "action_label": failure_info["label"],
                        "cell_id": cid,
                    })
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
                    execution_diagnostics["tool_steps"] += 1
                elif item["type"] == "usage":
                    _merge_llm_usage(execution_diagnostics["usage"], item.get("usage"))
                elif item["type"] == "diagnostic":
                    provider_diagnostics = item.get("diagnostics") or {}
                    for key in (
                        "transport", "ttft_ms", "duration_ms", "retry_count",
                        "request_ids", "response_ids",
                    ):
                        if key in provider_diagnostics:
                            execution_diagnostics[key] = provider_diagnostics[key]

                # Send streaming updates via WebSocket
                if item["type"] == "thinking_start":
                    await broadcast_json(nb_id, {"type": "thinking_start", "cell_id": cid})
                elif item["type"] == "thinking_end":
                    await broadcast_json(nb_id, {"type": "thinking_end", "cell_id": cid})
                elif item["type"] == "thinking":
                    await broadcast_json(nb_id, {"type": "stream_chunk", "cell_id": cid, "chunk": item["content"], "thinking": True})
                elif item["type"] == "var_substituted":
                    # Variable was substituted - notify client
                    await broadcast_json(nb_id, {
                        "type": "var_substituted",
                        "cell_id": cid,
                        "var_name": item.get("name", ""),
                        "var_value": item.get("value", "")
                    })
                elif item["type"] == "tool_available":
                    # Tool became available - notify client
                    await broadcast_json(nb_id, {
                        "type": "tool_available",
                        "cell_id": cid,
                        "tool_name": item.get("name", ""),
                        "tool_type": item.get("tool_type", "dynamic")
                    })
                elif item["type"] == "tool_call":
                    # AI is calling a tool - notify client
                    await broadcast_json(nb_id, {
                        "type": "tool_call",
                        "cell_id": cid,
                        "tool_id": item.get("id", ""),
                        "tool_name": item.get("name", ""),
                        "tool_input": item.get("input", {})
                    })
                elif item["type"] == "tool_result":
                    # Tool returned a result - notify client
                    await broadcast_json(nb_id, {
                        "type": "tool_result",
                        "cell_id": cid,
                        "tool_id": item.get("id", ""),
                        "tool_name": item.get("name", ""),
                        "result": item.get("result", {}),
                        "status": item.get("status", "success")
                    })
                elif item["type"] == "usage":
                    await broadcast_json(nb_id, {
                        "type": "llm_usage",
                        "cell_id": cid,
                        "provider": nb.llm_provider,
                        "usage": execution_diagnostics["usage"],
                    })
                elif item["type"] == "diagnostic":
                    pass
                elif item["type"] == "chunk":
                    await broadcast_json(nb_id, {"type": "stream_chunk", "cell_id": cid, "chunk": item["content"]})
        except asyncio.CancelledError:
            terminal_status = "cancelled"
            logger.info("Prompt generation cancelled (notebook=%s, cell=%s)", nb_id, cid)
        except Exception as e:
            terminal_status = "error"
            failure_info = {**_classify_llm_failure(str(e)), "message": str(e)}
            await broadcast_json(nb_id, {
                "type": "status_message",
                "message": f"Prompt failed: {e}",
                "tone": "error",
                "action": failure_info["action"],
                "action_label": failure_info["label"],
                "cell_id": cid,
            })
        finally:
            try:
                # Determine the final response text based on whether tools were used
                if had_any_tools:
                    response_text = "".join(post_tool_text)
                else:
                    response_text = "".join(pre_tool_text)
                response_text = _deduplicate_response_text(response_text)
                tool_steps_md = _format_tool_steps_markdown(tool_events)
                final_output = tool_steps_md + response_text if tool_steps_md else response_text
                if terminal_status == "error" and not final_output.strip():
                    final_output = previous_output
                execution_diagnostics["duration_ms"] = execution_diagnostics.get("duration_ms") or round(
                    (time.perf_counter() - execution_started) * 1000, 1,
                )
                execution_diagnostics["status"] = terminal_status
                if failure_info:
                    execution_diagnostics["failure"] = failure_info
                async with runtime_registry.mutation(nb_id):
                    current_cell = nb.get_cell(cid)
                    if current_cell is not None:
                        c = current_cell
                        output_changed = _commit_cell_output(
                            nb_id, nb, c, final_output, reason="prompt_output_commit",
                        )
                        c.time_run = datetime.now().strftime("%H:%M:%S")
                        previous_execution = c.llm_execution
                        c.llm_execution = dict(execution_diagnostics)
                        if not output_changed and previous_execution != c.llm_execution:
                            c.version += 1
                            c.last_modified = datetime.now()
                            _touch_notebook_revision(nb_id, "prompt_execution_commit")
                        nb.modified = True
                        await broadcast_json(nb_id, _prompt_output_payload(nb_id, c))
                        await broadcast_json(nb_id, {
                            "type": "prompt_execution_update",
                            "cell_id": cid,
                            "execution": c.llm_execution,
                            "notebook_usage": _notebook_llm_usage(nb),
                        })
                        log_operation(
                            logger,
                            "prompt_run_completed",
                            op_id=run_op_id,
                            notebook=nb_id,
                            cell=cid,
                            version=c.version,
                            notebook_revision=_get_notebook_revision(nb_id),
                            provider=nb.llm_provider,
                            model=nb.model,
                            transport=execution_diagnostics["transport"],
                            ttft_ms=execution_diagnostics["ttft_ms"],
                            duration_ms=execution_diagnostics["duration_ms"],
                            retry_count=execution_diagnostics["retry_count"],
                            usage=execution_diagnostics["usage"],
                            tool_steps=execution_diagnostics["tool_steps"],
                            status=terminal_status,
                            request_ids=execution_diagnostics["request_ids"],
                            response_ids=execution_diagnostics["response_ids"],
                        )

                if nb.get_cell(cid) is not None:
                    await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
                    await broadcast_json(nb_id, {
                        "type": "cell_class_update",
                        "cell_id": c.id,
                        "cls": get_cell_state_classes(c),
                    })
            except Exception:
                terminal_status = "error"
                logger.exception(
                    "Prompt terminal persistence failed (notebook=%s, cell=%s, operation_id=%s)",
                    nb_id, cid, run_op_id,
                )
            finally:
                # Never leave a notebook permanently blocked if persistence or
                # a UI broadcast fails during terminal processing.
                prompt_cancel_events.pop(cancel_key, None)

            terminal_payload = {
                "type": "prompt_run_terminal",
                "cell_id": cid,
                "operation_id": run_op_id,
                "status": terminal_status,
                "version": getattr(c, "version", None),
                "notebook_revision": _get_notebook_revision(nb_id),
            }
            for payload in (terminal_payload, {"type": "stream_end", "cell_id": cid}):
                try:
                    await broadcast_json(nb_id, payload)
                except Exception:
                    logger.exception(
                        "Prompt terminal broadcast failed (notebook=%s, cell=%s, event=%s)",
                        nb_id, cid, payload["type"],
                    )
            prompt_event = {
                "notebook_id": nb_id,
                "cell_id": cid,
                "provider": nb.llm_provider,
                "model": nb.model,
                "status": terminal_status,
                "duration_ms": execution_diagnostics.get("duration_ms"),
                "tool_steps": execution_diagnostics.get("tool_steps", 0),
            }
            if terminal_status == "completed":
                await event_bus.emit("prompt.completed", prompt_event)
            elif terminal_status == "error":
                await event_bus.emit("prompt.failed", {
                    **prompt_event,
                    "error": (failure_info or {}).get("message", "Prompt failed"),
                })

    # Selection and scroll position belong to the initiating browser. Prompt
    # output/header updates already arrive through granular WebSocket events, so
    # returning a full CellView here would replace the run cell after focus has
    # moved and can pull the viewport back to the top.
    return ""

@rt("/dialeng/{nb_id}/kernel/restart")
@serialized_notebook_mutation
async def post(nb_id: str):
    """Restart the kernel for a specific notebook."""
    await event_bus.emit("kernel.restarting", {"notebook_id": nb_id})
    await broadcast_kernel_status(nb_id, "restarting")
    logger.info("Kernel restart requested for notebook %s", nb_id)
    await _invalidate_kernel_background_work(nb_id, reason="kernel_restart")
    try:
        await kernel_service.restart_async(nb_id)
    except Exception as e:
        logger.exception("Kernel restart failed for notebook %s", nb_id)
        await broadcast_kernel_status(nb_id, "error")
        return Div(f"Kernel restart failed: {e}", cls="status error")

    # Clear CRAFT execution tracking so CRAFT cells re-execute on next page load
    from dialeng.services.craft_service import reset_craft_tracking
    reset_craft_tracking(nb_id)
    logger.info("Reset CRAFT execution tracking after restart for notebook %s", nb_id)

    craft_cells, craft_paths = _collect_craft_cells_for_setup(nb_id, only_unexecuted=False)
    logger.info(
        "Scheduling kernel restart setup (notebook=%s, craft_cells=%s)",
        nb_id, len(craft_cells),
    )
    _schedule_notebook_kernel_setup(
        nb_id,
        source="kernel_restart",
        craft_cells=craft_cells,
        craft_paths=craft_paths,
    )
    await _mark_outline_dirty(nb_id, reason="kernel_restart")

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
@serialized_notebook_mutation
async def post(nb_id: str):
    """Cancel running cell AND clear entire queue."""
    logger.info("Cancel-all requested for notebook %s", nb_id)
    queue = get_execution_queue(nb_id)

    # Get current queue status before cancelling
    status_before = queue.get_status(nb_id)
    logger.info(
        "Queue status before cancel-all (notebook=%s, running=%s, queued=%s)",
        nb_id, status_before.current_cell_id, status_before.queued_cell_ids,
    )

    # First interrupt the running cell
    kernel_service.interrupt(nb_id)
    logger.info("Kernel interrupt sent during cancel-all for notebook %s", nb_id)

    # Then clear the queue
    queue.cancel_all(nb_id)
    logger.info("Queue cleared during cancel-all for notebook %s", nb_id)

    # Get status after cancelling
    status_after = queue.get_status(nb_id)
    logger.info(
        "Queue status after cancel-all (notebook=%s, running=%s, queued=%s)",
        nb_id, status_after.current_cell_id, status_after.queued_cell_ids,
    )

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
@serialized_notebook_mutation
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
    logger.debug("[ADD_RELATIVE] dlg_name=%s, websocket_notebooks=%s", dlg_name, broadcast_manager.notebook_ids())
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
    nb.modified = True
    _touch_structure_revision(dlg_name, "dialoghelper_add_relative")

    # FOUST fix: broadcast granular cell_add JSON instead of AllCellsOOB.
    # This is the dialoghelper version of the add route — same pattern as /cell/add.
    try:
        await broadcast_json(dlg_name, _cell_add_payload(nb, dlg_name, new_cell.id))
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
@serialized_notebook_mutation
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
        _touch_structure_revision(dlg_name, "dialoghelper_rm_msg")
        await broadcast_json(dlg_name, _cell_delete_payload(nb, dlg_name, msid))
    return {"status": "ok", "id": msid}

def _str_to_bool(val: str) -> bool:
    """Convert string form value to boolean. Handles 'True', 'true', '1', etc."""
    if val is None:
        return None
    return str(val).lower() in ('true', '1', 'yes', 'on')


@rt("/update_msg_")
@serialized_notebook_mutation
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
    logger.debug("Dialoghelper update_msg requested (notebook=%s, cell=%s, pinned=%s)", dlg_name, id_, pinned)
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        logger.debug("Dialoghelper update_msg target before update (notebook=%s, cell=%s, pinned=%s)", dlg_name, cell.id, cell.pinned)

        # Track changes for logging
        changes = {}

        # Map and apply each field if provided (not None)
        if content is not None:
            if _str_to_bool(log_changed) and cell.source != content:
                changes["content"] = {"old": cell.source[:200], "new": content[:200]}
            _commit_cell_source(dlg_name, nb, cell, content, reason="dialoghelper_update_msg_source")
        if msg_type is not None:
            if _str_to_bool(log_changed) and cell.cell_type != msg_type:
                changes["msg_type"] = {"old": cell.cell_type, "new": msg_type}
            cell.cell_type = msg_type
            cell.version += 1
            cell.last_modified = datetime.now()
            nb.modified = True
            _touch_notebook_revision(dlg_name, "dialoghelper_update_msg_type")
        if output is not None:
            _commit_cell_output(dlg_name, nb, cell, output, reason="dialoghelper_update_msg_output")
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
            logger.debug("Dialoghelper update_msg set pinned (notebook=%s, cell=%s, pinned=%s)", dlg_name, cell.id, cell.pinned)

        if any(v is not None for v in [time_run, is_exported, skipped, pinned, i_collapsed, o_collapsed, heading_collapsed]):
            nb.modified = True
            _touch_notebook_revision(dlg_name, "dialoghelper_update_msg_metadata")

        # Log changes if requested
        if _str_to_bool(log_changed) and changes:
            log_change(dlg_name, "update", id_, changes)

        logger.debug("Dialoghelper update_msg target after update (notebook=%s, cell=%s, pinned=%s)", dlg_name, cell.id, cell.pinned)
        # Smart broadcast: use targeted updates to preserve Monaco editor DOM
        if msg_type is not None:
            # Type change requires full cell re-render
            await broadcast_to_notebook(dlg_name, CellViewOOB(cell, dlg_name))
        else:
            # Source change → JSON update (no FOUST)
            if content is not None:
                await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
            # Output change → targeted OOB
            if output is not None:
                await broadcast_to_notebook(dlg_name, CellOutputOOB(cell))
                if str(getattr(cell, "cell_type", "")) == "prompt":
                    await broadcast_json(dlg_name, _prompt_output_payload(dlg_name, cell))
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
@serialized_notebook_mutation
async def post(dlg_name: str, id_: str, insert_line: int, new_str: str):
    """Insert line at position in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines.insert(insert_line, new_str)
        _commit_cell_source(dlg_name, nb, cell, '\n'.join(lines), reason="dialoghelper_msg_insert_line")
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
    return {"status": "ok"}

@rt("/msg_str_replace_")
@serialized_notebook_mutation
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
        _commit_cell_source(dlg_name, nb, cell, '\n'.join(parts), reason="dialoghelper_msg_str_replace")
    else:
        count = n_matches if n_matches else 0
        _commit_cell_source(
            dlg_name,
            nb,
            cell,
            source.replace(old_str, new_str, count) if count else source.replace(old_str, new_str),
            reason="dialoghelper_msg_str_replace",
        )

    # JSON source update — preserves Monaco editor DOM (no FOUST)
    await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
    return {"status": "ok"}

@rt("/msg_strs_replace_")
@serialized_notebook_mutation
async def post(dlg_name: str, id_: str, old_strs: str, new_strs: str):
    """Replace multiple strings (JSON arrays)."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        old_list = json.loads(old_strs)
        new_list = json.loads(new_strs)
        source = cell.source
        for old, new in zip(old_list, new_list):
            source = source.replace(old, new, 1)
        _commit_cell_source(dlg_name, nb, cell, source, reason="dialoghelper_msg_strs_replace")
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
    return {"status": "ok"}

@rt("/msg_replace_lines_")
@serialized_notebook_mutation
async def post(dlg_name: str, id_: str, start_line: int, end_line: int, new_content: str):
    """Replace line range in message."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx >= 0:
        cell = nb.cells[idx]
        lines = cell.source.split('\n')
        lines[start_line:end_line] = [new_content]
        _commit_cell_source(dlg_name, nb, cell, '\n'.join(lines), reason="dialoghelper_msg_replace_lines")
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
    return {"status": "ok"}

@rt("/msg_del_lines_")
@serialized_notebook_mutation
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

    _commit_cell_source(dlg_name, nb, cell, '\n'.join(lines), reason="dialoghelper_msg_del_lines")
    # JSON source update — preserves Monaco editor DOM (no FOUST)
    await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
    return {"status": "ok"}

@rt("/msg_pyrun_")
@serialized_notebook_mutation
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
            _commit_cell_source(dlg_name, nb, cell, namespace["text"], reason="dialoghelper_msg_pyrun")
            # JSON source update — preserves Monaco editor DOM (no FOUST)
            await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Clipboard Operations
# ============================================================================

@rt("/msg_clipboard_")
@serialized_notebook_mutation
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
        _touch_structure_revision(dlg_name, "dialoghelper_clipboard_cut")
        # FOUST fix: granular cell_delete per cell instead of AllCellsOOB
        for cid in cell_ids:
            await broadcast_json(dlg_name, _cell_delete_payload(nb, dlg_name, cid))

    return result

@rt("/msg_paste_")
@serialized_notebook_mutation
async def post(dlg_name: str, id_: str = "", after: str = "True"):
    """Paste messages from clipboard."""
    nb = get_notebook(dlg_name)
    new_ids = clipboard_paste(nb, dlg_name, ref_id=id_, after=_str_to_bool(after))

    if new_ids:
        _touch_structure_revision(dlg_name, "dialoghelper_clipboard_paste")
        # FOUST fix: granular cell_add per pasted cell instead of AllCellsOOB
        for new_id in new_ids:
            for idx, c in enumerate(nb.cells):
                if c.id == new_id:
                    await broadcast_json(dlg_name, _cell_add_payload(nb, dlg_name, new_id))
                    break

    return {"status": "ok", "ids": new_ids}

# ============================================================================
# UI Toggle Operations
# ============================================================================

@rt("/toggle_header_collapse_")
@serialized_notebook_mutation
async def post(dlg_name: str, id_: str):
    """Toggle heading_collapsed state on a cell."""
    nb = get_notebook(dlg_name)
    idx = get_msg_idx(nb, id_)
    if idx < 0:
        return {"error": f"Message {id_} not found"}

    cell = nb.cells[idx]
    cell.heading_collapsed = not cell.heading_collapsed
    nb.modified = True
    _touch_notebook_revision(dlg_name, "dialoghelper_toggle_header_collapse")
    # Targeted header OOB (preserves editor DOM)
    await broadcast_to_notebook(dlg_name, CellHeaderOOB(cell, dlg_name))
    return {"status": "ok", "heading_collapsed": cell.heading_collapsed}

@rt("/dialeng/{nb_id}/cell/{cell_id}/toggle/{prop}")
@serialized_notebook_mutation
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
    nb.modified = True
    _touch_notebook_revision(nb_id, f"cell_toggle_{prop}")
    # Targeted header OOB + class update (preserves editor DOM)
    await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))
    await broadcast_json(nb_id, {"type": "cell_class_update", "cell_id": cell.id, "cls": get_cell_state_classes(cell)})
    return {"status": "ok", prop: getattr(cell, prop)}

@rt("/bookmark_")
@serialized_notebook_mutation
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
    nb.modified = True
    _touch_notebook_revision(dlg_name, "dialoghelper_bookmark")

    # Targeted header OOB (preserves editor DOM)
    await broadcast_to_notebook(dlg_name, CellHeaderOOB(cell, dlg_name))
    return {"status": "ok", "bookmark": cell.bookmark}

@rt("/toggle_comment_")
@serialized_notebook_mutation
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
            _commit_cell_source(dlg_name, nb, cell, '\n'.join(new_lines), reason="dialoghelper_toggle_comment")
        else:
            # Comment: add '# ' prefix
            _commit_cell_source(dlg_name, nb, cell, '\n'.join(
                (l[:len(l) - len(l.lstrip())] + '# ' + l.lstrip()) if l.strip() else l
                for l in lines
            ), reason="dialoghelper_toggle_comment")
        # JSON source update — preserves Monaco editor DOM (no FOUST)
        await broadcast_json(dlg_name, _cell_source_payload(dlg_name, cell))

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
            queue.cancel_all(name)
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"error": str(e)}

@rt("/rm_dialog_")
@serialized_notebook_mutation
async def post(name: str):
    """Delete a dialog/notebook from memory and optionally disk."""
    try:
        await _teardown_notebook_runtime(name, reason="memory_delete")
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

async def _cancel_prompt_cell(notebook_id: str, cell_id: str) -> None:
    cancelled_cells.add(cell_id)
    cancel_event = prompt_cancel_events.get(f"{notebook_id}:{cell_id}")
    if cancel_event:
        cancel_event.set()
    logger.info(
        "WebSocket cancel received (notebook=%s, cell=%s)",
        notebook_id,
        cell_id,
    )


async def _cancel_extension_command(operation_id: str) -> bool:
    cancel = extension_manager.context_services.get("cancel_extension_command")
    if cancel is None:
        return False
    result = cancel(operation_id)
    return bool(await result if inspect.isawaitable(result) else result)


register_websocket_routes(
    app,
    broadcast_manager=broadcast_manager,
    kernel_service=kernel_service,
    notebook_unavailable=lambda notebook_id: (
        notebook_id in deleting_notebooks or notebook_id in deleted_notebooks
    ),
    revision_snapshot=_get_revision_snapshot,
    send_kernel_snapshot=send_kernel_snapshot,
    cancel_cell=_cancel_prompt_cell,
    cancel_extension_command=_cancel_extension_command,
    logger=logger,
)

# ============================================================================
# Run
# ============================================================================

def main(root_dir: Path = None, port: int = 8000):
    """CLI entry point for dialeng."""
    if root_dir is not None:
        resolved_root = Path(root_dir).resolve()
        # Uvicorn's reload worker imports dialeng.app in a new process. Persist
        # the chosen project root so that worker does not fall back to ./notebooks.
        os.environ["DIALENG_NOTEBOOKS_DIR"] = str(resolved_root)
        set_root_dir(resolved_root)
    print(f"🚀 Dialeng starting at http://localhost:{port}")
    print(f"   Root directory: {NOTEBOOKS_DIR.resolve()}")
    print("   Format: Solveit-compatible .ipynb")
    print("")
    # Print credential status
    print_credential_status(CREDENTIAL_STATUS)
    print_credential_inventory(CREDENTIAL_INVENTORY)
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
    print("")
    print("   Press Ctrl+C to stop Dialeng and release attached kernels.")
    serve(
        appname="dialeng.app",
        port=port,
        reload_dirs=["dialeng"],
        log_config=build_log_config(),
    )


if __name__ == "__main__":
    main()
