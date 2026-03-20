"""Cell data model with streaming output support."""
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Any, List
from datetime import datetime
import uuid


class CellType(str, Enum):
    """Type of cell content."""
    CODE = "code"
    NOTE = "note"
    PROMPT = "prompt"
    RAW = "raw"

    def __str__(self): return self.value
    def __format__(self, format_spec): return str.__format__(self.value, format_spec)


class CellState(str, Enum):
    """Execution state of a cell."""
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    SUCCESS = "success"


class CollapseLevel(int, Enum):
    """Collapse level for cell sections."""
    EXPANDED = 0
    SCROLLABLE = 1
    SUMMARY = 2


@dataclass
class CellOutput:
    """
    Represents a single output item from cell execution.

    Supports streaming - cells can have multiple outputs appended
    as execution progresses.
    """
    output_type: str  # 'stream', 'execute_result', 'error', 'display_data', 'update_display_data', 'clear_output'
    content: Any = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Stream-specific
    stream_name: Optional[str] = None  # 'stdout' or 'stderr'

    # Error-specific
    ename: Optional[str] = None
    evalue: Optional[str] = None
    traceback: Optional[List[str]] = None

    # Display data metadata
    metadata: Optional[dict] = None

    # Display ID for update_display_data (tqdm, widgets)
    display_id: Optional[str] = None


def is_benign_display_formatter_error(output: CellOutput) -> bool:
    """Return True for formatter-only IPython repr failures during display.

    Some libraries emit rich HTML/image output successfully but trigger a
    secondary formatter failure when IPython also attempts a ``text/plain``
    representation. Those tracebacks are noisy, often duplicated, and are not
    evidence that the underlying computation failed.
    """
    if output.output_type != 'error' or output.ename != 'TypeError':
        return False

    evalue = output.evalue or ""
    if "__repr__ returned non-string" not in evalue:
        return False

    tb_text = "\n".join(output.traceback or [])
    return (
        "IPython/core/formatters.py" in tb_text and
        "IPython/lib/pretty.py" in tb_text
    )


def normalize_cell_outputs(outputs: List[CellOutput]) -> List[CellOutput]:
    """Collapse streaming kernel events into their final notebook-visible state.

    Jupyter kernels can emit transient output events such as:
    - ``update_display_data`` to mutate an existing rich output in place
    - ``clear_output`` to clear previously shown output before the next update

    Dialeng streams those events live to the browser, but static re-renders and
    notebook saves need the *final* visible output state rather than the raw
    event log. This function applies those semantics and returns a normalized
    output list suitable for OOB rendering and persistence.
    """
    has_rich_display = any(
        out.output_type in {'display_data', 'update_display_data'}
        for out in outputs
    )
    normalized: List[CellOutput] = []
    display_index_by_id: dict[str, int] = {}
    pending_clear = False

    def _reset_outputs():
        normalized.clear()
        display_index_by_id.clear()

    def _flush_pending_clear():
        nonlocal pending_clear
        if pending_clear:
            _reset_outputs()
            pending_clear = False

    for out in outputs:
        if out.output_type == 'clear_output':
            if out.content:
                pending_clear = True
            else:
                _reset_outputs()
                pending_clear = False
            continue

        _flush_pending_clear()

        if out.output_type == 'error':
            if is_benign_display_formatter_error(out) and has_rich_display:
                continue

            if normalized and normalized[-1].output_type == 'error':
                prev = normalized[-1]
                if (
                    prev.ename == out.ename and
                    prev.evalue == out.evalue and
                    (prev.traceback or []) == (out.traceback or [])
                ):
                    continue

        if out.output_type == 'update_display_data':
            merged = replace(out, output_type='display_data')
            if out.display_id and out.display_id in display_index_by_id:
                normalized[display_index_by_id[out.display_id]] = merged
            else:
                normalized.append(merged)
                if out.display_id:
                    display_index_by_id[out.display_id] = len(normalized) - 1
            continue

        if out.output_type == 'display_data' and out.display_id:
            if out.display_id in display_index_by_id:
                normalized[display_index_by_id[out.display_id]] = out
            else:
                normalized.append(out)
                display_index_by_id[out.display_id] = len(normalized) - 1
            continue

        normalized.append(out)

    return normalized


@dataclass
class Cell:
    """
    A single cell in a notebook.

    Enhanced from original to support:
    - Multiple outputs for streaming
    - Execution state tracking
    - Queue position awareness
    - Version tracking for change detection
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    cell_type: CellType = CellType.CODE
    source: str = ""
    outputs: List[CellOutput] = field(default_factory=list)

    # Execution state (runtime, not persisted)
    state: CellState = CellState.IDLE
    execution_count: Optional[int] = None
    time_run: Optional[str] = None

    # Version tracking (for detecting changes and rebuilding context)
    # Incremented whenever source is modified
    version: int = 0
    last_modified: Optional[datetime] = None

    # Cell metadata (persisted)
    skipped: bool = False
    pinned: bool = False
    use_thinking: bool = False
    is_exported: bool = False

    # UI collapse state
    collapsed: bool = False
    input_collapse: CollapseLevel = CollapseLevel.EXPANDED
    output_collapse: CollapseLevel = CollapseLevel.EXPANDED
    heading_collapsed: bool = False

    # Bookmark (0 = no bookmark, 1-9 = numbered bookmark)
    bookmark: int = 0

    def __post_init__(self):
        # Coerce string cell_type to CellType enum
        if isinstance(self.cell_type, str) and not isinstance(self.cell_type, CellType):
            try: self.cell_type = CellType(self.cell_type)
            except ValueError: pass  # Extension types like "shell" stay as string
        # Coerce int collapse levels to CollapseLevel enum
        if isinstance(self.input_collapse, int) and not isinstance(self.input_collapse, CollapseLevel):
            try: self.input_collapse = CollapseLevel(self.input_collapse)
            except ValueError: pass
        if isinstance(self.output_collapse, int) and not isinstance(self.output_collapse, CollapseLevel):
            try: self.output_collapse = CollapseLevel(self.output_collapse)
            except ValueError: pass

    @property
    def output(self) -> str:
        """
        Concatenated text output for display.
        Backwards compatible with original single-output model.
        """
        parts = []
        for out in self.normalized_outputs():
            if out.output_type == 'stream':
                parts.append(str(out.content))
            elif out.output_type == 'execute_result':
                parts.append(str(out.content))
            elif out.output_type == 'error':
                if out.traceback:
                    parts.extend(out.traceback)
                else:
                    parts.append(f"{out.ename}: {out.evalue}")
        return ''.join(parts)

    @output.setter
    def output(self, value: str):
        """For backwards compatibility - converts string to CellOutput."""
        if not value:
            self.outputs = []
        else:
            self.outputs = [CellOutput(
                output_type='stream',
                content=value,
                stream_name='stdout'
            )]

    def clear_outputs(self):
        """Clear all outputs and reset state for re-execution."""
        self.outputs = []
        self.state = CellState.IDLE
        self.execution_count = None
        self.time_run = None

    def update_source(self, new_source: str) -> bool:
        """
        Update cell source with change tracking.

        Returns True if source was actually changed.
        Increments version and clears outputs if source changed.
        """
        if self.source != new_source:
            self.source = new_source
            self.version += 1
            self.last_modified = datetime.now()
            self.clear_outputs()  # Clear stale output that doesn't match new source
            return True
        return False

    def append_output(self, output: CellOutput):
        """Append a new output (for streaming)."""
        self.outputs.append(output)

    def normalized_outputs(self) -> List[CellOutput]:
        """Return outputs collapsed to the final notebook-visible state."""
        return normalize_cell_outputs(self.outputs)

    def sync_export_directive(self):
        """Sync the #| export directive in source with the is_exported flag.

        - If is_exported is True and source doesn't start with #| export, prepend it
        - If is_exported is False and source starts with #| export, remove the line
        Only applies to code cells.
        """
        cell_type = self.cell_type
        if hasattr(cell_type, 'value'):
            cell_type = cell_type.value
        if cell_type != "code":
            return

        has_directive = self.source.startswith("#| export")
        if self.is_exported and not has_directive:
            self.source = "#| export\n" + self.source
        elif not self.is_exported and has_directive:
            lines = self.source.split('\n')
            # Remove the first line if it's the export directive
            if lines and lines[0].strip().startswith("#| export"):
                self.source = '\n'.join(lines[1:])

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'cell_type': self.cell_type.value,
            'source': self.source,
            'outputs': [
                {
                    'output_type': o.output_type,
                    'content': o.content,
                    'stream_name': o.stream_name,
                    'ename': o.ename,
                    'evalue': o.evalue,
                    'traceback': o.traceback,
                    'metadata': o.metadata,
                    'display_id': o.display_id,
                }
                for o in self.outputs
            ],
            'execution_count': self.execution_count,
            'time_run': self.time_run,
            'version': self.version,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'skipped': self.skipped,
            'pinned': self.pinned,
            'use_thinking': self.use_thinking,
            'is_exported': self.is_exported,
            'collapsed': self.collapsed,
            'input_collapse': self.input_collapse.value,
            'output_collapse': self.output_collapse.value,
            'heading_collapsed': self.heading_collapsed,
            'bookmark': self.bookmark,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Cell':
        """Create Cell from dictionary."""
        outputs = []
        for o in data.get('outputs', []):
            outputs.append(CellOutput(
                output_type=o.get('output_type', 'stream'),
                content=o.get('content', ''),
                stream_name=o.get('stream_name'),
                ename=o.get('ename'),
                evalue=o.get('evalue'),
                traceback=o.get('traceback'),
                metadata=o.get('metadata'),
                display_id=o.get('display_id'),
            ))

        # Parse last_modified if present
        last_modified = None
        if data.get('last_modified'):
            try:
                last_modified = datetime.fromisoformat(data['last_modified'])
            except (ValueError, TypeError):
                pass

        return cls(
            id=data.get('id', uuid.uuid4().hex[:8]),
            cell_type=CellType(data.get('cell_type', 'code')),
            source=data.get('source', ''),
            outputs=outputs,
            execution_count=data.get('execution_count'),
            time_run=data.get('time_run'),
            version=data.get('version', 0),
            last_modified=last_modified,
            skipped=data.get('skipped', False),
            pinned=data.get('pinned', False),
            use_thinking=data.get('use_thinking', False),
            is_exported=data.get('is_exported', False),
            collapsed=data.get('collapsed', False),
            input_collapse=CollapseLevel(data.get('input_collapse', 0)),
            output_collapse=CollapseLevel(data.get('output_collapse', 0)),
            heading_collapsed=data.get('heading_collapsed', False),
            bookmark=data.get('bookmark', 0),
        )

    # ========================================================================
    # Jupyter .ipynb serialization
    # ========================================================================

    def to_jupyter_cell(self) -> dict:
        """Convert to Jupyter .ipynb cell format following Solveit conventions."""
        from .prompt_utils import join_prompt_content

        cell_type_str = str(self.cell_type)

        if cell_type_str == "code":
            cell = {
                "cell_type": "code",
                "id": self.id,
                "metadata": {},
                "source": self._to_source_lines(self.source),
                "execution_count": self.execution_count,
                "outputs": self._format_outputs_jupyter() if self.outputs else []
            }
            if self.time_run: cell["metadata"]["time_run"] = self.time_run
            if self.skipped: cell["metadata"]["skipped"] = True
            if self.is_exported: cell["metadata"]["is_exported"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
            if self.input_collapse: cell["metadata"]["input_collapse"] = int(self.input_collapse)
            if self.output_collapse: cell["metadata"]["output_collapse"] = int(self.output_collapse)
            if self.heading_collapsed: cell["metadata"]["heading_collapsed"] = True
            if self.bookmark: cell["metadata"]["bookmark"] = self.bookmark
        elif cell_type_str == "note":
            cell = {
                "cell_type": "markdown",
                "id": self.id,
                "metadata": {"solveit_note": True},
                "source": self._to_source_lines(self.source)
            }
            if self.collapsed: cell["metadata"]["collapsed"] = True
            if self.pinned: cell["metadata"]["pinned"] = True
            if self.input_collapse: cell["metadata"]["input_collapse"] = int(self.input_collapse)
        else:  # prompt (and any other type)
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
            if self.input_collapse: cell["metadata"]["input_collapse"] = int(self.input_collapse)
            if self.output_collapse: cell["metadata"]["output_collapse"] = int(self.output_collapse)
        return cell

    @classmethod
    def from_jupyter_cell(cls, cell: dict) -> 'Cell':
        """Create Cell from Jupyter .ipynb cell format."""
        from .serialization import _jupyter_to_cell
        return _jupyter_to_cell(cell)

    @staticmethod
    def _to_source_lines(text: str) -> list:
        """Convert source text to Jupyter source line format."""
        if not text: return []
        lines = text.split('\n')
        return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

    @staticmethod
    def _from_source_lines(source) -> str:
        """Convert Jupyter source lines back to text."""
        if isinstance(source, list): return ''.join(source)
        return source or ""

    def _format_outputs_jupyter(self) -> list:
        """Convert CellOutput list to Jupyter output format."""
        jupyter_outputs = []
        for out in self.normalized_outputs():
            if out.output_type == 'stream':
                text = str(out.content)
                lines = text.split('\n')
                text_lines = [line + '\n' for line in lines[:-1]]
                if lines[-1]: text_lines.append(lines[-1])
                jupyter_outputs.append({
                    "output_type": "stream",
                    "name": out.stream_name or "stdout",
                    "text": text_lines if text_lines else [text]
                })
            elif out.output_type == 'execute_result':
                jupyter_outputs.append({
                    "output_type": "execute_result",
                    "data": {"text/plain": [str(out.content)]},
                    "metadata": out.metadata or {},
                    "execution_count": self.execution_count
                })
            elif out.output_type == 'display_data':
                jupyter_outputs.append({
                    "output_type": "display_data",
                    "data": out.content if isinstance(out.content, dict) else {"text/plain": [str(out.content)]},
                    "metadata": out.metadata or {}
                })
            elif out.output_type == 'error':
                jupyter_outputs.append({
                    "output_type": "error",
                    "ename": out.ename or "Error",
                    "evalue": out.evalue or "",
                    "traceback": out.traceback or []
                })
        return jupyter_outputs

    @staticmethod
    def _extract_output(outputs: list) -> str:
        """Extract flat output string from Jupyter output list."""
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
