"""Cell data model with streaming output support."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List
from datetime import datetime
import uuid


class CellType(str, Enum):
    """Type of cell content."""
    CODE = "code"
    NOTE = "note"
    PROMPT = "prompt"


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
    output_type: str  # 'stream', 'execute_result', 'error', 'display_data'
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


@dataclass
class Cell:
    """
    A single cell in a notebook.

    Enhanced from original to support:
    - Multiple outputs for streaming
    - Execution state tracking
    - Queue position awareness
    """
    id: str = field(default_factory=lambda: f"_{uuid.uuid4().hex[:8]}")
    cell_type: CellType = CellType.CODE
    source: str = ""
    outputs: List[CellOutput] = field(default_factory=list)

    # Execution state (runtime, not persisted)
    state: CellState = CellState.IDLE
    execution_count: Optional[int] = None
    time_run: Optional[str] = None

    # Cell metadata (persisted)
    skipped: bool = False
    pinned: bool = False
    use_thinking: bool = False
    is_exported: bool = False

    # UI collapse state
    collapsed: bool = False
    input_collapse: CollapseLevel = CollapseLevel.EXPANDED
    output_collapse: CollapseLevel = CollapseLevel.EXPANDED

    @property
    def output(self) -> str:
        """
        Concatenated text output for display.
        Backwards compatible with original single-output model.
        """
        parts = []
        for out in self.outputs:
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
        self.outputs = [CellOutput(
            output_type='stream',
            content=value,
            stream_name='stdout'
        )]

    def clear_outputs(self):
        """Clear all outputs and reset state for re-execution."""
        self.outputs = []
        self.state = CellState.IDLE

    def append_output(self, output: CellOutput):
        """Append a new output (for streaming)."""
        self.outputs.append(output)

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
                }
                for o in self.outputs
            ],
            'execution_count': self.execution_count,
            'time_run': self.time_run,
            'skipped': self.skipped,
            'pinned': self.pinned,
            'use_thinking': self.use_thinking,
            'is_exported': self.is_exported,
            'collapsed': self.collapsed,
            'input_collapse': self.input_collapse.value,
            'output_collapse': self.output_collapse.value,
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
            ))

        return cls(
            id=data.get('id', f"_{uuid.uuid4().hex[:8]}"),
            cell_type=CellType(data.get('cell_type', 'code')),
            source=data.get('source', ''),
            outputs=outputs,
            execution_count=data.get('execution_count'),
            time_run=data.get('time_run'),
            skipped=data.get('skipped', False),
            pinned=data.get('pinned', False),
            use_thinking=data.get('use_thinking', False),
            is_exported=data.get('is_exported', False),
            collapsed=data.get('collapsed', False),
            input_collapse=CollapseLevel(data.get('input_collapse', 0)),
            output_collapse=CollapseLevel(data.get('output_collapse', 0)),
        )
