"""
Execution queue for cell execution.

Provides FIFO queue semantics for cell execution, allowing
users to queue multiple cells while one is running. The UI
stays responsive while the queue is processed in the background.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Dict, List
from collections import deque
from datetime import datetime

from document.cell import Cell, CellState, CellOutput
from .kernel_service import KernelService


@dataclass
class QueuedExecution:
    """A cell execution request in the queue."""
    notebook_id: str
    cell: Cell
    queued_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher = more urgent (for "Run All" scenarios)


@dataclass
class QueueStatus:
    """Current status of an execution queue."""
    queued_count: int
    is_processing: bool
    current_cell_id: Optional[str]
    queued_cell_ids: List[str]


# Type aliases for callbacks
OutputCallback = Callable[[Cell, CellOutput], Awaitable[None]]
StateCallback = Callable[[Cell, CellState], Awaitable[None]]


class ExecutionQueue:
    """
    FIFO execution queue for notebook cells.

    Features:
    - Queue cells while one is running (FIFO order)
    - UI stays responsive (async background processing)
    - Cancel individual cells or all queued
    - Callbacks for streaming output and state changes
    - Support for WebSocket updates
    """

    def __init__(self, kernel_service: KernelService):
        """
        Initialize the execution queue.

        Args:
            kernel_service: KernelService instance for code execution
        """
        self.kernel = kernel_service
        self._queues: Dict[str, deque[QueuedExecution]] = {}
        self._processing: Dict[str, bool] = {}
        self._current_cell: Dict[str, Optional[str]] = {}

        # Callbacks for output streaming and state changes
        self._on_output: Dict[str, OutputCallback] = {}
        self._on_state_change: Dict[str, StateCallback] = {}

    def queue_cell(
        self,
        notebook_id: str,
        cell: Cell,
        priority: int = 0
    ) -> QueuedExecution:
        """
        Add cell to execution queue.

        Returns immediately - execution happens in background.

        Args:
            notebook_id: Notebook identifier
            cell: Cell to execute
            priority: Execution priority (higher = sooner)

        Returns:
            QueuedExecution object
        """
        # Initialize queue for notebook if needed
        if notebook_id not in self._queues:
            self._queues[notebook_id] = deque()
            self._processing[notebook_id] = False
            self._current_cell[notebook_id] = None

        # Create queued execution
        execution = QueuedExecution(
            notebook_id=notebook_id,
            cell=cell,
            priority=priority
        )

        # Update cell state to QUEUED
        cell.state = CellState.QUEUED

        # Add to queue (with priority handling)
        queue = self._queues[notebook_id]
        if priority > 0:
            # Insert based on priority (higher priority = earlier in queue)
            inserted = False
            for i, item in enumerate(queue):
                if item.priority < priority:
                    queue.insert(i, execution)
                    inserted = True
                    break
            if not inserted:
                queue.append(execution)
        else:
            queue.append(execution)

        # Start processing if not already running
        if not self._processing.get(notebook_id):
            asyncio.create_task(self._process_queue(notebook_id))

        return execution

    def queue_multiple(
        self,
        notebook_id: str,
        cells: List[Cell]
    ) -> List[QueuedExecution]:
        """
        Queue multiple cells (e.g., "Run All").

        Earlier cells get higher priority to maintain order.

        Args:
            notebook_id: Notebook identifier
            cells: List of cells to execute in order

        Returns:
            List of QueuedExecution objects
        """
        executions = []
        for i, cell in enumerate(cells):
            # Give earlier cells higher priority to maintain order
            priority = len(cells) - i
            executions.append(self.queue_cell(notebook_id, cell, priority))
        return executions

    def cancel_queued(self, notebook_id: str, cell_id: str) -> bool:
        """
        Remove a cell from queue if not yet running.

        Args:
            notebook_id: Notebook identifier
            cell_id: Cell to cancel

        Returns:
            True if cell was cancelled, False if not found or already running
        """
        if notebook_id not in self._queues:
            return False

        queue = self._queues[notebook_id]
        for execution in list(queue):
            if execution.cell.id == cell_id and execution.cell.state == CellState.QUEUED:
                queue.remove(execution)
                execution.cell.state = CellState.IDLE
                return True
        return False

    def cancel_all(self, notebook_id: str):
        """
        Cancel all queued cells and interrupt running execution.

        Args:
            notebook_id: Notebook identifier
        """
        if notebook_id not in self._queues:
            return

        # Cancel all queued cells
        queue = self._queues[notebook_id]
        while queue:
            execution = queue.popleft()
            if execution.cell.state == CellState.QUEUED:
                execution.cell.state = CellState.IDLE

        # Interrupt running execution
        self.kernel.interrupt(notebook_id)

    def get_status(self, notebook_id: str) -> QueueStatus:
        """
        Get current queue status for a notebook.

        Args:
            notebook_id: Notebook identifier

        Returns:
            QueueStatus object
        """
        queue = self._queues.get(notebook_id, deque())
        return QueueStatus(
            queued_count=len(queue),
            is_processing=self._processing.get(notebook_id, False),
            current_cell_id=self._current_cell.get(notebook_id),
            queued_cell_ids=[e.cell.id for e in queue]
        )

    def on_output(self, notebook_id: str, callback: OutputCallback):
        """
        Register callback for streaming output.

        The callback is called with (cell, output) for each
        chunk of output during execution.

        Args:
            notebook_id: Notebook identifier
            callback: Async callback function
        """
        self._on_output[notebook_id] = callback

    def on_state_change(self, notebook_id: str, callback: StateCallback):
        """
        Register callback for cell state changes.

        The callback is called with (cell, new_state) when
        a cell's state changes (QUEUED -> RUNNING -> SUCCESS/ERROR).

        Args:
            notebook_id: Notebook identifier
            callback: Async callback function
        """
        self._on_state_change[notebook_id] = callback

    def remove_callbacks(self, notebook_id: str):
        """Remove all callbacks for a notebook."""
        self._on_output.pop(notebook_id, None)
        self._on_state_change.pop(notebook_id, None)

    async def _process_queue(self, notebook_id: str):
        """
        Process execution queue for a notebook.

        This runs as a background task, processing cells one at a time.
        """
        self._processing[notebook_id] = True

        try:
            queue = self._queues.get(notebook_id, deque())

            while queue:
                execution = queue.popleft()
                cell = execution.cell

                # Skip if cell was cancelled
                if cell.state != CellState.QUEUED:
                    continue

                # Track current cell
                self._current_cell[notebook_id] = cell.id

                # Notify state change to RUNNING
                await self._notify_state(notebook_id, cell, CellState.RUNNING)

                # Execute with streaming
                async for output in self.kernel.execute_cell(notebook_id, cell):
                    # Notify output callback
                    await self._notify_output(notebook_id, cell, output)

                # Clear current cell
                self._current_cell[notebook_id] = None

                # Notify final state
                await self._notify_state(notebook_id, cell, cell.state)

        finally:
            self._processing[notebook_id] = False
            self._current_cell[notebook_id] = None

    async def _notify_output(
        self,
        notebook_id: str,
        cell: Cell,
        output: CellOutput
    ):
        """Notify registered callback of new output."""
        callback = self._on_output.get(notebook_id)
        if callback:
            try:
                await callback(cell, output)
            except Exception:
                pass  # Don't let callback errors break the queue

    async def _notify_state(
        self,
        notebook_id: str,
        cell: Cell,
        state: CellState
    ):
        """Notify registered callback of state change."""
        callback = self._on_state_change.get(notebook_id)
        if callback:
            try:
                await callback(cell, state)
            except Exception:
                pass  # Don't let callback errors break the queue
