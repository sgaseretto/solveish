"""
Dialeng 2-Way Callback System

Inspired by fastai's callback system, but adapted for cell execution.
Callbacks can both observe AND modify execution behavior.

Key features:
- ExecutionContext: Shared mutable context that callbacks can modify
- Callback ordering: Lower order values run first
- Exception-based flow control: CancelCellException, CancelQueueException
- Async support throughout

Example usage:
    class AutoImportCallback(Callback):
        order = 0  # Run early

        def before_execution(self, ctx: ExecutionContext):
            # Modify source code before execution
            if 'np.' in ctx.source and 'import numpy' not in ctx.source:
                ctx.source = "import numpy as np\\n" + ctx.source

    class OutputFilterCallback(Callback):
        def on_output(self, ctx, output):
            # Filter or transform output
            if len(output.content) > 10000:
                output.content = output.content[:10000] + "\\n... truncated"
            return output  # Return None to filter out
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Any
from datetime import datetime
import logging

if TYPE_CHECKING:
    from dialeng.document.cell import Cell, CellOutput

logger = logging.getLogger(__name__)


class CancelCellException(Exception):
    """Raise to skip remaining steps for current cell."""
    pass


class CancelQueueException(Exception):
    """Raise to cancel entire execution queue."""
    pass


@dataclass
class ExecutionContext:
    """
    Shared context passed to callbacks. Callbacks CAN modify these fields.

    This is the key to 2-way callbacks: instead of just observing,
    callbacks can transform the execution by modifying ctx.source,
    ctx.outputs, or setting ctx.skip_execution.

    Attributes:
        cell: The cell being executed (read-only reference)
        notebook_id: Parent notebook ID
        source: The source code to execute (MODIFIABLE - callbacks can transform)
        outputs: Accumulated outputs (MODIFIABLE - callbacks can filter/transform)
        execution_count: Will be set after execution
        start_time: Set by timing callbacks
        skip_execution: Set True to skip kernel execution (for custom cell types)
        metadata: Arbitrary dict for callback communication
    """
    cell: 'Cell'
    notebook_id: str
    source: str  # Modifiable! Callbacks can transform code
    outputs: List['CellOutput'] = field(default_factory=list)

    # Execution metadata
    execution_count: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Control flow
    skip_execution: bool = False  # Set True to skip kernel execution

    # Arbitrary metadata for callback communication
    metadata: dict = field(default_factory=dict)


class Callback:
    """
    Base class for execution callbacks.

    Subclass and override methods to hook into the execution lifecycle.
    All methods are optional - only override what you need.

    Ordering:
        Lower `order` values run first. Default is 0.
        Use negative values for callbacks that should run early (e.g., validation).
        Use positive values for callbacks that should run late (e.g., logging).

    Flow Control:
        Raise CancelCellException to skip remaining steps for current cell.
        Raise CancelQueueException to cancel entire queue.

    2-Way Modification:
        before_execution: Modify ctx.source to transform code before execution
        on_output: Return modified output, or None to filter it out
        after_execution: Access final ctx for post-processing
    """
    order: int = 0  # Lower = runs first
    name: str = ""  # Optional name for debugging

    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    def before_queue(self, ctx: ExecutionContext) -> None:
        """
        Called when cell is added to queue.

        Can cancel by raising CancelCellException.
        Useful for validation before queueing.
        """
        pass

    def before_execution(self, ctx: ExecutionContext) -> None:
        """
        Called before code runs. Can modify ctx.source!

        This is the primary hook for code transformation.
        Examples:
        - Auto-imports
        - Magic command expansion
        - Code instrumentation

        Set ctx.skip_execution = True to prevent kernel execution
        (useful for custom cell types that handle their own execution).
        """
        pass

    def on_output(self, ctx: ExecutionContext, output: 'CellOutput') -> Optional['CellOutput']:
        """
        Called for each output chunk during streaming.

        Return the output to keep it (possibly modified).
        Return None to filter it out.

        Examples:
        - Truncate long outputs
        - Filter sensitive data
        - Transform output format
        """
        return output

    def after_execution(self, ctx: ExecutionContext, error: Optional[Exception] = None) -> None:
        """
        Called after execution completes (success or error).

        The error parameter is the exception if execution failed, None otherwise.
        ctx.outputs contains all outputs that weren't filtered.

        Examples:
        - Timing/profiling
        - Logging
        - Cleanup
        """
        pass

    def after_cancel(self, ctx: ExecutionContext) -> None:
        """
        Called if execution was cancelled (CancelCellException raised).

        Useful for cleanup when a cell is skipped.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order})"


class CallbackHandler:
    """
    Manages a list of callbacks and runs them in order.

    Thread-safe for adding/removing callbacks during iteration.
    All `run_*` methods are async for compatibility with the execution queue.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = []
        if callbacks:
            for cb in callbacks:
                self.add(cb)

    def add(self, callback: Callback) -> 'CallbackHandler':
        """Add callback and re-sort by order. Returns self for chaining."""
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.order)
        return self

    def remove(self, callback_or_type) -> bool:
        """
        Remove callback by instance or type.

        Args:
            callback_or_type: Either a Callback instance or a Callback subclass

        Returns:
            True if any callbacks were removed
        """
        if isinstance(callback_or_type, type):
            original_len = len(self.callbacks)
            self.callbacks = [c for c in self.callbacks if not isinstance(c, callback_or_type)]
            return len(self.callbacks) < original_len
        else:
            if callback_or_type in self.callbacks:
                self.callbacks.remove(callback_or_type)
                return True
            return False

    def clear(self) -> None:
        """Remove all callbacks."""
        self.callbacks.clear()

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    async def run_before_queue(self, ctx: ExecutionContext) -> None:
        """Run before_queue on all callbacks. May raise CancelCellException."""
        for cb in self.callbacks:
            try:
                cb.before_queue(ctx)
            except CancelCellException:
                logger.info(f"Cell cancelled by {cb.name} in before_queue")
                await self.run_after_cancel(ctx)
                raise
            except CancelQueueException:
                logger.info(f"Queue cancelled by {cb.name} in before_queue")
                raise
            except Exception as e:
                logger.warning(f"Callback {cb.name}.before_queue failed: {e}")

    async def run_before_execution(self, ctx: ExecutionContext) -> None:
        """Run before_execution on all callbacks. May raise CancelCellException."""
        for cb in self.callbacks:
            try:
                cb.before_execution(ctx)
            except CancelCellException:
                logger.info(f"Cell cancelled by {cb.name} in before_execution")
                await self.run_after_cancel(ctx)
                raise
            except CancelQueueException:
                logger.info(f"Queue cancelled by {cb.name} in before_execution")
                raise
            except Exception as e:
                logger.warning(f"Callback {cb.name}.before_execution failed: {e}")

    async def run_on_output(
        self,
        ctx: ExecutionContext,
        output: 'CellOutput'
    ) -> Optional['CellOutput']:
        """
        Run on_output on all callbacks.

        Each callback can transform the output or return None to filter.
        Returns the final output (possibly transformed) or None if filtered.
        """
        current_output = output
        for cb in self.callbacks:
            try:
                result = cb.on_output(ctx, current_output)
                if result is None:
                    logger.debug(f"Output filtered by {cb.name}")
                    return None
                current_output = result
            except Exception as e:
                logger.warning(f"Callback {cb.name}.on_output failed: {e}")
        return current_output

    async def run_after_execution(
        self,
        ctx: ExecutionContext,
        error: Optional[Exception] = None
    ) -> None:
        """Run after_execution on all callbacks."""
        for cb in self.callbacks:
            try:
                cb.after_execution(ctx, error)
            except Exception as e:
                logger.warning(f"Callback {cb.name}.after_execution failed: {e}")

    async def run_after_cancel(self, ctx: ExecutionContext) -> None:
        """Run after_cancel on all callbacks."""
        for cb in self.callbacks:
            try:
                cb.after_cancel(ctx)
            except Exception as e:
                logger.warning(f"Callback {cb.name}.after_cancel failed: {e}")


# ============================================================================
# Built-in Callbacks
# ============================================================================

class TimingCallback(Callback):
    """
    Track execution time.

    Sets ctx.start_time before execution and ctx.end_time after.
    Updates cell.time_run with formatted duration.
    """
    order = -100  # Run very early (to capture full time)
    name = "TimingCallback"

    def before_execution(self, ctx: ExecutionContext) -> None:
        import time
        ctx.start_time = time.time()

    def after_execution(self, ctx: ExecutionContext, error: Optional[Exception] = None) -> None:
        import time
        ctx.end_time = time.time()
        if ctx.start_time:
            elapsed = ctx.end_time - ctx.start_time
            ctx.cell.time_run = f"{elapsed:.2f}s"


class LoggingCallback(Callback):
    """
    Log execution lifecycle events.

    Useful for debugging callback behavior.
    """
    order = 100  # Run late (to see final state)
    name = "LoggingCallback"

    def before_execution(self, ctx: ExecutionContext) -> None:
        logger.info(f"Executing cell {ctx.cell.id} ({ctx.cell.cell_type})")

    def after_execution(self, ctx: ExecutionContext, error: Optional[Exception] = None) -> None:
        if error:
            logger.info(f"Cell {ctx.cell.id} failed: {error}")
        else:
            logger.info(f"Cell {ctx.cell.id} completed with {len(ctx.outputs)} outputs")


class OutputTruncateCallback(Callback):
    """
    Truncate outputs that exceed a maximum length.

    Prevents memory issues and UI slowdowns from very large outputs.
    """
    order = 50
    name = "OutputTruncateCallback"

    def __init__(self, max_chars: int = 100000, max_lines: int = 5000):
        super().__init__()
        self.max_chars = max_chars
        self.max_lines = max_lines

    def on_output(self, ctx: ExecutionContext, output: 'CellOutput') -> Optional['CellOutput']:
        if output.output_type == 'stream' and isinstance(output.content, str):
            content = output.content

            # Truncate by lines
            lines = content.split('\n')
            if len(lines) > self.max_lines:
                content = '\n'.join(lines[:self.max_lines])
                content += f"\n... ({len(lines) - self.max_lines} more lines truncated)"

            # Truncate by chars
            if len(content) > self.max_chars:
                content = content[:self.max_chars]
                content += f"\n... (output truncated at {self.max_chars} chars)"

            output.content = content

        return output
