"""
Kernel service - manages kernels per notebook.

Provides a high-level interface for cell execution with
streaming output, managing one kernel per notebook.
Supports multiple kernel backends through the BaseKernel abstraction.
"""
import asyncio
import logging
from typing import Dict, AsyncIterator, Optional
from datetime import datetime

from dialeng.document.cell import (
    Cell,
    CellState,
    CellOutput,
    is_benign_display_formatter_error,
)
from .base_kernel import BaseKernel

logger = logging.getLogger(__name__)


class KernelService:
    """
    Service managing kernels per notebook.

    Each notebook gets its own kernel, providing isolated namespaces
    between notebooks. Supports multiple kernel backends (local subprocess,
    Google Colab, etc.) through the BaseKernel abstraction.
    """

    def __init__(self, lazy_start: bool = True):
        """
        Initialize the kernel service.

        Args:
            lazy_start: If True, kernels are started on first use.
                       If False, kernels must be explicitly started.
        """
        self._kernels: Dict[str, BaseKernel] = {}
        self._lazy_start = lazy_start
        self._colab_session_manager = None
        self._execution_locks: Dict[str, asyncio.Lock] = {}

    def set_colab_session_manager(self, manager):
        """Inject the Colab session manager for remote kernel support."""
        self._colab_session_manager = manager

    def get_kernel(self, notebook_id: str, kernel_type: str = "local",
                    runtime_type: str = "cpu") -> BaseKernel:
        """
        Get or create kernel for a notebook.

        Args:
            notebook_id: Unique identifier for the notebook
            kernel_type: Type of kernel ("local" or "colab")
            runtime_type: Colab runtime type ("cpu", "gpu", "tpu")

        Returns:
            BaseKernel instance for the notebook
        """
        if notebook_id not in self._kernels:
            from dialeng.core.registry import registry
            reg = registry.kernels.get(kernel_type)
            if kernel_type == "colab" and self._colab_session_manager:
                # Colab kernels are created via session manager (needs auth + API client)
                self._kernels[notebook_id] = self._colab_session_manager.get_kernel(
                    notebook_id, runtime_type=runtime_type
                )
            elif reg and reg.factory:
                self._kernels[notebook_id] = reg.factory(
                    start_immediately=self._lazy_start
                )
            else:
                # Fallback to local subprocess kernel
                from .subprocess_kernel import SubprocessKernel
                self._kernels[notebook_id] = SubprocessKernel(
                    start_immediately=self._lazy_start
                )
        return self._kernels[notebook_id]

    def _get_execution_lock(self, notebook_id: str) -> asyncio.Lock:
        """Return the per-notebook execution lock.

        Colab kernels multiplex all traffic over a single WebSocket, so background
        setup tasks and foreground cell execution must never overlap. The same
        lock also keeps local-kernel behavior deterministic.
        """
        lock = self._execution_locks.get(notebook_id)
        if lock is None:
            lock = asyncio.Lock()
            self._execution_locks[notebook_id] = lock
        return lock

    async def set_kernel_type(self, notebook_id: str, kernel_type: str,
                              runtime_type: str = "cpu") -> BaseKernel:
        """
        Switch a notebook's kernel type. Shuts down existing kernel first.

        Args:
            notebook_id: Notebook identifier
            kernel_type: New kernel type ("local" or "colab")
            runtime_type: Colab runtime type ("cpu", "gpu", "tpu")

        Returns:
            New BaseKernel instance
        """
        if notebook_id in self._kernels:
            old = self._kernels[notebook_id]
            if hasattr(old, 'shutdown_async'):
                await old.shutdown_async()
            else:
                old.shutdown()
            del self._kernels[notebook_id]
        return self.get_kernel(notebook_id, kernel_type, runtime_type)

    def set_kernel_instance(self, notebook_id: str, kernel: BaseKernel) -> None:
        """Replace the kernel object for a notebook without touching the lock."""
        self._kernels[notebook_id] = kernel

    def get_kernel_type(self, notebook_id: str) -> str:
        """Get the kernel type for a notebook."""
        if notebook_id in self._kernels:
            return self._kernels[notebook_id].get_info().kernel_type
        return "local"

    def has_kernel(self, notebook_id: str) -> bool:
        """Check if a kernel exists for the notebook."""
        return notebook_id in self._kernels

    def kernel_is_alive(self, notebook_id: str) -> bool:
        """Check if the notebook's kernel is running."""
        if notebook_id not in self._kernels:
            return False
        return self._kernels[notebook_id].is_alive

    def kernel_is_busy(self, notebook_id: str) -> bool:
        """Check if the notebook's kernel is busy executing."""
        if notebook_id not in self._kernels:
            return False
        return self._kernels[notebook_id].is_busy

    async def execute_cell(
        self,
        notebook_id: str,
        cell: Cell,
        source: Optional[str] = None
    ) -> AsyncIterator[CellOutput]:
        """
        Execute a code cell with streaming output.

        This is an async generator that:
        1. Sets cell state to RUNNING
        2. Clears previous outputs
        3. Yields CellOutput objects as they stream
        4. Updates cell state to SUCCESS/ERROR when done

        Args:
            notebook_id: Notebook identifier
            cell: Cell to execute
            source: Optional source code to execute (overrides cell.source).
                   This allows callbacks to transform code before execution.

        Yields:
            CellOutput objects for each chunk of output
        """
        lock = self._get_execution_lock(notebook_id)
        if lock.locked():
            logger.info(
                "Execution waiting for notebook lock",
                extra={"notebook_id": notebook_id, "cell_id": cell.id},
            )

        async with lock:
            kernel = self.get_kernel(notebook_id)

            # Use provided source or fall back to cell's source
            code_to_execute = source if source is not None else cell.source

            # Update cell state
            cell.state = CellState.RUNNING
            cell.outputs = []
            cell.time_run = datetime.now().strftime("%H:%M:%S")

            has_error = False
            saw_rich_display = False
            formatter_error_count = 0

            try:
                # Pass notebook_id and cell.id for dialoghelper magic variables
                async for output in kernel.execute_streaming(
                    code_to_execute,
                    notebook_id=notebook_id,
                    cell_id=cell.id
                ):
                    # Append to cell's outputs
                    cell.outputs.append(output)

                    if output.output_type in {'display_data', 'update_display_data'}:
                        saw_rich_display = True

                    # Track errors
                    if output.output_type == 'error':
                        if is_benign_display_formatter_error(output):
                            formatter_error_count += 1
                        else:
                            has_error = True

                    yield output

                # Update execution count
                cell.execution_count = kernel._execution_count

                if formatter_error_count:
                    if saw_rich_display:
                        logger.info(
                            "Suppressing %s formatter-only display error(s) for successful rich output",
                            formatter_error_count,
                            extra={"notebook_id": notebook_id, "cell_id": cell.id},
                        )
                    else:
                        has_error = True

            except Exception as e:
                # Unexpected error in the streaming itself
                error_output = CellOutput(
                    output_type='error',
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[f"{type(e).__name__}: {e}"]
                )
                cell.outputs.append(error_output)
                has_error = True
                yield error_output

            finally:
                # Set final state
                if cell.state == CellState.RUNNING:  # Wasn't interrupted
                    cell.state = CellState.ERROR if has_error else CellState.SUCCESS

    def interrupt(self, notebook_id: str) -> bool:
        """
        Interrupt the kernel for a notebook.

        Args:
            notebook_id: Notebook identifier

        Returns:
            True if interrupt was sent, False if no kernel/not running
        """
        if notebook_id not in self._kernels:
            return False
        return self._kernels[notebook_id].interrupt()

    async def interrupt_async(self, notebook_id: str) -> bool:
        """Async-safe interrupt for use from async route handlers."""
        if notebook_id not in self._kernels:
            return False
        kernel = self._kernels[notebook_id]
        # For Colab kernels, interrupt is sync but uses create_task internally
        # which needs an event loop - we're already in one here
        return kernel.interrupt()

    def restart(self, notebook_id: str) -> bool:
        """
        Restart the kernel for a notebook (sync - for local kernels only).

        Args:
            notebook_id: Notebook identifier

        Returns:
            True if restart succeeded
        """
        if notebook_id not in self._kernels:
            # Create a new kernel (default to local)
            from .subprocess_kernel import SubprocessKernel
            self._kernels[notebook_id] = SubprocessKernel()
            return True
        return self._kernels[notebook_id].restart()

    async def restart_async(self, notebook_id: str) -> bool:
        """Async-safe restart for use from async route handlers.

        Handles both local and Colab kernels correctly.
        """
        if notebook_id not in self._kernels:
            from .subprocess_kernel import SubprocessKernel
            self._kernels[notebook_id] = SubprocessKernel()
            return True
        kernel = self._kernels[notebook_id]
        info = kernel.get_info()
        if info.kernel_type == "colab" and hasattr(kernel, '_restart_async'):
            await kernel._restart_async()
            return True
        return kernel.restart()

    def shutdown(self, notebook_id: str):
        """
        Shutdown the kernel for a notebook.

        Args:
            notebook_id: Notebook identifier
        """
        if notebook_id in self._kernels:
            self._kernels[notebook_id].shutdown()
            del self._kernels[notebook_id]
        self._execution_locks.pop(notebook_id, None)

    def shutdown_all(self):
        """Shutdown all kernels."""
        for notebook_id in list(self._kernels.keys()):
            self.shutdown(notebook_id)

    async def complete(self, notebook_id: str, code: str) -> list[str]:
        """Get code completions for the given code text."""
        if notebook_id not in self._kernels:
            return []
        kernel = self._kernels[notebook_id]
        if not kernel.is_alive:
            return []
        return await kernel.complete(code)

    async def get_namespace_info(self, notebook_id: str) -> dict:
        """
        Get all user-defined variables and functions from the kernel namespace.

        Args:
            notebook_id: Notebook identifier

        Returns:
            Dict with 'variables' and 'functions' lists
        """
        if notebook_id not in self._kernels:
            return {'variables': [], 'functions': []}

        kernel = self._kernels[notebook_id]
        if not kernel.is_alive:
            return {'variables': [], 'functions': []}

        return await kernel.get_namespace_info()

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown_all()
