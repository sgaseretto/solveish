"""
Kernel service - manages kernels per notebook.

Provides a high-level interface for cell execution with
streaming output, managing one kernel per notebook.
"""
from typing import Dict, AsyncIterator, Optional
from datetime import datetime

from document.cell import Cell, CellState, CellOutput
from .subprocess_kernel import SubprocessKernel


class KernelService:
    """
    Service managing kernels per notebook.

    Each notebook gets its own kernel subprocess, providing
    isolated namespaces between notebooks while maintaining
    state within a notebook's cells.
    """

    def __init__(self, lazy_start: bool = True):
        """
        Initialize the kernel service.

        Args:
            lazy_start: If True, kernels are started on first use.
                       If False, kernels must be explicitly started.
        """
        self._kernels: Dict[str, SubprocessKernel] = {}
        self._lazy_start = lazy_start

    def get_kernel(self, notebook_id: str) -> SubprocessKernel:
        """
        Get or create kernel for a notebook.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            SubprocessKernel instance for the notebook
        """
        if notebook_id not in self._kernels:
            self._kernels[notebook_id] = SubprocessKernel(
                start_immediately=self._lazy_start
            )
        return self._kernels[notebook_id]

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
        return self._kernels[notebook_id]._is_busy

    async def execute_cell(
        self,
        notebook_id: str,
        cell: Cell
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

        Yields:
            CellOutput objects for each chunk of output
        """
        kernel = self.get_kernel(notebook_id)

        # Update cell state
        cell.state = CellState.RUNNING
        cell.outputs = []
        cell.time_run = datetime.now().strftime("%H:%M:%S")

        has_error = False

        try:
            # Pass notebook_id and cell.id for dialoghelper magic variables
            async for output in kernel.execute_streaming(
                cell.source,
                notebook_id=notebook_id,
                cell_id=cell.id
            ):
                # Append to cell's outputs
                cell.outputs.append(output)

                # Track errors
                if output.output_type == 'error':
                    has_error = True

                yield output

            # Update execution count
            cell.execution_count = kernel._execution_count

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

        Sends SIGINT to the kernel subprocess, which will
        raise KeyboardInterrupt in the running code.

        Args:
            notebook_id: Notebook identifier

        Returns:
            True if interrupt was sent, False if no kernel/not running
        """
        if notebook_id not in self._kernels:
            return False
        return self._kernels[notebook_id].interrupt()

    def restart(self, notebook_id: str) -> bool:
        """
        Restart the kernel for a notebook.

        This kills the subprocess and starts a new one,
        clearing all namespace state.

        Args:
            notebook_id: Notebook identifier

        Returns:
            True if restart succeeded
        """
        if notebook_id not in self._kernels:
            # Create a new kernel
            self._kernels[notebook_id] = SubprocessKernel()
            return True
        return self._kernels[notebook_id].restart()

    def shutdown(self, notebook_id: str):
        """
        Shutdown the kernel for a notebook.

        Args:
            notebook_id: Notebook identifier
        """
        if notebook_id in self._kernels:
            self._kernels[notebook_id].shutdown()
            del self._kernels[notebook_id]

    def shutdown_all(self):
        """Shutdown all kernels."""
        for notebook_id in list(self._kernels.keys()):
            self.shutdown(notebook_id)

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown_all()
