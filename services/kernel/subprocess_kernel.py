"""
Subprocess kernel manager.

This module manages the kernel subprocess and provides an async interface
for code execution with streaming output and hard interrupt support.
"""
import os
import signal
import asyncio
from multiprocessing import Process, Queue
from queue import Empty
from typing import AsyncIterator, Optional
from dataclasses import dataclass
from datetime import datetime

from document.cell import CellOutput


@dataclass
class KernelStatus:
    """Current status of the kernel."""
    is_alive: bool
    is_busy: bool
    execution_count: int
    pid: Optional[int] = None


class SubprocessKernel:
    """
    Kernel running in a subprocess with streaming output.

    Key features:
    - Hard interrupt via SIGINT to subprocess
    - Streaming output via multiprocessing Queue
    - Persistent namespace across cells (until restart)
    - Rich output support (images, plots, HTML)
    """

    def __init__(self, start_immediately: bool = True):
        self.process: Optional[Process] = None
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self._execution_count: int = 0
        self._is_busy: bool = False

        if start_immediately:
            self._start_process()

    def _start_process(self):
        """Start the kernel subprocess."""
        # Import here to avoid circular imports and ensure
        # the import happens in the right context
        from .kernel_worker import kernel_worker_main

        self.input_queue = Queue()
        self.output_queue = Queue()

        self.process = Process(
            target=kernel_worker_main,
            args=(self.input_queue, self.output_queue),
            daemon=True  # Die with parent process
        )
        self.process.start()

        # Wait for ready signal
        try:
            msg = self.output_queue.get(timeout=10)
            if msg.get('type') == 'status' and msg.get('status') == 'ready':
                return True
        except Empty:
            pass

        raise RuntimeError("Kernel subprocess failed to start")

    @property
    def is_alive(self) -> bool:
        """Check if kernel subprocess is running."""
        return self.process is not None and self.process.is_alive()

    @property
    def pid(self) -> Optional[int]:
        """Get kernel subprocess PID."""
        return self.process.pid if self.process else None

    def get_status(self) -> KernelStatus:
        """Get current kernel status."""
        return KernelStatus(
            is_alive=self.is_alive,
            is_busy=self._is_busy,
            execution_count=self._execution_count,
            pid=self.pid
        )

    async def execute_streaming(
        self,
        code: str,
        notebook_id: str = "",
        cell_id: str = ""
    ) -> AsyncIterator[CellOutput]:
        """
        Execute code and yield outputs as they stream.

        This is an async generator that yields CellOutput objects
        as stdout/stderr/display_data arrives from the subprocess.

        Args:
            code: Python code to execute
            notebook_id: Notebook identifier (for dialoghelper __dialog_name)
            cell_id: Cell identifier (for dialoghelper __msg_id)

        Yields:
            CellOutput objects for each chunk of output
        """
        if not self.is_alive:
            self._start_process()

        # Send execute request with context for dialoghelper magic variables
        self.input_queue.put({
            'type': 'execute',
            'code': code,
            'notebook_id': notebook_id,
            'cell_id': cell_id
        })
        self._is_busy = True

        loop = asyncio.get_event_loop()

        try:
            while True:
                try:
                    # Non-blocking queue read with timeout
                    msg = await loop.run_in_executor(
                        None,
                        lambda: self.output_queue.get(timeout=0.05)
                    )
                except Empty:
                    # Check if process is still alive
                    if not self.is_alive:
                        yield CellOutput(
                            output_type='error',
                            ename='KernelDied',
                            evalue='Kernel subprocess died unexpectedly',
                            traceback=['KernelDied: The kernel subprocess terminated unexpectedly']
                        )
                        break
                    continue

                msg_type = msg.get('type')

                if msg_type == 'execute_done':
                    self._execution_count = msg.get('execution_count', self._execution_count + 1)
                    break

                elif msg_type == 'stream':
                    yield CellOutput(
                        output_type='stream',
                        content=msg.get('text', ''),
                        stream_name=msg.get('name', 'stdout')
                    )

                elif msg_type == 'display_data':
                    yield CellOutput(
                        output_type='display_data',
                        content=msg.get('data', {}),
                        metadata=msg.get('metadata')
                    )

                elif msg_type == 'execute_result':
                    data = msg.get('data', {})
                    yield CellOutput(
                        output_type='execute_result',
                        content=data.get('text/plain', ''),
                        metadata=msg.get('metadata')
                    )

                elif msg_type == 'error':
                    yield CellOutput(
                        output_type='error',
                        ename=msg.get('ename', 'Error'),
                        evalue=msg.get('evalue', ''),
                        traceback=msg.get('traceback', [])
                    )

                elif msg_type == 'status':
                    # Status updates (busy/idle) - don't yield, just track
                    self._is_busy = msg.get('status') == 'busy'

                elif msg_type == 'clear_output':
                    # Clear output signal - could be used by frontend
                    yield CellOutput(
                        output_type='clear_output',
                        content={'wait': msg.get('wait', False)}
                    )

        finally:
            self._is_busy = False

    def interrupt(self) -> bool:
        """
        Send SIGINT to kernel subprocess - hard interrupt.

        This will interrupt any running Python code, including
        C extensions in tight loops.

        Returns:
            True if interrupt signal was sent, False if no kernel running
        """
        if self.process and self.process.is_alive():
            try:
                os.kill(self.process.pid, signal.SIGINT)
                return True
            except (ProcessLookupError, PermissionError):
                return False
        return False

    def restart(self) -> bool:
        """
        Kill and restart the kernel subprocess.

        This clears all namespace state and starts fresh.

        Returns:
            True if restart succeeded
        """
        self.shutdown()
        try:
            self._start_process()
            self._execution_count = 0
            return True
        except RuntimeError:
            return False

    def shutdown(self):
        """Shutdown the kernel subprocess cleanly."""
        if self.process is None:
            return

        # Try graceful shutdown first
        if self.input_queue:
            try:
                self.input_queue.put({'type': 'shutdown'})
                self.process.join(timeout=2)
            except Exception:
                pass

        # Force terminate if still alive
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)

        # Last resort: kill
        if self.process.is_alive():
            self.process.kill()

        self.process = None
        self.input_queue = None
        self.output_queue = None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown()


# Convenience function for one-off execution
async def execute_code(code: str) -> list[CellOutput]:
    """
    Execute code and return all outputs.

    This is a convenience function that creates a temporary kernel,
    runs the code, and returns all outputs.
    """
    kernel = SubprocessKernel()
    try:
        outputs = []
        async for output in kernel.execute_streaming(code):
            outputs.append(output)
        return outputs
    finally:
        kernel.shutdown()
