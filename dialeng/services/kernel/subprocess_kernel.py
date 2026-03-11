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

from dialeng.document.cell import CellOutput
from .base_kernel import BaseKernel, KernelInfo, KernelStatus


class SubprocessKernel(BaseKernel):
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
            kernel_type="local",
            pid=self.pid
        )

    def get_info(self) -> KernelInfo:
        """Get static kernel backend information."""
        return KernelInfo(
            kernel_type="local",
            display_name="Local Python",
            is_remote=False,
            supports_shell_cells=True,
            supports_interrupt=True,
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

    async def introspect_variable(self, name: str, timeout: float = 5.0) -> dict:
        """
        Introspect a variable in the kernel namespace.

        Args:
            name: Variable name to introspect
            timeout: Max time to wait for response

        Returns:
            Dict with 'exists', 'var_type', 'repr' on success,
            or 'exists': False, 'error' on failure
        """
        if not self.is_alive:
            self._start_process()

        self.input_queue.put({
            'type': 'introspect_var',
            'name': name
        })

        loop = asyncio.get_event_loop()
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                msg = await loop.run_in_executor(
                    None,
                    lambda: self.output_queue.get(timeout=0.1)
                )
                if msg.get('type') == 'introspect_var_reply':
                    return msg
            except Empty:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return {
                        'name': name,
                        'exists': False,
                        'error': 'Timeout waiting for introspection response'
                    }
                if not self.is_alive:
                    return {
                        'name': name,
                        'exists': False,
                        'error': 'Kernel died during introspection'
                    }

    async def introspect_function(self, name: str, timeout: float = 5.0) -> dict:
        """
        Introspect a function in the kernel namespace.

        Args:
            name: Function name to introspect
            timeout: Max time to wait for response

        Returns:
            Dict with 'exists', 'signature', 'docstring', 'parameters' on success,
            or 'exists': False, 'error' on failure
        """
        if not self.is_alive:
            self._start_process()

        self.input_queue.put({
            'type': 'introspect_function',
            'name': name
        })

        loop = asyncio.get_event_loop()
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                msg = await loop.run_in_executor(
                    None,
                    lambda: self.output_queue.get(timeout=0.1)
                )
                if msg.get('type') == 'introspect_function_reply':
                    return msg
            except Empty:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return {
                        'name': name,
                        'exists': False,
                        'error': 'Timeout waiting for introspection response'
                    }
                if not self.is_alive:
                    return {
                        'name': name,
                        'exists': False,
                        'error': 'Kernel died during introspection'
                    }

    async def execute_tool(self, name: str, kwargs: dict, timeout: float = 60.0) -> dict:
        """
        Execute a function as a tool with the given arguments.

        Args:
            name: Function name to execute
            kwargs: Keyword arguments to pass to the function
            timeout: Max time to wait for execution

        Returns:
            Dict with 'status', 'result' on success,
            or 'status': 'error', 'error' message on failure
        """
        if not self.is_alive:
            self._start_process()

        self.input_queue.put({
            'type': 'execute_tool',
            'name': name,
            'kwargs': kwargs
        })
        self._is_busy = True

        loop = asyncio.get_event_loop()
        start_time = asyncio.get_event_loop().time()

        try:
            while True:
                try:
                    msg = await loop.run_in_executor(
                        None,
                        lambda: self.output_queue.get(timeout=0.1)
                    )
                    msg_type = msg.get('type')

                    if msg_type == 'execute_tool_reply':
                        return msg
                    elif msg_type == 'status':
                        self._is_busy = msg.get('status') == 'busy'
                        # Keep waiting for the actual reply

                except Empty:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        return {
                            'name': name,
                            'status': 'error',
                            'error': f'Timeout after {timeout}s waiting for tool execution'
                        }
                    if not self.is_alive:
                        return {
                            'name': name,
                            'status': 'error',
                            'error': 'Kernel died during tool execution'
                        }
        finally:
            self._is_busy = False


    async def complete(self, code: str, timeout: float = 3.0) -> list[str]:
        """Get code completions from the kernel."""
        if not self.is_alive or self._is_busy:
            return []

        self.input_queue.put({
            'type': 'complete',
            'code': code,
        })

        loop = asyncio.get_event_loop()
        start_time = loop.time()

        while True:
            try:
                msg = await loop.run_in_executor(
                    None,
                    lambda: self.output_queue.get(timeout=0.1)
                )
                if msg.get('type') == 'complete_reply':
                    return msg.get('matches', [])
            except Empty:
                if loop.time() - start_time > timeout:
                    return []
                if not self.is_alive:
                    return []

    async def get_namespace_info(self, timeout: float = 5.0) -> dict:
        """
        Get all user-defined variables and functions from the kernel namespace.

        Args:
            timeout: Max time to wait for response

        Returns:
            Dict with 'variables' and 'functions' lists
        """
        if not self.is_alive:
            return {'variables': [], 'functions': []}

        self.input_queue.put({'type': 'list_namespace'})

        loop = asyncio.get_event_loop()
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                msg = await loop.run_in_executor(
                    None,
                    lambda: self.output_queue.get(timeout=0.1)
                )
                if msg.get('type') == 'list_namespace_reply':
                    return {
                        'variables': msg.get('variables', []),
                        'functions': msg.get('functions', [])
                    }
            except Empty:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return {'variables': [], 'functions': []}
                if not self.is_alive:
                    return {'variables': [], 'functions': []}


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


# Register as a kernel backend
def _register_local_kernel():
    from dialeng.core.registry import registry, KernelRegistration
    registry.register_kernel_type(KernelRegistration(
        name="local", label="Local Python", icon="house-plug",
        factory=lambda **kw: SubprocessKernel(start_immediately=kw.get("start_immediately", True)),
        description="Local Python subprocess with persistent namespace"
    ))

_register_local_kernel()
