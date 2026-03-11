"""Base kernel protocol for multi-kernel support.

Defines the abstract interface that all kernel backends must implement.
Both SubprocessKernel (local) and ColabKernel (remote) inherit from this.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field

from dialeng.document.cell import CellOutput


@dataclass
class KernelInfo:
    """Static metadata about a kernel backend."""
    kernel_type: str           # "local" | "colab"
    display_name: str          # "Local Python" | "Google Colab"
    is_remote: bool            # False for local, True for Colab
    supports_shell_cells: bool # Whether dedicated shell cells work
    supports_interrupt: bool   # Whether interrupt is supported


@dataclass
class KernelStatus:
    """Current status of any kernel."""
    is_alive: bool
    is_busy: bool
    execution_count: int
    kernel_type: str = "local"
    pid: Optional[int] = None
    # Remote-specific fields
    runtime_id: Optional[str] = None
    connection_state: Optional[str] = None  # "disconnected" | "connecting" | "connected"


class BaseKernel(ABC):
    """Abstract base class for all kernel implementations.

    Both SubprocessKernel and ColabKernel implement this interface.
    KernelService and ExecutionQueue interact only through this interface.
    """

    @abstractmethod
    async def execute_streaming(
        self,
        code: str,
        notebook_id: str = "",
        cell_id: str = ""
    ) -> AsyncIterator[CellOutput]:
        """Execute code and yield outputs as they stream."""
        ...

    @abstractmethod
    def interrupt(self) -> bool:
        """Interrupt currently running execution."""
        ...

    @abstractmethod
    def restart(self) -> bool:
        """Restart the kernel (clear state)."""
        ...

    @abstractmethod
    def shutdown(self):
        """Shutdown the kernel and release resources."""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the kernel is running and ready."""
        ...

    @property
    def is_busy(self) -> bool:
        """Whether the kernel is currently executing code."""
        return getattr(self, '_is_busy', False)

    @abstractmethod
    def get_status(self) -> KernelStatus:
        """Get current kernel status."""
        ...

    @abstractmethod
    def get_info(self) -> KernelInfo:
        """Get static kernel backend information."""
        ...

    @abstractmethod
    async def get_namespace_info(self, timeout: float = 5.0) -> dict:
        """Get user-defined variables and functions."""
        ...

    # Optional methods with default implementations
    async def introspect_variable(self, name: str, timeout: float = 5.0) -> dict:
        return {'name': name, 'exists': False, 'error': 'Not supported by this kernel'}

    async def introspect_function(self, name: str, timeout: float = 5.0) -> dict:
        return {'name': name, 'exists': False, 'error': 'Not supported by this kernel'}

    async def execute_tool(self, name: str, kwargs: dict, timeout: float = 60.0) -> dict:
        return {'name': name, 'status': 'error', 'error': 'Not supported by this kernel'}

    async def complete(self, code: str, timeout: float = 3.0) -> list[str]:
        """Get code completions for code text up to cursor position."""
        return []
