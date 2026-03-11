"""Kernel services - Code execution with streaming and queue."""
from .kernel_service import KernelService
from .execution_queue import ExecutionQueue
from .base_kernel import BaseKernel, KernelInfo, KernelStatus

# Import kernel implementations so they self-register with the registry
from . import subprocess_kernel  # noqa: F401 — registers "local" kernel

__all__ = ['KernelService', 'ExecutionQueue', 'BaseKernel', 'KernelInfo', 'KernelStatus']
