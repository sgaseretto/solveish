"""Colab session manager - manages ColabKernel instances per notebook.

Ties together authentication, API client, and kernel lifecycle.
Used by KernelService when a notebook is configured for Colab execution.
"""
import logging
from typing import Dict

from .colab_auth import ColabAuthService
from .colab_api import ColabAPIClient
from .colab_kernel import ColabKernel

logger = logging.getLogger(__name__)


class ColabSessionManager:
    """Manages Colab kernel sessions across notebooks.

    Each notebook gets its own ColabKernel instance. The session manager
    provides the API client (backed by the auth service) to each kernel.
    """

    def __init__(self, auth_service: ColabAuthService):
        self.auth = auth_service
        self._api = ColabAPIClient(auth_service)
        self._kernels: Dict[str, ColabKernel] = {}

    def get_kernel(self, notebook_id: str, runtime_type: str = "cpu") -> ColabKernel:
        """Get or create a ColabKernel for a notebook.

        The kernel is created but NOT connected. Connection happens
        automatically on first execute_streaming() call.

        Args:
            notebook_id: Notebook identifier
            runtime_type: "cpu", "gpu", or "tpu"
        """
        if notebook_id not in self._kernels:
            self._kernels[notebook_id] = ColabKernel(self._api, runtime_type=runtime_type)
            logger.info(f"Created ColabKernel for notebook {notebook_id} (runtime={runtime_type})")
        return self._kernels[notebook_id]

    async def set_runtime_type(self, notebook_id: str, runtime_type: str) -> ColabKernel:
        """Change the runtime type for a notebook's Colab kernel.

        Shuts down the existing kernel and creates a new one with the new runtime type.
        """
        if notebook_id in self._kernels:
            await self._kernels[notebook_id].shutdown_async()
            del self._kernels[notebook_id]
        return self.get_kernel(notebook_id, runtime_type)

    async def shutdown(self, notebook_id: str):
        """Shutdown a specific notebook's Colab kernel."""
        if notebook_id in self._kernels:
            await self._kernels[notebook_id].shutdown_async()
            del self._kernels[notebook_id]
            logger.info(f"Shutdown ColabKernel for notebook {notebook_id}")

    async def shutdown_all(self):
        """Shutdown all Colab kernels."""
        for notebook_id in list(self._kernels.keys()):
            await self.shutdown(notebook_id)

    def has_kernel(self, notebook_id: str) -> bool:
        """Check if a Colab kernel exists for the notebook."""
        return notebook_id in self._kernels
