"""Colab session manager - manages ColabKernel instances per notebook.

Ties together authentication, API client, and kernel lifecycle.
Used by KernelService when a notebook is configured for Colab execution.
"""
import logging
from typing import Dict

from .colab_auth import ColabAuthService
from .colab_api import ColabAPIClient
from .colab_kernel import ColabKernel
from .colab_state import ColabSessionStore

logger = logging.getLogger(__name__)


class ColabSessionManager:
    """Manages Colab kernel sessions across notebooks.

    Each notebook gets its own ColabKernel instance. The session manager
    provides the API client (backed by the auth service) to each kernel.
    """

    def __init__(self, auth_service: ColabAuthService, default_accelerator: str = ""):
        self.auth = auth_service
        self._api = ColabAPIClient(auth_service)
        self._kernels: Dict[str, ColabKernel] = {}
        self.default_accelerator = default_accelerator
        self._session_store = ColabSessionStore()

    def get_kernel(self, notebook_id: str, runtime_type: str = "cpu",
                   dialeng_port: int = 8000) -> ColabKernel:
        """Get or create a ColabKernel for a notebook.

        The kernel is created but NOT connected. Connection happens
        automatically on first execute_streaming() call.

        Args:
            notebook_id: Notebook identifier
            runtime_type: "cpu", "gpu", or "tpu"
            dialeng_port: Port of the Dialeng server for dialoghelper proxy
        """
        if notebook_id not in self._kernels:
            self._kernels[notebook_id] = ColabKernel(
                self._api,
                runtime_type=runtime_type,
                default_accelerator=self.default_accelerator,
                notebook_id=notebook_id,
                dialeng_port=dialeng_port
            )
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

    def forget_kernel(self, notebook_id: str) -> None:
        """Drop a cached kernel reference without shutting it down."""
        self._kernels.pop(notebook_id, None)

    def forget_all(self) -> None:
        """Drop all cached kernel references without touching the runtimes."""
        self._kernels.clear()

    def has_kernel(self, notebook_id: str) -> bool:
        """Check if a Colab kernel exists for the notebook."""
        return notebook_id in self._kernels

    async def prune_stale_metadata(self) -> int:
        """Drop persisted metadata for runtimes no longer listed by Colab."""
        assignments = await self._api.list_tunnel_assignments()
        active_endpoints = {assignment.endpoint for assignment in assignments}
        pruned = self._session_store.prune_stale_endpoints(active_endpoints)
        if pruned:
            logger.info("Pruned stale Colab session metadata entries: %s", pruned)
        return pruned
