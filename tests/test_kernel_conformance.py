"""Backend-agnostic kernel contract tests."""

import asyncio

from dialeng.document.cell import Cell, CellOutput, CellState
from dialeng.services.kernel.base_kernel import BaseKernel, KernelInfo, KernelStatus
from dialeng.services.kernel.kernel_service import KernelService
from dialeng.services.kernel.subprocess_kernel import SubprocessKernel


class _ToyKernel(BaseKernel):
    """Minimal conforming kernel used to test the shared service contract."""

    def __init__(self):
        self._execution_count = 0
        self._is_busy = False
        self.executions = []
        self.shutdown_called = False

    async def execute_streaming(self, code: str, notebook_id: str = "", cell_id: str = ""):
        self.executions.append((code, notebook_id, cell_id))
        self._is_busy = True
        try:
            yield CellOutput(output_type="stream", content=f"ran:{code}", stream_name="stdout")
            self._execution_count += 1
        finally:
            self._is_busy = False

    def interrupt(self) -> bool:
        return True

    def restart(self) -> bool:
        self._execution_count = 0
        return True

    def shutdown(self):
        self.shutdown_called = True

    @property
    def is_alive(self) -> bool:
        return True

    def get_status(self) -> KernelStatus:
        return KernelStatus(
            is_alive=self.is_alive,
            is_busy=self.is_busy,
            execution_count=self._execution_count,
            kernel_type="toy",
            last_activity_at=self.last_activity_at or None,
        )

    def get_info(self) -> KernelInfo:
        return KernelInfo(
            kernel_type="toy",
            display_name="Toy Kernel",
            is_remote=False,
            supports_shell_cells=True,
            supports_interrupt=True,
        )

    async def get_namespace_info(self, timeout: float = 5.0) -> dict:
        return {"variables": [], "functions": []}


def test_base_kernel_setup_code_collects_outputs_and_marks_activity():
    kernel = _ToyKernel()

    outputs = asyncio.run(
        kernel.run_setup_code(
            "bootstrap()",
            notebook_id="nb",
            cell_id="setup-cell",
            description="bootstrap",
        )
    )

    assert [output.content for output in outputs] == ["ran:bootstrap()"]
    assert kernel.executions == [("bootstrap()", "nb", "setup-cell")]
    assert kernel.last_activity_at > 0
    assert kernel.last_activity_reason == "setup:bootstrap"


def test_base_kernel_sync_project_files_contract_is_binary_safe():
    kernel = _ToyKernel()

    result = asyncio.run(
        kernel.sync_project_files(
            [("pkg/mod.py", "x = 1\n"), ("pkg/data.bin", b"\x00\xff")],
            notebook_id="nb",
            remote_root="remote",
        )
    )

    assert result["status"] == "noop"
    assert result["remote_root"] == "remote"
    assert result["file_count"] == 2
    assert result["total_bytes"] == len("x = 1\n".encode("utf-8")) + 2


def test_subprocess_kernel_sync_project_files_keeps_same_binary_safe_contract():
    kernel = SubprocessKernel()

    result = asyncio.run(
        kernel.sync_project_files(
            [("pkg/mod.py", "x = 1\n"), ("pkg/data.bin", b"\x00\xff")],
            notebook_id="nb",
            remote_root=".",
        )
    )

    assert result["status"] == "noop"
    assert result["file_count"] == 2
    assert result["total_bytes"] == len("x = 1\n".encode("utf-8")) + 2
    assert kernel.last_activity_reason == "sync_project_files:noop"


def test_kernel_service_executes_conforming_kernel_instances():
    service = KernelService()
    kernel = _ToyKernel()
    service.set_kernel_instance("nb", kernel)
    cell = Cell(id="cell-1", cell_type="code", source="print('hi')")

    async def _run():
        return [output async for output in service.execute_cell("nb", cell)]

    outputs = asyncio.run(_run())

    assert [output.content for output in outputs] == ["ran:print('hi')"]
    assert cell.state == CellState.SUCCESS
    assert cell.execution_count == 1
    assert kernel.get_status().execution_count == 1


def test_kernel_service_shutdown_uses_kernel_contract():
    service = KernelService()
    kernel = _ToyKernel()
    service.set_kernel_instance("nb", kernel)

    service.shutdown("nb")

    assert kernel.shutdown_called is True
    assert service.has_kernel("nb") is False
