"""Targeted tests for the Colab robustness work."""

import asyncio
import json
import time
from types import SimpleNamespace

from dialeng.document.cell import (
    Cell,
    CellOutput,
    normalize_cell_outputs,
    is_benign_display_formatter_error,
)
from dialeng.document.notebook import Notebook
from dialeng.document.serialization import save_notebook
from dialeng.services.kernel.base_kernel import BaseKernel, KernelInfo, KernelStatus
from dialeng.services.kernel.kernel_service import KernelService
from dialeng.ui.cells.code_cell import _render_cell_outputs


class _SerialFakeKernel(BaseKernel):
    """Minimal kernel that records concurrent executions."""

    def __init__(self):
        self._execution_count = 0
        self._is_busy = False
        self.active_executions = 0
        self.max_active_executions = 0

    async def execute_streaming(self, code: str, notebook_id: str = "", cell_id: str = ""):
        self._is_busy = True
        self.active_executions += 1
        self.max_active_executions = max(self.max_active_executions, self.active_executions)
        try:
            await asyncio.sleep(0.05)
            yield CellOutput(output_type="stream", content=f"ran:{cell_id}", stream_name="stdout")
            await asyncio.sleep(0.05)
            self._execution_count += 1
        finally:
            self.active_executions -= 1
            self._is_busy = False

    def interrupt(self) -> bool:
        return True

    def restart(self) -> bool:
        self._execution_count = 0
        return True

    def shutdown(self):
        return None

    @property
    def is_alive(self) -> bool:
        return True

    def get_status(self) -> KernelStatus:
        return KernelStatus(
            is_alive=True,
            is_busy=self._is_busy,
            execution_count=self._execution_count,
            kernel_type="fake",
        )

    def get_info(self) -> KernelInfo:
        return KernelInfo(
            kernel_type="fake",
            display_name="Fake",
            is_remote=False,
            supports_shell_cells=True,
            supports_interrupt=True,
        )

    async def get_namespace_info(self, timeout: float = 5.0) -> dict:
        return {"variables": [], "functions": []}


def test_kernel_service_serializes_concurrent_execute_cell_calls():
    service = KernelService()
    kernel = _SerialFakeKernel()
    service.set_kernel_instance("nb", kernel)

    cell1 = Cell(id="cell1", cell_type="code", source="print(1)")
    cell2 = Cell(id="cell2", cell_type="code", source="print(2)")

    async def _run_cell(cell):
        return [output async for output in service.execute_cell("nb", cell)]

    async def _run_both():
        return await asyncio.gather(_run_cell(cell1), _run_cell(cell2))

    outputs1, outputs2 = asyncio.run(_run_both())

    assert kernel.max_active_executions == 1
    assert outputs1[0].content == "ran:cell1"
    assert outputs2[0].content == "ran:cell2"
    assert cell1.execution_count == 1
    assert cell2.execution_count == 2


def test_colab_auth_state_is_single_use_and_status_keeps_email(monkeypatch, tmp_path):
    from dialeng.services.colab import colab_auth as mod

    token_dir = tmp_path / ".dialeng"
    token_file = token_dir / "colab_tokens.json"
    monkeypatch.setattr(mod, "TOKEN_DIR", token_dir)
    monkeypatch.setattr(mod, "TOKEN_FILE", token_file)

    creds = mod.OAuthClientCredentials("client-id", "client-secret", source="test")
    service = mod.ColabAuthService(credentials=creds)
    service._tokens = mod.ColabTokens(
        access_token="access-token",
        refresh_token="refresh-token",
        expires_at=time.time() + 3600,
        account_email="user@example.com",
    )
    service._validated = True
    service._save_tokens()

    restored = mod.ColabAuthService(credentials=creds)
    assert restored.account_email == "user@example.com"

    state = restored.create_auth_state()
    assert restored.validate_auth_state(state) is True
    assert restored.validate_auth_state(state) is False

    restored._validated = True
    status = restored.get_status()
    assert status["authenticated"] is True
    assert status["email"] == "user@example.com"


def test_colab_kernel_degraded_connection_counts_as_alive():
    from dialeng.services.colab.colab_kernel import ColabKernel

    kernel = ColabKernel(api_client=SimpleNamespace())
    kernel._connection_state = "degraded"
    kernel._ws = SimpleNamespace(closed=False)

    assert kernel.is_alive is True
    assert kernel.get_status().connection_state == "degraded"


def test_colab_kernel_keep_alive_is_activity_aware():
    from dialeng.services.colab.colab_kernel import ColabKernel, KEEP_ALIVE_ACTIVITY_GRACE

    kernel = ColabKernel(api_client=SimpleNamespace())
    kernel._assignment = SimpleNamespace(endpoint="runtime-1")
    kernel._is_busy = False
    kernel.set_client_count(0)
    kernel._last_activity_at = time.time() - (KEEP_ALIVE_ACTIVITY_GRACE + 30)

    assert kernel._should_keep_alive() is False

    kernel.set_client_count(1)
    assert kernel._should_keep_alive() is True

    kernel.set_client_count(0)
    kernel.mark_activity("test")
    assert kernel._should_keep_alive() is True


def test_colab_kernel_defers_recycle_while_busy():
    from dialeng.services.colab.colab_kernel import ColabKernel, FAILURE_RECYCLE_THRESHOLD

    kernel = ColabKernel(api_client=SimpleNamespace())
    kernel._assignment = SimpleNamespace(endpoint="runtime-1")
    kernel._is_busy = True

    kernel._request_connection_recycle("keep_alive", FAILURE_RECYCLE_THRESHOLD)

    assert kernel.connection_state == "degraded"
    assert kernel._recycle_reason == "keep_alive"
    assert kernel._recovery_task is None


def test_colab_kernel_recycles_idle_connection():
    from dialeng.services.colab.colab_kernel import ColabKernel, FAILURE_RECYCLE_THRESHOLD

    kernel = ColabKernel(api_client=SimpleNamespace())
    kernel._assignment = SimpleNamespace(endpoint="runtime-1")
    calls = []

    async def _fake_close_runtime_resources(*, cancel_background_tasks: bool):
        calls.append(cancel_background_tasks)
        kernel._assignment = None

    kernel._close_runtime_resources = _fake_close_runtime_resources

    async def _run():
        kernel._request_connection_recycle("keep_alive", FAILURE_RECYCLE_THRESHOLD)
        await kernel._recovery_task

    asyncio.run(_run())

    assert calls == [True]
    assert kernel._recovery_task is None
    assert kernel._recycle_reason == ""


def test_normalize_cell_outputs_merges_display_updates_and_wait_clear():
    outputs = [
        CellOutput(
            output_type="display_data",
            content={"text/html": '<div><progress max="10" value="0"></progress></div>'},
            display_id="progress-1",
        ),
        CellOutput(
            output_type="update_display_data",
            content={"text/html": '<div><progress max="10" value="10"></progress></div>'},
            display_id="progress-1",
        ),
        CellOutput(output_type="clear_output", content=True),
        CellOutput(output_type="stream", content="done\n", stream_name="stdout"),
    ]

    normalized = normalize_cell_outputs(outputs)

    assert len(normalized) == 1
    assert normalized[0].output_type == "stream"
    assert normalized[0].content == "done\n"


def test_render_cell_outputs_uses_final_display_update():
    cell = Cell(
        id="cell-progress",
        cell_type="code",
        source="download()",
        outputs=[
            CellOutput(
                output_type="display_data",
                content={"text/html": '<div><progress max="10" value="0"></progress></div>'},
                display_id="progress-1",
            ),
            CellOutput(
                output_type="update_display_data",
                content={"text/html": '<div><progress max="10" value="10"></progress></div>'},
                display_id="progress-1",
            ),
        ],
    )

    elements, has_error = _render_cell_outputs(cell)
    rendered = ''.join(str(getattr(el, 'children', [''])) for el in elements)

    assert has_error is False
    assert 'value="10"' in rendered
    assert 'value="0"' not in rendered


def test_save_notebook_persists_normalized_display_output(tmp_path):
    notebook = Notebook(
        id="nb",
        title="nb",
        cells=[
            Cell(
                id="cell-save",
                cell_type="code",
                source="learn.show_results()",
                outputs=[
                    CellOutput(
                        output_type="display_data",
                        content={"text/html": '<div><progress max="1" value="0"></progress></div>'},
                        display_id="progress-1",
                    ),
                    CellOutput(
                        output_type="update_display_data",
                        content={"text/html": '<div><progress max="1" value="1"></progress></div>'},
                        display_id="progress-1",
                    ),
                ],
                execution_count=1,
            )
        ],
        path=tmp_path / "progress.ipynb",
    )

    save_notebook(notebook)
    payload = json.loads((tmp_path / "progress.ipynb").read_text())
    outputs = payload["cells"][0]["outputs"]

    assert len(outputs) == 1
    assert outputs[0]["output_type"] == "display_data"
    assert outputs[0]["data"]["text/html"] == '<div><progress max="1" value="1"></progress></div>'


def _formatter_error_output():
    return CellOutput(
        output_type="error",
        ename="TypeError",
        evalue="__repr__ returned non-string (type NoneType)",
        traceback=[
            "/usr/local/lib/python3.12/dist-packages/IPython/core/formatters.py in __call__(self, obj)",
            "/usr/local/lib/python3.12/dist-packages/IPython/lib/pretty.py in pretty(self, obj)",
            "TypeError: __repr__ returned non-string (type NoneType)",
        ],
    )


def test_formatter_error_classifier_matches_ipython_repr_failure():
    assert is_benign_display_formatter_error(_formatter_error_output()) is True


def test_normalize_cell_outputs_drops_formatter_only_errors_when_rich_output_exists():
    outputs = [
        CellOutput(
            output_type="display_data",
            content={"image/png": "abc"},
            metadata={},
        ),
        _formatter_error_output(),
        _formatter_error_output(),
    ]

    normalized = normalize_cell_outputs(outputs)

    assert len(normalized) == 1
    assert normalized[0].output_type == "display_data"


def test_kernel_service_treats_formatter_only_error_with_rich_output_as_success():
    class _FormatterErrorKernel(BaseKernel):
        def __init__(self):
            self._execution_count = 0
            self._is_busy = False

        async def execute_streaming(self, code: str, notebook_id: str = "", cell_id: str = ""):
            self._is_busy = True
            try:
                yield CellOutput(output_type="display_data", content={"image/png": "abc"}, metadata={})
                yield _formatter_error_output()
                self._execution_count += 1
            finally:
                self._is_busy = False

        def interrupt(self) -> bool:
            return True

        def restart(self) -> bool:
            return True

        def shutdown(self):
            return None

        @property
        def is_alive(self) -> bool:
            return True

        def get_status(self) -> KernelStatus:
            return KernelStatus(
                is_alive=True,
                is_busy=self._is_busy,
                execution_count=self._execution_count,
                kernel_type="fake",
            )

        def get_info(self) -> KernelInfo:
            return KernelInfo(
                kernel_type="fake",
                display_name="Fake",
                is_remote=False,
                supports_shell_cells=True,
                supports_interrupt=True,
            )

        async def get_namespace_info(self, timeout: float = 5.0) -> dict:
            return {"variables": [], "functions": []}

    service = KernelService()
    service.set_kernel_instance("nb", _FormatterErrorKernel())
    cell = Cell(id="cell-fmt", cell_type="code", source="learn.show_results()")

    async def _run():
        return [output async for output in service.execute_cell("nb", cell)]

    outputs = asyncio.run(_run())

    assert any(output.output_type == "error" for output in outputs)
    assert cell.state == "success"


def test_kernel_service_applies_cached_client_count_to_kernel_instances():
    service = KernelService()
    service.set_client_count("nb", 3)
    kernel = _SerialFakeKernel()

    service.set_kernel_instance("nb", kernel)

    assert kernel.client_count == 3


def test_kernel_service_shutdown_async_prefers_async_kernel_shutdown():
    class _AsyncShutdownKernel(_SerialFakeKernel):
        def __init__(self):
            super().__init__()
            self.sync_shutdown_called = False
            self.async_shutdown_called = False

        def shutdown(self):
            self.sync_shutdown_called = True

        async def shutdown_async(self):
            self.async_shutdown_called = True

    service = KernelService()
    kernel = _AsyncShutdownKernel()
    service.set_kernel_instance("nb", kernel)

    asyncio.run(service.shutdown_async("nb"))

    assert kernel.async_shutdown_called is True
    assert kernel.sync_shutdown_called is False
    assert service.has_kernel("nb") is False
