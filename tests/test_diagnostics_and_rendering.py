"""Tests for diagnostics APIs and bounded large-notebook boot rendering."""

import json
import re
from pathlib import Path

from fasthtml.common import to_xml
from starlette.testclient import TestClient

import dialeng.app as appmod
from dialeng.document.cell import Cell
from dialeng.ui.cells.base import CellHeader
from dialeng.ui.cells.prompt_cell import _execution_diagnostics
from dialeng.ui.mime import render_mime_bundle


def _write_notebook(path: Path, cell_count: int = 1) -> None:
    path.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": f"cell-{index}",
            "metadata": {},
            "outputs": [],
            "source": [f"value_{index} = {index}"],
        } for index in range(cell_count)],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }), encoding="utf-8")


def test_diagnostics_endpoints_expose_content_free_runtime_state(tmp_path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR
    appmod.performance_monitor.reset()

    try:
        appmod.set_root_dir(root)
        appmod.get_notebook("demo")
        client = TestClient(appmod.app)
        browser = client.post("/dialeng/diagnostics/browser", data={
            "notebook_id": "demo",
            "long_task_count": 2,
            "long_task_total_ms": 180,
            "long_task_max_ms": 110,
            "editor_count": 4,
            "visible_cell_count": 3,
            "dom_cell_count": 100,
        })
        diagnostics = client.get("/dialeng/demo/diagnostics")
        performance = client.get("/dialeng/diagnostics/performance")

        assert browser.status_code == 200
        assert diagnostics.status_code == 200
        assert diagnostics.json()["runtime"]["lifecycle"] == "active"
        assert "cells" not in diagnostics.json()
        assert performance.json()["counters"]["browser.long_tasks"] == 2
        assert performance.json()["gauges"]["browser.last_editor_count"] == 4
        assert "runtime_registry" in performance.json()
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)


def test_large_notebook_renders_placeholders_without_eager_monaco_loader(tmp_path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "large.ipynb", cell_count=200)
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        appmod._touch_notebook_revision("large", "render_test")
        client = TestClient(appmod.app)
        response = client.get("/dialeng/large")
        html = response.text

        assert response.status_code == 200
        assert html.count("monaco-placeholder") == 200
        assert "monaco-editor@0.52.2/min/vs/loader.min.js" not in html
        assert "/static/js/event_dispatcher.js" in html
        assert "/static/js/notebook_client.js" in html
        assert "/static/js/prompt_response_editor.js" in html
        assert "/static/js/structure_controller.js" in html
        assert "/static/js/execution_ui.js" in html
        assert "/static/js/notifications.js" in html
        assert "/static/js/browser_diagnostics.js" in html
        assert re.search(r'/static/vendor/htmx-2\.0\.7\.min\.js\?v=[0-9a-f]+', html)
        assert re.search(r'/static/js/app\.js\?v=[0-9a-f]+', html)
        assert 'unpkg.com/htmx.org' not in html
        assert "window.INITIAL_REVISIONS" in html
        # Initial revisions are transport-event baselines, not the durable
        # notebook document revision advanced by the synthetic setup mutation.
        assert '&quot;notebook&quot;: 0' in html or '"notebook": 0' in html
    finally:
        appmod.runtime_registry.clear_runtime("large", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.notebook_revisions.pop("large", None)
        appmod.set_root_dir(previous_root)


def test_rich_images_request_lazy_decode():
    rendered = render_mime_bundle({"image/png": "YWJj"})
    assert 'loading="lazy"' in rendered
    assert 'decoding="async"' in rendered


def test_run_button_reads_live_source_at_request_time():
    rendered = to_xml(CellHeader(Cell(id="fresh-source", cell_type="code"), "demo"))

    assert "getCellSourceForRequest('fresh-source')" in rendered
    assert "syncMonacoToTextarea('fresh-source')" in rendered
    assert "onpointerdown=" in rendered


def test_prompt_execution_diagnostics_are_collapsed_by_default():
    cell = Cell(id="prompt-stats", cell_type="prompt", llm_execution={
        "operation_id": "prompt-run-123",
        "provider": "openai_codex",
        "model": "gpt-5.4",
        "duration_ms": 4600,
        "status": "completed",
        "usage": {"total_tokens": 2545},
    })

    rendered = to_xml(_execution_diagnostics(cell))

    assert rendered.startswith("<details")
    assert "<summary>gpt-5.4 · 4.6s · 2,545 tokens · completed</summary>" in rendered
    assert " open" not in rendered
    assert 'data-operation-id="prompt-run-123"' in rendered


def test_prompt_execution_diagnostics_placeholder_is_details():
    rendered = to_xml(_execution_diagnostics(Cell(id="prompt-empty", cell_type="prompt")))

    assert rendered.startswith("<details")
    assert "hidden" in rendered
