"""Regression tests for notebook deletion and runtime teardown."""

import json
from pathlib import Path
from types import SimpleNamespace

from starlette.testclient import TestClient

import dialeng.app as appmod


def _write_notebook(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_delete_notebook_cancels_queue_with_notebook_id(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    notebook_path = root / "queued.ipynb"
    _write_notebook(notebook_path)

    class _Queue:
        def __init__(self):
            self.cancelled_notebooks = []

        def cancel_all(self, notebook_id: str) -> None:
            self.cancelled_notebooks.append(notebook_id)

        def get_status(self, notebook_id: str):
            return SimpleNamespace(
                current_cell_id=None,
                queued_cell_ids=[],
                queued_count=0,
                is_processing=False,
            )

    previous_root = appmod.NOTEBOOKS_DIR
    queue = _Queue()
    notifications = [[], []]

    async def _send_first(message: str) -> None:
        notifications[0].append(json.loads(message))

    async def _send_second(message: str) -> None:
        notifications[1].append(json.loads(message))

    try:
        appmod.set_root_dir(root)
        appmod.get_notebook("queued")
        appmod.execution_queues["queued"] = queue
        appmod.broadcast_manager.add_connection("queued", _send_first)
        appmod.broadcast_manager.add_connection("queued", _send_second)

        client = TestClient(appmod.app)
        response = client.post(
            "/files/delete",
            data={"path": "queued.ipynb", "active_notebook_id": "queued"},
        )

        assert response.status_code == 200
        assert response.json()["deleted_notebook_id"] == "queued"
        assert queue.cancelled_notebooks == ["queued"]
        assert not notebook_path.exists()
        assert "queued" not in appmod.execution_queues
        assert "queued" not in appmod.notebooks
        assert "queued" in appmod.deleted_notebooks
        assert [events[0]["type"] for events in notifications] == ["notebook_deleted", "notebook_deleted"]
        assert not appmod.broadcast_manager.has_clients("queued")

        snapshot = client.get("/dialeng/queued/kernel/snapshot")
        assert snapshot.status_code == 410
        assert snapshot.json()["type"] == "notebook_deleted"
        assert "queued" not in appmod.notebooks

        stale_save = client.post("/dialeng/queued/save")
        assert stale_save.status_code == 410
        assert not notebook_path.exists()
        assert "queued" not in appmod.notebooks
    finally:
        appmod.execution_queues.pop("queued", None)
        appmod.notebooks.pop("queued", None)
        appmod.deleted_notebooks.pop("queued", None)
        appmod.deleting_notebooks.discard("queued")
        appmod.broadcast_manager.drop_notebook("queued")
        appmod.set_root_dir(previous_root)


def test_delete_notebook_keeps_file_when_runtime_teardown_fails(tmp_path: Path, monkeypatch):
    root = tmp_path / "notebooks"
    root.mkdir()
    notebook_path = root / "protected.ipynb"
    _write_notebook(notebook_path)

    async def _fail_teardown(notebook_id: str, reason: str, **kwargs) -> None:
        raise RuntimeError(f"teardown failed for {notebook_id}: {reason}")

    previous_root = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(root)
        monkeypatch.setattr(appmod, "_teardown_notebook_runtime", _fail_teardown)
        client = TestClient(appmod.app, raise_server_exceptions=False)

        response = client.post(
            "/files/delete",
            data={"path": "protected.ipynb", "active_notebook_id": "protected"},
        )

        assert response.status_code == 500
        assert notebook_path.exists()
        assert "protected" not in appmod.deleted_notebooks
    finally:
        appmod.notebooks.pop("protected", None)
        appmod.deleted_notebooks.pop("protected", None)
        appmod.deleting_notebooks.discard("protected")
        appmod.set_root_dir(previous_root)


def test_explicit_creation_clears_deleted_notebook_tombstone(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    appmod.deleted_notebooks["recreated"] = {
        "path": "recreated.ipynb",
        "deleted_at": "test",
    }

    previous_root = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(root)
        appmod.deleted_notebooks["recreated"] = {
            "path": "recreated.ipynb",
            "deleted_at": "test",
        }
        response = TestClient(appmod.app).get(
            "/dialeng/new",
            params={"name": "recreated"},
            follow_redirects=False,
        )

        assert response.status_code == 302
        assert (root / "recreated.ipynb").exists()
        assert "recreated" not in appmod.deleted_notebooks
        assert "recreated" in appmod.notebooks
    finally:
        appmod.notebooks.pop("recreated", None)
        appmod.deleted_notebooks.pop("recreated", None)
        appmod.set_root_dir(previous_root)


def test_missing_notebook_url_uses_workspace_without_creating_file(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    previous_root = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(root)
        client = TestClient(appmod.app)

        response = client.get(
            "/dialeng/",
            params={"name": "missing"},
            follow_redirects=False,
        )
        workspace = client.get("/dialeng/workspace")

        assert response.status_code == 302
        assert response.headers["location"] == "/dialeng/workspace"
        assert not (root / "missing.ipynb").exists()
        assert workspace.status_code == 200
        assert "No notebook open" in workspace.text
    finally:
        appmod.notebooks.pop("missing", None)
        appmod.set_root_dir(previous_root)
