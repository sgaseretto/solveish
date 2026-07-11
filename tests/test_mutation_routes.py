"""Route-level tests for expected revisions and idempotent structural retries."""

import json
import random
import time
from pathlib import Path

from starlette.testclient import TestClient

import dialeng.app as appmod


def _write_notebook(path: Path) -> None:
    path.write_text(json.dumps({
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }), encoding="utf-8")


def _next_json_event(websocket, event_type: str) -> dict:
    """Skip supplementary OOB HTML and unrelated status events."""
    for _ in range(20):
        raw = websocket.receive_text()
        if not raw.startswith("{"):
            continue
        event = json.loads(raw)
        if event.get("type") == event_type:
            return event
    raise AssertionError(f"Did not receive {event_type}")


def test_cell_add_retry_does_not_duplicate_and_stale_revision_is_rejected(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        client = TestClient(appmod.app)
        payload = {
            "pos": 0,
            "type": "code",
            "expected_revision": 0,
            "operation_id": "add-once",
        }

        first = client.post("/dialeng/demo/cell/add", data=payload)
        retry = client.post("/dialeng/demo/cell/add", data=payload)
        conflict = client.post("/dialeng/demo/cell/add", data={
            **payload,
            "operation_id": "different-operation",
        })

        assert first.status_code == 200
        assert retry.status_code == 200
        assert len(notebook.cells) == 1
        assert conflict.status_code == 409
        assert conflict.json()["actual_revision"] == 1
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)


def test_shift_enter_add_survives_run_source_revision(tmp_path: Path, monkeypatch):
    """A run source commit must not make the paired next-cell add stale."""
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    class QueueStub:
        queued = []

        def is_cell_queued(self, _notebook_id, _cell_id):
            return False

        def queue_cell(self, notebook_id, cell):
            self.queued.append((notebook_id, cell.id))

    async def ignore_queue_broadcast(_notebook_id):
        return None

    queue = QueueStub()
    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="code",
        )
        client = TestClient(appmod.app)
        monkeypatch.setattr(appmod.kernel_service, "has_kernel", lambda _nb_id: True)
        monkeypatch.setattr(appmod, "get_execution_queue", lambda _nb_id: queue)
        monkeypatch.setattr(appmod, "broadcast_queue_state", ignore_queue_broadcast)

        with client.websocket_connect("/ws/demo") as websocket:
            run = client.post(f"/dialeng/demo/cell/{cell.id}/run", data={
                "source": 'print("hello")',
                "expected_version": 0,
            })
            source_event = _next_json_event(websocket, "cell_source_update")

            # Shift+Enter captured structure revision 0 before the run request.
            # The source commit advanced notebook revision only, so this remains valid.
            add = client.post("/dialeng/demo/cell/add", data={
                "pos": 1,
                "type": "code",
                "expected_structure_revision": 0,
                "operation_id": "shift-enter-add",
            })
            add_event = _next_json_event(websocket, "cell_add")

        assert run.status_code == 200
        assert add.status_code == 200
        assert queue.queued == [("demo", cell.id)]
        assert source_event["revision"] == 1
        assert add_event["revision"] == 2
        assert add_event["structure_revision"] == 1
        assert [item.source for item in notebook.cells] == ['print("hello")', ""]

        stale_structure = client.post("/dialeng/demo/cell/add", data={
            "pos": 2,
            "type": "code",
            "expected_structure_revision": 0,
            "operation_id": "stale-structure",
        })
        assert stale_structure.status_code == 409
        assert stale_structure.json()["revision_domain"] == "structure"
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)


def test_stale_prompt_output_edit_restores_backend_response(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="prompt",
        )
        cell.update_output("Backend answer")
        client = TestClient(appmod.app)

        with client.websocket_connect("/ws/demo") as websocket:
            stale_output = client.post(
                f"/dialeng/demo/cell/{cell.id}/output",
                data={"output": "", "expected_version": 0},
            )
            conflict = _next_json_event(websocket, "cell_conflict")

        assert stale_output.status_code == 200
        assert stale_output.json()["status"] == "conflict"
        assert conflict["conflict_field"] == "output"
        assert conflict["output"] == "Backend answer"
        assert cell.output == "Backend answer"
    finally:
        appmod.runtime_registry.clear_runtime(
            "demo", lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.set_root_dir(previous_root)


def test_prompt_source_commit_preserves_response_and_broadcasts_it(tmp_path: Path):
    """Editing prompt input must not reset its authored assistant response."""
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="prompt",
        )
        cell.source = "My name is Joe"
        cell.output = "Nice to meet you Joe."
        client = TestClient(appmod.app)

        with client.websocket_connect("/ws/demo") as websocket:
            response = client.post(
                f"/dialeng/demo/cell/{cell.id}/source",
                data={
                    "source": "My name is John",
                    "expected_version": 0,
                    "operation_id": "edit-prompt-input",
                },
            )
            event = _next_json_event(websocket, "cell_source_update")

        assert response.status_code == 200
        assert cell.source == "My name is John"
        assert cell.output == "Nice to meet you Joe."
        assert event["source"] == "My name is John"
        assert event["output"] == "Nice to meet you Joe."
        assert event["version"] == 1
    finally:
        appmod.runtime_registry.clear_runtime(
            "demo", lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.set_root_dir(previous_root)


def test_prompt_run_does_not_append_or_return_a_replacement_cell(tmp_path: Path, monkeypatch):
    """Execution must not mutate structure or send a late focus-stealing swap."""
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        notebook.dialog_mode = "mock"
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="prompt",
        )
        cell.source = "My name is Joe"
        monkeypatch.setattr(appmod.kernel_service, "has_kernel", lambda _nb_id: True)
        client = TestClient(appmod.app)

        with client.websocket_connect("/ws/demo") as websocket:
            response = client.post(
                f"/dialeng/demo/cell/{cell.id}/run",
                data={"source": cell.source, "expected_version": 0},
            )
            terminal = _next_json_event(websocket, "prompt_run_terminal")

        assert response.status_code == 200
        assert response.text == ""
        assert [item.id for item in notebook.cells] == [cell.id]
        assert appmod._get_structure_revision("demo") == 0
        assert cell.output
        assert terminal["status"] == "completed"
        assert terminal["version"] == cell.version
        assert not appmod._active_prompt_cells("demo")
    finally:
        appmod.runtime_registry.clear_runtime(
            "demo", lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.set_root_dir(previous_root)


def test_prompt_mutations_conflict_immediately_while_stream_is_active(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="prompt",
        )
        cell.source = "Original prompt"
        appmod.prompt_cancel_events[f"demo:{cell.id}"] = appmod.asyncio.Event()
        client = TestClient(appmod.app)

        started = time.perf_counter()
        source = client.post(
            f"/dialeng/demo/cell/{cell.id}/source",
            data={"source": "Changed during run", "expected_version": 0},
        )
        save = client.post("/dialeng/demo/save")
        provider = client.post("/dialeng/demo/provider", data={"provider": "openai_codex"})
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5
        for response, operation in (
            (source, "post"),
            (save, "save"),
            (provider, "provider_change"),
        ):
            assert response.status_code == 409
            assert response.json()["type"] == "prompt_operation_conflict"
            assert response.json()["operation"] == operation
            assert response.json()["active_cell_ids"] == [cell.id]
        assert cell.source == "Original prompt"
    finally:
        appmod.prompt_cancel_events.pop(f"demo:{cell.id}", None)
        appmod.runtime_registry.clear_runtime(
            "demo", lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.set_root_dir(previous_root)


def test_generated_structural_mutations_preserve_order_ids_and_idempotency(tmp_path: Path):
    """Exercise mixed structural operations against a simple ordered-ID model."""
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR
    rng = random.Random(20260709)

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        client = TestClient(appmod.app)
        expected_ids: list[str] = []

        for step in range(60):
            revision = appmod.runtime_registry.get_or_create("demo").notebook_revision
            operation_id = f"generated-{step}"
            old_ids = list(expected_ids)
            possible = ["add"]
            if expected_ids:
                possible.extend(["duplicate", "move"])
            if len(expected_ids) > 1:
                possible.extend(["delete", "merge"])
            operation = rng.choice(possible)

            if operation == "add":
                pos = rng.randrange(len(expected_ids) + 1)
                response = client.post("/dialeng/demo/cell/add", data={
                    "pos": pos,
                    "type": "code",
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
                actual_ids = [cell.id for cell in notebook.cells]
                new_ids = set(actual_ids) - set(old_ids)
                assert len(new_ids) == 1
                expected_ids.insert(pos, new_ids.pop())
                retry = lambda: client.post("/dialeng/demo/cell/add", data={
                    "pos": pos,
                    "type": "code",
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
            elif operation == "duplicate":
                index = rng.randrange(len(expected_ids))
                cell_id = expected_ids[index]
                response = client.post(f"/dialeng/demo/cell/{cell_id}/duplicate", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
                actual_ids = [cell.id for cell in notebook.cells]
                new_ids = set(actual_ids) - set(old_ids)
                assert len(new_ids) == 1
                expected_ids.insert(index + 1, new_ids.pop())
                retry = lambda: client.post(f"/dialeng/demo/cell/{cell_id}/duplicate", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
            elif operation == "move":
                index = rng.randrange(len(expected_ids))
                choices = []
                if index > 0:
                    choices.append(("up", index - 1))
                if index + 1 < len(expected_ids):
                    choices.append(("down", index + 1))
                if not choices:
                    continue
                direction, new_index = rng.choice(choices)
                cell_id = expected_ids[index]
                response = client.post(f"/dialeng/demo/cell/{cell_id}/move/{direction}", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
                expected_ids[index], expected_ids[new_index] = (
                    expected_ids[new_index], expected_ids[index]
                )
                retry = lambda: client.post(f"/dialeng/demo/cell/{cell_id}/move/{direction}", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
            elif operation == "delete":
                index = rng.randrange(len(expected_ids))
                cell_id = expected_ids.pop(index)
                response = client.delete(f"/dialeng/demo/cell/{cell_id}", params={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
                retry = lambda: client.delete(f"/dialeng/demo/cell/{cell_id}", params={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
            else:
                index = rng.randrange(len(expected_ids) - 1)
                cell_id = expected_ids[index]
                expected_ids.pop(index + 1)
                response = client.post(f"/dialeng/demo/cell/{cell_id}/merge-below", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })
                retry = lambda: client.post(f"/dialeng/demo/cell/{cell_id}/merge-below", data={
                    "expected_revision": revision,
                    "operation_id": operation_id,
                })

            assert response.status_code == 200, (operation, response.text)
            actual_ids = [cell.id for cell in notebook.cells]
            assert actual_ids == expected_ids
            assert len(actual_ids) == len(set(actual_ids))
            completed_revision = appmod.runtime_registry.get_or_create("demo").notebook_revision
            assert completed_revision > revision

            retry_response = retry()
            assert retry_response.status_code == 200
            assert [cell.id for cell in notebook.cells] == expected_ids
            assert appmod.runtime_registry.get_or_create("demo").notebook_revision == completed_revision

        stale = client.post("/dialeng/demo/cell/add", data={
            "pos": len(expected_ids),
            "type": "code",
            "expected_revision": 0,
            "operation_id": "stale-after-sequence",
        })
        assert stale.status_code == 409
        assert [cell.id for cell in notebook.cells] == expected_ids
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)


def test_code_note_prompt_add_delete_sequence_keeps_revisions_consistent(tmp_path: Path):
    """Regression for type changes leaving later add/delete requests stale."""
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "demo.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        notebook = appmod.get_notebook("demo")
        cell = appmod.notebook_mutation_service.add(
            notebook, position=0, cell_type="code",
        )
        client = TestClient(appmod.app)

        with client.websocket_connect("/ws/demo") as websocket:
            response = client.post(f"/dialeng/demo/cell/{cell.id}/type", data={
                "cell_type": "note",
                "source": "",
                "expected_version": 0,
                "expected_cell_type": "code",
                "operation_id": "to-note",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_type_change")
            assert event["revision"] == 1
            assert event["cell_type"] == "note"

            response = client.post(f"/dialeng/demo/cell/{cell.id}/source", data={
                "source": "## A note",
                "expected_version": 1,
                "operation_id": "edit-note",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_source_update")
            assert event["revision"] == 2
            assert event["source"] == "## A note"

            response = client.post(f"/dialeng/demo/cell/{cell.id}/type", data={
                "cell_type": "prompt",
                "source": "## A note",
                "expected_version": 2,
                "expected_cell_type": "note",
                "operation_id": "to-prompt",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_type_change")
            assert event["revision"] == 3
            assert event["cell_type"] == "prompt"
            assert 'data-type="prompt"' in event["html"]

            response = client.post("/dialeng/demo/cell/add", data={
                "pos": 1,
                "type": "code",
                "expected_revision": 3,
                "operation_id": "add-after-prompt",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_add")
            assert event["revision"] == 4
            added_id = event["cell_id"]

            response = client.delete(f"/dialeng/demo/cell/{added_id}", params={
                "expected_revision": 4,
                "operation_id": "delete-added",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_delete")
            assert event["revision"] == 5

            response = client.delete(f"/dialeng/demo/cell/{cell.id}", params={
                "expected_revision": 5,
                "operation_id": "delete-prompt",
            })
            assert response.status_code == 200
            event = _next_json_event(websocket, "cell_delete")
            assert event["revision"] == 6

        assert notebook.cells == []
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)
