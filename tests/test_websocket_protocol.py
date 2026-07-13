"""Regression tests for revisioned WebSocket events and full recovery."""

import json
from pathlib import Path

from starlette.testclient import TestClient

import dialeng.app as appmod


def _write_notebook(path: Path) -> None:
    path.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "cell-one",
            "metadata": {},
            "outputs": [],
            "source": ["print('hello')"],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }), encoding="utf-8")


def test_protocol_events_use_independent_monotonic_streams():
    notebook_id = "protocol-test"
    appmod.notebook_revisions.pop(notebook_id, None)
    appmod.outline_revisions.pop(notebook_id, None)
    appmod.event_stream_revisions.pop(notebook_id, None)

    appmod._touch_notebook_revision(notebook_id, "test")
    structural = appmod._protocol_event(notebook_id, {
        "type": "cell_move",
        "notebook_revision": 1,
    })
    same_document_revision = appmod._protocol_event(notebook_id, {
        "type": "cell_class_update",
        "cell_id": "c1",
    })
    output_one = appmod._protocol_event(notebook_id, {
        "type": "code_stream_chunk", "cell_id": "c1", "stream": "stdout", "chunk": "a",
    })
    output_two = appmod._protocol_event(notebook_id, {
        "type": "code_stream_chunk", "cell_id": "c1", "stream": "stderr", "chunk": "b",
    })
    queue = appmod._protocol_event(notebook_id, {"type": "queue_update"})

    assert structural["stream"] == "notebook"
    assert structural["revision"] == 1
    assert same_document_revision["revision"] == 2
    assert same_document_revision["notebook_revision"] == 1
    assert output_one["stream"] == output_two["stream"] == "output"
    assert output_one["output_stream"] == "stdout"
    assert output_two["output_stream"] == "stderr"
    assert (output_one["revision"], output_two["revision"]) == (1, 2)
    assert queue["stream"] == "kernel"
    assert queue["authoritative"] is True
    assert len({structural["event_id"], output_one["event_id"], output_two["event_id"]}) == 3

    appmod.notebook_revisions.pop(notebook_id, None)
    appmod.event_stream_revisions.pop(notebook_id, None)


def test_recovery_endpoint_returns_cells_and_all_revision_domains(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "recover.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        appmod._touch_notebook_revision("recover", "test")
        appmod._protocol_event("recover", {"type": "cell_source_update"})
        appmod._next_stream_revision("recover", "output")
        runtime = appmod.runtime_registry.get_or_create("recover")
        runtime.kernel_generation = 3
        runtime.setup_last_generation = 3
        runtime.setup_last_outcome = "completed"
        client = TestClient(appmod.app)
        response = client.get("/dialeng/recover/recovery")

        assert response.status_code == 200
        payload = response.json()
        assert payload["type"] == "recovery_snapshot"
        assert 'id="cells"' in payload["cells_html"]
        assert 'id="cell-cell-one"' in payload["cells_html"]
        assert payload["snapshot"]["revisions"]["notebook"] == 1
        assert payload["snapshot"]["revisions"]["output"] == 1
        assert payload["snapshot"]["authoritative"] is True
        assert payload["snapshot"]["kernel"]["generation"] == 3
        assert payload["snapshot"]["setup"]["last_generation"] == 3
        assert payload["snapshot"]["setup"]["last_outcome"] == "completed"
    finally:
        appmod.runtime_registry.clear_runtime(
            "recover",
            lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.set_root_dir(previous_root)


def test_websocket_control_messages_use_fasthtml_msg_envelope():
    client = TestClient(appmod.app)

    with client.websocket_connect("/ws/heartbeat-test") as websocket:
        assert appmod.broadcast_manager.client_count("heartbeat-test") == 1
        websocket.send_json({"msg": json.dumps({"type": "heartbeat"})})
        payload = websocket.receive_json()

    assert payload["type"] == "heartbeat_ack"
    assert payload["timestamp"]
    assert appmod.broadcast_manager.client_count("heartbeat-test") == 0


def test_websocket_join_requests_recovery_for_missed_notebook_revision(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    _write_notebook(root / "join-recovery.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR

    try:
        appmod.set_root_dir(root)
        appmod._touch_notebook_revision("join-recovery", "test")
        appmod._protocol_event("join-recovery", {"type": "cell_source_update"})
        client = TestClient(appmod.app)

        with client.websocket_connect("/ws/join-recovery") as websocket:
            websocket.send_json({
                "msg": json.dumps({
                    "type": "join",
                    "revisions": {"notebook": 0, "output": 0, "persistence": 0},
                }),
            })
            resync = websocket.receive_json()
            snapshot = websocket.receive_json()

        assert resync["type"] == "resync_required"
        assert resync["streams"] == ["notebook"]
        assert resync["server_revisions"]["notebook"] == 1
        assert snapshot["type"] == "kernel_snapshot"
        assert appmod.broadcast_manager.client_count("join-recovery") == 0
    finally:
        appmod.runtime_registry.clear_runtime(
            "join-recovery",
            lifecycle=appmod.LifecycleState.STOPPED,
        )
        appmod.notebook_revisions.pop("join-recovery", None)
        appmod.event_stream_revisions.pop("join-recovery", None)
        appmod.set_root_dir(previous_root)
