"""Tests for atomic explicit saves and opt-in autosave/recovery."""

import asyncio
import errno
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import dialeng.app as appmod
from dialeng.document.cell import Cell
from dialeng.document.notebook import Notebook
from dialeng.services.dialeng_config import DEFAULT_CONFIG, load_config, reset_config_cache
from dialeng.services.persistence_service import (
    AutosaveManager,
    AutosaveSettings,
    ExternalFileChangedError,
    discard_recovery,
    find_newer_recovery,
)


def _notebook(source: str = "first") -> Notebook:
    return Notebook(id="demo", title="demo", cells=[Cell(cell_type="code", source=source)])


def test_atomic_save_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "demo.ipynb"
    notebook = _notebook("original")
    notebook.save(str(path))
    original_bytes = path.read_bytes()
    notebook.cells[0].source = "changed"

    import dialeng.services.persistence_service as persistence
    monkeypatch.setattr(persistence.os, "replace", lambda *_args: (_ for _ in ()).throw(
        OSError("replace failed")
    ))

    with pytest.raises(OSError, match="replace failed"):
        notebook.save(str(path))

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".demo.ipynb.*.tmp")) == []


def test_atomic_save_permission_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "demo.ipynb"
    notebook = _notebook("original")
    notebook.save(str(path))
    original_bytes = path.read_bytes()
    notebook.cells[0].source = "changed"

    import dialeng.services.persistence_service as persistence
    monkeypatch.setattr(
        persistence.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(PermissionError("read-only directory")),
    )

    with pytest.raises(PermissionError, match="read-only"):
        notebook.save(str(path))

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".demo.ipynb.*.tmp")) == []


def test_atomic_save_disk_full_failure_removes_partial_temp(tmp_path, monkeypatch):
    path = tmp_path / "demo.ipynb"
    notebook = _notebook("original")
    notebook.save(str(path))
    original_bytes = path.read_bytes()
    notebook.cells[0].source = "changed"

    import dialeng.services.persistence_service as persistence

    def fail_dump(_data, handle, **_kwargs):
        handle.write("{\"partial\":")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(persistence.json, "dump", fail_dump)

    with pytest.raises(OSError) as exc_info:
        notebook.save(str(path))

    assert exc_info.value.errno == errno.ENOSPC
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".demo.ipynb.*.tmp")) == []


def test_external_change_is_rejected_unless_overwrite_is_explicit(tmp_path):
    path = tmp_path / "demo.ipynb"
    notebook = _notebook()
    notebook.save(str(path))
    external = json.loads(path.read_text(encoding="utf-8"))
    external["metadata"]["external"] = True
    path.write_text(json.dumps(external), encoding="utf-8")
    notebook.cells[0].source = "dialeng edit"

    with pytest.raises(ExternalFileChangedError):
        notebook.save(str(path))

    notebook.save(str(path), overwrite_external=True)
    assert "dialeng edit" in path.read_text(encoding="utf-8")


def test_autosave_disabled_creates_no_task_or_file(tmp_path):
    async def run():
        notebook = _notebook()
        manager = AutosaveManager(lambda: tmp_path, lambda _id: True)
        scheduled = manager.schedule("demo", notebook, AutosaveSettings(enabled=False))
        await asyncio.sleep(0.06)
        assert scheduled is False
        assert manager.pending_count() == 0
        assert not (tmp_path / "demo.ipynb").exists()

    asyncio.run(run())


def test_autosave_coalesces_edits_and_bounds_recovery_files(tmp_path):
    async def run():
        saves = []

        async def on_saved(notebook_id, _notebook, _path):
            saves.append(notebook_id)

        notebook = _notebook()
        manager = AutosaveManager(
            lambda: tmp_path,
            lambda _id: True,
            on_saved=on_saved,
        )
        settings = AutosaveSettings(
            enabled=True,
            debounce_ms=50,
            recovery_snapshots=True,
            max_recovery_files=2,
        )
        for source in ("one", "two", "three"):
            notebook.cells[0].source = source
            manager.schedule("demo", notebook, settings)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)

        assert saves == ["demo"]
        assert "three" in (tmp_path / "demo.ipynb").read_text(encoding="utf-8")
        assert len(list((tmp_path / ".dialeng" / "recovery").glob("demo.*.ipynb"))) == 1

    asyncio.run(run())


def test_recovery_is_only_reported_when_newer_and_can_be_discarded(tmp_path):
    canonical = tmp_path / "demo.ipynb"
    canonical.write_text("{}", encoding="utf-8")
    recovery_dir = tmp_path / ".dialeng" / "recovery"
    recovery_dir.mkdir(parents=True)
    recovery = recovery_dir / "demo.99999999T999999Z.ipynb"
    recovery.write_text("{}", encoding="utf-8")
    # Ensure a strictly newer mtime even on coarse filesystems.
    recovery.touch()
    canonical_stat = canonical.stat()
    recovery_stat = recovery.stat()
    if recovery_stat.st_mtime_ns <= canonical_stat.st_mtime_ns:
        import os
        os.utime(recovery, ns=(canonical_stat.st_mtime_ns + 1, canonical_stat.st_mtime_ns + 1))

    assert find_newer_recovery(tmp_path, "demo", canonical) == recovery
    discard_recovery(recovery, tmp_path)
    assert not recovery.exists()


def test_autosave_defaults_to_disabled_in_new_and_minimal_config(tmp_path):
    assert DEFAULT_CONFIG["persistence"]["autosave"]["enabled"] is False
    config_path = tmp_path / "dialeng_config.json"
    config_path.write_text("{}", encoding="utf-8")
    reset_config_cache()
    config = load_config(config_path=config_path, force_reload=True)
    assert config.persistence_atomic_explicit_save is True
    assert config.autosave_enabled is False
    reset_config_cache()


def test_corrupted_recovery_never_replaces_canonical_notebook(tmp_path):
    root = tmp_path / "notebooks"
    root.mkdir()
    canonical = root / "demo.ipynb"
    notebook = _notebook("canonical")
    notebook.save(str(canonical))
    canonical_bytes = canonical.read_bytes()
    recovery_dir = root / ".dialeng" / "recovery"
    recovery_dir.mkdir(parents=True)
    recovery = recovery_dir / "demo.99999999T999999Z.ipynb"
    recovery.write_text("{not valid json", encoding="utf-8")
    canonical_stat = canonical.stat()
    import os
    os.utime(
        recovery,
        ns=(canonical_stat.st_mtime_ns + 1, canonical_stat.st_mtime_ns + 1),
    )
    previous_root = appmod.NOTEBOOKS_DIR
    previous_config = appmod.DIALENG_CONFIG

    try:
        appmod.set_root_dir(root)
        appmod.DIALENG_CONFIG.autosave_enabled = True
        appmod.DIALENG_CONFIG.autosave_recovery_snapshots = True
        client = TestClient(appmod.app)

        response = client.post("/dialeng/demo/recovery", data={"action": "restore"})

        assert response.status_code == 422
        assert response.json()["status"] == "invalid_recovery"
        assert canonical.read_bytes() == canonical_bytes
        assert recovery.exists()
    finally:
        appmod.runtime_registry.clear_runtime("demo", lifecycle=appmod.LifecycleState.STOPPED)
        appmod.set_root_dir(previous_root)
        appmod.DIALENG_CONFIG = previous_config
