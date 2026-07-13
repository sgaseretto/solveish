"""Tests for deterministic extension environment locking and restoration."""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from dialeng.extensions.lockfile import ExtensionLockService
from dialeng.extensions.manifest import ExtensionSource, extension_source_hash, load_extension_manifest
from dialeng.extensions.state import ExtensionStateStore


def _extension(root: Path, extension_id: str = "test.locked") -> Path:
    root.mkdir(parents=True)
    (root / "extension.toml").write_text(
        f'''schema_version = 1
id = "{extension_id}"
name = "Locked"
version = "1.0.0"
description = "Lock fixture"
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "project"

[permissions]
filesystem_read = ["."]

[contributes]
commands = true
''',
        encoding="utf-8",
    )
    (root / "extension.py").write_text("def setup(api, ctx):\n    return None\n", encoding="utf-8")
    return root


def _service(project: Path, tmp_path: Path, *, development_paths=(), cache_root=None):
    state = ExtensionStateStore(tmp_path / "state.json")
    return ExtensionLockService(
        project,
        state_store=state,
        user_root=tmp_path / "no-user",
        development_paths=tuple(development_paths),
        cache_root=cache_root or (tmp_path / "cache"),
    ), state


def test_lockfile_is_deterministic_relative_and_excludes_trust(tmp_path):
    project = tmp_path / "project"
    extension = _extension(project / ".dialeng" / "extensions" / "locked")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.locked", True)

    first = service.write()
    first_bytes = service.path.read_bytes()
    second = service.write()

    assert first["ok"] and second["ok"]
    assert service.path.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert payload["extensions"][0]["source"] == {
        "type": "project_path",
        "path": ".dialeng/extensions/locked",
    }
    assert str(project) not in first_bytes.decode()
    assert "trust" not in first_bytes.decode().lower()
    assert payload["environment"]["uv_lock"]["path"] == "uv.lock"
    assert payload["extensions"][0]["source_hash"]
    assert extension.is_dir()


def test_check_reports_source_permission_environment_and_enabled_drift(tmp_path):
    project = tmp_path / "project"
    extension = _extension(project / ".dialeng" / "extensions" / "locked")
    (project / "uv.lock").write_text("one", encoding="utf-8")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.locked", True)
    assert service.write()["ok"]
    assert service.check()["ok"]

    manifest = extension / "extension.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'filesystem_read = ["."]', 'filesystem_read = [".", "docs"]'
        ),
        encoding="utf-8",
    )
    (project / "uv.lock").write_text("two", encoding="utf-8")
    state.set_enabled(project, "test.locked", False)

    checked = service.check()
    fields = {item["field"] for item in checked["drift"]}
    assert checked["ok"] is False
    assert {"source_hash", "manifest_hash", "permission_hash", "environment", "enabled_extensions"} <= fields


def test_sync_restores_exact_enabled_set_without_granting_trust(tmp_path):
    project = tmp_path / "project"
    _extension(project / ".dialeng" / "extensions" / "one", "test.one")
    _extension(project / ".dialeng" / "extensions" / "two", "test.two")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.one", True)
    assert service.write()["ok"]
    state.set_enabled(project, "test.one", False)
    state.set_enabled(project, "test.two", True)

    synced = service.sync()

    assert synced["ok"] is True
    assert synced["enabled"] == ["test.one"]
    assert synced["trust_granted"] is False
    assert state.enabled(project) == {"test.one"}


def test_git_lock_uses_immutable_commit_and_sync_restores_from_local_remote(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    repository = tmp_path / "extension-repository"
    extension = _extension(repository / "plugin", "test.git")
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(repository), "remote", "add", "origin", str(repository)], check=True)
    service, state = _service(project, tmp_path, development_paths=(extension,))
    state.set_enabled(project, "test.git", True)

    locked = service.write()
    coordinate = locked["lock"]["extensions"][0]["source"]
    assert coordinate["type"] == "git"
    assert len(coordinate["commit"]) == 40
    assert coordinate["subdirectory"] == "plugin"

    restored_service, restored_state = _service(project, tmp_path / "restored")
    restored_service.path.parent.mkdir(parents=True, exist_ok=True)
    restored_service.path.write_bytes(service.path.read_bytes())
    synced = restored_service.sync()

    assert synced["ok"] is True
    assert restored_state.enabled(project) == {"test.git"}
    assert (project / ".dialeng" / "extensions" / "test.git" / "extension.toml").is_file()


def test_update_surfaces_permission_and_source_review(tmp_path):
    project = tmp_path / "project"
    extension = _extension(project / ".dialeng" / "extensions" / "locked")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.locked", True)
    assert service.write()["ok"]
    manifest = extension / "extension.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'filesystem_read = ["."]', 'filesystem_read = [".", "docs"]'
        ),
        encoding="utf-8",
    )

    updated = service.update("test.locked")

    assert updated["ok"] is True
    changes = {(item["field"], item["requires_trust_review"]) for item in updated["changes"]}
    assert ("permissions", True) in changes
    assert ("source_hash", True) in changes


def test_offline_sync_restores_git_artifact_from_populated_cache(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    repository = tmp_path / "remote"
    extension = _extension(repository / "plugin", "test.offline")
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(repository), "remote", "add", "origin", str(repository)], check=True)
    shared_cache = tmp_path / "shared-cache"
    service, state = _service(
        project, tmp_path / "author", development_paths=(extension,), cache_root=shared_cache,
    )
    state.set_enabled(project, "test.offline", True)
    assert service.write()["ok"]

    clone = tmp_path / "clone"
    clone.mkdir()
    restored, restored_state = _service(clone, tmp_path / "restored", cache_root=shared_cache)
    restored.path.parent.mkdir(parents=True, exist_ok=True)
    restored.path.write_bytes(service.path.read_bytes())
    shutil.rmtree(repository)

    result = restored.sync(offline=True)

    assert result["ok"] is True
    assert result["offline"] is True
    assert restored_state.enabled(clone) == {"test.offline"}
    assert (clone / ".dialeng/extensions/test.offline/extension.toml").is_file()


def test_offline_sync_reports_unavailable_artifact_without_partial_enablement(tmp_path):
    project = tmp_path / "project"
    extension = _extension(project / ".dialeng/extensions/missing", "test.missing")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.missing", True)
    assert service.write()["ok"]
    shutil.rmtree(extension)
    shutil.rmtree(service.cache_root)
    state.set_enabled(project, "test.missing", False)

    result = service.sync(offline=True)

    assert result["ok"] is False
    assert "unavailable in cache" in result["error"]
    assert state.enabled(project) == set()


def test_wheel_coordinate_is_immutable_and_offline_cache_installs_real_wheel(tmp_path, monkeypatch):
    source = tmp_path / "package"
    _extension(source, "test.wheel")
    (source / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "dialeng-lock-wheel"
version = "1.0.0"
requires-python = ">=3.11"

[project.entry-points."dialeng.extensions"]
"test.wheel" = "extension:setup"

[tool.setuptools]
py-modules = ["extension"]

[tool.setuptools.data-files]
"dialeng-extension-manifests/lock-wheel" = ["extension.toml"]
''', encoding="utf-8")
    dist = tmp_path / "dist"
    subprocess.run(["uv", "build", str(source), "--out-dir", str(dist)], check=True, capture_output=True)
    wheel = next(dist.glob("*.whl"))
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["uv", "venv", str(project / ".venv")], check=True, capture_output=True)
    service, _state = _service(project, tmp_path)

    class Distribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({
                "url": wheel.as_uri(), "archive_info": {"hash": f"sha256={digest}"},
            })

    monkeypatch.setattr("dialeng.extensions.lockfile.metadata.distribution", lambda _name: Distribution())
    manifest = load_extension_manifest(source / "extension.toml")
    package_source = ExtensionSource(
        source, source / "extension.toml", manifest, extension_source_hash(source),
        "package", 0, package_entrypoint=manifest.id,
        package_name="dialeng-lock-wheel", package_version="1.0.0",
    )
    coordinate = service._source_coordinate(package_source)
    assert coordinate == {"type": "wheel", "filename": wheel.name, "sha256": digest}
    cached = service._wheel_cache_path(digest, wheel.name)
    assert cached.is_file()
    shutil.rmtree(dist)

    service._restore_wheel({"id": manifest.id, "source": coordinate}, offline=True)

    probe = subprocess.run(
        [str(project / ".venv/bin/python"), "-c", "import importlib.metadata as m; print([e.name for e in m.entry_points(group='dialeng.extensions')])"],
        check=True, text=True, capture_output=True,
    )
    assert manifest.id in probe.stdout


def test_concurrent_lock_and_sync_are_serialized_and_leave_valid_state(tmp_path):
    project = tmp_path / "project"
    _extension(project / ".dialeng/extensions/locked")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.locked", True)
    assert service.write()["ok"]
    barrier = threading.Barrier(8)
    results = []

    def operation(index):
        barrier.wait()
        results.append(service.write() if index % 2 == 0 else service.sync(offline=True))

    threads = [threading.Thread(target=operation, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert all(not thread.is_alive() for thread in threads)
    assert all(result["ok"] for result in results)
    assert json.loads(service.path.read_text(encoding="utf-8"))["extensions"][0]["id"] == "test.locked"
    assert not [path for path in service.dialeng_root.glob(".extensions.lock.*") if path != service.guard_path]


def test_interrupted_atomic_lock_write_preserves_previous_lock(tmp_path, monkeypatch):
    project = tmp_path / "project"
    extension = _extension(project / ".dialeng/extensions/locked")
    service, state = _service(project, tmp_path)
    state.set_enabled(project, "test.locked", True)
    assert service.write()["ok"]
    previous = service.path.read_bytes()
    (extension / "extension.py").write_text("def setup(api, ctx):\n    value = 2\n", encoding="utf-8")
    real_replace = __import__("os").replace

    def interrupted(source, destination):
        if Path(destination) == service.path:
            raise OSError("simulated interruption")
        return real_replace(source, destination)

    monkeypatch.setattr("dialeng.extensions.lockfile.os.replace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        service.write()

    assert service.path.read_bytes() == previous
    assert not [path for path in service.dialeng_root.glob(".extensions.lock.*") if path != service.guard_path]
