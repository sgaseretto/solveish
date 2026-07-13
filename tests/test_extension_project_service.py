"""Tests for notebook-first extension scaffolding and isolated validation."""
from pathlib import Path

import pytest

from dialeng.extensions.manager import ExtensionManager
from dialeng.services.extension_project_service import (
    ExtensionProjectService,
    get_extension_project_service,
    request_extension_activation,
    set_extension_project_root,
    update_extension_file,
)


@pytest.mark.parametrize("template", ["basic", "tool", "cell", "lifecycle", "provider", "package"])
def test_all_extension_templates_build_and_validate_syntax(tmp_path, template):
    service = ExtensionProjectService(tmp_path)

    result = service.create(f"fixture-{template}", template=template)

    assert result["exported_cells"] >= 1
    assert Path(result["manifest"]).is_file()
    assert Path(result["notebook"]).is_file()
    assert Path(result["module"]).is_file()
    assert result["validation"]["ok"] is True
    assert "#| export" not in Path(result["module"]).read_text(encoding="utf-8")


def test_prepare_runs_isolated_activation_and_extension_tests(tmp_path):
    service = ExtensionProjectService(tmp_path)
    created = service.create("prepared-extension", template="basic")

    prepared = service.prepare(Path(created["root"]))

    assert prepared["ok"] is True
    assert prepared["validation"]["isolated"] is True
    assert prepared["validation"]["subprocess"]["ok"] is True
    assert prepared["tests"]["ok"] is True
    assert "1 passed" in prepared["tests"]["stdout"]
    assert Path(prepared["build_record"]).stat().st_mode & 0o777 == 0o600


def test_rebuild_is_reproducible_and_stale_validation_cannot_activate(tmp_path):
    import asyncio

    service = ExtensionProjectService(tmp_path)
    created = service.create("reproducible-extension", template="basic")
    root = Path(created["root"])
    first = service.prepare(root)
    module = Path(first["module"])
    first_bytes = module.read_bytes()

    service.build(root)
    assert module.read_bytes() == first_bytes

    module.write_text(module.read_text(encoding="utf-8") + "\nCHANGED = True\n", encoding="utf-8")
    manager = ExtensionManager(
        tmp_path,
        user_root=tmp_path / "no-user",
        development_paths=(root,),
    )
    with pytest.raises(RuntimeError, match="changed after validation"):
        asyncio.run(manager.activate("local.reproducible-extension", require_trust=False))


def test_create_is_non_destructive_and_rejects_invalid_names(tmp_path):
    service = ExtensionProjectService(tmp_path)
    service.create("first-extension")

    with pytest.raises(FileExistsError):
        service.create("first-extension")
    with pytest.raises(ValueError):
        service.create("../escape")


def test_notebook_is_source_of_truth_on_rebuild(tmp_path):
    service = ExtensionProjectService(tmp_path)
    created = service.create("source-of-truth")
    root = Path(created["root"])
    module = root / "extension.py"
    module.write_text("BROKEN = True\n", encoding="utf-8")

    rebuilt = service.build(root)

    assert rebuilt["validation"]["ok"] is True
    assert "BROKEN" not in module.read_text(encoding="utf-8")


def test_constrained_authoring_files_use_hashes_and_never_activate(tmp_path):
    service = set_extension_project_root(tmp_path)
    created = service.create("safe-authoring")
    inspected = service.read_file("local.safe-authoring", "README.md")

    updated = update_extension_file(
        "local.safe-authoring",
        "README.md",
        inspected["content"] + "\nReviewed.\n",
        inspected["sha256"],
    )
    assert "Reviewed" in updated["content"]
    with pytest.raises(ValueError, match="changed since"):
        update_extension_file(
            "local.safe-authoring", "README.md", "stale", inspected["sha256"],
        )
    with pytest.raises(ValueError, match="escapes"):
        service.read_file("local.safe-authoring", "../outside.py")
    with pytest.raises(ValueError, match="manifest editor"):
        service.update_file("local.safe-authoring", "extension.toml", "bad")

    request = request_extension_activation("local.safe-authoring")
    assert request["requires_user_confirmation"] is True
    assert request["extension_id"] == "local.safe-authoring"
    assert get_extension_project_service() is service
    assert created["validation"]["ok"] is True


def test_package_template_builds_a_wheel(tmp_path):
    service = ExtensionProjectService(tmp_path)
    created = service.create("packaged-reference", template="package")

    packaged = service.package(Path(created["root"]))

    assert packaged["ok"] is True
    assert any(path.endswith(".whl") for path in packaged["artifacts"])
