"""Configurable originals, relative aliases, and recoverable skill installation."""
import json
import os
import shutil
from pathlib import Path

import pytest

from dialeng.services.dialeng_config import DialengConfig
from dialeng.services.skill_creation_service import SkillCreationService, build_skill_from_notebooks
from dialeng.services.skill_service import SkillService, set_skill_root


@pytest.fixture
def creation(tmp_path, monkeypatch):
    (tmp_path / "workflow.md").write_text("# Workflow\nRead the source first.", encoding="utf-8")
    config = DialengConfig()
    monkeypatch.setattr("dialeng.services.dialeng_config.get_config", lambda: config)
    set_skill_root(tmp_path)
    return SkillCreationService(tmp_path), config


def draft(service, **options):
    return service.create_build("shared-workflow", "Use for project work.", ["workflow.md"], **options)


@pytest.mark.parametrize("install_root", ["agents", "dialeng"])
@pytest.mark.parametrize("linked", [False, True])
def test_install_update_and_discover_original_with_optional_link(creation, tmp_path, install_root, linked):
    service, _ = creation
    other = "dialeng" if install_root == "agents" else "agents"
    preview = draft(service, install_root=install_root, link_other_root=linked)
    original = tmp_path / f".{install_root}/skills/shared-workflow"
    alias = tmp_path / f".{other}/skills/shared-workflow"
    assert preview["target"] == original.relative_to(tmp_path).as_posix()
    assert preview["link_target"] == (alias.relative_to(tmp_path).as_posix() if linked else None)
    assert not original.exists() and not alias.exists()

    result = service.commit_build(preview["build_id"], preview["build_hash"])
    assert original.is_dir() and not original.is_symlink()
    assert (original / "SKILL.md").is_file()
    assert alias.is_symlink() is linked
    assert result["link_path"] == preview["link_target"]
    if linked:
        assert not Path(os.readlink(alias)).is_absolute()
        assert alias.resolve() == original
        assert (alias / "SKILL.md").read_text() == (original / "SKILL.md").read_text()
    skills = SkillService(tmp_path)
    assert [skill.name for skill in skills.skills] == ["shared-workflow"]
    assert skills.diagnostics == []
    assert "Read the source" in skills.read_skill("shared-workflow", next(
        path.relative_to(original).as_posix() for path in (original / "references").iterdir()
    ))

    updated = draft(service, install_root=install_root, link_other_root=linked, instructions="Updated instructions.")
    service.commit_build(updated["build_id"], updated["build_hash"], overwrite=True)
    assert "Updated instructions" in skills.read_skill("shared-workflow")
    if linked:
        assert "Updated instructions" in (alias / "SKILL.md").read_text()


def test_relative_link_survives_project_move(creation, tmp_path):
    service, _ = creation
    preview = draft(service, install_root="agents", link_other_root=True)
    service.commit_build(preview["build_id"], preview["build_hash"])
    moved = tmp_path / "relocated"
    moved.mkdir()
    for folder in (".agents", ".dialeng"):
        shutil.move(str(tmp_path / folder), moved / folder)
    assert (moved / ".dialeng/skills/shared-workflow").resolve() == moved / ".agents/skills/shared-workflow"
    assert SkillService(moved).read_skill("shared-workflow")


def test_model_tool_uses_config_and_reviewed_plan_survives_setting_changes(creation, tmp_path):
    service, config = creation
    config.skills_install_root = "agents"
    config.skills_link_other_root = True
    preview = build_skill_from_notebooks("shared-workflow", "Use for project work.", ["workflow.md"])
    config.skills_install_root = "dialeng"
    config.skills_link_other_root = False
    reloaded = SkillCreationService(tmp_path)
    assert reloaded.inspect_build(preview["build_id"])["target"] == ".agents/skills/shared-workflow"
    result = reloaded.commit_build(preview["build_id"], preview["build_hash"])
    assert result["path"] == ".agents/skills/shared-workflow"
    assert result["link_path"] == ".dialeng/skills/shared-workflow"


@pytest.mark.parametrize("change", [{"install_root": "agents"}, {"link_other_root": True}])
def test_destination_changes_invalidate_preview_hash(creation, tmp_path, change):
    service, _ = creation
    preview = draft(service)
    manifest = tmp_path / ".dialeng/skill-builds" / preview["build_id"] / "build.json"
    data = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**data, **change}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preview"):
        service.commit_build(preview["build_id"], preview["build_hash"])
    assert not (tmp_path / ".dialeng/skills/shared-workflow").exists()
    assert not (tmp_path / ".agents/skills/shared-workflow").exists()


@pytest.mark.parametrize("occupied", ["directory", "file", "symlink"])
def test_link_conflict_preserves_both_existing_skills(creation, tmp_path, occupied):
    service, _ = creation
    first = draft(service, install_root="agents", instructions="Keep original.")
    service.commit_build(first["build_id"], first["build_hash"])
    alias = tmp_path / ".dialeng/skills/shared-workflow"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if occupied == "directory":
        alias.mkdir()
        (alias / "keep.txt").write_text("Keep independent skill.")
    elif occupied == "file":
        alias.write_text("Keep independent file.")
    else:
        alias.symlink_to("missing-target")
    preview = draft(service, install_root="agents", link_other_root=True)
    with pytest.raises(FileExistsError, match="already occupied"):
        service.commit_build(preview["build_id"], preview["build_hash"], overwrite=True)
    assert "Keep original" in (tmp_path / ".agents/skills/shared-workflow/SKILL.md").read_text()
    assert alias.exists() or alias.is_symlink()


@pytest.mark.parametrize("existing", [False, True])
def test_symlink_failure_rolls_back_installation(creation, tmp_path, monkeypatch, existing):
    service, _ = creation
    if existing:
        first = draft(service, install_root="agents", instructions="Original content.")
        service.commit_build(first["build_id"], first["build_hash"])
    preview = draft(service, install_root="agents", link_other_root=True)

    def unavailable(*args, **kwargs):
        raise OSError("Symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", unavailable)
    with pytest.raises(OSError, match="Symlinks unavailable"):
        service.commit_build(preview["build_id"], preview["build_hash"], overwrite=existing)
    original = tmp_path / ".agents/skills/shared-workflow"
    assert original.exists() is existing
    if existing:
        assert "Original content" in (original / "SKILL.md").read_text()
    assert not (tmp_path / ".dialeng/skills/shared-workflow").exists()
    assert service.inspect_build(preview["build_id"])["valid"]
    assert not list(original.parent.glob(".*"))


@pytest.mark.parametrize("escape_root", [".agents", ".dialeng"])
def test_installation_rejects_escaping_original_or_link_root(creation, tmp_path, escape_root):
    service, _ = creation
    preview = draft(service, install_root="agents", link_other_root=True)
    outside = tmp_path.parent / f"outside-{escape_root}"
    outside.mkdir(exist_ok=True)
    folder = tmp_path / escape_root
    folder.mkdir(exist_ok=True)
    (folder / "skills").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes the project"):
        service.commit_build(preview["build_id"], preview["build_hash"])
    assert list(outside.iterdir()) == []


def test_update_rejects_using_alias_as_original(creation, tmp_path):
    service, _ = creation
    first = draft(service, install_root="agents", link_other_root=True, instructions="Keep original.")
    service.commit_build(first["build_id"], first["build_hash"])
    second = draft(service, install_root="dialeng")
    with pytest.raises(ValueError, match="select its original install_root"):
        service.commit_build(second["build_id"], second["build_hash"], overwrite=True)
    assert "Keep original" in (tmp_path / ".agents/skills/shared-workflow/SKILL.md").read_text()


@pytest.mark.parametrize("options", [{"install_root": "../outside"}, {"link_other_root": "false"}])
def test_invalid_install_options_are_rejected_before_staging(creation, tmp_path, options):
    service, _ = creation
    with pytest.raises(ValueError):
        draft(service, **options)
    assert service.list_builds() == []
