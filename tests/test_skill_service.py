import os

import pytest

from dialeng.services.skill_service import SkillService, set_skill_root


def _write_skill(root, directory, name="test-skill", description="Useful for tests", body="Do the test.", *, folder=".dialeng"):
    path = root / folder / "skills" / directory / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("folder", [".dialeng", ".agents"])
def test_skills_discover_refresh_and_read_resources(tmp_path, folder):
    service = SkillService(tmp_path)
    assert service.skills == []
    path = _write_skill(tmp_path, "testing", folder=folder)
    resource = path.parent / "reference.md"
    resource.write_text("reference one", encoding="utf-8")

    assert [skill.name for skill in service.skills] == ["test-skill"]
    assert service.read_skill("test-skill", "reference.md") == "reference one"
    assert "read_skill" in service.augment_system_prompt("system")

    path.write_text(path.read_text().replace("Useful for tests", "Updated description"), encoding="utf-8")
    os.utime(path, None)
    assert service.skills[0].description == "Updated description"
    source = "agents" if folder == ".agents" else "project"
    assert service.skills[0].source == source
    assert any(item["name"] == "test-skill" and item["source"] == source
               for item in service.execution_metadata())
    path.unlink()
    assert service.skills == []


@pytest.mark.parametrize("folder", [".dialeng", ".agents"])
def test_skills_reject_duplicates_malformed_and_traversal(tmp_path, folder):
    _write_skill(tmp_path, "one", name="duplicate", folder=folder)
    _write_skill(tmp_path, "two", name="duplicate", folder=folder)
    malformed = tmp_path / folder / "skills" / "bad" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("no frontmatter", encoding="utf-8")
    service = SkillService(tmp_path)

    assert len(service.skills) == 1
    assert len(service.diagnostics) == 2
    with pytest.raises(ValueError, match="escapes"):
        service.read_skill("duplicate", "../outside.txt")


@pytest.mark.parametrize("folder", [".dialeng", ".agents"])
def test_disabled_skill_is_not_model_invokable(tmp_path, folder):
    path = _write_skill(tmp_path, "manual", folder=folder)
    path.write_text(path.read_text().replace("description:", "disable-model-invocation: true\ndescription:"), encoding="utf-8")
    service = SkillService(tmp_path)
    assert service.skills[0].disable_model_invocation is True
    assert "skill-creator" in service.augment_system_prompt("system")
    assert "test-skill" not in service.augment_system_prompt("system")
    with pytest.raises(ValueError, match="disables model invocation"):
        service.read_skill("test-skill")


def test_builtin_skill_creator_is_available_and_project_can_override_it(tmp_path):
    service = SkillService(tmp_path)
    creator = next(skill for skill in service.all_skills if skill.name == "skill-creator")
    assert creator.source == "builtin"
    assert "build_skill_from_notebooks" in service.read_skill("skill-creator")

    _write_skill(
        tmp_path,
        "skill-creator",
        name="skill-creator",
        description="Project-specific creator",
        body="Use the project workflow.",
    )
    service.refresh(force=True)
    creator = next(skill for skill in service.all_skills if skill.name == "skill-creator")
    assert creator.source == "project"
    assert creator.description == "Project-specific creator"


def test_builtin_extension_creator_documents_constrained_activation(tmp_path):
    service = SkillService(tmp_path)
    creator = next(skill for skill in service.all_skills if skill.name == "extension-creator")

    assert creator.source == "builtin"
    content = service.read_skill("extension-creator")
    assert "request_extension_activation" in content
    assert "only creates review instructions" in content
    assert "Never silently trust, activate" in content
    assert "supervised `python -I` host" in content
    assert "package_extension" in content
    assert "extension sync --path . --offline" in content
    assert "Commands And The Palette" in content
    assert "Extension code runs in the Dialeng server process after trust" not in content


def test_extension_creator_registers_packaging_and_inspection_tools(tmp_path):
    from dialeng.services.tool_registry import get_tool_registry

    set_skill_root(tmp_path)
    registry = get_tool_registry()
    package_schema = registry.get_builtin_schema("package_extension")
    inspect_schema = registry.get_builtin_schema("inspect_extension_api")

    assert package_schema["input_schema"]["required"] == ["extension_id"]
    assert inspect_schema["input_schema"]["properties"]["topic"]["type"] == "string"


def test_builtin_skill_creator_can_be_hidden_without_hiding_project_skills(tmp_path):
    _write_skill(tmp_path, "testing")
    prompt = SkillService(tmp_path).augment_system_prompt("system", include_creator=False)
    assert "skill-creator" not in prompt
    assert "test-skill" in prompt


@pytest.mark.parametrize("folder", [".dialeng", ".agents"])
@pytest.mark.parametrize("late_symlink", [False, True])
def test_symlinked_skill_root_cannot_escape_project(tmp_path, folder, late_symlink):
    outside = tmp_path / "outside"
    _write_skill(outside, "escaped")
    project = tmp_path / "project"
    (project / folder).mkdir(parents=True)
    if late_symlink:
        service = SkillService(project)
        assert service.skills == []
    (project / folder / "skills").symlink_to(outside / ".dialeng" / "skills", target_is_directory=True)
    if not late_symlink:
        service = SkillService(project)
    assert service.skills == []
    assert service.diagnostics[0].message == "skills root resolves outside the project"
    (project / folder / "skills").unlink()
    _write_skill(project, "valid", folder=folder)
    assert [skill.name for skill in service.skills] == ["test-skill"]
    assert service.diagnostics == []


@pytest.mark.parametrize("base_source", ["builtin", "extension"])
def test_agents_override_packaged_skills_and_dialeng_overrides_agents(tmp_path, monkeypatch, base_source):
    from dialeng.extensions import resources

    packaged = _write_skill(tmp_path / "package", "shared", body="Packaged instructions")
    service = SkillService(tmp_path)
    if base_source == "builtin":
        service.builtin_skills_root = packaged.parent.parent
        monkeypatch.setattr(resources, "bundle_roots", lambda kind: [])
    else:
        monkeypatch.setattr(resources, "bundle_roots", lambda kind: [("example", packaged.parent.parent)])
    assert "Packaged instructions" in service.read_skill("test-skill")

    shared = _write_skill(tmp_path, "shared", body="Shared instructions", folder=".agents")
    assert "Shared instructions" in service.read_skill("test-skill")
    assert service.skills[0].source == "agents"
    local = _write_skill(tmp_path, "local", body="Dialeng instructions")
    assert "Dialeng instructions" in service.read_skill("test-skill")
    assert service.skills[0].source == "project"
    assert len(service.skills) == 1
    assert service.diagnostics == []

    local.unlink()
    assert "Shared instructions" in service.read_skill("test-skill")
    shared.unlink()
    assert "Packaged instructions" in service.read_skill("test-skill")


def test_agents_resource_symlink_cannot_escape_skill_directory(tmp_path):
    skill = _write_skill(tmp_path, "shared", folder=".agents")
    outside = tmp_path / "private.txt"
    outside.write_text("Private content", encoding="utf-8")
    (skill.parent / "reference.md").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        SkillService(tmp_path).read_skill("test-skill", "reference.md")


def test_invalid_override_falls_back_but_disabled_override_masks_shared_skill(tmp_path):
    _write_skill(tmp_path, "shared", body="Shared instructions", folder=".agents")
    override = _write_skill(tmp_path, "override", body="Dialeng instructions")
    valid = override.read_text()
    override.write_text("invalid frontmatter", encoding="utf-8")
    service = SkillService(tmp_path)
    assert "Shared instructions" in service.read_skill("test-skill")
    assert len(service.diagnostics) == 1

    override.write_text(valid.replace("description:", "disable-model-invocation: true\ndescription:"), encoding="utf-8")
    assert service.skills[0].source == "project"
    assert "test-skill" not in service.augment_system_prompt("system")
    assert service.diagnostics == []
    with pytest.raises(ValueError, match="disables model invocation"):
        service.read_skill("test-skill")


def test_agents_cli_list_and_validate(tmp_path, capsys):
    from dialeng.cli import cli

    _write_skill(tmp_path, "shared", folder=".agents")
    cli(["skill", "list", "--path", str(tmp_path)])
    assert "test-skill\tagents\tUseful for tests" in capsys.readouterr().out
    cli(["skill", "validate", "--path", str(tmp_path)])
    assert "Validated 1 skill(s)." in capsys.readouterr().out
