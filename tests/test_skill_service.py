import os

import pytest

from dialeng.services.skill_service import SkillService


def _write_skill(root, directory, name="test-skill", description="Useful for tests", body="Do the test."):
    path = root / ".dialeng" / "skills" / directory / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_skills_discover_refresh_and_read_resources(tmp_path):
    path = _write_skill(tmp_path, "testing")
    resource = path.parent / "reference.md"
    resource.write_text("reference one", encoding="utf-8")
    service = SkillService(tmp_path)

    assert [skill.name for skill in service.skills] == ["test-skill"]
    assert service.read_skill("test-skill", "reference.md") == "reference one"
    assert "read_skill" in service.augment_system_prompt("system")

    path.write_text(path.read_text().replace("Useful for tests", "Updated description"), encoding="utf-8")
    os.utime(path, None)
    assert service.skills[0].description == "Updated description"


def test_skills_reject_duplicates_malformed_and_traversal(tmp_path):
    _write_skill(tmp_path, "one", name="duplicate")
    _write_skill(tmp_path, "two", name="duplicate")
    malformed = tmp_path / ".dialeng" / "skills" / "bad" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("no frontmatter", encoding="utf-8")
    service = SkillService(tmp_path)

    assert len(service.skills) == 1
    assert len(service.diagnostics) == 2
    with pytest.raises(ValueError, match="escapes"):
        service.read_skill("duplicate", "../outside.txt")


def test_disabled_skill_is_not_model_invokable(tmp_path):
    path = _write_skill(tmp_path, "manual")
    path.write_text(path.read_text().replace("description:", "disable-model-invocation: true\ndescription:"), encoding="utf-8")
    service = SkillService(tmp_path)
    assert service.skills[0].disable_model_invocation is True
    assert "<skills>" not in service.augment_system_prompt("system")
    with pytest.raises(ValueError, match="disables model invocation"):
        service.read_skill("test-skill")


def test_symlinked_skill_root_cannot_escape_project(tmp_path):
    outside = tmp_path / "outside"
    _write_skill(outside, "escaped")
    project = tmp_path / "project"
    (project / ".dialeng").mkdir(parents=True)
    (project / ".dialeng" / "skills").symlink_to(outside / ".dialeng" / "skills", target_is_directory=True)
    service = SkillService(project)
    assert service.skills == []
    assert service.diagnostics[0].message == "skills root resolves outside the project"
