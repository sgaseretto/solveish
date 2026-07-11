import asyncio
import json
from pathlib import Path

import pytest
from fasthtml.common import to_xml
from starlette.testclient import TestClient

from dialeng.services.notebook_context_service import NotebookContextService
from dialeng.services.skill_creation_service import SkillCreationService
from dialeng.services.skill_service import SkillService, set_skill_root
from dialeng.ui.skill_creator import SkillCreatorModal


def _write_notebook(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"solveit_note": True, "id": "note"},
                "source": "# Workflow\nAPI_KEY=super-secret-value",
            },
            {
                "cell_type": "markdown",
                "metadata": {"solveit_ai": True, "id": "prompt"},
                "source": "My name is Joe\n\n##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_aabbccdd -->\n\nNice to meet you Joe",
            },
            {
                "cell_type": "code",
                "metadata": {"id": "code", "pinned": True},
                "source": "print('hello')",
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "hello\n"}],
                "execution_count": 1,
            },
            {
                "cell_type": "code",
                "metadata": {"id": "skip", "skipped": True},
                "source": "print('do not include')",
                "outputs": [],
                "execution_count": None,
            },
        ],
    }), encoding="utf-8")


def test_notebook_context_preserves_prompt_semantics_and_bounds_outputs(tmp_path):
    _write_notebook(tmp_path / "course" / "lesson.ipynb")
    service = NotebookContextService(tmp_path)

    without_outputs = service.extract(["course"], include_outputs=False)
    text = without_outputs.documents[0].markdown
    assert "### User\n\nMy name is Joe" in text
    assert "### Assistant\n\nNice to meet you Joe" in text
    assert "do not include" not in text
    assert "### Text output" not in text
    assert "super-secret-value" not in text
    assert "[REDACTED]" in text
    assert without_outputs.documents[0].redaction_count == 1
    assert "<document index=\"1\">" in without_outputs.as_xml()

    with_outputs = service.extract(["course/lesson.ipynb"], include_outputs=True)
    assert "### Text output" in with_outputs.documents[0].markdown


def test_notebook_context_rejects_escape_missing_and_size_limit(tmp_path):
    notebook_path = tmp_path / "lesson.ipynb"
    _write_notebook(notebook_path)
    service = NotebookContextService(tmp_path, max_source_bytes=1024)
    with pytest.raises(ValueError, match="escapes"):
        service.extract(["../outside.ipynb"])
    with pytest.raises(ValueError, match="does not exist"):
        service.extract(["missing.ipynb"])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][2]["source"] += "\n# " + "x" * 2000
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        service.extract(["lesson.ipynb"])


def test_skill_context_combines_notebook_text_and_web2md_sources(tmp_path, monkeypatch):
    _write_notebook(tmp_path / "lesson.ipynb")
    (tmp_path / "guide.md").write_text("# Local guide\n\nPASSWORD=hidden", encoding="utf-8")

    class Response:
        text = "# Web guide\n\nUse the remote workflow."
        def raise_for_status(self):
            return None

    def fake_post(url, *, data, timeout, follow_redirects):
        assert url == "https://web2md.answer.ai/api"
        assert data == {"url": "https://example.com/guide", "extractor": "traf"}
        return Response()

    monkeypatch.setattr("dialeng.services.notebook_context_service.httpx.post", fake_post)
    service = NotebookContextService(tmp_path)
    bundle = service.extract([
        "lesson.ipynb", "guide.md", "https://example.com/guide",
    ])

    assert [document.kind for document in bundle.documents] == ["file", "notebook", "url"]
    assert "[REDACTED]" in bundle.documents[0].markdown
    assert bundle.documents[-1].fetched_at
    assert "Web guide" in bundle.documents[-1].markdown


def test_skill_build_preview_stale_hash_and_atomic_commit(tmp_path):
    _write_notebook(tmp_path / "lesson.ipynb")
    set_skill_root(tmp_path)
    service = SkillCreationService(tmp_path)
    preview = service.create_build(
        "lesson-helper",
        "Use when applying the lesson workflow.",
        ["lesson.ipynb"],
    )

    assert preview["valid"] is True
    assert "references/lesson-" in "\n".join(preview["files"])
    assert not (tmp_path / ".dialeng" / "skills" / "lesson-helper").exists()
    old_hash = preview["build_hash"]
    updated = service.update_build(preview["build_id"], "# Workflow\n\nRead the lesson reference first.")
    with pytest.raises(ValueError, match="changed after preview"):
        service.commit_build(preview["build_id"], old_hash)

    result = service.commit_build(updated["build_id"], updated["build_hash"])
    assert result["name"] == "lesson-helper"
    installed = tmp_path / ".dialeng" / "skills" / "lesson-helper"
    assert (installed / "SKILL.md").is_file()
    assert (installed / ".dialeng-sources.json").is_file()
    assert SkillService(tmp_path).skills[0].name == "lesson-helper"


def test_skill_build_does_not_overwrite_without_explicit_update(tmp_path):
    _write_notebook(tmp_path / "lesson.ipynb")
    set_skill_root(tmp_path)
    service = SkillCreationService(tmp_path)
    first = service.create_build("lesson", "Use for lessons.", ["lesson.ipynb"])
    service.commit_build(first["build_id"], first["build_hash"])
    second = service.create_build("lesson", "Use for revised lessons.", ["lesson.ipynb"])
    with pytest.raises(FileExistsError):
        service.commit_build(second["build_id"], second["build_hash"])


def test_skill_build_can_be_refined_through_provider_manager(tmp_path, monkeypatch):
    _write_notebook(tmp_path / "lesson.ipynb")
    service = SkillCreationService(tmp_path)
    preview = service.create_build("lesson", "Use for lessons.", ["lesson.ipynb"])

    from dialeng.services.llm import llm_service

    monkeypatch.setattr(llm_service, "resolve_provider_id", lambda provider: "openai_codex")

    async def fake_stream(*args, **kwargs):
        yield {"type": "chunk", "content": '{"instructions":"# Refined workflow\\n\\nRead the source before acting."}'}

    monkeypatch.setattr(llm_service, "stream_response", fake_stream)
    refined = asyncio.run(service.refine_build(preview["build_id"], model="gpt-5.4"))
    assert refined["refined_by_llm"] is True
    assert refined["provider"] == "openai_codex"
    assert "# Refined workflow" in refined["files"]["SKILL.md"]


def test_skill_creator_routes_stage_preview_and_commit(tmp_path):
    from dialeng import app as appmod
    from dialeng.services.dialeng_config import reset_config_cache

    _write_notebook(tmp_path / "lesson.ipynb")
    previous_root = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(tmp_path)
        client = TestClient(appmod.app)
        snapshot = client.get("/dialeng/skills")
        assert snapshot.status_code == 200
        assert "lesson.ipynb" in snapshot.json()["notebooks"]
        assert any(skill["name"] == "skill-creator" for skill in snapshot.json()["skills"])

        staged = client.post("/dialeng/skills/build", json={
            "name": "lesson-web",
            "description": "Use for browser-created lessons.",
            "sources": ["lesson.ipynb"],
            "use_llm": False,
        })
        assert staged.status_code == 200
        preview = staged.json()
        assert preview["valid"] is True
        installed = client.post(
            f"/dialeng/skills/build/{preview['build_id']}/commit",
            json={"expected_hash": preview["build_hash"]},
        )
        assert installed.status_code == 200
        assert (tmp_path / ".dialeng" / "skills" / "lesson-web" / "SKILL.md").is_file()
    finally:
        appmod.set_root_dir(previous_root)
        reset_config_cache()


def test_skill_creator_modal_is_review_first():
    html = to_xml(SkillCreatorModal())
    assert "Create Skill From Sources" in html
    assert 'id="skill-creator-urls"' in html
    assert 'id="skill-creator-preview"' in html
    assert 'id="skill-install-btn"' in html
    assert "disabled" in html


def test_skill_cli_creates_from_notebook_non_interactively(tmp_path, capsys):
    from dialeng.cli import cli
    from dialeng.services.dialeng_config import reset_config_cache

    _write_notebook(tmp_path / "lesson.ipynb")
    try:
        cli([
            "skill", "create", "lesson-cli",
            "--from", "lesson.ipynb",
            "--description", "Use for CLI-created lessons.",
            "--no-llm", "--yes", "--path", str(tmp_path),
        ])
        assert (tmp_path / ".dialeng" / "skills" / "lesson-cli" / "SKILL.md").is_file()
        assert "Installed lesson-cli" in capsys.readouterr().out
    finally:
        reset_config_cache()


def test_skill_creator_tool_schema_accepts_notebook_arrays(tmp_path):
    from dialeng.services.tool_registry import get_tool_registry

    set_skill_root(tmp_path)
    schema = get_tool_registry().get_builtin_schema("build_skill_from_notebooks")
    assert schema["input_schema"]["properties"]["sources"]["type"] == "array"
