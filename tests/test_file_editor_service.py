"""Tests for standalone text file editing support."""

from pathlib import Path

from starlette.testclient import TestClient

from dialeng.services.file_editor_service import FileEditorService
import dialeng.app as appmod


def test_open_file_acquires_single_writer_lock(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "README.md"
    path.write_text("# Hello\n", encoding="utf-8")

    service = FileEditorService(root)

    opened = service.open_file("README.md", "client-a")
    blocked = service.open_file("README.md", "client-b")

    assert opened.status == "editable"
    assert opened.language == "markdown"
    assert blocked.status == "locked"


def test_release_allows_another_client_to_edit(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "config.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")

    service = FileEditorService(root)
    first = service.open_file("config.json", "client-a")
    assert first.status == "editable"

    service.release("config.json", "client-a")
    second = service.open_file("config.json", "client-b")

    assert second.status == "editable"
    assert second.language == "json"


def test_save_requires_lock_owner_and_updates_content(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "script.py"
    path.write_text('print("old")\n', encoding="utf-8")

    service = FileEditorService(root)
    assert service.open_file("script.py", "client-a").status == "editable"

    denied = service.save_file("script.py", "client-b", 'print("nope")\n')
    saved = service.save_file("script.py", "client-a", 'print("new")\n')

    assert denied.status == "locked"
    assert saved.status == "editable"
    assert path.read_text(encoding="utf-8") == 'print("new")\n'


def test_binary_files_are_noneditable(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "archive.bin"
    path.write_bytes(b"\x00\x01\x02\x03")

    service = FileEditorService(root)
    result = service.open_file("archive.bin", "client-a")

    assert result.status == "noneditable"


def test_utf16_text_file_is_editable(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "notes.txt"
    path.write_text("hola\n", encoding="utf-16")

    service = FileEditorService(root)
    result = service.open_file("notes.txt", "client-a")

    assert result.status == "editable"
    assert result.content == "hola\n"


def test_ipynb_files_remain_notebook_resources(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    path = root / "demo.ipynb"
    path.write_text("{}", encoding="utf-8")

    service = FileEditorService(root)
    result = service.open_file("demo.ipynb", "client-a")

    assert result.status == "notebook"


def test_file_view_route_returns_fragment_only(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    (root / "helpers.py").write_text('print("hello")\n', encoding="utf-8")

    previous_root = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(root)
        client = TestClient(appmod.app)
        response = client.get(
            "/dialeng/file/view",
            params={"path": "helpers.py", "client_id": "client-a"},
            headers={"HX-Request": "true"},
        )
    finally:
        appmod.set_root_dir(previous_root)

    assert response.status_code == 200
    assert "<!doctype html>" not in response.text.lower()
    assert '<input type="hidden" value="editable" id="file-editor-state"' in response.text
    assert '<textarea id="file-source"' in response.text
