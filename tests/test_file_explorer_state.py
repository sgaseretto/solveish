"""Tests for file explorer notebook state rendering."""

from pathlib import Path

from fasthtml.common import to_xml

from dialeng.ui.file_explorer import FileListContent


def test_nested_notebook_shows_active_and_running_state(tmp_path: Path):
    root = tmp_path / "notebooks"
    nested = root / "demo_project"
    nested.mkdir(parents=True)
    (nested / "colab_test.ipynb").write_text("{}", encoding="utf-8")

    html = to_xml(
        FileListContent(
            nested,
            root,
            active_notebook_id="demo_project~colab_test",
            kernel_notebooks={"demo_project~colab_test"},
        )
    )

    assert 'href="/dialeng/?name=demo_project/colab_test"' in html
    assert 'class="file-explorer-item active has-kernel"' in html
    assert '<use href="#notebook-text"></use>' in html


def test_root_notebook_shows_running_state_with_shared_id_encoding(tmp_path: Path):
    root = tmp_path / "notebooks"
    root.mkdir()
    (root / "root_demo.ipynb").write_text("{}", encoding="utf-8")

    html = to_xml(
        FileListContent(
            root,
            root,
            active_notebook_id="root_demo",
            kernel_notebooks={"root_demo"},
        )
    )

    assert 'href="/dialeng/?name=root_demo"' in html
    assert 'class="file-explorer-item active has-kernel"' in html


def test_plain_file_renders_file_editor_link_and_active_state(tmp_path: Path):
    root = tmp_path / "notebooks"
    nested = root / "demo_project"
    nested.mkdir(parents=True)
    (nested / "notes.md").write_text("# Notes\n", encoding="utf-8")

    html = to_xml(
        FileListContent(
            nested,
            root,
            active_notebook_id="",
            active_file_relpath="demo_project/notes.md",
            kernel_notebooks=set(),
        )
    )

    assert 'href="/dialeng/file?path=demo_project/notes.md"' in html
    assert 'class="file-explorer-item active"' in html
    assert '<use href="#file-text"></use>' in html
