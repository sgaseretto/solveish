"""Tests for expanded built-in file tools."""

from dialeng.services.builtin_tools import file_del_lines, replace_lines, strs_replace


def test_strs_replace_updates_multiple_strings(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = strs_replace(str(path), ["alpha", "gamma"], ["one", "three"])

    assert "Applied 2 string replacement" in result
    assert path.read_text(encoding="utf-8") == "one\nbeta\nthree\n"


def test_replace_lines_replaces_inclusive_line_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = replace_lines(str(path), 2, 3, "x\ny")

    assert "Replaced lines 2-3" in result
    assert path.read_text(encoding="utf-8") == "a\nx\ny\nd\n"


def test_file_del_lines_deletes_inclusive_line_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = file_del_lines(str(path), 2, 3)

    assert "Deleted lines 2-3" in result
    assert path.read_text(encoding="utf-8") == "a\nd\n"
