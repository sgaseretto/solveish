from pathlib import Path

from dialeng_project_inspector import _stats, setup


def test_stats_are_bounded(tmp_path: Path):
    (tmp_path / "one.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "two.py").write_text("print(1)", encoding="utf-8")
    assert _stats(tmp_path)["files"] == 2
    assert _stats(tmp_path)["notebooks"] == 1
    assert callable(setup)
