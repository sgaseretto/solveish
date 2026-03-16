"""Tests for save-hook notebook export to _lib/."""
import json
import pytest
from pathlib import Path


def _make_notebook(cells, path):
    """Helper: write a minimal .ipynb with given code cell sources."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": src if isinstance(src, list) else [src],
                "metadata": {},
                "outputs": [],
                "id": f"cell_{i}",
            }
            for i, src in enumerate(cells)
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


class TestFindDefaultExp:
    def test_finds_default_exp(self, tmp_path):
        from dialeng.services.lib_export_service import find_default_exp
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp mymodule\nimport os"], nb)
        assert find_default_exp(nb) == "mymodule"

    def test_returns_none_when_missing(self, tmp_path):
        from dialeng.services.lib_export_service import find_default_exp
        nb = tmp_path / "test.ipynb"
        _make_notebook(["import os"], nb)
        assert find_default_exp(nb) is None

    def test_strips_whitespace(self, tmp_path):
        from dialeng.services.lib_export_service import find_default_exp
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp  my_utils \n"], nb)
        assert find_default_exp(nb) == "my_utils"

    def test_dotted_module_name(self, tmp_path):
        from dialeng.services.lib_export_service import find_default_exp
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp data.loaders\nimport os"], nb)
        assert find_default_exp(nb) == "data.loaders"


class TestMaybeExtract:
    def test_extracts_export_cells_to_lib(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "utils.ipynb"
        _make_notebook([
            "#| default_exp helpers",
            "#| export\ndef greet(): return 'hello'",
            "# scratch code, not exported",
        ], nb)
        result = maybe_extract(nb, root_dir=tmp_path)
        assert result is not None
        lib_file = tmp_path / "_lib" / "helpers.py"
        assert lib_file.exists()
        content = lib_file.read_text()
        assert "def greet():" in content
        assert "scratch code" not in content

    def test_skips_notebook_without_default_exp(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "scratch.ipynb"
        _make_notebook(["#| export\nx = 1"], nb)
        result = maybe_extract(nb, root_dir=tmp_path)
        assert result is None
        assert not (tmp_path / "_lib").exists()

    def test_creates_init_py(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp foo", "#| export\nX = 1"], nb)
        maybe_extract(nb, root_dir=tmp_path)
        assert (tmp_path / "_lib" / "__init__.py").exists()

    def test_dotted_module_creates_subdirs(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp data.loaders", "#| export\ndef load(): pass"], nb)
        maybe_extract(nb, root_dir=tmp_path)
        assert (tmp_path / "_lib" / "data" / "loaders.py").exists()
        assert (tmp_path / "_lib" / "data" / "__init__.py").exists()

    def test_no_export_cells_removes_stale_file(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp old", "#| export\nX = 1"], nb)
        maybe_extract(nb, root_dir=tmp_path)
        assert (tmp_path / "_lib" / "old.py").exists()
        _make_notebook(["#| default_exp old", "X = 1"], nb)
        maybe_extract(nb, root_dir=tmp_path)
        assert not (tmp_path / "_lib" / "old.py").exists()

    def test_returns_extraction_info(self, tmp_path):
        from dialeng.services.lib_export_service import maybe_extract
        nb = tmp_path / "test.ipynb"
        _make_notebook(["#| default_exp mymod", "#| export\nA = 1", "#| export\nB = 2"], nb)
        result = maybe_extract(nb, root_dir=tmp_path)
        assert result["module"] == "mymod"
        assert result["cells_exported"] == 2
