"""Tests for dialeng package init scaffolding."""
import json
import pytest
from pathlib import Path


def _make_notebook(cells, path):
    """Helper: write a minimal .ipynb."""
    nb = {
        "cells": [
            {"cell_type": "code", "source": src if isinstance(src, list) else [src],
             "metadata": {}, "outputs": [], "id": f"cell_{i}"}
            for i, src in enumerate(cells)
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


class TestScaffoldPackage:
    def test_creates_pyproject_toml(self, tmp_path):
        from dialeng.services.package_scaffold_service import scaffold_package
        _make_notebook(["#| default_exp core", "#| export\nX = 1"], tmp_path / "core.ipynb")
        scaffold_package(tmp_path, package_name="mylib")
        assert (tmp_path / "pyproject.toml").exists()

    def test_pyproject_has_nbdev_section(self, tmp_path):
        from dialeng.services.package_scaffold_service import scaffold_package
        _make_notebook(["#| default_exp core", "#| export\nX = 1"], tmp_path / "core.ipynb")
        scaffold_package(tmp_path, package_name="mylib")
        content = (tmp_path / "pyproject.toml").read_text()
        assert "[tool.nbdev]" in content

    def test_defaults_name_to_directory(self, tmp_path):
        from dialeng.services.package_scaffold_service import scaffold_package
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        _make_notebook(["#| default_exp core", "#| export\nX = 1"], project_dir / "core.ipynb")
        scaffold_package(project_dir)
        content = (project_dir / "pyproject.toml").read_text()
        assert 'name = "my_project"' in content

    def test_creates_package_directory(self, tmp_path):
        from dialeng.services.package_scaffold_service import scaffold_package
        _make_notebook(["#| default_exp core", "#| export\nX = 1"], tmp_path / "core.ipynb")
        scaffold_package(tmp_path, package_name="mylib")
        assert (tmp_path / "mylib").is_dir()
        assert (tmp_path / "mylib" / "__init__.py").exists()

    def test_does_not_overwrite_existing_pyproject(self, tmp_path):
        from dialeng.services.package_scaffold_service import scaffold_package
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'existing'\n")
        _make_notebook(["#| default_exp core", "#| export\nX = 1"], tmp_path / "core.ipynb")
        with pytest.raises(FileExistsError):
            scaffold_package(tmp_path, package_name="mylib")

    def test_scans_notebooks_for_modules(self, tmp_path):
        from dialeng.services.package_scaffold_service import scan_notebook_modules
        _make_notebook(["#| default_exp utils", "#| export\nX = 1"], tmp_path / "utils.ipynb")
        _make_notebook(["#| default_exp data.loaders", "#| export\ndef load(): pass"], tmp_path / "data.ipynb")
        _make_notebook(["import os"], tmp_path / "scratch.ipynb")
        modules = scan_notebook_modules(tmp_path)
        assert set(modules.keys()) == {"utils", "data.loaders"}
