"""Tests for dialeng CLI argument parsing."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_no_args_defaults_to_cwd(self):
        from dialeng.cli import parse_args
        args = parse_args([])
        assert args.path == "."

    def test_relative_path(self):
        from dialeng.cli import parse_args
        args = parse_args(["./my-project"])
        assert args.path == "./my-project"

    def test_absolute_path(self):
        from dialeng.cli import parse_args
        args = parse_args(["/tmp/notebooks"])
        assert args.path == "/tmp/notebooks"

    def test_port_flag(self):
        from dialeng.cli import parse_args
        args = parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_default_port(self):
        from dialeng.cli import parse_args
        args = parse_args([])
        assert args.port == 8000

    def test_path_with_port(self):
        from dialeng.cli import parse_args
        args = parse_args(["./projects", "--port", "3000"])
        assert args.path == "./projects"
        assert args.port == 3000


class TestResolveRootDir:
    """Test directory resolution logic."""

    def test_dot_resolves_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from dialeng.cli import resolve_root_dir
        result = resolve_root_dir(".")
        assert result == tmp_path

    def test_relative_path_resolved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        subdir = tmp_path / "notebooks"
        subdir.mkdir()
        from dialeng.cli import resolve_root_dir
        result = resolve_root_dir("notebooks")
        assert result == subdir

    def test_absolute_path_used_directly(self, tmp_path):
        from dialeng.cli import resolve_root_dir
        result = resolve_root_dir(str(tmp_path))
        assert result == tmp_path

    def test_nonexistent_path_created(self, tmp_path):
        from dialeng.cli import resolve_root_dir
        new_dir = tmp_path / "brand-new"
        result = resolve_root_dir(str(new_dir))
        assert result == new_dir
        assert new_dir.exists()

    def test_env_var_override_still_works(self, tmp_path, monkeypatch):
        """DIALENG_NOTEBOOKS_DIR env var should override CLI path."""
        env_dir = tmp_path / "from-env"
        env_dir.mkdir()
        monkeypatch.setenv("DIALENG_NOTEBOOKS_DIR", str(env_dir))
        from dialeng.cli import resolve_root_dir
        result = resolve_root_dir(".", respect_env=True)
        assert result == env_dir


class TestPackageSubcommand:
    """Test that 'dialeng package init' subcommand is parsed."""

    def test_package_init_parsed(self):
        from dialeng.cli import parse_args
        args = parse_args(["package", "init"])
        assert args.subcommand == "package"
        assert args.package_action == "init"

    def test_package_init_with_name(self):
        from dialeng.cli import parse_args
        args = parse_args(["package", "init", "--name", "my_lib"])
        assert args.subcommand == "package"
        assert args.package_name == "my_lib"

    def test_no_subcommand_means_serve(self):
        from dialeng.cli import parse_args
        args = parse_args([])
        assert args.subcommand is None
