"""Tests for AUTORUN hot-reload and action registry."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============================================================================
# reload_autorun_extensions tests
# ============================================================================

class TestReloadAutorunExtensions:
    """Test the reload_autorun_extensions() function."""

    def test_returns_empty_when_no_autorun_dir(self, tmp_path):
        from dialeng.services.autorun_service import reload_autorun_extensions, AUTORUN_DIR
        with patch("dialeng.services.autorun_service.AUTORUN_DIR", tmp_path / "nonexistent"):
            result = reload_autorun_extensions()
        assert result == {"extracted": [], "loaded": [], "errors": []}

    def test_returns_dict_structure(self, tmp_path):
        from dialeng.services.autorun_service import reload_autorun_extensions
        autorun_dir = tmp_path / "AUTORUN"
        autorun_dir.mkdir()
        with patch("dialeng.services.autorun_service.AUTORUN_DIR", autorun_dir), \
             patch("dialeng.services.autorun_service.CACHE_DIR", tmp_path / ".autorun_modules"):
            result = reload_autorun_extensions()
        assert "extracted" in result
        assert "loaded" in result
        assert "errors" in result

    def test_extracts_from_notebooks(self, tmp_path):
        """Test that extract_extension is called for each .ipynb in AUTORUN/."""
        autorun_dir = tmp_path / "AUTORUN"
        autorun_dir.mkdir()
        cache_dir = tmp_path / ".autorun_modules"

        # Create a minimal notebook file
        import json
        nb = {
            "cells": [{
                "cell_type": "code",
                "source": ["#| export\n", "X = 42\n"],
                "metadata": {},
                "outputs": []
            }],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        (autorun_dir / "test_ext.ipynb").write_text(json.dumps(nb))

        with patch("dialeng.services.autorun_service.AUTORUN_DIR", autorun_dir), \
             patch("dialeng.services.autorun_service.CACHE_DIR", cache_dir):
            from dialeng.services.autorun_service import reload_autorun_extensions
            result = reload_autorun_extensions()

        assert "test_ext.ipynb" in result["extracted"]
        # The generated .py file should exist
        assert (cache_dir / "test_ext.py").exists()

    def test_handles_extract_errors_gracefully(self, tmp_path):
        """Errors in one notebook don't prevent others from loading."""
        autorun_dir = tmp_path / "AUTORUN"
        autorun_dir.mkdir()
        cache_dir = tmp_path / ".autorun_modules"

        # Write an invalid notebook
        (autorun_dir / "bad.ipynb").write_text("not valid json")

        with patch("dialeng.services.autorun_service.AUTORUN_DIR", autorun_dir), \
             patch("dialeng.services.autorun_service.CACHE_DIR", cache_dir):
            from dialeng.services.autorun_service import reload_autorun_extensions
            result = reload_autorun_extensions()

        assert len(result["errors"]) > 0
        assert "bad.ipynb" in result["errors"][0]


# ============================================================================
# Registry action tests
# ============================================================================

class TestRegistryActions:
    """Test the action registration system."""

    def setup_method(self):
        """Get a fresh reference to the registry."""
        from dialeng.core.registry import registry
        self.registry = registry
        # Clean up any actions from previous tests
        self._original_actions = dict(self.registry.actions)

    def teardown_method(self):
        """Restore original actions."""
        self.registry.actions = self._original_actions

    def test_register_action(self):
        def my_handler(nb_id, **kwargs):
            return {"ok": True}
        self.registry.register_action("test_action", my_handler)
        assert "test_action" in self.registry.actions
        assert self.registry.actions["test_action"] is my_handler

    def test_register_action_overwrites(self):
        def handler1(nb_id, **kwargs):
            return {"v": 1}
        def handler2(nb_id, **kwargs):
            return {"v": 2}
        self.registry.register_action("overwrite_test", handler1)
        self.registry.register_action("overwrite_test", handler2)
        assert self.registry.actions["overwrite_test"] is handler2

    def test_unknown_action_returns_none(self):
        assert self.registry.actions.get("nonexistent") is None

    def test_repr_includes_actions(self):
        self.registry.register_action("repr_test", lambda **kw: {})
        r = repr(self.registry)
        assert "actions=" in r
        assert "repr_test" in r


class TestRegisterActionDecorator:
    """Test the @register_action decorator."""

    def setup_method(self):
        from dialeng.core.registry import registry
        self.registry = registry
        self._original_actions = dict(self.registry.actions)

    def teardown_method(self):
        self.registry.actions = self._original_actions

    def test_decorator_registers_handler(self):
        from dialeng.core.registry import register_action

        @register_action("decorated_action")
        def my_action(nb_id, **kwargs):
            return {"decorated": True}

        assert "decorated_action" in self.registry.actions
        result = self.registry.actions["decorated_action"](nb_id="test")
        assert result == {"decorated": True}

    def test_decorator_preserves_function(self):
        from dialeng.core.registry import register_action

        @register_action("preserve_test")
        def my_func(nb_id, **kwargs):
            """My docstring."""
            return {}

        assert my_func.__doc__ == "My docstring."
        assert my_func.__name__ == "my_func"
