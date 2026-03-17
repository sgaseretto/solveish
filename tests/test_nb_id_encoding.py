"""Tests for collision-proof notebook ID encoding/decoding.

The encoding uses ~ as path separator with ~~ for literal tildes:
  _nb_id_from_path: Path → ID  (encode)
  _nb_id_to_relpath: ID → Path (decode)
"""
import pytest
from pathlib import Path
from unittest.mock import patch


# We test the encoding functions directly by importing from app.
# They depend on NOTEBOOKS_DIR, so we patch it for tests.
FAKE_NOTEBOOKS_DIR = Path("/fake/notebooks")


@pytest.fixture(autouse=True)
def patch_notebooks_dir():
    with patch("dialeng.app.NOTEBOOKS_DIR", FAKE_NOTEBOOKS_DIR):
        yield


from dialeng.app import _nb_id_from_path, _nb_id_to_relpath, _nb_id_encode_part


class TestEncoding:
    """Test _nb_id_from_path produces correct IDs."""

    def test_root_notebook(self):
        p = FAKE_NOTEBOOKS_DIR / "test.ipynb"
        assert _nb_id_from_path(p) == "test"

    def test_one_level_subdirectory(self):
        p = FAKE_NOTEBOOKS_DIR / "subfolder" / "test.ipynb"
        assert _nb_id_from_path(p) == "subfolder~test"

    def test_two_level_subdirectory(self):
        p = FAKE_NOTEBOOKS_DIR / "a" / "b" / "test.ipynb"
        assert _nb_id_from_path(p) == "a~b~test"

    def test_underscores_preserved(self):
        p = FAKE_NOTEBOOKS_DIR / "my_project" / "data_analysis.ipynb"
        assert _nb_id_from_path(p) == "my_project~data_analysis"

    def test_tilde_in_filename_escaped(self):
        p = FAKE_NOTEBOOKS_DIR / "has~tilde.ipynb"
        assert _nb_id_from_path(p) == "has~~tilde"

    def test_tilde_in_dirname_escaped(self):
        p = FAKE_NOTEBOOKS_DIR / "dir~name" / "test.ipynb"
        assert _nb_id_from_path(p) == "dir~~name~test"

    def test_multiple_tildes_escaped(self):
        p = FAKE_NOTEBOOKS_DIR / "a~~b.ipynb"
        assert _nb_id_from_path(p) == "a~~~~b"

    def test_fallback_for_outside_notebooks_dir(self):
        p = Path("/other/location/test.ipynb")
        assert _nb_id_from_path(p) == "test"

    def test_fallback_with_tilde(self):
        p = Path("/other/location/has~tilde.ipynb")
        assert _nb_id_from_path(p) == "has~~tilde"


class TestDecoding:
    """Test _nb_id_to_relpath reverses the encoding."""

    def test_root_notebook(self):
        assert _nb_id_to_relpath("test") == Path("test")

    def test_one_level_subdirectory(self):
        assert _nb_id_to_relpath("subfolder~test") == Path("subfolder/test")

    def test_two_level_subdirectory(self):
        assert _nb_id_to_relpath("a~b~test") == Path("a/b/test")

    def test_underscores_preserved(self):
        assert _nb_id_to_relpath("my_project~data_analysis") == Path("my_project/data_analysis")

    def test_escaped_tilde(self):
        assert _nb_id_to_relpath("has~~tilde") == Path("has~tilde")

    def test_escaped_tilde_in_dir(self):
        assert _nb_id_to_relpath("dir~~name~test") == Path("dir~name/test")

    def test_multiple_escaped_tildes(self):
        assert _nb_id_to_relpath("a~~~~b") == Path("a~~b")


class TestCollisionProof:
    """Verify that previously-ambiguous paths now produce distinct IDs."""

    def test_no_collision_underscore_in_name_vs_subdir(self):
        """The old scheme made these collide as 'my_project_analysis'."""
        p1 = FAKE_NOTEBOOKS_DIR / "my_project" / "analysis.ipynb"
        p2 = FAKE_NOTEBOOKS_DIR / "my" / "project_analysis.ipynb"
        p3 = FAKE_NOTEBOOKS_DIR / "my_project_analysis.ipynb"

        id1 = _nb_id_from_path(p1)
        id2 = _nb_id_from_path(p2)
        id3 = _nb_id_from_path(p3)

        assert id1 == "my_project~analysis"
        assert id2 == "my~project_analysis"
        assert id3 == "my_project_analysis"
        assert len({id1, id2, id3}) == 3  # All distinct

    def test_no_collision_tilde_in_name_vs_subdir(self):
        p1 = FAKE_NOTEBOOKS_DIR / "a~b" / "c.ipynb"   # dir with tilde
        p2 = FAKE_NOTEBOOKS_DIR / "a" / "b" / "c.ipynb"  # two subdirs

        id1 = _nb_id_from_path(p1)
        id2 = _nb_id_from_path(p2)

        assert id1 == "a~~b~c"
        assert id2 == "a~b~c"
        assert id1 != id2


class TestRoundtrip:
    """Encode → decode must recover the original relative path."""

    @pytest.mark.parametrize("rel_parts", [
        ("test",),
        ("subfolder", "test"),
        ("a", "b", "c"),
        ("my_project", "data_analysis"),
        ("has~tilde",),
        ("dir~name", "file~name"),
        ("a~~b",),
        ("~leading",),
        ("trailing~",),
        ("a", "b~c", "d"),
    ])
    def test_roundtrip(self, rel_parts):
        original = Path(*rel_parts)
        p = FAKE_NOTEBOOKS_DIR / original.parent / f"{original.name}.ipynb"
        nb_id = _nb_id_from_path(p)
        recovered = _nb_id_to_relpath(nb_id)
        assert recovered == original, f"Roundtrip failed: {original} → {nb_id} → {recovered}"


class TestEncodePart:
    """Test the per-component escape function."""

    def test_no_tilde(self):
        assert _nb_id_encode_part("hello_world") == "hello_world"

    def test_single_tilde(self):
        assert _nb_id_encode_part("a~b") == "a~~b"

    def test_multiple_tildes(self):
        assert _nb_id_encode_part("~a~b~") == "~~a~~b~~"

    def test_empty_string(self):
        assert _nb_id_encode_part("") == ""
