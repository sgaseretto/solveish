"""Tests for file explorer and outline incremental caches."""

from types import SimpleNamespace

from dialeng.services.file_listing_cache import DirectoryListingCache
from dialeng.services.outline_cache import OutlineCache


def test_directory_cache_uses_signature_and_targeted_invalidation(tmp_path):
    root = tmp_path
    folder = root / "folder"
    folder.mkdir()
    scans = 0
    cache = DirectoryListingCache()

    def loader():
        nonlocal scans
        scans += 1
        return sorted(path.name for path in folder.iterdir())

    first, first_hit = cache.get(folder, root, loader)
    second, second_hit = cache.get(folder, root, loader)
    assert first == second == []
    assert first_hit is False and second_hit is True
    assert scans == 1

    (folder / "new.py").write_text("pass", encoding="utf-8")
    third, third_hit = cache.get(folder, root, loader)
    assert third == ["new.py"]
    assert third_hit is False
    assert scans == 2

    cache.invalidate(folder)
    cache.get(folder, root, loader)
    assert scans == 3


def test_outline_cache_reuses_unchanged_headings_and_namespace():
    cache = OutlineCache()
    cell = SimpleNamespace(id="note", version=1, source="# Heading")
    extractions = 0

    def extract(source):
        nonlocal extractions
        extractions += 1
        return [{"level": 1, "text": source[2:]}]

    first, first_hit = cache.headings(cell, extract)
    second, second_hit = cache.headings(cell, extract)
    assert first == second
    assert first_hit is False and second_hit is True
    assert extractions == 1

    cell.version = 2
    cache.headings(cell, extract)
    assert extractions == 2

    namespace = {"variables": [{"name": "x"}], "functions": []}
    cache.set_namespace("demo", 1, 3, namespace)
    assert cache.get_namespace("demo", 1, 3) == namespace
    assert cache.get_namespace("demo", 1, 4) is None
    cache.invalidate_notebook("demo")
    assert cache.get_namespace("demo", 1, 3) is None
