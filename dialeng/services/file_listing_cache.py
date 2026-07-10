"""Filesystem-signature cache for directory explorer listings."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class DirectoryListingCache(Generic[T]):
    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple[Path, Path], tuple[tuple[int, int], T]] = OrderedDict()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_ino

    def get(self, path: Path, root: Path, loader: Callable[[], T]) -> tuple[T, bool]:
        key = (root.resolve(), path.resolve())
        signature = self._signature(path)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == signature:
            self._entries.move_to_end(key)
            return cached[1], True
        value = loader()
        self._entries[key] = (signature, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value, False

    def invalidate(self, path: Path | None = None) -> None:
        if path is None:
            self._entries.clear()
            return
        resolved = path.resolve()
        for key in list(self._entries):
            if key[1] == resolved or resolved.is_relative_to(key[1]):
                self._entries.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)


directory_listing_cache: DirectoryListingCache = DirectoryListingCache()
