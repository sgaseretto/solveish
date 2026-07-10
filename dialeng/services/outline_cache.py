"""Versioned heading and kernel-namespace caches for the outline."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Callable


class OutlineCache:
    def __init__(self, max_heading_entries: int = 2048) -> None:
        self.max_heading_entries = max(1, max_heading_entries)
        self._headings: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._namespaces: dict[tuple[str, int, int], dict[str, Any]] = {}

    def headings(self, cell, extractor: Callable[[str], list[dict]]) -> tuple[list[dict], bool]:
        key = (cell.id, cell.version)
        cached = self._headings.get(key)
        if cached is not None:
            self._headings.move_to_end(key)
            return copy.deepcopy(cached), True
        headings = extractor(cell.source)
        self._headings[key] = copy.deepcopy(headings)
        self._headings.move_to_end(key)
        while len(self._headings) > self.max_heading_entries:
            self._headings.popitem(last=False)
        return headings, False

    def get_namespace(
        self,
        notebook_id: str,
        kernel_generation: int,
        execution_count: int,
    ) -> dict[str, Any] | None:
        value = self._namespaces.get((notebook_id, kernel_generation, execution_count))
        return copy.deepcopy(value) if value is not None else None

    def set_namespace(
        self,
        notebook_id: str,
        kernel_generation: int,
        execution_count: int,
        value: dict[str, Any],
    ) -> None:
        self._namespaces = {
            key: item
            for key, item in self._namespaces.items()
            if key[0] != notebook_id
        }
        self._namespaces[(notebook_id, kernel_generation, execution_count)] = copy.deepcopy(value)

    def invalidate_notebook(self, notebook_id: str) -> None:
        self._namespaces = {
            key: item for key, item in self._namespaces.items() if key[0] != notebook_id
        }


outline_cache = OutlineCache()
