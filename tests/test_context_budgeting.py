"""Tests for size-aware prompt context budgeting."""

from types import SimpleNamespace

from dialeng.document.cell import Cell, CellType
from dialeng.services import dialoghelper_service


def _note(cell_id: str, text: str, *, pinned: bool = False) -> Cell:
    return Cell(id=cell_id, cell_type=CellType.NOTE, source=text, pinned=pinned)


def _prompt(cell_id: str, text: str) -> Cell:
    return Cell(id=cell_id, cell_type=CellType.PROMPT, source=text)


def test_context_budget_prefers_recent_non_pinned_cells(monkeypatch):
    monkeypatch.setattr(dialoghelper_service, "MAX_CONTEXT_CHARS", 220)
    monkeypatch.setattr(dialoghelper_service, "MAX_CONTEXT_CELLS", 100)

    notebook = SimpleNamespace(
        cells=[
            _note("old", "old context " * 6),
            _note("pinned", "pinned context " * 4, pinned=True),
            _note("recent", "recent context " * 5),
            _prompt("target", "What should I keep?"),
        ],
        path=None,
    )

    messages = dialoghelper_service.build_context_messages(notebook, "target")
    flattened = [msg["content"] for msg in messages]

    assert any("pinned context" in content for content in flattened)
    assert any("recent context" in content for content in flattened)
    assert not any("old context" in content for content in flattened)


def test_context_budget_keeps_pinned_cells_even_when_they_fill_budget(monkeypatch):
    monkeypatch.setattr(dialoghelper_service, "MAX_CONTEXT_CHARS", 80)
    monkeypatch.setattr(dialoghelper_service, "MAX_CONTEXT_CELLS", 100)

    notebook = SimpleNamespace(
        cells=[
            _note("pinned", "pinned context " * 10, pinned=True),
            _note("recent", "recent context " * 2),
            _prompt("target", "What should I keep?"),
        ],
        path=None,
    )

    messages = dialoghelper_service.build_context_messages(notebook, "target")
    flattened = [msg["content"] for msg in messages]

    assert any("pinned context" in content for content in flattened)
    assert not any("recent context" in content for content in flattened)
