"""Pure structural notebook mutations used by HTTP and compatibility routes."""

from __future__ import annotations

from dataclasses import dataclass

from dialeng.document.cell import Cell
from dialeng.document.notebook import Notebook


@dataclass(frozen=True)
class MergeResult:
    target: Cell
    removed: Cell
    source_changed: bool


class NotebookMutationService:
    """Apply canonical cell-order mutations without transport side effects."""

    def add(self, notebook: Notebook, *, position: int, cell_type: str) -> Cell:
        position = max(0, min(position, len(notebook.cells)))
        cell = Cell(cell_type=cell_type)
        notebook.cells.insert(position, cell)
        notebook.modified = True
        return cell

    def delete(self, notebook: Notebook, cell_id: str) -> Cell | None:
        index = notebook.get_cell_index(cell_id)
        if index < 0:
            return None
        notebook.modified = True
        return notebook.cells.pop(index)

    def move(self, notebook: Notebook, cell_id: str, direction: int) -> bool:
        return notebook.move_cell(cell_id, direction)

    def duplicate(self, notebook: Notebook, cell_id: str) -> Cell | None:
        index = notebook.get_cell_index(cell_id)
        if index < 0:
            return None
        source = notebook.cells[index]
        duplicate = Cell(
            cell_type=source.cell_type,
            source=source.source,
            skipped=source.skipped,
            pinned=source.pinned,
            is_exported=source.is_exported,
        )
        notebook.cells.insert(index + 1, duplicate)
        notebook.modified = True
        return duplicate

    def merge_below(self, notebook: Notebook, cell_id: str) -> MergeResult | None:
        index = notebook.get_cell_index(cell_id)
        if index < 0 or index + 1 >= len(notebook.cells):
            return None
        target = notebook.cells[index]
        removed = notebook.cells[index + 1]
        merged_source = target.source.rstrip("\n") + "\n" + removed.source
        source_changed = target.update_source(merged_source)
        del notebook.cells[index + 1]
        notebook.modified = True
        return MergeResult(target=target, removed=removed, source_changed=source_changed)


notebook_mutation_service = NotebookMutationService()
