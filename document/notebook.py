"""Notebook data model."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Iterator
import uuid

from .cell import Cell, CellType, CellState


@dataclass
class Notebook:
    """
    A notebook document containing cells.

    Manages cell operations and provides context gathering
    for LLM prompts.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "Untitled"
    cells: List[Cell] = field(default_factory=list)
    dialog_mode: str = "learning"

    # File info
    path: Optional[Path] = None
    modified: bool = False

    # Solveit metadata
    solveit_ver: int = 2

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Get cell by ID."""
        return next((c for c in self.cells if c.id == cell_id), None)

    def get_cell_index(self, cell_id: str) -> int:
        """Get index of cell, -1 if not found."""
        return next((i for i, c in enumerate(self.cells) if c.id == cell_id), -1)

    def add_cell(self, cell_type: CellType, after_id: Optional[str] = None) -> Cell:
        """Add a new cell, optionally after a specific cell."""
        cell = Cell(cell_type=cell_type)
        if after_id:
            idx = self.get_cell_index(after_id)
            if idx >= 0:
                self.cells.insert(idx + 1, cell)
            else:
                self.cells.append(cell)
        else:
            self.cells.append(cell)
        self.modified = True
        return cell

    def delete_cell(self, cell_id: str) -> bool:
        """Delete a cell by ID."""
        idx = self.get_cell_index(cell_id)
        if idx >= 0:
            self.cells.pop(idx)
            self.modified = True
            return True
        return False

    def move_cell(self, cell_id: str, direction: int) -> bool:
        """Move cell up (-1) or down (+1)."""
        idx = self.get_cell_index(cell_id)
        new_idx = idx + direction
        if 0 <= idx < len(self.cells) and 0 <= new_idx < len(self.cells):
            self.cells[idx], self.cells[new_idx] = self.cells[new_idx], self.cells[idx]
            self.modified = True
            return True
        return False

    def cells_before(self, cell_id: str, include_current: bool = False) -> Iterator[Cell]:
        """Iterate cells before the given cell."""
        idx = self.get_cell_index(cell_id)
        if idx < 0:
            return
        end = idx + 1 if include_current else idx
        for cell in self.cells[:end]:
            yield cell

    def visible_cells(self, before_id: str) -> List[Cell]:
        """Get cells visible in context (not skipped)."""
        return [c for c in self.cells_before(before_id) if not c.skipped]

    def pinned_cells(self) -> List[Cell]:
        """Get all pinned cells."""
        return [c for c in self.cells if c.pinned]

    def code_cells(self) -> List[Cell]:
        """Get all code cells."""
        return [c for c in self.cells if c.cell_type == CellType.CODE]

    def queued_cells(self) -> List[Cell]:
        """Get all cells currently queued for execution."""
        return [c for c in self.cells if c.state == CellState.QUEUED]

    def running_cell(self) -> Optional[Cell]:
        """Get the currently running cell, if any."""
        return next((c for c in self.cells if c.state == CellState.RUNNING), None)

    def ensure_trailing_cell(self) -> bool:
        """Ensure notebook ends with an empty cell. Returns True if cell was added."""
        if not self.cells or self.cells[-1].source.strip():
            self.add_cell(CellType.CODE)
            return True
        return False
