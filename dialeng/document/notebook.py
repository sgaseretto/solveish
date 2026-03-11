"""Notebook data model."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Iterator, Dict, Any
import json
import uuid

from .cell import Cell, CellType, CellState


@dataclass
class Notebook:
    """
    A notebook document containing cells.

    Manages cell operations, serialization, and provides context gathering
    for LLM prompts.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "Untitled Notebook"
    cells: List[Cell] = field(default_factory=list)
    dialog_mode: str = "learning"
    model: Optional[str] = None  # LLM model to use (None = use config default)

    # File info
    path: Optional[Path] = None
    modified: bool = False

    # Solveit metadata
    solveit_ver: int = 2

    # Shell execution settings
    safe_mode: bool = False  # Enable safecmd validation for shell commands

    # Kernel settings
    kernel_type: str = "local"  # "local" or "colab"
    colab_runtime_type: str = "cpu"  # "cpu", "gpu", or "tpu"

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

    @property
    def default_export_module(self) -> Optional[str]:
        """Get the nbdev default export module name from #| default_exp directive."""
        for cell in self.cells:
            source = cell.source.strip()
            if source.startswith("#| default_exp"):
                parts = source.split()
                if len(parts) >= 2:
                    return parts[-1]
        return None

    def ensure_trailing_cell(self) -> bool:
        """Ensure notebook ends with an empty cell. Returns True if cell was added."""
        if not self.cells or self.cells[-1].source.strip():
            self.add_cell(CellType.CODE)
            return True
        return False

    def to_ipynb(self) -> Dict[str, Any]:
        """Convert notebook to .ipynb format."""
        return {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11.0"},
                "title": self.title,
                "solveit_dialog_mode": self.dialog_mode,
                "dialeng_model": self.model,
                "solveit_ver": self.solveit_ver,
                "dialeng_safe_mode": self.safe_mode,
                "dialeng_kernel_type": self.kernel_type,
                "dialeng_colab_runtime_type": self.colab_runtime_type,
            },
            "cells": [cell.to_jupyter_cell() for cell in self.cells]
        }

    @classmethod
    def from_ipynb(cls, data: Dict[str, Any], notebook_id: str = None,
                   default_dialog_mode: str = "learning",
                   model_validator=None) -> "Notebook":
        """Load notebook from .ipynb data.

        Args:
            data: Parsed .ipynb JSON data
            notebook_id: Override notebook ID (defaults to random)
            default_dialog_mode: Fallback dialog mode
            model_validator: Optional callable(model_id) -> valid_model_id
        """
        metadata = data.get("metadata", {})
        cells = [Cell.from_jupyter_cell(c) for c in data.get("cells", [])]
        saved_mode = metadata.get("solveit_dialog_mode", default_dialog_mode)
        saved_model = metadata.get("dialeng_model") or metadata.get("solveit_model", "")
        effective_model = model_validator(saved_model) if model_validator and saved_model else saved_model or None
        return cls(
            id=notebook_id or uuid.uuid4().hex[:8],
            title=metadata.get("title", "Imported Notebook"),
            cells=cells,
            dialog_mode=saved_mode,
            model=effective_model,
            solveit_ver=metadata.get("solveit_ver", 2),
            safe_mode=metadata.get("dialeng_safe_mode", False),
            kernel_type=metadata.get("dialeng_kernel_type", "local"),
            colab_runtime_type=metadata.get("dialeng_colab_runtime_type", "cpu"),
        )

    def save(self, path: str):
        """Save notebook to .ipynb file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(self.to_ipynb(), f, indent=2)
        self.path = p
        self.modified = False

    @classmethod
    def load(cls, path: str, default_dialog_mode: str = "learning",
             model_validator=None) -> "Notebook":
        """Load notebook from .ipynb file.

        Args:
            path: Path to .ipynb file
            default_dialog_mode: Fallback dialog mode
            model_validator: Optional callable(model_id) -> valid_model_id
        """
        with open(path) as f:
            data = json.load(f)
        nb_id = Path(path).stem
        nb = cls.from_ipynb(data, nb_id,
                            default_dialog_mode=default_dialog_mode,
                            model_validator=model_validator)
        nb.title = Path(path).stem
        nb.path = Path(path)
        return nb
