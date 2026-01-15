"""
Dialeng UI Package

FastHTML UI components for the Dialeng notebook interface.

Usage:
    from ui import CellView, NotebookPage, AllCells, AddButtons

    # Or import specific components
    from ui.cells import CodeCellView, NoteCellView, PromptCellView
    from ui.controls import TypeSelect, CollapseBtn
    from ui.layout import NotebookPage, AllCells
    from ui.oob import AllCellsOOB, CellViewOOB
"""

# Base utilities
from .base import get_collapse_class

# Cell components
from .cells import CellView, CellHeader, CodeCellView, NoteCellView, PromptCellView

# Controls
from .controls import TypeSelect, CollapseBtn, AddButtons

# Layout
from .layout import NotebookPage, AllCells, AllCellsContent

# OOB (Out-of-Band) components for WebSocket
from .oob import AllCellsOOB, CellViewOOB

__all__ = [
    # Base
    'get_collapse_class',
    # Cells
    'CellView',
    'CellHeader',
    'CodeCellView',
    'NoteCellView',
    'PromptCellView',
    # Controls
    'TypeSelect',
    'CollapseBtn',
    'AddButtons',
    # Layout
    'NotebookPage',
    'AllCells',
    'AllCellsContent',
    # OOB
    'AllCellsOOB',
    'CellViewOOB',
]
