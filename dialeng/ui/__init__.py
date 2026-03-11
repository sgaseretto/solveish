"""
Dialeng UI Package

FastHTML UI components for the Dialeng notebook interface.

Usage:
    from dialeng.ui import CellView, NotebookPage, AllCells, AddButtons

    # Or import specific components
    from dialeng.ui.cells import CodeCellView, NoteCellView, PromptCellView
    from dialeng.ui.controls import TypeSelect, CollapseBtn
    from dialeng.ui.layout import NotebookPage, AllCells
    from dialeng.ui.settings import SettingsSidebar, SettingsOverlay
    from dialeng.ui.outline import OutlineSidebar, OutlineToggleButton
    from dialeng.ui.oob import AllCellsOOB, CellViewOOB
"""

# Base utilities
from .base import get_collapse_class

# Cell components
from .cells import CellView, CellHeader, CodeCellView, NoteCellView, PromptCellView

# Controls
from .controls import TypeSelect, CollapseBtn, AddButtons

# Layout
from .layout import NotebookPage, AllCells, AllCellsContent

# Settings components
from .settings import SettingsSidebar, SettingsOverlay, SettingsGroup

# Outline sidebar components
from .outline import OutlineSidebar, OutlineToggleButton, OutlineSection

# Kernel selection components
from .kernel_modal import KernelToolbarButton, KernelModal

# File explorer components
from .file_explorer import FileExplorerSidebar, FileListContent, NewItemModal

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
    # Settings
    'SettingsSidebar',
    'SettingsOverlay',
    'SettingsGroup',
    # Outline
    'OutlineSidebar',
    'OutlineToggleButton',
    'OutlineSection',
    # Kernel
    'KernelToolbarButton',
    'KernelModal',
    # File Explorer
    'FileExplorerSidebar',
    'FileListContent',
    'NewItemModal',
    # OOB
    'AllCellsOOB',
    'CellViewOOB',
]
