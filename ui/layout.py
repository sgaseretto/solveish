"""
Dialeng UI - Layout Components

Page layout and container components.
"""

from fasthtml.common import *
from typing import List, Optional
from .cells import CellView
from .controls import AddButtons
from .settings import SettingsSidebar, SettingsOverlay
from .outline import OutlineSidebar, OutlineToggleButton
from services.dialeng_config import DialengConfig


def AllCellsContent(nb):
    """Returns just the cell content (for innerHTML swaps).

    Args:
        nb: Notebook instance

    Returns:
        Div containing all cells without wrapper ID (for content swaps)
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i + 1, nb.id)])
    return Div(*items)


def AllCells(nb):
    """Returns all cells wrapped in a container with ID.

    Args:
        nb: Notebook instance

    Returns:
        Div with id="cells" containing all cells
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i + 1, nb.id)])
    return Div(*items, id="cells")


def NotebookPage(nb, notebook_list: List[str], available_dialog_modes: list, available_models: list,
                 config: Optional[DialengConfig] = None, shfmt_available: bool = True):
    """Render the complete notebook page.

    Args:
        nb: Notebook instance
        notebook_list: List of notebook IDs for the file list
        available_dialog_modes: List of (mode_id, label) tuples
        available_models: List of (model_id, label) tuples
        config: Optional DialengConfig for settings sidebar
        shfmt_available: Whether shfmt binary is installed (for safe mode)

    Returns:
        Complete page with Titled wrapper
    """
    return Titled(
        f"{nb.title} - Dialeng",
        # Main layout wrapper - flex container for outline sidebar + content
        Div(
            # Outline sidebar (push style - on left)
            OutlineSidebar(nb.id, is_open=False),
            # Main content container
            Div(
                Div(
                    Div(Span("📓", cls="title-icon"), Span(nb.title, cls="title")),
                    Div(
                        OutlineToggleButton(),
                        Button("☀️", cls="theme-toggle", id="theme-toggle",
                               onclick="toggleTheme()", title="Toggle light/dark theme"),
                        Select(
                            *[Option(label, value=mode_id, selected=nb.dialog_mode == mode_id)
                              for mode_id, label in available_dialog_modes],
                            cls="mode-select", name="mode", id="mode-select",
                            hx_post=f"/notebook/{nb.id}/mode", hx_swap="none", title="AI Mode",
                            onchange="toggleModelSelect(this.value)"
                        ),
                        Select(
                            *[Option(label, value=model_id, selected=getattr(nb, 'model', None) == model_id)
                              for model_id, label in available_models],
                            cls="model-select", name="model", id="model-select",
                            hx_post=f"/notebook/{nb.id}/model", hx_swap="none", title="Model",
                            style="display: none;" if nb.dialog_mode == "mock" else ""
                        ),
                        # Safe mode toggle for shell commands
                        Label(
                            Input(type="checkbox", name="safe_mode", id="safe-mode-toggle",
                                  checked=getattr(nb, 'safe_mode', False),
                                  disabled=not shfmt_available,
                                  hx_post=f"/notebook/{nb.id}/safe_mode",
                                  hx_swap="none",
                                  hx_vals="js:{safe_mode: event.target.checked}",
                                  cls="safe-mode-checkbox"),
                            Span("Safe", cls="safe-mode-label"),
                            cls="safe-mode-toggle",
                            title="Safe Mode: Validate shell commands against allowlist" if shfmt_available else "Safe Mode unavailable - install shfmt"
                        ),
                        Button("Restart", cls="btn btn-sm",
                               hx_post=f"/notebook/{nb.id}/kernel/restart", hx_target="#status", title="Restart kernel"),
                        Button("⏹ Cancel All", cls="btn btn-sm btn-cancel-all", id="cancel-all-btn",
                               onclick="cancelAllExecution()", title="Cancel running cell and clear queue (Esc Esc)",
                               style="display: none;"),
                        Button("💾 Save", cls="btn btn-sm btn-save", id="save-btn",
                               hx_post=f"/notebook/{nb.id}/save", hx_target="#status", title="Save (Ctrl+S)"),
                        Button("📥 Export", cls="btn btn-sm",
                               hx_get=f"/notebook/{nb.id}/export", title="Download .ipynb"),
                        Button("⚙️", cls="btn btn-sm settings-btn", id="settings-btn",
                               onclick="toggleSettings()", title="Settings"),
                        cls="toolbar"
                    ),
                    cls="header"
                ),
                Div(id="status"),
                Div(
                    *[A(name, href=f"/notebook/{name}",
                        cls=f"file-item{' active' if name == nb.id else ''}")
                      for name in notebook_list],
                    A("+ New", href="/notebook/new", cls="file-item"),
                    cls="file-list"
                ) if notebook_list else None,
                AllCells(nb),
                Script(f"window.NOTEBOOK_ID = '{nb.id}';"),
                Script(f"document.addEventListener('DOMContentLoaded', () => connectWebSocket('{nb.id}'));"),
                Div(id="ephemeral"),  # Container for dialoghelper script injection (matches add_scr default)
                cls="container"
            ),
            cls="main-layout"
        ),
        # Settings sidebar and overlay (outside main layout - overlay style)
        SettingsOverlay() if config else None,
        SettingsSidebar(config) if config else None
    )
