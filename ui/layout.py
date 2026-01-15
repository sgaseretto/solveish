"""
Dialeng UI - Layout Components

Page layout and container components.
"""

from fasthtml.common import *
from typing import List
from .cells import CellView
from .controls import AddButtons


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


def NotebookPage(nb, notebook_list: List[str], available_dialog_modes: list, available_models: list):
    """Render the complete notebook page.

    Args:
        nb: Notebook instance
        notebook_list: List of notebook IDs for the file list
        available_dialog_modes: List of (mode_id, label) tuples
        available_models: List of (model_id, label) tuples

    Returns:
        Complete page with Titled wrapper
    """
    return Titled(
        f"{nb.title} - Dialeng",
        Div(
            Div(
                Div(Span("📓", cls="title-icon"), Span(nb.title, cls="title")),
                Div(
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
                        *[Option(label, value=model_id, selected=nb.model == model_id)
                          for model_id, label in available_models],
                        cls="model-select", name="model", id="model-select",
                        hx_post=f"/notebook/{nb.id}/model", hx_swap="none", title="Model",
                        style="display: none;" if nb.dialog_mode == "mock" else ""
                    ),
                    Button("🔄 Restart", cls="btn btn-sm",
                           hx_post=f"/notebook/{nb.id}/kernel/restart", hx_target="#status", title="Restart kernel"),
                    Button("⏹ Cancel All", cls="btn btn-sm btn-cancel-all", id="cancel-all-btn",
                           onclick="cancelAllExecution()", title="Cancel running cell and clear queue (Esc Esc)",
                           style="display: none;"),
                    Button("💾 Save", cls="btn btn-sm btn-save", id="save-btn",
                           hx_post=f"/notebook/{nb.id}/save", hx_target="#status", title="Save (Ctrl+S)"),
                    Button("📥 Export", cls="btn btn-sm",
                           hx_get=f"/notebook/{nb.id}/export", title="Download .ipynb"),
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
            Div(id="js-script"),  # Container for dialoghelper script injection
            cls="container"
        )
    )
