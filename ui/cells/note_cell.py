"""
Dialeng UI - Note Cell Component

Renders markdown note cells with preview/edit toggle.
"""

from fasthtml.common import *
from ..base import get_collapse_class
from .base import CellHeader


def NoteCellView(cell, notebook_id: str):
    """Render a note cell with markdown preview.

    Args:
        cell: Cell dataclass instance with cell_type="note"
        notebook_id: Parent notebook ID

    Returns:
        Complete note cell Div with header and markdown preview
    """
    input_collapse_cls = get_collapse_class(cell.input_collapse)

    header = CellHeader(cell, notebook_id)

    body = Div(
        # Hidden textarea for editing (shown on double-click)
        Textarea(cell.source, cls="source", name="source", id=f"source-{cell.id}",
                 placeholder="# Markdown notes...",
                 hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                 hx_trigger="blur changed", hx_swap="none",
                 style="display: none;",
                 onblur=f"switchToPreview('{cell.id}', 'source')"),
        # Markdown preview (double-click to edit)
        Div(
            Div(id=f"preview-{cell.id}", cls="md-preview",
                data_cell_id=cell.id, data_field="source"),
            cls=f"cell-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        Div("Double-click to edit | Escape to finish | Z to cycle collapse", cls="edit-hint"),
        cls="cell-body"
    )

    collapsed_cls = " collapsed" if cell.collapsed else ""
    return Div(header, body, id=f"cell-{cell.id}", cls=f"cell{collapsed_cls}", data_type=cell.cell_type)
