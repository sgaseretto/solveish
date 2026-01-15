"""
Dialeng UI - Code Cell Component

Renders code cells with Ace editor and output display.
"""

from fasthtml.common import *
from ..base import get_collapse_class
from .base import CellHeader


def CodeCellView(cell, notebook_id: str):
    """Render a code cell with Ace editor and output.

    Args:
        cell: Cell dataclass instance with cell_type="code"
        notebook_id: Parent notebook ID

    Returns:
        Complete code cell Div with header, editor, and output
    """
    input_collapse_cls = get_collapse_class(cell.input_collapse)
    output_collapse_cls = get_collapse_class(cell.output_collapse)

    header = CellHeader(cell, notebook_id)

    body = Div(
        # Hidden textarea for form submission - Ace reads from this
        Textarea(cell.source, name="source", id=f"source-{cell.id}",
                 style="display: none;",
                 hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                 hx_trigger="blur changed", hx_swap="none"),
        # Ace Editor container - with collapse support
        Div(
            Div(id=f"ace-{cell.id}", cls="ace-container"),
            cls=f"cell-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        # Output section
        Div(
            Pre(NotStr(cell.output), cls="stream-output") if cell.output else "",
            id=f"output-{cell.id}",
            cls=f"cell-output{' error' if cell.output and ('Error' in cell.output or 'Traceback' in cell.output) else ''} {output_collapse_cls}".strip(),
            data_collapse_section="output"
        ),
        # Initialize Ace editor
        Script(f"setTimeout(() => initAceEditor('{cell.id}'), 0);"),
        cls="cell-body"
    )

    collapsed_cls = " collapsed" if cell.collapsed else ""
    return Div(header, body, id=f"cell-{cell.id}", cls=f"cell{collapsed_cls}", data_type=cell.cell_type)
