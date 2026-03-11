"""
Dialeng UI - OOB (Out-of-Band) Components

Components with hx-swap-oob for WebSocket broadcasting.
HTMX will automatically swap these elements by ID when received via WebSocket.
"""

from fasthtml.common import *
from .cells import CellView
from .cells.base import CellHeader
from .controls import AddButtons
from .base import get_collapse_class, get_cell_state_classes


def AllCellsOOB(nb):
    """Returns AllCells with hx-swap-oob for WebSocket broadcasting.

    Args:
        nb: Notebook instance

    Returns:
        Div with id="cells" and hx-swap-oob="true" for automatic OOB swapping
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i + 1, nb.id)])
    return Div(*items, id="cells", hx_swap_oob="true")


def CellViewOOB(cell, notebook_id: str):
    """Returns CellView with hx-swap-oob for WebSocket broadcasting.

    Args:
        cell: Cell dataclass instance
        notebook_id: Parent notebook ID

    Returns:
        Cell Div with hx-swap-oob="true" for automatic OOB swapping
    """
    cell_div = CellView(cell, notebook_id)
    # Recreate with OOB attribute since CellView returns a complete Div
    return Div(
        *cell_div.children,
        id=f"cell-{cell.id}",
        cls=cell_div.attrs.get('class', ''),
        hx_swap_oob="true",
        **{k: v for k, v in cell_div.attrs.items() if k not in ('id', 'class')}
    )


def CellOutputOOB(cell):
    """OOB swap for just the output section of a code/shell cell.

    Only replaces the output div — the Monaco editor DOM is untouched.
    This eliminates FOUST (Flash of Unstyled Text) after execution.
    """
    output_collapse_cls = get_collapse_class(cell.output_collapse)
    has_error = cell.output and ('Error' in cell.output or 'Traceback' in cell.output)
    return Div(
        Pre(NotStr(cell.output), cls="stream-output") if cell.output else "",
        id=f"output-{cell.id}",
        cls=f"cell-output{' error' if has_error else ''} {output_collapse_cls}".strip(),
        data_collapse_section="output",
        hx_swap_oob="true"
    )


def CellHeaderOOB(cell, notebook_id: str):
    """OOB swap for just the cell header (execution count, time, state badges).

    Only replaces the header div — the Monaco editor DOM is untouched.
    """
    header = CellHeader(cell, notebook_id)
    return Div(
        *header.children,
        id=f"header-{cell.id}",
        cls="cell-header",
        hx_swap_oob="true"
    )


def CellClassOOB(cell):
    """OOB swap to update the cell wrapper's CSS classes (skipped, pinned, etc).

    Uses hx-swap-oob="outerHTML" with the full cell state classes.
    Note: This replaces the entire cell div including body, so it should only
    be used when the full cell needs re-rendering. For class-only updates,
    use a JSON WebSocket message instead.
    """
    cls = get_cell_state_classes(cell)
    # We can't do class-only OOB swaps in HTMX without replacing content.
    # Instead, we send a JSON message from the server to update classes client-side.
    # This function is kept for reference but prefer broadcast_json for class updates.
    raise NotImplementedError("Use broadcast_json with cell_class_update type instead")
