"""
Dialeng UI - OOB (Out-of-Band) Components

Components with hx-swap-oob for WebSocket broadcasting.
HTMX will automatically swap these elements by ID when received via WebSocket.
"""

from fasthtml.common import *
from .cells import CellView
from .controls import AddButtons


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
