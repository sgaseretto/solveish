"""
Dialeng UI - Control Components

Button, select, and interactive control components.
"""

from fasthtml.common import *


def TypeSelect(cell_id: str, current: str, nb_id: str):
    """Cell type selector dropdown.

    Args:
        cell_id: The cell's unique ID
        current: Current cell type ("code", "note", or "prompt")
        nb_id: Parent notebook ID

    Returns:
        Select element that changes cell type on selection
    """
    return Select(
        Option("code", value="code", selected=current == "code"),
        Option("note", value="note", selected=current == "note"),
        Option("prompt", value="prompt", selected=current == "prompt"),
        cls="type-select",
        name="cell_type",
        hx_post=f"/notebook/{nb_id}/cell/{cell_id}/type",
        hx_target=f"#cell-{cell_id}",
        hx_swap="outerHTML"
    )


def CollapseBtn(cell_id: str, section: str, level: int) -> Button:
    """Create a collapse button for input/output section.

    Args:
        cell_id: The cell's unique ID
        section: "input" or "output"
        level: Current collapse level (0=expanded, 1=scrollable, 2=summary)

    Returns:
        Button that cycles through collapse levels on click
    """
    labels = {0: "▼", 1: "◐", 2: "▬"}
    tooltips = {0: "Expanded", 1: "Scrollable", 2: "Summary"}
    return Button(
        labels.get(level, "▼"),
        cls="section-collapse-btn",
        data_collapse_btn=section,
        data_level=str(level),
        onclick=f"cycleCollapseLevel('{cell_id}', '{section}')",
        title=f"{section.capitalize()}: {tooltips.get(level, 'Expanded')} (click to cycle)"
    )


def AddButtons(pos: int, nb_id: str):
    """Buttons for adding new cells at a position.

    Args:
        pos: Position index for the new cell
        nb_id: Parent notebook ID

    Returns:
        Div containing add cell buttons
    """
    return Div(
        Button("+ Code", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=code",
               hx_target="#cells", hx_swap="outerHTML"),
        Button("+ Note", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=note",
               hx_target="#cells", hx_swap="outerHTML"),
        Button("+ Prompt", cls="btn btn-sm",
               hx_post=f"/notebook/{nb_id}/cell/add?pos={pos}&type=prompt",
               hx_target="#cells", hx_swap="outerHTML"),
        cls="add-row"
    )
