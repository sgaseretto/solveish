"""
Dialeng UI - Cell Base Components

Contains CellView dispatcher and shared CellHeader component.
"""

from fasthtml.common import *
from ..controls import TypeSelect, CollapseBtn


def CellHeader(cell, notebook_id: str, collapse_controls: list = None):
    """Shared header component for all cell types.

    Args:
        cell: Cell dataclass instance
        notebook_id: Parent notebook ID
        collapse_controls: Optional list of collapse control elements

    Returns:
        Div containing the cell header with badge, meta, and actions
    """
    meta_info = []
    if cell.execution_count:
        meta_info.append(Span(f"[{cell.execution_count}]"))
    if cell.time_run:
        meta_info.append(Span(cell.time_run))

    # Build collapse controls based on cell type if not provided
    if collapse_controls is None:
        collapse_controls = []
        if cell.cell_type == "code":
            collapse_controls = [
                Span("In:", cls="label"),
                CollapseBtn(cell.id, "input", cell.input_collapse),
                Span("Out:", cls="label"),
                CollapseBtn(cell.id, "output", cell.output_collapse),
            ]
        elif cell.cell_type == "note":
            collapse_controls = [
                CollapseBtn(cell.id, "input", cell.input_collapse),
            ]
        else:  # prompt
            collapse_controls = [
                Span("In:", cls="label"),
                CollapseBtn(cell.id, "input", cell.input_collapse),
                Span("Out:", cls="label"),
                CollapseBtn(cell.id, "output", cell.output_collapse),
            ]

    return Div(
        Div(
            Button("▼", cls="collapse-btn",
                   onclick=f"toggleCollapse('{cell.id}')",
                   title="Collapse/Expand (full)"),
            Span(cell.cell_type.upper(), cls=f"cell-badge {cell.cell_type}"),
            Span(*meta_info, cls="cell-meta") if meta_info else None,
            Div(*collapse_controls, cls="collapse-controls") if collapse_controls else None,
        ),
        Div(
            TypeSelect(cell.id, cell.cell_type, notebook_id),
            _run_button(cell, notebook_id),
            _cancel_button(cell, notebook_id),
            Button("↑", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/up",
                   hx_target="#cells", hx_swap="outerHTML", title="Move up"),
            Button("↓", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/down",
                   hx_target="#cells", hx_swap="outerHTML", title="Move down"),
            Button("×", cls="btn btn-sm btn-icon",
                   hx_delete=f"/notebook/{notebook_id}/cell/{cell.id}",
                   hx_target="#cells",
                   hx_swap="outerHTML", title="Delete (D D)"),
            cls="cell-actions"
        ),
        cls="cell-header"
    )


def _run_button(cell, notebook_id: str):
    """Create run button based on cell type."""
    if cell.cell_type == "note":
        return None

    if cell.cell_type == "prompt":
        onclick = (f"syncAceToTextarea('{cell.id}'); "
                   f"syncPromptContent('{cell.id}'); "
                   f"startStreaming('{cell.id}', {str(cell.use_thinking).lower()});")
    else:
        onclick = f"syncAceToTextarea('{cell.id}'); prepareCodeRun('{cell.id}');"

    return Button("▶", cls="btn btn-sm btn-run",
                  hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/run",
                  hx_target=f"#cell-{cell.id}",
                  hx_swap="none" if cell.cell_type == "code" else "outerHTML",
                  hx_vals=f"js:{{source: document.getElementById('source-{cell.id}')?.value || ''}}",
                  hx_timeout="120s",
                  onclick=onclick,
                  title="Run (Shift+Enter)")


def _cancel_button(cell, notebook_id: str):
    """Create cancel button based on cell type."""
    if cell.cell_type not in ("prompt", "code"):
        return None

    if cell.cell_type == "prompt":
        onclick = f"cancelStreaming('{cell.id}')"
        title = "Cancel generation"
    else:
        onclick = f"interruptCodeCell('{notebook_id}', '{cell.id}')"
        title = "Interrupt execution (Ctrl+C)"

    return Button("⏹", cls="btn btn-sm btn-cancel",
                  onclick=onclick,
                  title=title,
                  style="display: none;")


def CellView(cell, notebook_id: str):
    """Dispatch to appropriate cell view based on cell type.

    Args:
        cell: Cell dataclass instance
        notebook_id: Parent notebook ID

    Returns:
        Complete cell Div with header and body
    """
    from .code_cell import CodeCellView
    from .note_cell import NoteCellView
    from .prompt_cell import PromptCellView

    views = {
        "code": CodeCellView,
        "note": NoteCellView,
        "prompt": PromptCellView,
    }

    view_func = views.get(cell.cell_type, CodeCellView)
    return view_func(cell, notebook_id)
