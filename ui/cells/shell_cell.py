"""
Dialeng UI - Shell Cell Component

Renders shell cells with Ace editor (bash mode) and output display.
Shell cells execute bash commands via pshnb with optional safecmd validation.
"""

from fasthtml.common import *
from ..base import get_collapse_class, get_cell_state_classes
from ..controls import TypeSelect, CollapseBtn
from ..icons import sprites as ss


def ShellCellHeader(cell, notebook_id: str, safe_mode: bool = False):
    """Shell cell header with safe mode indicator.

    Args:
        cell: Cell dataclass instance with cell_type="shell"
        notebook_id: Parent notebook ID
        safe_mode: Whether safe mode is enabled for this notebook

    Returns:
        Div containing the cell header with badge, meta, and actions
    """
    meta_info = []
    if cell.execution_count:
        meta_info.append(Span(f"[{cell.execution_count}]"))
    if cell.time_run:
        meta_info.append(Span(cell.time_run))

    # Safe mode indicator
    if safe_mode:
        meta_info.append(Span("Safe", cls="safe-mode-badge", title="Safe mode enabled - commands validated"))

    collapse_controls = [
        Span("In:", cls="label"),
        CollapseBtn(cell.id, "input", cell.input_collapse),
        Span("Out:", cls="label"),
        CollapseBtn(cell.id, "output", cell.output_collapse),
    ]

    # Run button with bash-specific handling
    run_onclick = f"syncAceToTextarea('{cell.id}'); prepareCodeRun('{cell.id}');"
    run_button = Button(
        "▶", cls="btn btn-sm btn-run",
        hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/run",
        hx_target=f"#cell-{cell.id}",
        hx_swap="none",
        hx_vals=f"js:{{source: document.getElementById('source-{cell.id}')?.value || ''}}",
        hx_timeout="120s",
        onclick=run_onclick,
        title="Run (Shift+Enter)"
    )

    # Cancel button
    cancel_onclick = f"interruptCodeCell('{notebook_id}', '{cell.id}')"
    cancel_button = Button(
        "Stop", cls="btn btn-sm btn-cancel",
        onclick=cancel_onclick,
        title="Interrupt execution (Ctrl+C)",
        style="display: none;"
    )

    # State indicators (visible badges when state is active)
    state_indicators = []
    if cell.skipped:
        state_indicators.append(Span("HIDDEN", cls="state-indicator skipped-indicator"))
    if cell.pinned:
        state_indicators.append(Span("PINNED", cls="state-indicator pinned-indicator"))
    if cell.is_exported:
        state_indicators.append(Span("EXPORT", cls="state-indicator exported-indicator"))

    return Div(
        Div(
            Button("V", cls="collapse-btn",
                   onclick=f"toggleCollapse('{cell.id}')",
                   title="Collapse/Expand (full)"),
            Span("SHELL", cls="cell-badge shell"),
            *state_indicators,
            Span(*meta_info, cls="cell-meta") if meta_info else None,
            Div(*collapse_controls, cls="collapse-controls") if collapse_controls else None,
        ),
        Div(
            Button(ss('eye-closed' if cell.skipped else 'eye'),
                   cls=f"btn btn-sm btn-icon state-toggle{' active' if cell.skipped else ''}",
                   onclick=f"toggleCellState('{cell.id}', 'skipped')",
                   title="Toggle AI visibility (h)"),
            Button(ss('pin' if cell.pinned else 'pin-off'),
                   cls=f"btn btn-sm btn-icon state-toggle{' active' if cell.pinned else ''}",
                   onclick=f"toggleCellState('{cell.id}', 'pinned')",
                   title="Toggle pin (p)"),
            Button(ss('bookmark-check' if cell.is_exported else 'bookmark'),
                   cls=f"btn btn-sm btn-icon state-toggle{' active' if cell.is_exported else ''}",
                   onclick=f"toggleCellState('{cell.id}', 'is_exported')",
                   title="Toggle export (e)"),
            TypeSelect(cell.id, cell.cell_type, notebook_id),
            run_button,
            cancel_button,
            Button("Up", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/up",
                   hx_target="#cells", hx_swap="outerHTML", title="Move up"),
            Button("Down", cls="btn btn-sm btn-icon",
                   hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/move/down",
                   hx_target="#cells", hx_swap="outerHTML", title="Move down"),
            Button("X", cls="btn btn-sm btn-icon",
                   hx_delete=f"/notebook/{notebook_id}/cell/{cell.id}",
                   hx_target="#cells",
                   hx_swap="outerHTML", title="Delete (D D)"),
            cls="cell-actions"
        ),
        cls="cell-header"
    )


def ShellCellView(cell, notebook_id: str, safe_mode: bool = False):
    """Render a shell cell with Ace editor (bash mode) and output.

    Args:
        cell: Cell dataclass instance with cell_type="shell"
        notebook_id: Parent notebook ID
        safe_mode: Whether safe mode is enabled for this notebook

    Returns:
        Complete shell cell Div with header, editor, and output
    """
    input_collapse_cls = get_collapse_class(cell.input_collapse)
    output_collapse_cls = get_collapse_class(cell.output_collapse)

    header = ShellCellHeader(cell, notebook_id, safe_mode)

    body = Div(
        # Hidden textarea for form submission - Ace reads from this
        Textarea(cell.source, name="source", id=f"source-{cell.id}",
                 style="display: none;",
                 hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                 hx_trigger="blur changed", hx_swap="none"),
        # Ace Editor container - with collapse support
        # Uses bash mode for shell syntax highlighting
        Div(
            Div(id=f"ace-{cell.id}", cls="ace-container"),
            cls=f"cell-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        # Output section
        Div(
            Pre(NotStr(cell.output), cls="stream-output") if cell.output else "",
            id=f"output-{cell.id}",
            cls=f"cell-output{' error' if cell.output and ('Error' in cell.output or 'Traceback' in cell.output or 'DisallowedCmd' in cell.output) else ''} {output_collapse_cls}".strip(),
            data_collapse_section="output"
        ),
        # Initialize Ace editor with bash mode
        Script(f"setTimeout(() => initAceEditor('{cell.id}', 'sh'), 0);"),
        cls="cell-body"
    )

    return Div(header, body, id=f"cell-{cell.id}", cls=get_cell_state_classes(cell), data_type="shell")
