"""
Dialeng UI - Code Cell Component

Renders code cells with Monaco editor and output display.
"""

from fasthtml.common import *
from ..base import get_collapse_class, get_cell_state_classes
from .base import CellHeader


def _process_carriage_returns(text: str) -> str:
    """Process \\r (carriage return) to collapse progress bar updates.

    tqdm and similar tools use \\r to overwrite the current line.
    This mimics terminal behavior: \\r moves cursor to start of line,
    subsequent text overwrites from there.
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        if '\r' in line:
            # Process \r: each \r resets to start of line
            parts = line.split('\r')
            # The last non-empty part is what's visible
            final = parts[-1]
            if not final and len(parts) > 1:
                final = parts[-2]
            result.append(final)
        else:
            result.append(line)
    return '\n'.join(result)


def _render_cell_outputs(cell) -> tuple:
    """Render all cell outputs (stream + display_data) for static and OOB rendering.

    Returns:
        (output_elements, has_error): list of FT elements and whether any errors exist
    """
    from dialeng.ui.mime import render_mime_bundle, ansi_to_html

    elements = []
    has_error = False
    stream_parts = []  # raw text, before ansi conversion

    for out in cell.outputs:
        if out.output_type in ('stream', 'execute_result'):
            stream_parts.append(str(out.content))
        elif out.output_type == 'error':
            has_error = True
            if out.traceback:
                stream_parts.extend(out.traceback)
            else:
                stream_parts.append(f"{out.ename}: {out.evalue}")
        elif out.output_type == 'display_data':
            # Flush any accumulated stream text before display_data
            if stream_parts:
                text = _process_carriage_returns(''.join(stream_parts))
                elements.append(Pre(NotStr(ansi_to_html(text)), cls="stream-output"))
                stream_parts = []
            # Render the MIME bundle to HTML
            html = render_mime_bundle(
                out.content if isinstance(out.content, dict) else {'text/plain': str(out.content)},
                out.metadata
            )
            elements.append(Div(NotStr(html), cls="display-data"))

    # Flush remaining stream text
    if stream_parts:
        text = _process_carriage_returns(''.join(stream_parts))
        has_error = has_error or ('Error' in text or 'Traceback' in text)
        elements.append(Pre(NotStr(ansi_to_html(text)), cls="stream-output"))

    return elements, has_error


def CodeCellView(cell, notebook_id: str):
    """Render a code cell with Monaco editor and output.

    Args:
        cell: Cell dataclass instance with cell_type="code"
        notebook_id: Parent notebook ID

    Returns:
        Complete code cell Div with header, editor, and output
    """
    input_collapse_cls = get_collapse_class(cell.input_collapse)
    output_collapse_cls = get_collapse_class(cell.output_collapse)

    header = CellHeader(cell, notebook_id)

    output_elements, has_error = _render_cell_outputs(cell)

    body = Div(
        # Hidden textarea for form submission - Monaco reads from this
        Textarea(cell.source, name="source", id=f"source-{cell.id}",
                 style="display: none;",
                 hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                 hx_trigger="blur changed", hx_swap="none"),
        # Monaco Editor container - with collapse support
        Div(
            Div(id=f"monaco-{cell.id}", cls="monaco-container"),
            cls=f"cell-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        # Output section
        Div(
            *output_elements,
            id=f"output-{cell.id}",
            cls=f"cell-output{' error' if has_error else ''} {output_collapse_cls}".strip(),
            data_collapse_section="output"
        ),
        cls="cell-body"
    )

    return Div(header, body, id=f"cell-{cell.id}", cls=get_cell_state_classes(cell), data_type=cell.cell_type)
