"""
Dialeng UI - Prompt Cell Component

Renders prompt cells with user input and AI response sections.
"""

from fasthtml.common import *
from ..base import get_collapse_class
from .base import CellHeader


def PromptCellView(cell, notebook_id: str):
    """Render a prompt cell with user input and AI response.

    Args:
        cell: Cell dataclass instance with cell_type="prompt"
        notebook_id: Parent notebook ID

    Returns:
        Complete prompt cell Div with header, prompt input, and AI output
    """
    has_output = bool(cell.output and cell.output.strip())
    input_collapse_cls = get_collapse_class(cell.input_collapse)
    output_collapse_cls = get_collapse_class(cell.output_collapse)

    header = CellHeader(cell, notebook_id)

    # User prompt section - show preview if has output, else show textarea
    if has_output:
        user_prompt_section = _user_prompt_section_preview(
            cell, notebook_id, input_collapse_cls
        )
    else:
        user_prompt_section = _user_prompt_section_editable(
            cell, notebook_id, input_collapse_cls
        )

    body = Div(
        # Hidden source field for hx-vals
        Input(type="hidden", id=f"source-{cell.id}", name="source", value=cell.source),
        user_prompt_section,
        # AI response section with preview/edit toggle
        Div(
            Div(Span("🤖"), " AI Response ",
                Span("(double-click to edit)", style="font-weight: normal; opacity: 0.6;"),
                cls="prompt-label ai"),
            Div(
                Textarea(cell.output if cell.output else "",
                         cls="prompt-content", name="output", id=f"output-{cell.id}",
                         placeholder="Click Run to generate response...",
                         hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/output",
                         hx_trigger="blur changed", hx_swap="none",
                         style="display: none; min-height: 80px;",
                         onblur=f"switchToPreview('{cell.id}', 'output')"),
                Div(cls="ai-preview", data_cell_id=cell.id, data_field="output"),
                cls=f"prompt-output {output_collapse_cls}".strip(),
                data_collapse_section="output"
            ),
            cls="prompt-section"
        ),
        cls="cell-body"
    )

    collapsed_cls = " collapsed" if cell.collapsed else ""
    return Div(header, body, id=f"cell-{cell.id}", cls=f"cell{collapsed_cls}", data_type=cell.cell_type)


def _user_prompt_section_preview(cell, notebook_id: str, input_collapse_cls: str):
    """User prompt section after run (markdown preview mode)."""
    return Div(
        Div(Span("👤"), " Your Prompt ",
            Span("(double-click to edit)", style="font-weight: normal; opacity: 0.6;"),
            cls="prompt-label user"),
        Div(
            Textarea(cell.source, cls="prompt-content", name="prompt_source",
                     id=f"prompt-{cell.id}",
                     placeholder="Ask the AI anything...",
                     hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                     hx_include=f"#source-{cell.id}",
                     hx_trigger="blur changed", hx_swap="none",
                     style="display: none;",
                     oninput=f"document.getElementById('source-{cell.id}').value = this.value",
                     onblur=f"switchToPreview('{cell.id}', 'prompt')"),
            Div(cls="md-preview prompt-preview", data_cell_id=cell.id, data_field="prompt"),
            cls=f"prompt-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        cls="prompt-section"
    )


def _user_prompt_section_editable(cell, notebook_id: str, input_collapse_cls: str):
    """User prompt section before run (direct edit mode)."""
    return Div(
        Div(Span("👤"), " Your Prompt", cls="prompt-label user"),
        Div(
            Textarea(cell.source, cls="prompt-content", name="prompt_source",
                     id=f"prompt-{cell.id}",
                     placeholder="Ask the AI anything...",
                     hx_post=f"/notebook/{notebook_id}/cell/{cell.id}/source",
                     hx_include=f"#source-{cell.id}",
                     hx_trigger="blur changed", hx_swap="none",
                     oninput=f"document.getElementById('source-{cell.id}').value = this.value"),
            cls=f"prompt-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        cls="prompt-section"
    )
