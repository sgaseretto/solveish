"""
Dialeng UI - Prompt Cell Component

Renders prompt cells with user input and AI response sections.
"""

from fasthtml.common import *
from ..base import get_collapse_class, get_cell_state_classes
from .base import CellHeader


def _execution_diagnostics(cell):
    execution = getattr(cell, "llm_execution", None) or {}
    if not execution:
        return Details(
            id=f"prompt-diagnostics-{cell.id}",
            cls="prompt-diagnostics",
            hidden=True,
        )
    usage = execution.get("usage") or {}
    total = int(usage.get("total_tokens") or 0)
    duration = execution.get("duration_ms")
    summary = " · ".join(filter(None, [
        execution.get("model"),
        f"{duration / 1000:.1f}s" if isinstance(duration, (int, float)) else None,
        f"{total:,} tokens" if total else None,
        execution.get("status"),
    ]))
    detail = [
        ("Operation", execution.get("operation_id")),
        ("Provider", execution.get("provider")),
        ("Transport", execution.get("transport")),
        ("First token", f"{execution['ttft_ms']:.1f} ms" if isinstance(execution.get("ttft_ms"), (int, float)) else None),
        ("Retries", execution.get("retry_count")),
        ("Input / cached / output", f"{usage.get('input_tokens', 0):,} / {usage.get('cached_tokens', 0):,} / {usage.get('output_tokens', 0):,}"),
        ("Tool steps", execution.get("tool_steps")),
    ]
    return Details(
        Summary(summary or "Execution details"),
        Div(*[
            Div(Span(label), Code(str(value)), cls="prompt-diagnostic-row")
            for label, value in detail if value is not None
        ], cls="prompt-diagnostic-grid"),
        id=f"prompt-diagnostics-{cell.id}",
        cls="prompt-diagnostics",
        data_operation_id=execution.get("operation_id") or "",
    )


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
        Input(type="hidden", id=f"version-{cell.id}", name="expected_version", value=str(cell.version)),
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
                         style="display: none; min-height: 80px;"),
                Div(cls="ai-preview", data_cell_id=cell.id, data_field="output"),
                cls=f"prompt-output {output_collapse_cls}".strip(),
                data_collapse_section="output"
            ),
            cls="prompt-section"
        ),
        _execution_diagnostics(cell),
        cls="cell-body"
    )

    return Div(header, body, id=f"cell-{cell.id}", cls=get_cell_state_classes(cell), data_type=cell.cell_type)


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
                     hx_post=f"/dialeng/{notebook_id}/cell/{cell.id}/source",
                     hx_include=f"#source-{cell.id}, #version-{cell.id}",
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
                     hx_post=f"/dialeng/{notebook_id}/cell/{cell.id}/source",
                     hx_include=f"#source-{cell.id}, #version-{cell.id}",
                     hx_trigger="blur changed", hx_swap="none",
                     oninput=f"document.getElementById('source-{cell.id}').value = this.value"),
            cls=f"prompt-input {input_collapse_cls}".strip(),
            data_collapse_section="input"
        ),
        cls="prompt-section"
    )
