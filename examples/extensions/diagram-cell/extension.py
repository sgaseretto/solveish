from fasthtml.common import Div, Input, P, Pre, Textarea
from dataclasses import dataclass

from dialeng.core.registry import CellTypeRegistration
from dialeng.document.cell import Cell
from dialeng.ui.cells.base import CellHeader


@dataclass
class DiagramCell(Cell):
    cell_type: str = "diagram"


def render_diagram(cell, notebook_id):
    return Div(
        CellHeader(cell, notebook_id, collapse_controls=[]),
        Div(
            P("Diagram source", cls="diagram-label"),
            Textarea(
                cell.source,
                id=f"source-{cell.id}",
                name="source",
                cls="diagram-source",
                hx_post=f"/dialeng/{notebook_id}/cell/{cell.id}/source",
                hx_trigger="blur",
                hx_include=f"#version-{cell.id}",
                hx_swap="none",
            ),
            Pre(cell.source, id=f"diagram-preview-{cell.id}", cls="diagram-preview"),
            Input(type="hidden", id=f"version-{cell.id}", name="expected_version", value=str(cell.version)),
            cls="cell-body diagram-cell-body",
        ),
        id=f"cell-{cell.id}",
        cls="cell diagram-cell",
        data_cell_id=cell.id,
        data_cell_type="diagram",
    )


def serialize_diagram(cell):
    metadata = dict(getattr(cell, "raw_metadata", {}) or {})
    metadata.update({
        "id": cell.id,
        "dialeng_cell_type": "diagram",
        "collapsed": bool(cell.collapsed),
        "input_collapse": int(cell.input_collapse),
        "output_collapse": int(cell.output_collapse),
    })
    return {"cell_type": "raw", "source": cell.source, "metadata": metadata}


def deserialize_diagram(raw):
    metadata = raw.get("metadata", {})
    if metadata.get("dialeng_cell_type") != "diagram":
        return None
    source = raw.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    return DiagramCell(
        id=metadata.get("id") or raw.get("id") or "diagram",
        source=source,
        raw_metadata=dict(metadata),
        raw_jupyter_cell_type=raw.get("cell_type", "raw"),
    )


def diagram_to_messages(cell):
    return [{"role": "user", "content": f"[Diagram]\n{cell.source}"}]


def setup(api, ctx):
    api.register_cell_type(
        CellTypeRegistration(DiagramCell, "diagram", "workflow", "Diagram"),
        renderer=render_diagram,
        serializer=serialize_diagram,
        deserializer=deserialize_diagram,
        llm_converter=diagram_to_messages,
    )

    async def convert_cell(command_context):
        cell_id = str(command_context.get("args", {}).get("cell_id", ""))
        if not cell_id:
            raise ValueError("cell_id is required")
        return await ctx.change_cell_type(command_context["notebook_id"], cell_id, "diagram")

    api.register_command(
        "convert-to-diagram",
        title="Convert to Diagram",
        description="Convert one selected cell to a Diagram cell.",
        args_schema={
            "type": "object",
            "required": ["cell_id"],
            "properties": {"cell_id": {"type": "string"}},
        },
        handler=convert_cell,
    )
    api.register_asset("diagram-style", kind="css", path="assets/diagram.css")
    api.register_asset("diagram-client", kind="js", path="assets/diagram.js")
