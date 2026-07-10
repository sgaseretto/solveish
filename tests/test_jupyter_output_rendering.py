"""Jupyter-compatible output capture, rendering, and persistence tests."""

import asyncio

from dialeng.document.cell import Cell, CellOutput, normalize_cell_outputs
from dialeng.document.serialization import _cell_to_jupyter, _jupyter_to_cell
from dialeng.services.kernel.kernel_worker import (
    StreamingDisplayPublisher,
    _wire_safe_mime_bundle,
)
from dialeng.services.kernel.subprocess_kernel import SubprocessKernel
from dialeng.ui.cells.code_cell import _render_cell_outputs
from dialeng.ui.mime import render_mime_bundle


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_execute_result_mime_bundle_round_trips_without_flattening():
    bundle = {
        "text/plain": ["fallback"],
        "text/html": ["<strong>rich</strong>"],
        "application/json": {"value": 42},
    }
    cell = Cell(
        id="result",
        cell_type="code",
        execution_count=7,
        outputs=[CellOutput(
            output_type="execute_result",
            content=bundle,
            metadata={"text/html": {"isolated": False}},
        )],
    )

    serialized = _cell_to_jupyter(cell)
    restored = _jupyter_to_cell(serialized)

    assert serialized["outputs"][0]["output_type"] == "execute_result"
    assert serialized["outputs"][0]["data"] == bundle
    assert restored.outputs[0].content == bundle
    assert restored.output == "fallback"


def test_static_renderer_preserves_stream_order_and_error_identity():
    cell = Cell(
        id="ordered",
        cell_type="code",
        outputs=[
            CellOutput(output_type="stream", content="out\n", stream_name="stdout"),
            CellOutput(output_type="stream", content="warning\n", stream_name="stderr"),
            CellOutput(
                output_type="execute_result",
                content={"text/html": "<b>result</b>", "text/plain": "result"},
            ),
            CellOutput(output_type="stream", content="tail\n", stream_name="stdout"),
        ],
    )

    elements, has_error = _render_cell_outputs(cell)
    classes = [element.attrs.get("class", "") for element in elements]

    assert has_error is False
    assert classes == [
        "output-item stream-output stdout",
        "output-item stream-output stderr",
        "output-item display-data execute-result",
        "output-item stream-output stdout",
    ]


def test_traceback_words_in_stdout_do_not_mark_cell_as_error():
    cell = Cell(
        id="not-error",
        cell_type="code",
        outputs=[CellOutput(
            output_type="stream",
            content="The documentation says Traceback and ValueError.",
            stream_name="stdout",
        )],
    )

    _elements, has_error = _render_cell_outputs(cell)

    assert has_error is False


def test_mime_renderer_supports_line_arrays_nested_image_metadata_and_widgets():
    html = render_mime_bundle({"text/html": ["<b>", "ok", "</b>"]})
    image = render_mime_bundle(
        {"image/png": ["YWJj\n"]},
        {"image/png": {"width": 320, "height": "50%", "alt": "plot"}},
    )
    widget = render_mime_bundle({
        "application/vnd.jupyter.widget-view+json": {"model_id": "abc"},
        "text/plain": "Widget",
    })

    assert "<b>ok</b>" in html
    assert 'width:320px' in image
    assert 'height:50%' in image
    assert 'alt="plot"' in image
    assert "widget manager" in widget


def test_formatter_error_is_suppressed_for_rich_execute_result():
    outputs = [
        CellOutput(
            output_type="execute_result",
            content={"image/png": "abc", "text/plain": "image"},
        ),
        CellOutput(
            output_type="error",
            ename="TypeError",
            evalue="__repr__ returned non-string (type NoneType)",
            traceback=[
                "IPython/core/formatters.py in __call__",
                "IPython/lib/pretty.py in pretty",
            ],
        ),
    ]

    normalized = normalize_cell_outputs(outputs)

    assert [output.output_type for output in normalized] == ["execute_result"]


def test_clear_output_supports_immediate_wait_and_legacy_dict_payloads():
    old = CellOutput(output_type="stream", content="old", stream_name="stdout")

    assert normalize_cell_outputs([
        old,
        CellOutput(output_type="clear_output", content={"wait": False}),
    ]) == []
    assert normalize_cell_outputs([
        old,
        CellOutput(output_type="clear_output", content=True),
    ]) == [old]
    replacement = normalize_cell_outputs([
        old,
        CellOutput(output_type="clear_output", content=True),
        CellOutput(output_type="stream", content="new", stream_name="stdout"),
    ])
    assert [output.content for output in replacement] == ["new"]


def test_display_update_replaces_every_output_with_shared_display_id():
    normalized = normalize_cell_outputs([
        CellOutput(
            output_type="display_data",
            content={"text/plain": "first-a"},
            display_id="shared",
        ),
        CellOutput(
            output_type="display_data",
            content={"text/plain": "first-b"},
            display_id="shared",
        ),
        CellOutput(
            output_type="update_display_data",
            content={"text/plain": "updated"},
            display_id="shared",
        ),
    ])

    assert len(normalized) == 2
    assert [output.content["text/plain"] for output in normalized] == [
        "updated", "updated",
    ]


def test_binary_image_formatter_values_are_encoded_for_jupyter_transport():
    bundle = _wire_safe_mime_bundle({
        "image/png": b"png-bytes",
        "image/svg+xml": b"<svg></svg>",
        "text/plain": "fallback",
    })

    assert bundle["image/png"] == "cG5nLWJ5dGVz"
    assert bundle["image/svg+xml"] == "<svg></svg>"


def test_local_display_publisher_preserves_display_id_and_update_type():
    queue = _Queue()
    publisher = StreamingDisplayPublisher(queue)

    publisher.publish(
        {"text/plain": "first"},
        transient={"display_id": "progress"},
    )
    publisher.publish(
        {"text/plain": "second"},
        transient={"display_id": "progress"},
        update=True,
    )

    assert queue.items[0]["type"] == "display_data"
    assert queue.items[1]["type"] == "update_display_data"
    assert [item["display_id"] for item in queue.items] == ["progress", "progress"]


def test_real_local_kernel_emits_jupyter_display_updates():
    async def run():
        kernel = SubprocessKernel()
        try:
            updates = [output async for output in kernel.execute_streaming(
                "from IPython.display import display\n"
                "handle = display('first', display_id=True)\n"
                "handle.update('second')",
                "output-test",
                "cell",
            )]
            result = [output async for output in kernel.execute_streaming(
                "class Rich:\n"
                "    def _repr_html_(self): return '<b>rich</b>'\n"
                "Rich()",
                "output-test",
                "rich-cell",
            )]
            return updates, result
        finally:
            kernel.shutdown()

    outputs, result_outputs = asyncio.run(run())

    assert [output.output_type for output in outputs] == [
        "display_data", "update_display_data",
    ]
    assert outputs[0].display_id == outputs[1].display_id
    assert outputs[1].content["text/plain"] == "'second'"
    assert [output.output_type for output in result_outputs] == ["execute_result"]
    assert result_outputs[0].content["text/html"] == "<b>rich</b>"
    assert "text/plain" in result_outputs[0].content


def test_real_local_kernel_emits_one_structured_traceback():
    async def run(code):
        kernel = SubprocessKernel()
        try:
            return [output async for output in kernel.execute_streaming(
                code,
                "output-test",
                "error-cell",
            )]
        finally:
            kernel.shutdown()

    runtime_outputs = asyncio.run(run(
        'import sys\nprint("before")\nprint("warning", file=sys.stderr)\n'
        'raise ValueError("broken")'
    ))
    syntax_outputs = asyncio.run(run("if True print('broken')"))

    runtime_streams = [
        output for output in runtime_outputs if output.output_type == "stream"
    ]
    assert "".join(
        str(output.content)
        for output in runtime_streams
        if output.stream_name == "stdout"
    ) == "before\n"
    assert "".join(
        str(output.content)
        for output in runtime_streams
        if output.stream_name == "stderr"
    ) == "warning\n"
    assert sum(output.output_type == "error" for output in runtime_outputs) == 1
    assert runtime_outputs[-1].ename == "ValueError"
    assert sum(output.output_type == "error" for output in syntax_outputs) == 1
    assert all(
        "Traceback" not in str(output.content)
        for output in runtime_outputs
        if output.output_type == "stream"
    )
