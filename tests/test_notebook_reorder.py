"""Tests for notebook cell reordering semantics."""

from dialeng.document.cell import Cell, CellOutput
from dialeng.document.notebook import Notebook


def test_prompt_cell_can_move_down_multiple_times():
    prompt = Cell(
        cell_type="prompt",
        source="Explain this",
        outputs=[CellOutput(output_type="stream", content="Answer", stream_name="stdout")],
    )
    code_a = Cell(cell_type="code", source="a = 1")
    code_b = Cell(cell_type="code", source="b = 2")
    code_c = Cell(cell_type="code", source="c = 3")
    nb = Notebook(cells=[code_a, prompt, code_b, code_c])

    assert nb.move_cell(prompt.id, 1) is True
    assert [cell.id for cell in nb.cells] == [code_a.id, code_b.id, prompt.id, code_c.id]

    assert nb.move_cell(prompt.id, 1) is True
    assert [cell.id for cell in nb.cells] == [code_a.id, code_b.id, code_c.id, prompt.id]


def test_move_cell_respects_boundaries():
    first = Cell(cell_type="note", source="top")
    second = Cell(
        cell_type="prompt",
        source="middle",
        outputs=[CellOutput(output_type="stream", content="done", stream_name="stdout")],
    )
    nb = Notebook(cells=[first, second])

    assert nb.move_cell(first.id, -1) is False
    assert nb.move_cell(second.id, 1) is False
    assert [cell.id for cell in nb.cells] == [first.id, second.id]
