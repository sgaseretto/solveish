"""Tests for prompt-cell save/load persistence."""

import json

from dialeng.document import Cell, CellType, Notebook
from dialeng.document.prompt_utils import LEGACY_SEPARATOR_PREFIX


def test_prompt_cell_roundtrips_through_notebook_save_load(tmp_path):
    """Prompt cells should preserve both prompt and edited assistant response."""
    path = tmp_path / "prompt_roundtrip.ipynb"
    prompt_cell = Cell(
        id="prompt1",
        cell_type=CellType.PROMPT,
        source="My name is Joe Doe",
    )
    prompt_cell.output = "Nice to meet you Joe Doe."
    notebook = Notebook(
        id="prompt-roundtrip",
        title="Prompt Roundtrip",
        cells=[prompt_cell],
    )

    notebook.save(str(path))
    loaded = Notebook.load(str(path))

    assert len(loaded.cells) == 1
    assert loaded.cells[0].cell_type == CellType.PROMPT
    assert loaded.cells[0].source == "My name is Joe Doe"
    assert loaded.cells[0].output == "Nice to meet you Joe Doe."

    saved_nb = json.loads(path.read_text(encoding="utf-8"))
    saved_source = "".join(saved_nb["cells"][0]["source"])
    assert "##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_" in saved_source


def test_prompt_cell_loads_legacy_separator_format(tmp_path):
    """Legacy prompt separators should still load into source/output fields."""
    path = tmp_path / "legacy_prompt.ipynb"
    source = (
        "My name is John Doe\n\n"
        f"{LEGACY_SEPARATOR_PREFIX}abc123ef -->\n\n"
        "Nice to meet you John Doe."
    )
    notebook_dict = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "id": "prompt1",
                "metadata": {"solveit_ai": True},
                "source": source,
            }
        ],
    }
    path.write_text(json.dumps(notebook_dict), encoding="utf-8")

    loaded = Notebook.load(str(path))

    assert len(loaded.cells) == 1
    assert loaded.cells[0].cell_type == CellType.PROMPT
    assert loaded.cells[0].source == "My name is John Doe"
    assert loaded.cells[0].output == "Nice to meet you John Doe."
