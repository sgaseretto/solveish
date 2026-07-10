"""Tests for versioned prompt-context block caching and inspection metadata."""

from dialeng.document.cell import Cell
from dialeng.document.notebook import Notebook
from dialeng.services import dialoghelper_service
from dialeng.services.performance_service import performance_monitor


def _notebook() -> Notebook:
    return Notebook(
        id="context-cache",
        cells=[
            Cell(id="note", cell_type="note", source="My name is Joe. $`dynamic_name`"),
            Cell(id="hidden", cell_type="note", source="secret", skipped=True),
            Cell(id="prompt", cell_type="prompt", source="What is my name?"),
        ],
    )


def test_unchanged_context_hits_cache_and_one_edit_invalidates_one_block():
    notebook = _notebook()
    dialoghelper_service.clear_context_cache()
    performance_monitor.reset()

    first = dialoghelper_service.build_context_messages(notebook, "prompt")
    counters_after_first = performance_monitor.snapshot()["counters"]
    second = dialoghelper_service.build_context_messages(notebook, "prompt")
    counters_after_second = performance_monitor.snapshot()["counters"]

    assert first == second
    assert counters_after_first["prompt.context_cell_cache.miss"] == 1
    assert counters_after_second["prompt.context_cell_cache.hit"] > counters_after_first.get(
        "prompt.context_cell_cache.hit", 0
    )

    notebook.get_cell("note").update_source("My name is Jane. $`dynamic_name`")
    third = dialoghelper_service.build_context_messages(notebook, "prompt")
    counters_after_third = performance_monitor.snapshot()["counters"]

    assert "Jane" in str(third)
    assert counters_after_third["prompt.context_cell_cache.miss"] == 2


def test_context_inspection_matches_included_and_skipped_cells():
    notebook = _notebook()
    dialoghelper_service.clear_context_cache()
    messages = dialoghelper_service.build_context_messages(notebook, "prompt")
    report = dialoghelper_service.get_context_inspection(notebook.id, "prompt")

    assert report["included_cells"][0]["cell_id"] == "note"
    assert report["skipped_cells"] == [
        {"cell_id": "hidden", "index": 1, "reason": "skipped"}
    ]
    assert report["truncated_cells"] == []
    assert report["messages"] == [
        {"role": message["role"], "content_chars": len(message["content"])}
        for message in messages
    ]
    assert "$`dynamic_name`" in str(messages)
    assert report["dynamic_injections"] == "evaluated_fresh_during_prompt_execution"
