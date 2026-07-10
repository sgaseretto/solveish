"""Smoke tests for reproducible performance fixtures."""

from benchmarks.performance_baseline import build_notebook, run_benchmarks


def test_build_notebook_uses_requested_size_and_final_prompt():
    notebook = build_notebook(20)

    assert len(notebook.cells) == 20
    assert notebook.cells[-1].cell_type == "prompt"


def test_small_benchmark_returns_json_safe_measurements():
    payload = run_benchmarks([5], repeats=1)
    result = payload["results"][0]

    assert payload["schema_version"] == 1
    assert result["cell_count"] == 5
    assert result["serialized_bytes"] > 0
    assert result["save"]["min_ms"] >= 0
    assert result["context_build"]["median_ms"] >= 0
