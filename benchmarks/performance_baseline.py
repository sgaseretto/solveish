"""Reproducible Dialeng server-side performance baseline.

Run with:
    uv run python benchmarks/performance_baseline.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

from dialeng.document.cell import Cell
from dialeng.document.notebook import Notebook
from dialeng.services.dialoghelper_service import build_context_messages
from dialeng.ui.file_explorer import list_directory


def build_notebook(cell_count: int) -> Notebook:
    notebook = Notebook(id=f"benchmark-{cell_count}", title=f"Benchmark {cell_count}")
    cells: list[Cell] = []
    for index in range(max(1, cell_count - 1)):
        if index % 3 == 0:
            cells.append(Cell(cell_type="note", source=f"## Section {index}\n" + "context " * 20))
        elif index % 3 == 1:
            cell = Cell(
                cell_type="code",
                source=f"value_{index} = {index}\nvalue_{index}",
            )
            cell.output = str(index)
            cells.append(cell)
        else:
            cell = Cell(
                cell_type="prompt",
                source=f"Summarize section {index}",
            )
            cell.output = f"Summary for section {index}"
            cells.append(cell)
    cells.append(Cell(cell_type="prompt", source="Summarize the current notebook."))
    notebook.cells = cells
    return notebook


def _measure(repeats: int, operation: Callable[[], None]) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return {
        "min_ms": round(ordered[0], 3),
        "median_ms": round(statistics.median(ordered), 3),
        "max_ms": round(ordered[-1], 3),
    }


def benchmark_size(cell_count: int, repeats: int) -> dict:
    notebook = build_notebook(cell_count)
    current_cell_id = notebook.cells[-1].id
    with tempfile.TemporaryDirectory(prefix="dialeng-benchmark-") as temp_dir:
        root = Path(temp_dir)
        notebook_path = root / "benchmark.ipynb"
        notebook.path = notebook_path

        save = _measure(repeats, lambda: notebook.save(notebook_path))
        load = _measure(repeats, lambda: Notebook.load(notebook_path))
        context = _measure(repeats, lambda: build_context_messages(notebook, current_cell_id))

        explorer_dir = root / "files"
        explorer_dir.mkdir()
        for index in range(cell_count):
            suffix = ".ipynb" if index % 2 == 0 else ".py"
            (explorer_dir / f"item_{index:04d}{suffix}").touch()
        directory_scan = _measure(repeats, lambda: list_directory(explorer_dir, root))

        return {
            "cell_count": cell_count,
            "serialized_bytes": notebook_path.stat().st_size,
            "save": save,
            "load": load,
            "context_build": context,
            "directory_scan": directory_scan,
        }


def run_benchmarks(sizes: list[int], repeats: int) -> dict:
    return {
        "schema_version": 1,
        "repeats": repeats,
        "results": [benchmark_size(size, repeats) for size in sizes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dialeng performance baselines")
    parser.add_argument("--sizes", default="20,100,250", help="Comma-separated cell counts")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    payload = run_benchmarks(sizes, max(1, args.repeats))
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
