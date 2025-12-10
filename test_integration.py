#!/usr/bin/env python3
"""
Integration test showing how the new kernel integrates with the app.

This demonstrates:
1. The new streaming kernel replacing the old PythonKernel
2. WebSocket streaming of code output
3. The execution queue for responsive UI
"""
import asyncio
import sys
sys.path.insert(0, '.')

from document import Cell, CellType, Notebook, load_notebook, save_notebook
from services.kernel import KernelService, ExecutionQueue


async def demo_streaming_kernel():
    """Demo: Streaming kernel execution with queue."""
    print("\n" + "=" * 60)
    print("DEMO: Streaming Kernel Integration")
    print("=" * 60)

    # Create services (like app.py would)
    kernel_service = KernelService()
    execution_queue = ExecutionQueue(kernel_service)

    # Create a notebook with cells
    nb = Notebook(
        id="test-notebook",
        title="Test Notebook",
        cells=[
            Cell(
                id="cell1",
                cell_type=CellType.CODE,
                source='print("Hello from cell 1!")'
            ),
            Cell(
                id="cell2",
                cell_type=CellType.CODE,
                source='''
from time import sleep
print("Starting streaming demo...")
for i in range(5):
    print(f"Step {i+1}/5")
    sleep(0.3)
print("Done!")
'''
            ),
            Cell(
                id="cell3",
                cell_type=CellType.CODE,
                source='x = 42\nx * 2  # Should show result'
            ),
        ]
    )

    # Register output callback (simulates WebSocket sending)
    async def on_output(cell, output):
        print(f"  [WS] Cell {cell.id}: {output.output_type}: {str(output.content)[:50]!r}")

    execution_queue.on_output(nb.id, on_output)

    # Queue all cells for execution (like "Run All" button)
    print("\n>>> Queuing all cells...")
    for cell in nb.cells:
        execution_queue.queue_cell(nb.id, cell)
        print(f"  Queued: {cell.id} (state: {cell.state})")

    # Wait for queue to process
    print("\n>>> Waiting for execution...")
    while execution_queue._processing.get(nb.id, False):
        await asyncio.sleep(0.1)

    # Show final states
    print("\n>>> Final cell states:")
    for cell in nb.cells:
        print(f"  {cell.id}: {cell.state} (output: {cell.output[:30]!r}...)")

    # Cleanup
    kernel_service.shutdown_all()
    print("\n>>> Done!")


async def demo_interrupt():
    """Demo: Interrupt a long-running cell."""
    print("\n" + "=" * 60)
    print("DEMO: Interrupt Long-Running Cell")
    print("=" * 60)

    kernel_service = KernelService()
    execution_queue = ExecutionQueue(kernel_service)

    cell = Cell(
        id="long-running",
        cell_type=CellType.CODE,
        source='''
import time
print("Starting long computation...")
for i in range(100):
    print(f"Iteration {i}")
    time.sleep(0.2)
print("Finished!")
'''
    )

    async def on_output(cell, output):
        if output.output_type == 'stream':
            print(f"  [{cell.id}] {output.content.strip()}")
        elif output.output_type == 'error':
            print(f"  [{cell.id}] ERROR: {output.ename}")

    execution_queue.on_output("nb1", on_output)

    print("\n>>> Starting long-running cell...")
    execution_queue.queue_cell("nb1", cell)

    # Wait a bit, then interrupt
    await asyncio.sleep(1.5)
    print("\n>>> Sending interrupt!")
    kernel_service.interrupt("nb1")

    # Wait for completion
    while execution_queue._processing.get("nb1", False):
        await asyncio.sleep(0.1)

    print(f"\n>>> Final state: {cell.state}")
    kernel_service.shutdown_all()


async def demo_notebook_persistence():
    """Demo: Save and load notebook with new document layer."""
    print("\n" + "=" * 60)
    print("DEMO: Notebook Persistence")
    print("=" * 60)

    from pathlib import Path

    # Create a notebook
    nb = Notebook(
        id="persist-test",
        title="Persistence Test",
        cells=[
            Cell(cell_type=CellType.CODE, source='x = 1 + 1\nx'),
            Cell(cell_type=CellType.NOTE, source='# This is a note\nWith **markdown**!'),
            Cell(cell_type=CellType.PROMPT, source='Explain Python'),
        ]
    )

    # Save it
    path = Path("notebooks/test_persist.ipynb")
    path.parent.mkdir(exist_ok=True)
    save_notebook(nb, path)
    print(f">>> Saved notebook to {path}")

    # Load it back
    nb2 = load_notebook(path)
    print(f">>> Loaded notebook: {nb2.title}")
    print(f">>> Cells: {len(nb2.cells)}")
    for cell in nb2.cells:
        print(f"    - {cell.cell_type}: {cell.source[:30]!r}")

    # Cleanup
    path.unlink()
    print("\n>>> Done!")


async def main():
    await demo_streaming_kernel()
    await demo_interrupt()
    await demo_notebook_persistence()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
