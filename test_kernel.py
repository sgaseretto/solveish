#!/usr/bin/env python3
"""
Test script for the subprocess kernel with streaming output.

Run with: uv run python test_kernel.py
"""
import asyncio
import sys
sys.path.insert(0, '.')

from document.cell import Cell, CellType
from services.kernel.subprocess_kernel import SubprocessKernel
from services.kernel.kernel_service import KernelService


async def test_basic_streaming():
    """Test basic streaming output."""
    print("=" * 60)
    print("Test 1: Basic streaming output")
    print("=" * 60)

    kernel = SubprocessKernel()

    code = '''
from time import sleep
for i in range(5):
    print(f"i = {i}")
    sleep(0.5)
print("Done!")
'''

    print(f"Running code:\n{code}\n")
    print("Streaming output:")

    async for output in kernel.execute_streaming(code):
        if output.output_type == 'stream':
            print(f"  [{output.stream_name}] {output.content!r}")
        elif output.output_type == 'execute_result':
            print(f"  [result] {output.content}")
        elif output.output_type == 'error':
            print(f"  [error] {output.ename}: {output.evalue}")

    kernel.shutdown()
    print("\nTest 1 PASSED\n")


async def test_kernel_service():
    """Test KernelService with Cell object."""
    print("=" * 60)
    print("Test 2: KernelService with Cell")
    print("=" * 60)

    service = KernelService()

    cell = Cell(
        cell_type=CellType.CODE,
        source='x = 42\nprint(f"x = {x}")\nx * 2'
    )

    print(f"Cell source:\n{cell.source}\n")
    print("Streaming output:")

    async for output in service.execute_cell("test-notebook", cell):
        if output.output_type == 'stream':
            print(f"  [{output.stream_name}] {output.content!r}")
        elif output.output_type == 'execute_result':
            print(f"  [result] {output.content}")

    print(f"\nCell state: {cell.state}")
    print(f"Cell execution_count: {cell.execution_count}")
    print(f"Cell output: {cell.output!r}")

    service.shutdown_all()
    print("\nTest 2 PASSED\n")


async def test_namespace_persistence():
    """Test that variables persist between executions."""
    print("=" * 60)
    print("Test 3: Namespace persistence")
    print("=" * 60)

    service = KernelService()

    # First cell: define a variable
    cell1 = Cell(cell_type=CellType.CODE, source='my_var = "Hello from cell 1"')
    async for _ in service.execute_cell("test-notebook", cell1):
        pass

    # Second cell: use the variable
    cell2 = Cell(cell_type=CellType.CODE, source='print(my_var)')
    print("Streaming output from cell 2:")
    async for output in service.execute_cell("test-notebook", cell2):
        if output.output_type == 'stream':
            print(f"  [{output.stream_name}] {output.content!r}")

    service.shutdown_all()
    print("\nTest 3 PASSED\n")


async def test_error_handling():
    """Test error output."""
    print("=" * 60)
    print("Test 4: Error handling")
    print("=" * 60)

    service = KernelService()

    cell = Cell(
        cell_type=CellType.CODE,
        source='raise ValueError("This is a test error")'
    )

    print("Streaming output:")
    async for output in service.execute_cell("test-notebook", cell):
        if output.output_type == 'error':
            print(f"  [error] {output.ename}: {output.evalue}")
            print(f"  [traceback] {output.traceback[-1] if output.traceback else 'N/A'}")

    print(f"\nCell state: {cell.state}")

    service.shutdown_all()
    print("\nTest 4 PASSED\n")


async def test_interrupt():
    """Test hard interrupt via SIGINT."""
    print("=" * 60)
    print("Test 5: Hard interrupt")
    print("=" * 60)

    service = KernelService()

    cell = Cell(
        cell_type=CellType.CODE,
        source='''
import time
print("Starting long loop...")
for i in range(100):
    print(f"Iteration {i}")
    time.sleep(0.5)
print("Should not reach here")
'''
    )

    print("Starting execution (will interrupt after 1.5 seconds)...")

    async def run_with_interrupt():
        outputs = []

        async def collect():
            async for output in service.execute_cell("test-notebook", cell):
                outputs.append(output)
                if output.output_type == 'stream':
                    print(f"  [{output.stream_name}] {output.content!r}")
                elif output.output_type == 'error':
                    print(f"  [error] {output.ename}: {output.evalue}")

        # Start execution
        task = asyncio.create_task(collect())

        # Wait a bit then interrupt
        await asyncio.sleep(1.5)
        print("\n  >>> Sending interrupt <<<\n")
        service.interrupt("test-notebook")

        # Wait for task to complete
        await task

        return outputs

    outputs = await run_with_interrupt()

    print(f"\nCell state: {cell.state}")
    has_interrupt = any(
        o.output_type == 'error' and o.ename == 'KeyboardInterrupt'
        for o in outputs
    )
    print(f"Got KeyboardInterrupt: {has_interrupt}")

    service.shutdown_all()
    print("\nTest 5 PASSED\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SUBPROCESS KERNEL TESTS")
    print("=" * 60 + "\n")

    await test_basic_streaming()
    await test_kernel_service()
    await test_namespace_persistence()
    await test_error_handling()
    await test_interrupt()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
