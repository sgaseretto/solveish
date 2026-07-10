"""Tests for authoritative notebook runtime ownership and mutation ordering."""

import asyncio

import pytest

from dialeng.services.runtime_registry import (
    LifecycleState,
    NotebookRuntimeRegistry,
    RuntimeRevisionConflict,
)


def test_field_maps_share_one_runtime_and_preserve_mapping_behavior():
    registry = NotebookRuntimeRegistry()
    notebooks = registry.field_map("notebook")
    queues = registry.field_map("execution_queue")
    revisions = registry.field_map("notebook_revision")

    notebooks["demo"] = "notebook"
    queues["demo"] = "queue"
    revisions["demo"] = 2

    runtime = registry.get("demo")
    assert runtime.notebook == "notebook"
    assert runtime.execution_queue == "queue"
    assert runtime.notebook_revision == 2
    assert notebooks.get("demo") == "notebook"
    assert list(queues) == ["demo"]

    queues.pop("demo")
    assert queues.get("demo") is None
    assert notebooks["demo"] == "notebook"


def test_mutations_are_serialized_and_revision_conflicts_are_explicit():
    async def run():
        registry = NotebookRuntimeRegistry()
        active = 0
        maximum_active = 0
        order = []

        async def mutate(name):
            nonlocal active, maximum_active
            async with registry.mutation("demo"):
                active += 1
                maximum_active = max(maximum_active, active)
                order.append(f"{name}:start")
                await asyncio.sleep(0.01)
                order.append(f"{name}:end")
                active -= 1

        await asyncio.gather(mutate("one"), mutate("two"))
        assert maximum_active == 1
        assert order in (
            ["one:start", "one:end", "two:start", "two:end"],
            ["two:start", "two:end", "one:start", "one:end"],
        )

        registry.next_notebook_revision("demo")
        with pytest.raises(RuntimeRevisionConflict) as exc_info:
            async with registry.mutation("demo", expected_revision=0):
                pass
        assert exc_info.value.actual == 1

        registry.next_structure_revision("demo")
        # Durable source/output revisions do not invalidate cell placement.
        async with registry.mutation("demo", expected_structure_revision=1):
            pass
        with pytest.raises(RuntimeRevisionConflict) as structure_exc:
            async with registry.mutation("demo", expected_structure_revision=0):
                pass
        assert structure_exc.value.domain == "structure"
        assert structure_exc.value.actual == 1

    asyncio.run(run())


def test_operation_ids_are_idempotent_and_bounded():
    async def run():
        registry = NotebookRuntimeRegistry(max_completed_operations=2)
        async with registry.mutation("demo", operation_id="one") as lease:
            lease.result = "first"
        async with registry.mutation("demo", operation_id="one") as duplicate:
            assert duplicate.duplicate is True
            assert duplicate.previous_result == "first"
        async with registry.mutation("demo", operation_id="two") as lease:
            lease.result = "second"
        async with registry.mutation("demo", operation_id="three") as lease:
            lease.result = "third"

        runtime = registry.get("demo")
        assert list(runtime.completed_operations) == ["two", "three"]

    asyncio.run(run())


def test_clear_runtime_is_idempotent_and_keeps_terminal_lifecycle():
    registry = NotebookRuntimeRegistry()
    runtime = registry.get_or_create("demo")
    runtime.notebook = object()
    runtime.notebook_revision = 4
    runtime.structure_revision = 2
    runtime.stream_revisions["output"] = 3

    registry.clear_runtime("demo", lifecycle=LifecycleState.DELETED)
    registry.clear_runtime("demo", lifecycle=LifecycleState.DELETED)

    runtime = registry.get("demo")
    assert runtime.lifecycle == LifecycleState.DELETED
    assert runtime.notebook is None
    assert runtime.notebook_revision == 0
    assert runtime.structure_revision == 0
    assert runtime.stream_revisions == {}
    assert registry.diagnostics()["lifecycle_counts"]["deleted"] == 1
