"""Authoritative ownership and mutation serialization for notebook runtimes."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Generic, TypeVar


class LifecycleState(str, Enum):
    ACTIVE = "active"
    DELETING = "deleting"
    DELETED = "deleted"
    STOPPED = "stopped"


class RuntimeRevisionConflict(Exception):
    def __init__(
        self,
        notebook_id: str,
        expected: int,
        actual: int,
        *,
        domain: str = "notebook",
    ) -> None:
        self.notebook_id = notebook_id
        self.expected = expected
        self.actual = actual
        self.domain = domain
        super().__init__(
            f"Notebook {notebook_id} {domain} revision conflict: "
            f"expected {expected}, actual {actual}"
        )


@dataclass
class MutationLease:
    runtime: "NotebookRuntime"
    operation_id: str | None
    duplicate: bool = False
    previous_result: Any = None
    result: Any = None


@dataclass
class NotebookRuntime:
    notebook_id: str
    lifecycle: LifecycleState = LifecycleState.ACTIVE
    notebook: Any = None
    mutation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    execution_queue: Any = None
    kernel_generation: int = 0
    notebook_revision: int = 0
    structure_revision: int = 0
    outline_revision: int = 0
    setup_state: dict[str, Any] | None = None
    setup_task: asyncio.Task | None = None
    sync_task: asyncio.Task | None = None
    data_queues: dict[str, asyncio.Queue] = field(default_factory=dict)
    stream_revisions: dict[str, int] = field(default_factory=dict)
    completed_operations: OrderedDict[str, Any] = field(default_factory=OrderedDict)


T = TypeVar("T")


class RuntimeFieldMap(MutableMapping[str, T], Generic[T]):
    """Compatibility mapping backed by fields on registry-owned runtimes."""

    def __init__(self, registry: "NotebookRuntimeRegistry", field_name: str) -> None:
        self.registry = registry
        self.field_name = field_name

    def __getitem__(self, key: str) -> T:
        runtime = self.registry._runtimes[key]
        value = getattr(runtime, self.field_name)
        if not self.registry._field_present(runtime, self.field_name, value):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: T) -> None:
        runtime = self.registry.get_or_create(key)
        setattr(runtime, self.field_name, value)
        if self.field_name == "notebook" and value is not None:
            runtime.lifecycle = LifecycleState.ACTIVE

    def __delitem__(self, key: str) -> None:
        runtime = self.registry._runtimes.get(key)
        if runtime is None:
            raise KeyError(key)
        self.registry._clear_field(runtime, self.field_name)
        self.registry._prune_if_empty(key)

    def __iter__(self) -> Iterator[str]:
        for key, runtime in list(self.registry._runtimes.items()):
            value = getattr(runtime, self.field_name)
            if self.registry._field_present(runtime, self.field_name, value):
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def clear(self) -> None:
        for key in list(self):
            self.pop(key, None)


class NotebookRuntimeRegistry:
    """Own all notebook-scoped mutable runtime state in one lifecycle object."""

    _OPTIONAL_FIELDS = {
        "notebook", "execution_queue", "setup_state", "setup_task", "sync_task",
    }
    _NUMERIC_FIELDS = {
        "kernel_generation", "notebook_revision", "structure_revision", "outline_revision",
    }
    _DICT_FIELDS = {"data_queues", "stream_revisions"}

    def __init__(self, *, max_completed_operations: int = 512) -> None:
        self._runtimes: dict[str, NotebookRuntime] = {}
        self.max_completed_operations = max(1, max_completed_operations)

    def get_or_create(self, notebook_id: str) -> NotebookRuntime:
        runtime = self._runtimes.get(notebook_id)
        if runtime is None:
            runtime = NotebookRuntime(notebook_id=notebook_id)
            self._runtimes[notebook_id] = runtime
        return runtime

    def get(self, notebook_id: str) -> NotebookRuntime | None:
        return self._runtimes.get(notebook_id)

    def field_map(self, field_name: str) -> RuntimeFieldMap[Any]:
        if field_name not in NotebookRuntime.__dataclass_fields__:
            raise ValueError(f"Unknown runtime field: {field_name}")
        return RuntimeFieldMap(self, field_name)

    def _field_present(self, runtime: NotebookRuntime, field_name: str, value: Any) -> bool:
        if field_name in self._OPTIONAL_FIELDS:
            return value is not None
        if field_name in self._NUMERIC_FIELDS:
            return bool(value)
        if field_name in self._DICT_FIELDS:
            return bool(value)
        return True

    def _clear_field(self, runtime: NotebookRuntime, field_name: str) -> None:
        if field_name in self._OPTIONAL_FIELDS:
            setattr(runtime, field_name, None)
        elif field_name in self._NUMERIC_FIELDS:
            setattr(runtime, field_name, 0)
        elif field_name in self._DICT_FIELDS:
            getattr(runtime, field_name).clear()
        else:
            raise ValueError(f"Field cannot be cleared through a mapping: {field_name}")

    def _prune_if_empty(self, notebook_id: str) -> None:
        runtime = self._runtimes.get(notebook_id)
        if runtime is None:
            return
        if (
            runtime.notebook is None
            and runtime.execution_queue is None
            and runtime.kernel_generation == 0
            and runtime.notebook_revision == 0
            and runtime.structure_revision == 0
            and runtime.outline_revision == 0
            and runtime.setup_state is None
            and runtime.setup_task is None
            and runtime.sync_task is None
            and not runtime.data_queues
            and not runtime.stream_revisions
            and not runtime.completed_operations
        ):
            self._runtimes.pop(notebook_id, None)

    def set_lifecycle(self, notebook_id: str, lifecycle: LifecycleState) -> None:
        self.get_or_create(notebook_id).lifecycle = lifecycle

    def next_notebook_revision(self, notebook_id: str) -> int:
        runtime = self.get_or_create(notebook_id)
        runtime.notebook_revision += 1
        return runtime.notebook_revision

    def next_structure_revision(self, notebook_id: str) -> int:
        runtime = self.get_or_create(notebook_id)
        runtime.structure_revision += 1
        return runtime.structure_revision

    def next_outline_revision(self, notebook_id: str) -> int:
        runtime = self.get_or_create(notebook_id)
        runtime.outline_revision += 1
        return runtime.outline_revision

    def next_stream_revision(self, notebook_id: str, stream: str) -> int:
        revisions = self.get_or_create(notebook_id).stream_revisions
        revisions[stream] = revisions.get(stream, 0) + 1
        return revisions[stream]

    @asynccontextmanager
    async def mutation(
        self,
        notebook_id: str,
        *,
        expected_revision: int | None = None,
        expected_structure_revision: int | None = None,
        operation_id: str | None = None,
    ) -> AsyncIterator[MutationLease]:
        runtime = self.get_or_create(notebook_id)
        async with runtime.mutation_lock:
            if operation_id and operation_id in runtime.completed_operations:
                yield MutationLease(
                    runtime=runtime,
                    operation_id=operation_id,
                    duplicate=True,
                    previous_result=runtime.completed_operations[operation_id],
                )
                return
            if expected_revision is not None and expected_revision >= 0:
                if expected_revision != runtime.notebook_revision:
                    raise RuntimeRevisionConflict(
                        notebook_id,
                        expected_revision,
                        runtime.notebook_revision,
                    )
            if expected_structure_revision is not None and expected_structure_revision >= 0:
                if expected_structure_revision != runtime.structure_revision:
                    raise RuntimeRevisionConflict(
                        notebook_id,
                        expected_structure_revision,
                        runtime.structure_revision,
                        domain="structure",
                    )
            lease = MutationLease(runtime=runtime, operation_id=operation_id)
            yield lease
            if operation_id:
                runtime.completed_operations[operation_id] = lease.result
                runtime.completed_operations.move_to_end(operation_id)
                while len(runtime.completed_operations) > self.max_completed_operations:
                    runtime.completed_operations.popitem(last=False)

    def clear_runtime(self, notebook_id: str, *, lifecycle: LifecycleState) -> None:
        runtime = self.get_or_create(notebook_id)
        runtime.lifecycle = lifecycle
        runtime.notebook = None
        runtime.execution_queue = None
        runtime.kernel_generation = 0
        runtime.notebook_revision = 0
        runtime.structure_revision = 0
        runtime.outline_revision = 0
        runtime.setup_state = None
        runtime.setup_task = None
        runtime.sync_task = None
        runtime.data_queues.clear()
        runtime.stream_revisions.clear()
        runtime.completed_operations.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "runtime_count": len(self._runtimes),
            "lifecycle_counts": {
                state.value: sum(1 for runtime in self._runtimes.values() if runtime.lifecycle == state)
                for state in LifecycleState
            },
            "mutation_locks_held": sum(
                1 for runtime in self._runtimes.values() if runtime.mutation_lock.locked()
            ),
            "completed_operation_ids": sum(
                len(runtime.completed_operations) for runtime in self._runtimes.values()
            ),
        }
