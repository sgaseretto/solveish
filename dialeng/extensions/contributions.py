"""Atomic application of extension-owned registry contributions."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

from .context import ExtensionContribution


MISSING = object()


class ContributionConflictError(ValueError):
    pass


@dataclass(frozen=True)
class ContributionOwner:
    extension_id: str
    contributions: tuple[ExtensionContribution, ...]


_MAP_KINDS = {
    "registry.cell_type", "registry.service", "registry.kernel",
    "registry.provider", "registry.toolbar", "registry.settings_section",
    "registry.action", "dispatch.renderer", "dispatch.serializer",
    "dispatch.llm_converter", "settings.definition",
    "extension.command", "extension.action", "extension.toolbar", "extension.shortcut", "extension.panel",
    "extension.status", "extension.asset", "extension.tool", "extension.event",
}
_LIST_KINDS = {"registry.callback", "dispatch.deserializer"}


def _mapping(kind: str) -> dict:
    from dialeng.core import dispatch
    from dialeng.core.registry import registry
    from dialeng.services import settings_service
    from dialeng.extensions import events, resources

    targets = {
        "registry.cell_type": registry.cell_types,
        "registry.service": registry.services,
        "registry.kernel": registry.kernels,
        "registry.provider": registry.providers,
        "registry.toolbar": registry.toolbar_items,
        "registry.settings_section": registry.settings_sections,
        "registry.action": registry.actions,
        "dispatch.renderer": dispatch._renderers,
        "dispatch.serializer": dispatch._jupyter_serializers,
        "dispatch.llm_converter": dispatch._llm_converters,
        "settings.definition": settings_service._DYNAMIC_DEFINITIONS,
        "extension.command": resources.commands,
        "extension.action": resources.actions,
        "extension.toolbar": resources.toolbar_items,
        "extension.shortcut": resources.shortcuts,
        "extension.panel": resources.panels,
        "extension.status": resources.status_items,
        "extension.asset": resources.assets,
        "extension.tool": resources.tools,
        "extension.event": events.handlers,
    }
    return targets[kind]


def _sequence(kind: str) -> list:
    from dialeng.core import dispatch
    from dialeng.core.registry import registry

    return {
        "registry.callback": registry.callbacks,
        "dispatch.deserializer": dispatch._jupyter_deserializers,
    }[kind]


class ContributionStore:
    """Own live extension contributions and rebuild affected registry entries."""

    def __init__(self):
        self._owners: "OrderedDict[str, ContributionOwner]" = OrderedDict()
        self._base_maps: dict[tuple[str, str], Any] = {}
        self._base_lists: dict[str, list[Any]] = {}

    @property
    def owners(self) -> dict[str, ContributionOwner]:
        return dict(self._owners)

    def _validate(self, owner: str, contributions: Iterable[ExtensionContribution]) -> tuple[ExtensionContribution, ...]:
        values = tuple(contributions)
        keys: set[tuple[str, str]] = set()
        for contribution in values:
            if contribution.kind not in _MAP_KINDS | _LIST_KINDS:
                raise ValueError(f"Unsupported contribution kind: {contribution.kind}")
            key = (contribution.kind, contribution.name)
            if key in keys:
                raise ValueError(f"Duplicate contribution in {owner}: {contribution.kind}:{contribution.name}")
            keys.add(key)
            for other_owner, active in self._owners.items():
                if other_owner == owner:
                    continue
                if any((item.kind, item.name) == key for item in active.contributions):
                    raise ContributionConflictError(
                        f"{contribution.kind}:{contribution.name} is owned by {other_owner}"
                    )
            if contribution.kind in _MAP_KINDS:
                target = _mapping(contribution.kind)
                base_key = (contribution.kind, contribution.name)
                if base_key not in self._base_maps:
                    self._base_maps[base_key] = target.get(contribution.name, MISSING)
                if self._base_maps[base_key] is not MISSING:
                    raise ContributionConflictError(
                        f"{contribution.kind}:{contribution.name} conflicts with a core registration"
                    )
        return values

    def replace(self, owner: str, contributions: Iterable[ExtensionContribution]) -> None:
        candidate = self._validate(owner, contributions)
        previous = self._owners.copy()
        if owner in self._owners:
            items = list(self._owners.items())
            index = next(index for index, item in enumerate(items) if item[0] == owner)
            items[index] = (owner, ContributionOwner(owner, candidate))
            self._owners = OrderedDict(items)
        else:
            self._owners[owner] = ContributionOwner(owner, candidate)
        try:
            self._rebuild(previous_owner=previous.get(owner))
            from dialeng.extensions.events import event_bus
            for contribution in candidate:
                if contribution.kind == "extension.event":
                    event_bus.reset_handler(contribution.name)
        except Exception:
            self._owners = previous
            self._rebuild()
            raise

    def remove(self, owner: str) -> bool:
        active = self._owners.pop(owner, None)
        if active is None:
            return False
        self._rebuild(previous_owner=active)
        return True

    def _rebuild(self, previous_owner: ContributionOwner | None = None) -> None:
        affected: set[tuple[str, str]] = set()
        list_kinds: set[str] = set()
        owners = list(self._owners.values())
        if previous_owner:
            owners = [previous_owner, *owners]
        for active in owners:
            for contribution in active.contributions:
                if contribution.kind in _MAP_KINDS:
                    affected.add((contribution.kind, contribution.name))
                else:
                    list_kinds.add(contribution.kind)

        for kind, name in affected:
            target = _mapping(kind)
            value = self._base_maps.get((kind, name), MISSING)
            for active in self._owners.values():
                for contribution in active.contributions:
                    if contribution.kind == kind and contribution.name == name:
                        value = contribution.value
            if value is MISSING:
                target.pop(name, None)
            else:
                target[name] = value

        for kind in list_kinds:
            target = _sequence(kind)
            if kind not in self._base_lists:
                owned_values = {
                    id(item.value)
                    for active in self._owners.values()
                    for item in active.contributions
                    if item.kind == kind
                }
                if previous_owner:
                    owned_values.update(
                        id(item.value) for item in previous_owner.contributions if item.kind == kind
                    )
                self._base_lists[kind] = [item for item in target if id(item) not in owned_values]
            values = list(self._base_lists[kind])
            for active in self._owners.values():
                values.extend(item.value for item in active.contributions if item.kind == kind)
            if kind == "registry.callback":
                values.sort(key=lambda callback: callback.order)
            target[:] = values

        from dialeng.extensions import events
        live_handler_names = set(events.handlers)
        events.event_bus.disabled_handlers.intersection_update(live_handler_names)
        for name in list(events.event_bus.failure_counts):
            if name not in live_handler_names:
                events.event_bus.failure_counts.pop(name, None)

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        return {
            owner: [
                {
                    "kind": item.kind,
                    "name": item.name,
                    "owner": item.owner or owner,
                    "version": item.extension_version,
                    "source_location": item.source_location,
                    "replace_policy": item.replace_policy,
                }
                for item in active.contributions
            ]
            for owner, active in self._owners.items()
        }
