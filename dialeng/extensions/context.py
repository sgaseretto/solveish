"""Registration staging context used by the extension runtime."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
import inspect


@dataclass(frozen=True)
class ExtensionContribution:
    """One extension-owned mutation proposed for a live Dialeng registry."""

    kind: str
    name: str
    value: Any
    owner: str = ""
    extension_version: str = ""
    source_location: str = ""
    replace_policy: str = "replace"


@dataclass
class RegistrationScope:
    """Collect registrations without mutating live global registries."""

    extension_id: str
    extension_version: str = ""
    contributions: list[ExtensionContribution] = field(default_factory=list)
    _keys: set[tuple[str, str]] = field(default_factory=set, init=False)

    def add(self, kind: str, name: str, value: Any) -> None:
        key = (kind, name)
        if key in self._keys:
            raise ValueError(
                f"Extension {self.extension_id!r} registered {kind}:{name} more than once"
            )
        self._keys.add(key)
        source_location = ""
        for frame in inspect.stack()[2:]:
            filename = frame.filename.replace("\\", "/")
            if "/dialeng/extensions/" not in filename and "/dialeng/core/" not in filename:
                source_location = f"{frame.filename}:{frame.lineno}"
                break
        self.contributions.append(ExtensionContribution(
            kind,
            name,
            value,
            owner=self.extension_id,
            extension_version=self.extension_version,
            source_location=source_location,
        ))


_CURRENT_SCOPE: ContextVar[Optional[RegistrationScope]] = ContextVar(
    "dialeng_extension_registration_scope", default=None
)


def current_registration_scope() -> Optional[RegistrationScope]:
    return _CURRENT_SCOPE.get()


def capture_registration(kind: str, name: str, value: Any) -> bool:
    """Capture a registration when an extension candidate is being staged."""
    scope = current_registration_scope()
    if scope is None:
        return False
    scope.add(kind, str(name), value)
    return True


@contextmanager
def registration_scope(scope: RegistrationScope) -> Iterator[RegistrationScope]:
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)
