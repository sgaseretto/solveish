"""Secret-redacted diagnostics for extension inventory and browser responses."""
from __future__ import annotations

import os
import re
import json
import hashlib
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .contracts import API_POLICY


_ASSIGNMENT = re.compile(
    r"(?i)(authorization|cookie|set-cookie|access[_-]?token|refresh[_-]?token|api[_-]?key|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)


def redact_text(value: Any) -> str:
    text = str(value)
    text = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    for name, secret in os.environ.items():
        if not secret or len(secret) < 8:
            continue
        upper = name.upper()
        if any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "COOKIE")):
            text = text.replace(secret, "[REDACTED]")
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(
                marker in str(key).lower()
                for marker in ("token", "secret", "password", "cookie", "authorization", "api_key")
            ) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def build_extension_doctor(
    project_root: Path,
    discovery,
    trust_store,
    state_store,
    *,
    runtime_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bounded project-wide extension health report."""

    project_root = Path(project_root).resolve()
    runtime_items = {
        item.get("id"): item
        for item in (runtime_snapshot or {}).get("extensions", [])
        if isinstance(item, dict) and item.get("id")
    }
    enabled = state_store.enabled(project_root)
    extensions = []
    for extension_id, source in sorted(discovery.active.items()):
        runtime = runtime_items.get(extension_id, {})
        build_path = (
            project_root / ".dialeng" / "extension-builds"
            / f"{hashlib.sha256(extension_id.encode()).hexdigest()}.json"
        )
        build = None
        if build_path.is_file():
            try:
                build = json.loads(build_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                build = {"corrupted": True}
        extensions.append({
            "id": extension_id,
            "version": source.manifest.version,
            "origin": source.origin,
            "root": str(source.root),
            "source_hash": source.source_hash,
            "dialeng_api": source.manifest.dialeng_api,
            "compatible": True,
            "trusted": trust_store.is_trusted(source),
            "enabled": extension_id in enabled,
            "shadowed": [item.public() for item in discovery.shadowed.get(extension_id, [])],
            "build": build,
            "build_fresh": bool(build and build.get("source_hash") == source.source_hash),
            "runtime_state": runtime.get("state", "not_running"),
            "runtime_mode": runtime.get("runtime_mode", "unknown"),
            "host": runtime.get("host"),
            "registrations": runtime.get("contributions", []),
            "managed_tasks": runtime.get("managed_tasks", 0),
            "failure": runtime.get("error"),
            "cleanup_errors": runtime.get("cleanup_errors", []),
        })
    discovery_diagnostics = [item.__dict__ for item in discovery.diagnostics]
    try:
        dialeng_version = metadata.version("dialeng")
    except metadata.PackageNotFoundError:
        dialeng_version = "source-checkout"
    return redact_value({
        "ok": not discovery_diagnostics and not any(item["failure"] for item in extensions),
        "project_root": str(project_root),
        "dialeng_version": dialeng_version,
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "api_policy": API_POLICY.public(),
        "extensions": extensions,
        "diagnostics": discovery_diagnostics,
        "runtime_available": runtime_snapshot is not None,
        "event_diagnostics": (runtime_snapshot or {}).get("event_diagnostics", {}),
    })
