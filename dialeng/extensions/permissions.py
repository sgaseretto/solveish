"""Permission matching shared by extension contexts and the core-side broker."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


def resolve_permitted_project_path(
    project_root: str | Path,
    path: str | Path,
    permissions: dict[str, Any],
    *,
    access: str = "read",
    must_exist: bool = False,
) -> Path:
    """Resolve a project path and enforce the matching manifest path rules."""

    project = Path(project_root).resolve()
    candidate = (project / Path(path)).resolve()
    if not candidate.is_relative_to(project):
        raise ValueError("Extension path escapes the project root")
    relative = candidate.relative_to(project).as_posix() or "."
    permission = "filesystem_write" if access == "write" else "filesystem_read"
    patterns = permissions.get(permission, [])
    if not isinstance(patterns, list) or not any(
        _path_matches(relative, str(pattern)) for pattern in patterns
    ):
        raise PermissionError(
            f"Extension manifest does not permit {access} access to {relative!r}"
        )
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _path_matches(relative: str, pattern: str) -> bool:
    normalized = pattern.strip().replace("\\", "/").strip("/")
    if normalized in {"", ".", "*", "**", "project"}:
        return True
    relative_path = PurePosixPath(relative)
    if any(character in normalized for character in "*?["):
        return relative_path.match(normalized) or relative_path.match(f"{normalized}/**")
    return relative == normalized or relative.startswith(f"{normalized}/")
