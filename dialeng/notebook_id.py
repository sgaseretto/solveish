"""Shared notebook ID helpers.

Dialeng uses notebook IDs as stable in-memory keys and URL path parameters.
For notebooks in subdirectories, IDs encode path boundaries with ``~`` while
escaping literal tildes as ``~~`` so the mapping remains reversible.
"""

from pathlib import Path


def nb_id_encode_part(part: str) -> str:
    """Escape tildes in a single path component."""
    return part.replace("~", "~~")


def nb_id_from_relpath(relpath: str | Path) -> str:
    """Encode a notebooks-relative path into a Dialeng notebook ID.

    The input may be a path with or without the ``.ipynb`` suffix.
    """
    rel = Path(relpath)
    parts = list(rel.parts)
    if not parts:
        return ""
    parts[-1] = rel.stem if rel.suffix == ".ipynb" else rel.name
    encoded = [nb_id_encode_part(part) for part in parts]
    return "~".join(encoded) if len(encoded) > 1 else encoded[0]


def nb_id_from_path(path: Path, notebooks_root: Path) -> str:
    """Derive a collision-proof, URL-safe notebook ID from an absolute path."""
    try:
        rel = path.resolve().relative_to(notebooks_root.resolve())
    except ValueError:
        return nb_id_encode_part(path.stem)
    return nb_id_from_relpath(rel)


def nb_id_to_relpath(notebook_id: str) -> Path:
    """Decode a Dialeng notebook ID back to a notebooks-relative path."""
    placeholder = "\x00"
    safe = notebook_id.replace("~~", placeholder)
    parts = safe.split("~")
    parts = [part.replace(placeholder, "~") for part in parts]
    return Path(*parts) if len(parts) > 1 else Path(parts[0])
