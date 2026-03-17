"""Save-hook extraction: notebooks with #| default_exp auto-export to a lib directory.

When a notebook containing `#| default_exp module_name` is saved,
cells marked with `#| export` are extracted to `{lib_name}/{module_name}.py`.
The lib directory name is read from pyproject.toml [tool.dialeng] lib_name,
defaulting to '_lib' when no configuration is present.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

LIB_DIR_NAME = "_lib"


def get_lib_name(root_dir: Path) -> str:
    """Read lib_name from pyproject.toml [tool.dialeng], defaulting to '_lib'."""
    pyproject = root_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            return data.get("tool", {}).get("dialeng", {}).get("lib_name", LIB_DIR_NAME)
        except Exception:
            pass
    return LIB_DIR_NAME


def find_default_exp(notebook_path: Path) -> Optional[str]:
    """Find the #| default_exp directive in a notebook.

    Returns the module name string (e.g. 'core', 'data.loaders') or None.
    """
    try:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read notebook {notebook_path}: {e}")
        return None

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#| default_exp"):
                parts = stripped.split()
                # parts: ['#|', 'default_exp', 'module_name']
                if len(parts) >= 3:
                    return parts[2].strip()
    return None


def _extract_export_cells(notebook_path: Path) -> list[str]:
    """Extract source from cells marked with #| export."""
    try:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    cells = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        if not source.strip():
            continue
        first_line = source.lstrip().split("\n", 1)[0].strip()
        if first_line == "#| export":
            lines = source.split("\n", 1)
            body = lines[1] if len(lines) > 1 else ""
            if body.strip():
                cells.append(body)
    return cells


def _ensure_init_files(lib_dir: Path, module_name: str):
    """Create __init__.py files for the _lib package and any sub-packages."""
    init = lib_dir / "__init__.py"
    if not init.exists():
        init.write_text("")
    parts = module_name.split(".")
    if len(parts) > 1:
        current = lib_dir
        for part in parts[:-1]:
            current = current / part
            current.mkdir(exist_ok=True)
            pkg_init = current / "__init__.py"
            if not pkg_init.exists():
                pkg_init.write_text("")


def maybe_extract(notebook_path: Path, root_dir: Path) -> Optional[Dict]:
    """Extract #| export cells from a notebook to the lib directory if it has #| default_exp."""
    module_name = find_default_exp(notebook_path)
    if module_name is None:
        return None

    lib_name = get_lib_name(root_dir)
    lib_dir = root_dir / lib_name
    export_cells = _extract_export_cells(notebook_path)

    parts = module_name.split(".")
    if len(parts) > 1:
        output_path = lib_dir / Path(*parts[:-1]) / f"{parts[-1]}.py"
    else:
        output_path = lib_dir / f"{module_name}.py"

    if not export_cells:
        if output_path.exists():
            output_path.unlink()
            logger.info(f"Removed stale {lib_name} export: {output_path}")
        return None

    lib_dir.mkdir(exist_ok=True)
    _ensure_init_files(lib_dir, module_name)

    header = f'# Auto-extracted from {notebook_path.name} — do not edit directly\n\n'
    content = header + "\n\n".join(export_cells)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    logger.info(f"Extracted {len(export_cells)} cells from {notebook_path.name} → {output_path}")
    return {"module": module_name, "cells_exported": len(export_cells), "path": str(output_path)}
