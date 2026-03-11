"""
CRAFT service - discovers and processes CRAFT.ipynb files.

CRAFT.ipynb files provide shared context and setup for notebooks:
- Note/prompt cells are prepended to the LLM context
- Code cells are auto-executed in the kernel when a notebook is opened

CRAFT files are resolved hierarchically: parent-first order,
so org-wide settings come before project-specific ones.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

logger = logging.getLogger(__name__)

# Track which CRAFT code cells have been executed per notebook
_executed_craft: Dict[str, Set[str]] = {}


def find_craft_files(notebook_path: Optional[Path], root: Path) -> List[Path]:
    """Find CRAFT.ipynb files from notebook's directory up to root (parent-first order).

    Args:
        notebook_path: Path to the current notebook (or its directory)
        root: Project root (stop searching here)

    Returns:
        List of CRAFT.ipynb paths, parent-first order
    """
    if notebook_path is None:
        return []

    crafts = []
    current = Path(notebook_path).resolve()
    # If it's a file, start from its parent directory
    if current.is_file():
        current = current.parent
    root = root.resolve()

    while True:
        craft_path = current / "CRAFT.ipynb"
        if craft_path.exists():
            crafts.append(craft_path)

        # Stop at root
        if current == root or current.parent == current:
            break
        current = current.parent

    # Reverse to get parent-first order
    crafts.reverse()
    return crafts


def get_craft_context(craft_paths: List[Path]) -> List[Dict]:
    """Extract LLM context messages from CRAFT notebook note/prompt cells.

    Args:
        craft_paths: List of CRAFT.ipynb paths (parent-first order)

    Returns:
        List of message dicts in LLM format (role + content)
    """
    from document.serialization import load_notebook
    from core.dispatch import cell_to_llm_messages

    messages = []
    for path in craft_paths:
        try:
            nb = load_notebook(path)
            for cell in nb.cells:
                cell_type = cell.cell_type
                if hasattr(cell_type, 'value'):
                    cell_type = cell_type.value

                # Only include note and prompt cells in context
                if cell_type in ("note", "prompt"):
                    cell_msgs = cell_to_llm_messages(cell)
                    messages.extend(cell_msgs)
        except Exception as e:
            logger.error(f"Failed to load CRAFT {path}: {e}")

    return messages


def get_craft_code_cells(craft_paths: List[Path]) -> List[Tuple[str, str]]:
    """Extract code cells from CRAFT notebooks for kernel execution.

    Args:
        craft_paths: List of CRAFT.ipynb paths (parent-first order)

    Returns:
        List of (cell_id, source) tuples for code cells
    """
    from document.serialization import load_notebook

    code_cells = []
    for path in craft_paths:
        try:
            nb = load_notebook(path)
            for cell in nb.cells:
                cell_type = cell.cell_type
                if hasattr(cell_type, 'value'):
                    cell_type = cell_type.value

                if cell_type == "code" and cell.source.strip():
                    # Use path + cell.id as unique key
                    unique_id = f"{path}:{cell.id}"
                    code_cells.append((unique_id, cell.source))
        except Exception as e:
            logger.error(f"Failed to load CRAFT {path}: {e}")

    return code_cells


def mark_craft_executed(notebook_id: str, craft_cell_id: str) -> None:
    """Mark a CRAFT code cell as executed for a notebook."""
    if notebook_id not in _executed_craft:
        _executed_craft[notebook_id] = set()
    _executed_craft[notebook_id].add(craft_cell_id)


def is_craft_executed(notebook_id: str, craft_cell_id: str) -> bool:
    """Check if a CRAFT code cell has been executed for a notebook."""
    return craft_cell_id in _executed_craft.get(notebook_id, set())


def reset_craft_tracking(notebook_id: str) -> None:
    """Reset CRAFT execution tracking for a notebook (e.g., on kernel restart)."""
    _executed_craft.pop(notebook_id, None)
