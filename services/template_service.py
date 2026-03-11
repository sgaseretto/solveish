"""
Template service - discovers and loads TEMPLATE.ipynb files.

TEMPLATE.ipynb files provide default cells for new notebooks created
in the same directory or subdirectories. Templates are resolved
hierarchically: parent templates are included first, then child templates.
"""
import uuid
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def find_templates(target_dir: Path, root: Path) -> List[Path]:
    """Find TEMPLATE.ipynb files from target_dir up to root (parent-first order).

    Args:
        target_dir: Directory where the new notebook will be created
        root: Project root (stop searching here)

    Returns:
        List of TEMPLATE.ipynb paths, parent-first order
    """
    templates = []
    current = target_dir.resolve()
    root = root.resolve()

    while True:
        template_path = current / "TEMPLATE.ipynb"
        if template_path.exists():
            templates.append(template_path)

        # Stop at root
        if current == root or current.parent == current:
            break
        current = current.parent

    # Reverse to get parent-first order
    templates.reverse()
    return templates


def load_template_cells(template_paths: List[Path]) -> List:
    """Load cells from template notebooks with fresh IDs.

    Args:
        template_paths: List of TEMPLATE.ipynb paths (parent-first order)

    Returns:
        List of Cell objects with fresh IDs
    """
    from document.serialization import load_notebook

    all_cells = []
    for path in template_paths:
        try:
            nb = load_notebook(path)
            for cell in nb.cells:
                # Generate fresh ID so template cells don't conflict
                cell.id = uuid.uuid4().hex[:8]
                all_cells.append(cell)
        except Exception as e:
            logger.error(f"Failed to load template {path}: {e}")

    return all_cells
