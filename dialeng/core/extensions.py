"""
Dialeng Extension Loading

Provides utilities for loading extensions from:
- Python files in the extensions/ directory
- Notebooks with @extension markers

Extension Discovery:
    Extensions are Python files in the `extensions/` directory.
    Each file is imported as a module, which triggers any decorators
    (e.g., @register_cell_type, @register_callback).

Notebook-to-Extension Workflow:
    1. Experiment in a dialeng notebook
    2. Mark cells with `# @extension` comment
    3. Use `extract_extension()` to create a standalone file
    4. Place in extensions/ directory
    5. Restart dialeng to load

Example Extension File:
    ```python
    # extensions/diagram_cell.py
    from dialeng.core.registry import register_cell_type, register_callback
    from dialeng.core.dispatch import register_renderer, register_llm_converter
    from dialeng.core.callbacks import Callback
    from dialeng.document.cell import Cell
    from fasthtml.common import Div, Pre

    class DiagramCell(Cell):
        cell_type = "diagram"

    @register_renderer("diagram")
    def render_diagram(cell, notebook_id):
        return Div(Pre(cell.source), id=f"cell-{cell.id}")

    @register_llm_converter("diagram")
    def diagram_to_messages(cell):
        return [{"role": "user", "content": f"[Diagram]\\n{cell.source}"}]

    register_cell_type(DiagramCell, icon="📊", label="Diagram")
    ```
"""

from __future__ import annotations
import importlib
import importlib.util
import sys
import logging
from pathlib import Path
from typing import List, Optional

from .registry import registry

logger = logging.getLogger(__name__)


def load_extensions(
    extensions_dir: Optional[Path] = None,
    silent: bool = False
) -> List[str]:
    """
    Load all extensions from the extensions directory.

    Extensions are Python files (*.py) that are imported as modules.
    Files starting with underscore are ignored.

    Args:
        extensions_dir: Directory containing extension files.
                       Defaults to 'extensions/' in the project root.
        silent: If True, don't log warnings for missing directory.

    Returns:
        List of loaded extension module names.
    """
    if extensions_dir is None:
        # Default to extensions/ relative to this file's parent (project root)
        project_root = Path(__file__).parent.parent
        extensions_dir = project_root / "extensions"

    extensions_dir = Path(extensions_dir)

    if not extensions_dir.exists():
        if not silent:
            logger.info(f"Extensions directory not found: {extensions_dir}")
        return []

    if not extensions_dir.is_dir():
        logger.warning(f"Extensions path is not a directory: {extensions_dir}")
        return []

    loaded = []

    # Add extensions dir to path for imports
    str_path = str(extensions_dir)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

    # Load each .py file
    for py_file in sorted(extensions_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem

        # Check if already loaded
        if registry.is_extension_loaded(module_name):
            logger.debug(f"Extension already loaded: {module_name}")
            continue

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load extension: {py_file}")
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            registry.mark_extension_loaded(module_name)
            loaded.append(module_name)
            logger.info(f"Loaded extension: {module_name}")

        except Exception as e:
            logger.error(f"Failed to load extension {module_name}: {e}")

    return loaded


def reload_extension(name: str, extensions_dir: Optional[Path] = None) -> bool:
    """
    Reload a specific extension.

    Useful during development to pick up changes without restarting.

    Args:
        name: Extension module name (without .py)
        extensions_dir: Directory containing extension files.

    Returns:
        True if reload succeeded, False otherwise.
    """
    if extensions_dir is None:
        project_root = Path(__file__).parent.parent
        extensions_dir = project_root / "extensions"

    py_file = extensions_dir / f"{name}.py"

    if not py_file.exists():
        logger.error(f"Extension file not found: {py_file}")
        return False

    try:
        # Remove from sys.modules to force reimport
        if name in sys.modules:
            del sys.modules[name]

        # Remove from registry tracking
        if name in registry._loaded_extensions:
            registry._loaded_extensions.remove(name)

        # Reimport
        spec = importlib.util.spec_from_file_location(name, py_file)
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        registry.mark_extension_loaded(name)
        logger.info(f"Reloaded extension: {name}")
        return True

    except Exception as e:
        logger.error(f"Failed to reload extension {name}: {e}")
        return False


def extract_extension(
    notebook_path: Path,
    output_path: Path,
    marker: str = "#| export"
) -> int:
    """
    Extract extension code from a notebook.

    Cells containing the marker directive (or with is_exported=True) are
    extracted and combined into a standalone Python file.

    Args:
        notebook_path: Path to the notebook (.ipynb)
        output_path: Path for the output Python file
        marker: Comment marker to identify extension cells (default: "#| export")

    Returns:
        Number of cells extracted.

    Example notebook cell:
        ```python
        #| export
        from dialeng.core.registry import register_cell_type

        class MyCell(Cell):
            cell_type = "my_cell"
        ```
    """
    from dialeng.document.serialization import load_notebook

    notebook = load_notebook(notebook_path)

    extension_cells = []
    for cell in notebook.cells:
        # Only look at code cells
        cell_type = cell.cell_type
        if hasattr(cell_type, 'value'):
            cell_type = cell_type.value

        if cell_type != "code":
            continue

        # Check for marker in source or is_exported metadata flag
        if marker in cell.source or cell.is_exported:
            # Remove the marker line itself
            lines = cell.source.split('\n')
            lines = [line for line in lines if marker not in line]
            source = '\n'.join(lines).strip()
            if source:
                extension_cells.append(source)

    if not extension_cells:
        logger.info(f"No extension cells found in {notebook_path}")
        return 0

    # Combine cells with separator
    content = f'''"""
Extension extracted from: {notebook_path.name}

Auto-generated by dialeng. Edit as needed.
"""

''' + '\n\n'.join(extension_cells)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    logger.info(f"Extracted {len(extension_cells)} cells to {output_path}")
    return len(extension_cells)


def list_extensions(extensions_dir: Optional[Path] = None) -> List[dict]:
    """
    List available extensions and their status.

    Args:
        extensions_dir: Directory to scan for extensions.

    Returns:
        List of dicts with name, path, loaded status.
    """
    if extensions_dir is None:
        project_root = Path(__file__).parent.parent
        extensions_dir = project_root / "extensions"

    extensions_dir = Path(extensions_dir)

    if not extensions_dir.exists():
        return []

    result = []
    for py_file in sorted(extensions_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        name = py_file.stem
        result.append({
            "name": name,
            "path": str(py_file),
            "loaded": registry.is_extension_loaded(name)
        })

    return result
