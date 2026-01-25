"""
Dialeng Type Dispatch System

Provides extensible dispatch functions for cell operations.
Extensions can register handlers for new cell types.

Note: While fastcore's @typedispatch works with actual Python types,
dialeng's Cell class uses a cell_type string attribute. This module
provides a dispatch pattern that routes based on cell_type while
maintaining extensibility.

Usage:
    # The built-in cell types are already registered
    from core.dispatch import render_cell, cell_to_llm_messages

    # Render a cell
    html = render_cell(cell, notebook_id)

    # Convert to LLM messages
    messages = cell_to_llm_messages(cell)

    # Extensions register handlers:
    from core.dispatch import register_renderer, register_llm_converter

    @register_renderer("diagram")
    def render_diagram_cell(cell, notebook_id):
        return Div(...)

    @register_llm_converter("diagram")
    def diagram_to_messages(cell):
        return [{"role": "user", "content": f"[Diagram]\\n{cell.source}"}]
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Callable, Any, Optional
import logging

if TYPE_CHECKING:
    from document.cell import Cell, CellOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Dispatch Registries
# ============================================================================

# Renderer functions: cell_type -> (cell, notebook_id) -> HTML component
_renderers: Dict[str, Callable] = {}

# Jupyter serializers: cell_type -> (cell) -> dict
_jupyter_serializers: Dict[str, Callable] = {}

# Jupyter deserializers: (jcell_dict) -> Cell (routes based on metadata)
# This is a list because we may need to try multiple deserializers
_jupyter_deserializers: List[Callable] = []

# LLM message converters: cell_type -> (cell) -> List[Dict]
_llm_converters: Dict[str, Callable] = {}


# ============================================================================
# Registration Decorators
# ============================================================================

def register_renderer(cell_type: str):
    """
    Register a renderer for a cell type.

    The renderer function receives (cell, notebook_id) and returns
    a FastHTML component (Div, etc.).

    Example:
        @register_renderer("diagram")
        def render_diagram(cell, notebook_id):
            return Div(...)
    """
    def decorator(func: Callable) -> Callable:
        _renderers[cell_type] = func
        logger.debug(f"Registered renderer for cell type: {cell_type}")
        return func
    return decorator


def register_jupyter_serializer(cell_type: str):
    """
    Register a Jupyter serializer for a cell type.

    The serializer receives (cell) and returns a dict in Jupyter format.

    Example:
        @register_jupyter_serializer("diagram")
        def serialize_diagram(cell):
            return {"cell_type": "markdown", "source": cell.source, ...}
    """
    def decorator(func: Callable) -> Callable:
        _jupyter_serializers[cell_type] = func
        logger.debug(f"Registered Jupyter serializer for cell type: {cell_type}")
        return func
    return decorator


def register_jupyter_deserializer(func: Callable) -> Callable:
    """
    Register a Jupyter deserializer.

    Deserializers are tried in order. Each receives a jcell dict and should
    return a Cell if it can handle it, or None to try the next deserializer.

    Example:
        @register_jupyter_deserializer
        def deserialize_diagram(jcell):
            if jcell.get("metadata", {}).get("dialeng_diagram"):
                return DiagramCell(...)
            return None
    """
    _jupyter_deserializers.append(func)
    return func


def register_llm_converter(cell_type: str):
    """
    Register an LLM message converter for a cell type.

    The converter receives (cell) and returns a List[Dict] with
    role/content message format.

    Example:
        @register_llm_converter("diagram")
        def diagram_to_messages(cell):
            return [{"role": "user", "content": f"[Diagram]\\n{cell.source}"}]
    """
    def decorator(func: Callable) -> Callable:
        _llm_converters[cell_type] = func
        logger.debug(f"Registered LLM converter for cell type: {cell_type}")
        return func
    return decorator


# ============================================================================
# Dispatch Functions
# ============================================================================

def render_cell(cell: 'Cell', notebook_id: str) -> Any:
    """
    Render a cell to HTML using the appropriate renderer.

    Dispatches based on cell.cell_type. Falls back to code cell renderer
    for unknown types.

    Args:
        cell: Cell to render
        notebook_id: Parent notebook ID

    Returns:
        FastHTML component (Div, etc.)
    """
    cell_type = _get_cell_type_str(cell)

    if cell_type in _renderers:
        return _renderers[cell_type](cell, notebook_id)

    # Fall back to default renderers
    if cell_type == "code":
        return _default_code_renderer(cell, notebook_id)
    elif cell_type == "note":
        return _default_note_renderer(cell, notebook_id)
    elif cell_type == "prompt":
        return _default_prompt_renderer(cell, notebook_id)

    # Unknown type - use code renderer as fallback
    logger.warning(f"No renderer for cell type '{cell_type}', using code renderer")
    return _default_code_renderer(cell, notebook_id)


def cell_to_jupyter(cell: 'Cell') -> dict:
    """
    Convert a cell to Jupyter notebook format.

    Dispatches based on cell.cell_type.

    Args:
        cell: Cell to serialize

    Returns:
        Dict in Jupyter cell format
    """
    cell_type = _get_cell_type_str(cell)

    if cell_type in _jupyter_serializers:
        return _jupyter_serializers[cell_type](cell)

    # Use default serializers from document.serialization
    # Import here to avoid circular imports
    from document.serialization import _cell_to_jupyter
    return _cell_to_jupyter(cell)


def jupyter_to_cell(jcell: dict, index: int = 0) -> 'Cell':
    """
    Convert a Jupyter cell dict to internal Cell.

    Tries registered deserializers first, then falls back to default.

    Args:
        jcell: Jupyter cell dict
        index: Cell index (for logging)

    Returns:
        Cell instance
    """
    # Try registered deserializers
    for deserializer in _jupyter_deserializers:
        try:
            result = deserializer(jcell)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Deserializer failed: {e}")

    # Fall back to default
    from document.serialization import _jupyter_to_cell
    return _jupyter_to_cell(jcell, index)


def cell_to_llm_messages(cell: 'Cell') -> List[Dict]:
    """
    Convert a cell to LLM message format.

    Dispatches based on cell.cell_type.

    Args:
        cell: Cell to convert

    Returns:
        List of message dicts with "role" and "content" keys
    """
    cell_type = _get_cell_type_str(cell)

    if cell_type in _llm_converters:
        return _llm_converters[cell_type](cell)

    # Default converters for built-in types
    if cell_type == "code":
        content = f"```python\n{cell.source}\n```"
        if cell.output:
            content += f"\nOutput:\n```\n{cell.output}\n```"
        return [{"role": "user", "content": content}]

    elif cell_type == "note":
        return [{"role": "user", "content": cell.source}]

    elif cell_type == "prompt":
        msgs = [{"role": "user", "content": cell.source}]
        if cell.output:
            # Strip LLM Steps HTML from output before including in context
            # This prevents the LLM from seeing/reproducing our formatting HTML
            clean_output = _strip_llm_steps_html(cell.output)
            if clean_output.strip():
                msgs.append({"role": "assistant", "content": clean_output})
        return msgs

    # Unknown type - treat as user message
    logger.warning(f"No LLM converter for cell type '{cell_type}'")
    return [{"role": "user", "content": cell.source}]


# ============================================================================
# Helpers
# ============================================================================

def _strip_llm_steps_html(output: str) -> str:
    """
    Strip LLM Steps HTML from cell output before including in LLM context.

    The <details class="tool-steps-container"> blocks contain our formatting
    for tool call visualization. If included in context, the LLM might try
    to reproduce this HTML in its response, causing duplication issues.

    Args:
        output: Cell output that may contain LLM Steps HTML

    Returns:
        Output with LLM Steps HTML removed
    """
    import re

    if not output or '<details class="tool-steps-container">' not in output:
        return output

    # Remove the entire <details class="tool-steps-container">...</details> block
    # This regex matches the opening tag through the closing tag
    pattern = r'<details class="tool-steps-container">.*?</details>\s*'
    cleaned = re.sub(pattern, '', output, flags=re.DOTALL)

    return cleaned.strip()


def _get_cell_type_str(cell: 'Cell') -> str:
    """Get cell type as string (handles both enum and string)."""
    cell_type = cell.cell_type
    if hasattr(cell_type, 'value'):
        return cell_type.value
    return str(cell_type)


def get_registered_cell_types() -> List[str]:
    """Get list of all cell types with registered renderers."""
    return list(_renderers.keys())


def has_renderer(cell_type: str) -> bool:
    """Check if a renderer is registered for a cell type."""
    return cell_type in _renderers


def has_llm_converter(cell_type: str) -> bool:
    """Check if an LLM converter is registered for a cell type."""
    return cell_type in _llm_converters


# ============================================================================
# Default Renderers (imported lazily to avoid circular imports)
# ============================================================================

def _default_code_renderer(cell: 'Cell', notebook_id: str):
    """Default renderer for code cells."""
    from ui.cells.code_cell import CodeCellView
    return CodeCellView(cell, notebook_id)


def _default_note_renderer(cell: 'Cell', notebook_id: str):
    """Default renderer for note cells."""
    from ui.cells.note_cell import NoteCellView
    return NoteCellView(cell, notebook_id)


def _default_prompt_renderer(cell: 'Cell', notebook_id: str):
    """Default renderer for prompt cells."""
    from ui.cells.prompt_cell import PromptCellView
    return PromptCellView(cell, notebook_id)


# ============================================================================
# Register Built-in Types
# ============================================================================

# Note: Built-in types use the default renderers above, so we don't need
# to explicitly register them. The dispatch functions check for registered
# handlers first, then fall back to defaults.
#
# Extensions should register their handlers explicitly:
#
#     @register_renderer("diagram")
#     def render_diagram(cell, notebook_id):
#         ...
