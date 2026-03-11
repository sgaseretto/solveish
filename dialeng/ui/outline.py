"""
Dialeng UI - Outline Sidebar Components

Sidebar that shows notebook structure: headings, variables, and functions.
Unlike the settings sidebar (overlay), this sidebar pushes the main content.
"""

from fasthtml.common import *
from fastlucide import SvgSprites
from typing import List, Dict, Any, Optional
import re


# Create sprite manager for icons
sprites = SvgSprites('lc-')


def extract_headings_from_markdown(source: str) -> List[Dict[str, Any]]:
    """Extract markdown headings from cell source.

    Args:
        source: Markdown cell source

    Returns:
        List of dicts with 'level', 'text', and 'indent' keys
    """
    headings = []
    for line in source.split('\n'):
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            # Calculate indent based on heading level (2 spaces per level after h1)
            indent = (level - 1) * 2
            headings.append({
                'level': level,
                'text': text,
                'indent': indent
            })
    return headings


def OutlineSection(title: str, items: List[Any], icon_name: str = None, empty_message: str = "No items found"):
    """Collapsible section in the outline sidebar.

    Args:
        title: Section title (e.g., "Headings", "Variables")
        items: List of Li elements to display
        icon_name: Optional Lucide icon name for the section header
        empty_message: Message shown when no items

    Returns:
        Div containing the section
    """
    icon = sprites(icon_name) if icon_name else None

    if not items:
        items = [Li(Em(empty_message, cls="text-muted"), cls="outline-empty")]

    return Details(
        Summary(
            icon,
            Span(title, cls="outline-section-title"),
            cls="outline-section-header"
        ),
        Ul(*items, cls="outline-section-list"),
        cls="outline-section",
        open=True
    )


def HeadingItem(text: str, cell_id: str, level: int = 2):
    """Clickable heading item that scrolls to the cell.

    Args:
        text: Heading text
        cell_id: ID of the cell containing the heading
        level: Heading level (1-6) for indentation

    Returns:
        Li element with click handler
    """
    indent_style = f"padding-left: {(level - 1) * 12}px;"
    return Li(
        A(
            text,
            href=f"#cell-{cell_id}",
            onclick=f"scrollToCell('{cell_id}'); return false;",
            cls=f"outline-heading outline-heading-{level}",
            style=indent_style
        ),
        cls="outline-item"
    )


def VariableItem(name: str, var_type: str, value_preview: str = None, cell_id: str = None):
    """Variable item showing name, type, and optional preview.

    Args:
        name: Variable name
        var_type: Type string (e.g., "int", "DataFrame")
        value_preview: Optional short preview of the value
        cell_id: Optional cell ID where the variable was defined (makes it clickable)

    Returns:
        Li element with variable info
    """
    content = Div(
        Div(
            Span(name, cls="outline-var-name"),
            Span(var_type, cls="outline-var-type"),
            cls="outline-var-row"
        ),
        Span(value_preview, cls="outline-var-preview") if value_preview else None,
    )

    if cell_id:
        # Wrap in an anchor to make it clickable
        return Li(
            A(
                content,
                href=f"#cell-{cell_id}",
                onclick=f"scrollToCell('{cell_id}'); return false;",
                cls="outline-var-link"
            ),
            cls="outline-item outline-var-item outline-clickable"
        )
    else:
        return Li(
            content,
            cls="outline-item outline-var-item"
        )


def FunctionItem(name: str, signature: str = None, cell_id: str = None):
    """Function item showing name and optional signature.

    Args:
        name: Function name
        signature: Optional function signature (already includes parentheses)
        cell_id: Optional cell ID where the function was defined (makes it clickable)

    Returns:
        Li element with function info
    """
    # Signature from kernel introspection already includes parentheses e.g. "(x, y)"
    display = f"{name}{signature}" if signature else f"{name}()"

    if cell_id:
        return Li(
            A(
                Span(display, cls="outline-func-name"),
                href=f"#cell-{cell_id}",
                onclick=f"scrollToCell('{cell_id}'); return false;",
                cls="outline-func-link"
            ),
            cls="outline-item outline-func-item outline-clickable"
        )
    else:
        return Li(
            Span(display, cls="outline-func-name"),
            cls="outline-item outline-func-item"
        )


def OutlineSidebar(notebook_id: str, headings: List[Dict] = None,
                   variables: List[Dict] = None, functions: List[Dict] = None,
                   is_open: bool = True):
    """Main outline sidebar component.

    This sidebar shows the notebook structure and pushes the main content
    when open (unlike the settings sidebar which overlays).

    Args:
        notebook_id: ID of the current notebook
        headings: List of dicts with 'text', 'cell_id', 'level' keys
        variables: List of dicts with 'name', 'type', 'preview' keys
        functions: List of dicts with 'name', 'signature' keys
        is_open: Whether the sidebar starts open

    Returns:
        Aside element containing the outline sidebar
    """
    headings = headings or []
    variables = variables or []
    functions = functions or []

    # Build heading items
    heading_items = [
        HeadingItem(h['text'], h['cell_id'], h.get('level', 2))
        for h in headings
    ]

    # Build variable items (include cell_id if available for click-to-scroll)
    variable_items = [
        VariableItem(v['name'], v['type'], v.get('preview'), v.get('cell_id'))
        for v in variables
    ]

    # Build function items (include cell_id if available for click-to-scroll)
    function_items = [
        FunctionItem(f['name'], f.get('signature'), f.get('cell_id'))
        for f in functions
    ]

    return Aside(
        # Header with title and close button
        Div(
            Span(sprites('table-of-contents'), "Outline", cls="outline-title"),
            Button(
                sprites('x'),
                cls="outline-close-btn",
                onclick="toggleOutline()",
                title="Close outline"
            ),
            cls="outline-header"
        ),
        # Scrollable content
        Div(
            OutlineSection("Headings", heading_items, "list", "No headings found"),
            OutlineSection("Variables", variable_items, "database", "No variables found"),
            OutlineSection("Functions", function_items, "code", "No functions found"),
            cls="outline-content"
        ),
        # Include the SVG sprite definitions
        sprites,
        id="outline-sidebar",
        cls=f"outline-sidebar {'outline-open' if is_open else ''}",
        hx_get=f"/notebook/{notebook_id}/outline",
        hx_trigger="outline-refresh from:body",
        hx_target="#outline-sidebar",
        hx_swap="outerHTML"
    )


def OutlineToggleButton():
    """Button to toggle the outline sidebar.

    Returns:
        Button element with click handler
    """
    return Button(
        sprites('table-of-contents'),
        sprites,  # Include sprites for the button icon
        cls="btn btn-sm outline-toggle-btn",
        id="outline-toggle-btn",
        onclick="toggleOutline()",
        title="Toggle outline (Ctrl+Shift+O)"
    )
