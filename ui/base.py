"""
Dialeng UI - Base Utilities

Common utilities shared across UI components.
"""

from fasthtml.common import *


def get_collapse_class(level: int) -> str:
    """Get CSS class for collapse level.

    Args:
        level: CollapseLevel value (0=expanded, 1=scrollable, 2=summary)

    Returns:
        CSS class string for the collapse state
    """
    if level == 1:
        return "collapse-scrollable"
    if level == 2:
        return "collapse-summary"
    return ""


def get_cell_state_classes(cell) -> str:
    """Build CSS class string reflecting cell metadata states."""
    classes = ["cell"]
    if cell.collapsed: classes.append("collapsed")
    if cell.skipped: classes.append("skipped")
    if cell.pinned: classes.append("pinned")
    if cell.is_exported: classes.append("exported")
    return " ".join(classes)
