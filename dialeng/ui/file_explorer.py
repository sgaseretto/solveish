"""
Dialeng UI - File Explorer Sidebar

Directory-aware file browser with breadcrumbs, folder navigation,
and notebook creation. Replaces the old flat file list.
"""

from fasthtml.common import *
from pathlib import Path
from typing import List, Tuple, Optional
from .icons import sprites as icon_sprites


def list_directory(path: Path, root: Path) -> Tuple[List[str], List[str]]:
    """List folders and notebooks in a directory.

    Args:
        path: Directory to list
        root: Root notebooks directory (for traversal protection)

    Returns:
        Tuple of (folder_names, notebook_names), sorted alphabetically
    """
    # Prevent directory traversal
    if not path.resolve().is_relative_to(root.resolve()):
        return [], []
    if not path.is_dir():
        return [], []

    folders = sorted(
        d.name for d in path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
        and d.name not in ('__pycache__', '.ipynb_checkpoints')
    )
    notebooks = sorted(
        f.stem for f in path.iterdir()
        if f.suffix == '.ipynb' and not f.name.startswith('.')
    )
    return folders, notebooks


def BreadcrumbNav(current_path: Path, root: Path, nb_id: str):
    """Breadcrumb navigation showing current path relative to root.

    Args:
        current_path: Currently browsed directory
        root: Root notebooks directory
        nb_id: Active notebook ID
    """
    try:
        rel = current_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(".")

    segments = []
    # Root segment
    accumulated = ""
    segments.append(
        Span("Home", cls="breadcrumb-segment" + (" current" if str(rel) == "." else ""),
             hx_get=f"/files?path=", hx_target="#file-list-content", hx_swap="innerHTML")
    )

    if str(rel) != ".":
        parts = rel.parts
        for i, part in enumerate(parts):
            accumulated = str(Path(*parts[:i+1]))
            is_last = i == len(parts) - 1
            segments.append(icon_sprites('chevron-right', sz=10))
            segments.append(
                Span(part,
                     cls="breadcrumb-segment" + (" current" if is_last else ""),
                     hx_get=f"/files?path={accumulated}",
                     hx_target="#file-list-content", hx_swap="innerHTML")
            )

    return Div(*segments, cls="breadcrumb-nav")


def FolderItem(name: str, path: str, nb_id: str):
    """A folder item in the file explorer.

    Args:
        name: Folder display name
        path: Relative path to folder
        nb_id: Active notebook ID
    """
    return Div(
        icon_sprites('folder', sz=16),
        Span(name, cls="file-explorer-item-name"),
        cls="file-explorer-item folder",
        hx_get=f"/files?path={path}",
        hx_target="#file-list-content",
        hx_swap="innerHTML",
    )


def FileItem(name: str, path: str, is_active: bool, nb_id: str,
             has_kernel: bool = False):
    """A notebook file item in the file explorer.

    Args:
        name: Notebook name (without .ipynb)
        path: Relative path for navigation (relative to NOTEBOOKS_DIR)
        is_active: Whether this is the currently open notebook
        nb_id: Active notebook ID
        has_kernel: Whether this notebook has a running kernel
    """
    icon_name = 'notebook-text' if is_active else 'notebook'
    cls_parts = ["file-explorer-item"]
    if is_active:
        cls_parts.append("active")
    if has_kernel:
        cls_parts.append("has-kernel")
    file_path = f"{path}/{name}" if path and path != "." else name
    # Use query param approach: /dialeng/?name=subfolder/test
    nb_name_param = f"{path}/{name}" if path and path != "." else name
    return Div(
        A(
            icon_sprites(icon_name, sz=16),
            Span(name, cls="file-explorer-item-name"),
            href=f"/dialeng/?name={nb_name_param}",
            cls="file-explorer-item-link",
        ),
        Button(
            icon_sprites('trash', sz=14),
            cls="file-explorer-delete-btn",
            onclick=f"showDeleteConfirm('{file_path}', '{name}')",
            title=f"Delete {name}",
        ),
        cls=" ".join(cls_parts),
    )


def FileListContent(path: Path, root: Path, active_notebook_id: str,
                    kernel_notebooks: set = None):
    """File list content (breadcrumbs + folders + files) for HTMX partial swaps.

    Also stores the current relative path in a hidden input so JS can read it
    when creating new items (the modal is rendered once at page load, so it
    can't have the path baked in).

    Args:
        path: Current directory to display
        root: Root notebooks directory
        active_notebook_id: Currently open notebook ID
        kernel_notebooks: Set of notebook IDs that have running kernels
    """
    if kernel_notebooks is None:
        kernel_notebooks = set()
    folders, notebooks = list_directory(path, root)

    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = ""

    # Hidden input that JS reads to know the currently browsed folder
    current_path_val = rel if rel and rel != "." else ""
    items = [
        Input(type="hidden", id="current-explorer-path", value=current_path_val),
        BreadcrumbNav(path, root, active_notebook_id),
    ]

    # Folders
    for folder in folders:
        folder_path = f"{rel}/{folder}" if rel and rel != "." else folder
        items.append(FolderItem(folder, folder_path, active_notebook_id))

    # Notebooks
    for nb_name in notebooks:
        nb_rel_path = f"{rel}/{nb_name}" if rel and rel != "." else nb_name
        is_active = nb_rel_path == active_notebook_id or nb_name == active_notebook_id
        items.append(FileItem(nb_name, rel, is_active, active_notebook_id,
                              has_kernel=nb_rel_path in kernel_notebooks or nb_name in kernel_notebooks))

    if not folders and not notebooks:
        items.append(Div("Empty folder", cls="file-explorer-item",
                         style="color: var(--text-muted); font-style: italic;"))

    return Div(*items, id="file-list-content")


def FileExplorerSidebar(current_path: Path, root: Path, active_notebook_id: str,
                        is_open: bool = True, kernel_notebooks: set = None):
    """File explorer sidebar component.

    Args:
        current_path: Directory to display
        root: Root notebooks directory
        active_notebook_id: Currently open notebook ID
        is_open: Whether sidebar starts open
        kernel_notebooks: Set of notebook IDs that have running kernels
    """
    cls = "file-explorer-sidebar"
    if is_open:
        cls += " file-explorer-open"

    return Aside(
        Div(
            icon_sprites('folder-open', sz=16),
            Span("Files", cls="file-explorer-title"),
            Button(icon_sprites('refresh-cw', sz=14),
                   onclick="refreshFileExplorer()",
                   title="Refresh file explorer"),
            Button(icon_sprites('file-plus', sz=14), onclick="toggleNewItemModal()",
                   title="New notebook or folder"),
            Button(icon_sprites('panel-left-close', sz=14), onclick="toggleFileExplorer()",
                   title="Collapse file explorer"),
            cls="file-explorer-header"
        ),
        Div(
            FileListContent(current_path, root, active_notebook_id,
                           kernel_notebooks=kernel_notebooks or set()),
            cls="file-explorer-content"
        ),
        id="file-explorer-sidebar",
        cls=cls,
    )


def NewItemModal():
    """Modal for creating new notebooks or folders.

    The current path is read dynamically from the hidden #current-explorer-path
    input inside #file-list-content, so it always reflects the folder the user
    is currently browsing (not the path at page-load time).
    """
    return Div(
        Div(
            H3("Create New"),
            Div(
                Div(
                    Label("Name", cls="setting-label", style="margin-bottom: 4px;"),
                    Input(type="text", name="item_name", placeholder="Enter name...",
                          id="new-item-name", cls="setting-text",
                          style="margin-bottom: 12px;"),
                    cls="new-item-name-row"
                ),
                Div(
                    Label("Type", cls="setting-label", style="margin-bottom: 4px;"),
                    Div(
                        Button(icon_sprites('notebook', sz=14), " Dialog",
                               cls="btn btn-sm new-item-type-btn active",
                               id="new-item-type-dialog",
                               onclick="selectNewItemType('dialog')"),
                        Button(icon_sprites('folder', sz=14), " Folder",
                               cls="btn btn-sm new-item-type-btn",
                               id="new-item-type-folder",
                               onclick="selectNewItemType('folder')"),
                        cls="new-item-type-group"
                    ),
                    cls="new-item-type-row"
                ),
                Input(type="hidden", id="new-item-type", value="dialog"),
                cls="new-item-form"
            ),
            Div(
                Button("Cancel", cls="btn btn-sm", onclick="toggleNewItemModal()"),
                Button("Create", cls="btn btn-sm btn-save",
                       onclick="createNewItem()"),
                cls="new-item-modal-actions"
            ),
            cls="new-item-panel"
        ),
        id="new-item-modal",
        cls="new-item-modal",
        onclick="if(event.target===this) toggleNewItemModal()"
    )


def DeleteConfirmModal():
    """Modal for confirming file deletion."""
    return Div(
        Div(
            H3("Delete File"),
            P("Are you sure you want to delete ", Span(id="delete-file-display", style="font-weight: 600;"), "?",
              style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 16px;"),
            Input(type="hidden", id="delete-file-path"),
            Div(
                Button("Cancel", cls="btn btn-sm", onclick="hideDeleteConfirm()"),
                Button(icon_sprites('trash', sz=14), " Delete", cls="btn btn-sm btn-cancel",
                       onclick="confirmDeleteFile()"),
                cls="new-item-modal-actions"
            ),
            cls="new-item-panel"
        ),
        id="delete-confirm-modal",
        cls="new-item-modal",
        onclick="if(event.target===this) hideDeleteConfirm()"
    )
