"""
Dialeng UI - File Explorer Sidebar

Directory-aware file browser with breadcrumbs, folder navigation,
and notebook/file navigation. Replaces the old flat file list.
"""

from fasthtml.common import *
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import quote
from dialeng.notebook_id import nb_id_from_relpath
from .icons import sprites as icon_sprites


def list_directory(path: Path, root: Path) -> Tuple[List[str], List[str], List[str]]:
    """List folders, notebooks, and plain files in a directory.

    Args:
        path: Directory to list
        root: Root notebooks directory (for traversal protection)

    Returns:
        Tuple of (folder_names, notebook_names, other_file_names), sorted alphabetically
    """
    # Prevent directory traversal
    if not path.resolve().is_relative_to(root.resolve()):
        return [], [], []
    if not path.is_dir():
        return [], [], []

    folders = sorted(
        d.name for d in path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
        and d.name not in ('__pycache__', '.ipynb_checkpoints')
    )
    notebooks = sorted(
        f.stem for f in path.iterdir()
        if f.suffix == '.ipynb' and not f.name.startswith('.')
    )
    other_files = sorted(
        f.name for f in path.iterdir()
        if f.is_file() and f.suffix != '.ipynb' and not f.name.startswith('.')
    )
    return folders, notebooks, other_files


def BreadcrumbNav(current_path: Path, root: Path, nb_id: str, active_file_relpath: str = ""):
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
    active_query = ""
    if nb_id:
        active_query += f"&active_notebook_id={nb_id}"
    if active_file_relpath:
        active_query += f"&active_file_relpath={quote(active_file_relpath)}"
    # Root segment
    accumulated = ""
    segments.append(
        Span("Home", cls="breadcrumb-segment" + (" current" if str(rel) == "." else ""),
             hx_get=f"/files?path={active_query}", hx_target="#file-list-content", hx_swap="innerHTML")
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
                     hx_get=f"/files?path={accumulated}{active_query}",
                     hx_target="#file-list-content", hx_swap="innerHTML")
            )

    return Div(*segments, cls="breadcrumb-nav")


def FolderItem(name: str, path: str, nb_id: str, active_file_relpath: str = ""):
    """A folder item in the file explorer.

    Args:
        name: Folder display name
        path: Relative path to folder
        nb_id: Active notebook ID
    """
    active_query = ""
    if nb_id:
        active_query += f"&active_notebook_id={nb_id}"
    if active_file_relpath:
        active_query += f"&active_file_relpath={quote(active_file_relpath)}"
    return Div(
        icon_sprites('folder', sz=16),
        Span(name, cls="file-explorer-item-name"),
        cls="file-explorer-item folder",
        hx_get=f"/files?path={path}{active_query}",
        hx_target="#file-list-content",
        hx_swap="innerHTML",
    )


def NotebookItem(name: str, path: str, is_active: bool, has_kernel: bool = False):
    """A notebook file item in the file explorer.

    Args:
        name: Notebook name (without .ipynb)
        path: Relative path for navigation (relative to NOTEBOOKS_DIR)
        is_active: Whether this is the currently open notebook
        has_kernel: Whether this notebook has a running kernel
    """
    icon_name = 'notebook-text' if is_active else 'notebook'
    cls_parts = ["file-explorer-item"]
    if is_active:
        cls_parts.append("active")
    if has_kernel:
        cls_parts.append("has-kernel")
    file_path = f"{path}/{name}.ipynb" if path and path != "." else f"{name}.ipynb"
    # Use query param approach: /dialeng/?name=subfolder/test
    nb_name_param = f"{path}/{name}" if path and path != "." else name
    notebook_id = nb_id_from_relpath(nb_name_param)
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
        data_notebook_id=notebook_id,
        cls=" ".join(cls_parts),
    )


def FileItem(name: str, path: str, is_active: bool):
    """A plain file item in the file explorer."""
    cls_parts = ["file-explorer-item"]
    if is_active:
        cls_parts.append("active")
    rel_path = f"{path}/{name}" if path and path != "." else name
    href = f"/dialeng/file?path={quote(rel_path)}"
    icon_name = "file-text" if any(name.endswith(ext) for ext in (".md", ".txt", ".json", ".py", ".js", ".css", ".html", ".toml", ".yaml", ".yml", ".xml", ".sql", ".sh", ".ts")) else "file"
    return Div(
        A(
            icon_sprites(icon_name, sz=16),
            Span(name, cls="file-explorer-item-name"),
            href=href,
            cls="file-explorer-item-link",
        ),
        Button(
            icon_sprites('trash', sz=14),
            cls="file-explorer-delete-btn",
            onclick=f"showDeleteConfirm('{rel_path}', '{name}')",
            title=f"Delete {name}",
        ),
        data_file_relpath=rel_path,
        cls=" ".join(cls_parts),
    )


def FileListContent(path: Path, root: Path, active_notebook_id: str,
                    active_file_relpath: str = "",
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
    folders, notebooks, other_files = list_directory(path, root)

    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = ""

    # Hidden input that JS reads to know the currently browsed folder
    current_path_val = rel if rel and rel != "." else ""
    items = [
        Input(type="hidden", id="current-explorer-path", value=current_path_val),
        Input(type="hidden", id="current-explorer-active-notebook", value=active_notebook_id),
        Input(type="hidden", id="current-explorer-active-file", value=active_file_relpath),
        BreadcrumbNav(path, root, active_notebook_id, active_file_relpath),
    ]

    # Folders
    for folder in folders:
        folder_path = f"{rel}/{folder}" if rel and rel != "." else folder
        items.append(FolderItem(folder, folder_path, active_notebook_id, active_file_relpath))

    # Notebooks
    for nb_name in notebooks:
        nb_rel_path = f"{rel}/{nb_name}" if rel and rel != "." else nb_name
        explorer_nb_id = nb_id_from_relpath(nb_rel_path)
        is_active = explorer_nb_id == active_notebook_id
        items.append(NotebookItem(nb_name, rel, is_active,
                                  has_kernel=explorer_nb_id in kernel_notebooks))

    for file_name in other_files:
        file_rel_path = f"{rel}/{file_name}" if rel and rel != "." else file_name
        items.append(FileItem(file_name, rel, file_rel_path == active_file_relpath))

    if not folders and not notebooks and not other_files:
        items.append(Div("Empty folder", cls="file-explorer-item",
                         style="color: var(--text-muted); font-style: italic;"))

    return Div(*items, id="file-list-content")


def FileExplorerSidebar(current_path: Path, root: Path, active_notebook_id: str,
                        active_file_relpath: str = "",
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
                           active_file_relpath=active_file_relpath,
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
