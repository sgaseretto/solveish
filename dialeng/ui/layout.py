"""
Dialeng UI - Layout Components

Page layout and container components.
"""

from fasthtml.common import *
from typing import List, Optional
from pathlib import Path
from .cells import CellView
from .controls import AddButtons
from .settings import SettingsSidebar, SettingsOverlay
from .outline import OutlineSidebar, OutlineToggleButton
from .kernel_modal import KernelToolbarButton, KernelModal
from .file_explorer import FileExplorerSidebar, NewItemModal, DeleteConfirmModal
from .icons import sprites as icon_sprites
from dialeng.services.dialeng_config import DialengConfig
from dialeng.core.registry import registry


def AllCellsContent(nb):
    """Returns just the cell content (for innerHTML swaps).

    Args:
        nb: Notebook instance

    Returns:
        Div containing all cells without wrapper ID (for content swaps)
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i + 1, nb.id)])
    return Div(*items)


def AllCells(nb):
    """Returns all cells wrapped in a container with ID.

    Args:
        nb: Notebook instance

    Returns:
        Div with id="cells" containing all cells
    """
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i + 1, nb.id)])
    return Div(*items, id="cells")


def NotebookPage(nb, notebook_list: List[str], available_dialog_modes: list, available_models: list,
                 config: Optional[DialengConfig] = None, shfmt_available: bool = True,
                 colab_enabled: bool = False, colab_authenticated: bool = False,
                 notebooks_dir: Optional[Path] = None, kernel_alive: bool = False,
                 kernel_notebooks: Optional[set] = None):
    """Render the complete notebook page.

    Args:
        nb: Notebook instance
        notebook_list: List of notebook IDs for the file list (legacy, used as fallback)
        available_dialog_modes: List of (mode_id, label) tuples
        available_models: List of (model_id, label) tuples
        config: Optional DialengConfig for settings sidebar
        shfmt_available: Whether shfmt binary is installed (for safe mode)
        colab_enabled: Whether Colab integration is configured
        colab_authenticated: Whether user is authenticated with Google
        notebooks_dir: Root notebooks directory for file explorer
    """
    # File explorer sidebar (replaces old flat file list)
    file_sidebar = None
    if notebooks_dir:
        nb_path = nb.path
        nb_dir = Path(nb_path).parent if nb_path else notebooks_dir
        file_sidebar = FileExplorerSidebar(nb_dir, notebooks_dir, nb.id,
                                           kernel_notebooks=kernel_notebooks or set())

    # Initial kernel dot state
    dot_cls = "kernel-dot connected" if kernel_alive else "kernel-dot"
    dot_title = "Kernel: idle" if kernel_alive else "Kernel: not connected"

    # Display settings CSS custom properties
    display_style = ""
    if config:
        notebook_width = config.display_notebook_width
        font_size = config.display_font_size
        btn_size = config.display_button_size
        btn_padding = {"compact": "3px 6px", "normal": "5px 10px", "large": "8px 14px"}.get(btn_size, "5px 10px")
        btn_font = {"compact": "0.75rem", "normal": "0.8rem", "large": "0.9rem"}.get(btn_size, "0.8rem")
        btn_sm_padding = {"compact": "2px 5px", "normal": "4px 8px", "large": "6px 12px"}.get(btn_size, "4px 8px")
        display_style = f":root {{ --notebook-width: {notebook_width}px; --base-font-size: {font_size}px; --btn-padding: {btn_padding}; --btn-font-size: {btn_font}; --btn-sm-padding: {btn_sm_padding}; }}"

    return Titled(
        f"{nb.title} - Dialeng",
        # Display settings as CSS custom properties
        Style(display_style) if display_style else None,
        # Lucide icon sprite sheet (must be in DOM before any <use href="#..."> references)
        icon_sprites,
        # Main layout wrapper - flex container for sidebars + content
        Div(
            # File explorer sidebar (push style - on left)
            file_sidebar,
            # Outline sidebar (push style - on left)
            OutlineSidebar(nb.id, is_open=False),
            # Main content container
            Div(
                # Toolbar container (rounded card)
                Div(
                    Div(
                        # Left group: sidebar toggles + notebook name
                        Button(icon_sprites('panel-left-open', sz=16), cls="btn btn-sm file-explorer-toggle-btn",
                               onclick="toggleFileExplorer()", title="Toggle file explorer (Ctrl+Shift+E)"),
                        OutlineToggleButton(),
                        Span(icon_sprites('notebook', sz=16), nb.title,
                             Span(cls=dot_cls, id="kernel-dot", title=dot_title),
                             cls="toolbar-title"),
                        cls="toolbar-left"
                    ),
                    Div(
                        # Right group: controls
                        Button(icon_sprites('sun', sz=16), cls="theme-toggle", id="theme-toggle",
                               onclick="toggleTheme()", title="Toggle light/dark theme"),
                        Select(
                            *[Option(label, value=mode_id, selected=nb.dialog_mode == mode_id)
                              for mode_id, label in available_dialog_modes],
                            cls="mode-select", name="mode", id="mode-select",
                            hx_post=f"/notebook/{nb.id}/mode", hx_swap="none", title="AI Mode",
                            onchange="toggleModelSelect(this.value)"
                        ),
                        Select(
                            *[Option(label, value=model_id, selected=nb.model == model_id)
                              for model_id, label in available_models],
                            cls="model-select", name="model", id="model-select",
                            hx_post=f"/notebook/{nb.id}/model", hx_swap="none", title="Model",
                            style="display: none;" if nb.dialog_mode == "mock" else ""
                        ),
                        # Safe mode toggle for shell commands
                        Button(
                            icon_sprites('shield-check' if nb.safe_mode else 'shield-off', sz=14),
                            cls=f"btn btn-sm btn-icon safe-mode-btn{' active' if nb.safe_mode else ''}",
                            id="safe-mode-toggle",
                            onclick=f"toggleSafeMode('{nb.id}')",
                            disabled=not shfmt_available,
                            title="Safe Mode: Validate shell commands against allowlist" if shfmt_available else "Safe Mode unavailable - install shfmt"
                        ),
                        # Kernel selector (compact toolbar button → opens modal)
                        KernelToolbarButton(nb),
                        Button(icon_sprites('rotate-ccw', sz=14), cls="btn btn-sm",
                               hx_post=f"/notebook/{nb.id}/kernel/restart", hx_target="#status", title="Restart kernel"),
                        Button(icon_sprites('square', sz=14), " Cancel", cls="btn btn-sm btn-cancel-all", id="cancel-all-btn",
                               onclick="cancelAllExecution()", title="Cancel running cell and clear queue (Esc Esc)",
                               style="display: none;"),
                        Button(icon_sprites('save', sz=14), cls="btn btn-sm btn-save", id="save-btn",
                               hx_post=f"/notebook/{nb.id}/save", hx_target="#status", title="Save (Ctrl+S)"),
                        Button(icon_sprites('download', sz=14), cls="btn btn-sm",
                               hx_get=f"/notebook/{nb.id}/export", title="Download .ipynb"),
                        Button(icon_sprites('settings', sz=16), cls="btn btn-sm settings-btn", id="settings-btn",
                               onclick="toggleSettings()", title="Settings"),
                        # Extension toolbar items
                        *[reg.renderer(nb, config)
                          for reg in sorted(registry.toolbar_items.values(), key=lambda r: r.order)],
                        cls="toolbar-right"
                    ),
                    cls="toolbar-container"
                ),
                Div(id="status"),
                AllCells(nb),
                Script(f"window.NOTEBOOK_ID = '{nb.id}';"),
                Script(f"document.addEventListener('DOMContentLoaded', () => connectWebSocket('{nb.id}'));"),
                Div(id="ephemeral"),  # Container for dialoghelper script injection (matches add_scr default)
                cls="container"
            ),
            cls="main-layout"
        ),
        # New item modal (for file explorer)
        NewItemModal(str(Path(nb_path).parent.relative_to(notebooks_dir)) if nb_path and notebooks_dir else ""),
        # Delete confirmation modal (for file explorer)
        DeleteConfirmModal(),
        # Kernel selection modal (overlay style, hidden by default)
        KernelModal(nb.id, nb.kernel_type, colab_authenticated, nb.colab_runtime_type),
        # Settings sidebar and overlay (outside main layout - overlay style)
        SettingsOverlay() if config else None,
        SettingsSidebar(config) if config else None
    )
