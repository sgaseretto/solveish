"""
Dialeng UI - Layout Components

Page layout and container components.
"""

from fasthtml.common import *
import json
from typing import List, Optional
from pathlib import Path
from .cells import CellView
from .controls import AddButtons
from .settings import SettingsSidebar, SettingsOverlay
from .outline import OutlineSidebar, OutlineToggleButton
from .kernel_modal import KernelToolbarButton, KernelModal
from .skill_creator import SkillCreatorButton, SkillCreatorModal
from .extensions import ExtensionManagerButton, ExtensionManagerModal
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


def ProviderSelect(nb, available_providers: list):
    """Backend-authored provider selector, including an unavailable saved value."""
    options = []
    for provider in available_providers:
        provider_id = provider["id"]
        authenticated = bool(provider.get("authenticated"))
        validation_state = provider.get("validation_state", "ready")
        label = provider.get("label", provider_id)
        if validation_state == "validating":
            label = f"{label} (validating...)"
        elif not authenticated:
            label = f"{label} (unavailable)"
        options.append(Option(
            label,
            value=provider_id,
            selected=nb.llm_provider == provider_id,
            disabled=not authenticated,
        ))
    return Select(
        *options,
        cls="model-select provider-select",
        name="provider",
        id="provider-select",
        hx_post=f"/dialeng/{nb.id}/provider",
        hx_target="#model-select",
        hx_swap="outerHTML",
        title="LLM provider",
        style="display: none;" if nb.dialog_mode == "mock" else "",
    )


def ModelSelect(nb, available_models: list):
    """Backend-authored model selector for the notebook's selected provider."""
    return Select(
        *[Option(label, value=model_id, selected=nb.model == model_id)
          for model_id, label in available_models],
        cls="model-select",
        name="model",
        id="model-select",
        hx_post=f"/dialeng/{nb.id}/model",
        hx_swap="none",
        title="Model",
        style="display: none;" if nb.dialog_mode == "mock" else "",
    )


def _notebook_usage_total(nb) -> int:
    return sum(
        int(((getattr(cell, "llm_execution", None) or {}).get("usage") or {}).get("total_tokens") or 0)
        for cell in nb.cells
    )


def WorkspacePage(root: Path, config: Optional[DialengConfig] = None,
                  kernel_notebooks: Optional[set] = None,
                  notebook_deleted: bool = False):
    """Render the project shell without opening or creating a notebook."""
    title = "Notebook deleted" if notebook_deleted else "No notebook open"
    description = (
        "The notebook and its attached kernel were removed. Select another file "
        "from the explorer or create a new notebook."
        if notebook_deleted else
        "Select a notebook from the explorer or create one to start working."
    )
    return Titled(
        "Dialeng Workspace",
        icon_sprites,
        Div(
            FileExplorerSidebar(
                root,
                root,
                "",
                active_file_relpath="",
                kernel_notebooks=kernel_notebooks or set(),
            ),
            Div(
                Div(
                    Div(
                        Button(
                            icon_sprites('panel-left-open', sz=16),
                            cls="btn btn-sm file-explorer-toggle-btn",
                            onclick="toggleFileExplorer()",
                            title="Toggle file explorer (Ctrl+Shift+E)",
                        ),
                        Span(icon_sprites('notebook', sz=16), "Dialeng", cls="toolbar-title"),
                        cls="toolbar-left",
                    ),
                    Div(
                        Button(
                            icon_sprites('settings', sz=16),
                            cls="btn btn-sm settings-btn",
                            id="settings-btn",
                            onclick="toggleSettings()",
                            title="Settings",
                        ),
                        ExtensionManagerButton(),
                        Div(id="extension-toolbar-items", cls="extension-toolbar-items"),
                        SkillCreatorButton(bool(config and config.skills_enabled and config.skills_creator_enabled)),
                        Button(
                            icon_sprites('sun', sz=16),
                            cls="theme-toggle",
                            id="theme-toggle",
                            onclick="toggleTheme()",
                            title="Toggle light/dark theme",
                        ),
                        cls="toolbar-right",
                    ),
                    cls="toolbar-container",
                ),
                Div(id="extension-status-items", cls="extension-status-items"),
                Div(
                    Div(icon_sprites('notebook-text', sz=34), cls="workspace-empty-icon"),
                    H2(title),
                    P(description),
                    Button(
                        icon_sprites('file-plus', sz=15),
                        " New notebook",
                        cls="btn btn-primary",
                        onclick="toggleNewItemModal()",
                    ),
                    cls="workspace-empty-card",
                ),
                Script("window.NOTEBOOK_ID = ''; window.NOTEBOOK_TERMINAL_STATE = true;"),
                cls="container workspace-container",
            ),
            cls="main-layout",
        ),
        Div(id="extension-panel-host", cls="extension-panel-host"),
        NewItemModal(),
        DeleteConfirmModal(),
        SettingsOverlay() if config else None,
        SettingsSidebar(config) if config else None,
        SkillCreatorModal() if config and config.skills_enabled and config.skills_creator_enabled else None,
        ExtensionManagerModal(),
    )


def NotebookPage(nb, notebook_list: List[str], available_dialog_modes: list, available_models: list,
                 available_providers: Optional[list] = None,
                 config: Optional[DialengConfig] = None, shfmt_available: bool = True,
                 colab_enabled: bool = False, colab_authenticated: bool = False,
                 colab_account_email: Optional[str] = None,
                 notebooks_dir: Optional[Path] = None, kernel_alive: bool = False,
                 kernel_notebooks: Optional[set] = None, has_craft_code: bool = False,
                 initial_revisions: Optional[dict] = None):
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
                                           active_file_relpath="",
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

    return Titled(f"{nb.title} - Dialeng",
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
                        SkillCreatorButton(bool(config and config.skills_enabled and config.skills_creator_enabled)),
                        ExtensionManagerButton(),
                        Button(icon_sprites('settings', sz=16), cls="btn btn-sm settings-btn", id="settings-btn",
                               onclick="toggleSettings()", title="Settings"),
                        Button(icon_sprites('sun', sz=16), cls="theme-toggle", id="theme-toggle",
                               onclick="toggleTheme()", title="Toggle light/dark theme"),
                        Select(
                            *[Option(label, value=mode_id, selected=nb.dialog_mode == mode_id)
                              for mode_id, label in available_dialog_modes],
                            cls="mode-select", name="mode", id="mode-select",
                            hx_post=f"/dialeng/{nb.id}/mode", hx_swap="none", title="AI Mode",
                            onchange="toggleModelSelect(this.value)"
                        ),
                        ProviderSelect(nb, available_providers or []),
                        ModelSelect(nb, available_models),
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
                               hx_post=f"/dialeng/{nb.id}/kernel/restart", hx_target="#status", title="Restart kernel"),
                        Button(icon_sprites('square', sz=14), " Cancel", cls="btn btn-sm btn-cancel-all", id="cancel-all-btn",
                               onclick="cancelAllExecution()", title="Cancel running cell and clear queue (Esc Esc)",
                               style="display: none;"),
                        Button(icon_sprites('save', sz=14), cls="btn btn-sm btn-save", id="save-btn",
                               hx_post=f"/dialeng/{nb.id}/save", hx_target="#status", title="Save (Ctrl+S)"),
                        Span(
                            "Modified" if nb.modified else "Saved",
                            id="notebook-save-state",
                            cls=f"notebook-save-state {'modified' if nb.modified else 'saved'}",
                        ),
                        Span(
                            f"{_notebook_usage_total(nb):,} tok",
                            id="notebook-llm-usage",
                            cls="notebook-llm-usage",
                            title="Cumulative prompt token usage stored in this notebook",
                        ),
                        Button(icon_sprites('download', sz=14), cls="btn btn-sm",
                               hx_get=f"/dialeng/{nb.id}/export", title="Download .ipynb"),
                        Button(icon_sprites('activity', sz=14), cls="btn btn-sm btn-icon",
                               onclick="showKernelDiagnostics()", title="Kernel diagnostics"),
                        Button(icon_sprites('keyboard', sz=14), cls="btn btn-sm",
                               onclick="toggleKeyboardShortcuts()", title="Keyboard shortcuts (?)"),
                        # Extension toolbar items
                        Div(id="extension-toolbar-items", cls="extension-toolbar-items"),
                        *[reg.renderer(nb, config)
                          for reg in sorted(registry.toolbar_items.values(), key=lambda r: r.order)],
                        cls="toolbar-right"
                    ),
                    cls="toolbar-container"
                ),
                Div(id="extension-status-items", cls="extension-status-items"),
                Div(id="status", aria_live="polite", aria_atomic="true"),
                AllCells(nb),
                Script(f"""
                    window.NOTEBOOK_ID = '{nb.id}';
                    window.INITIAL_REVISIONS = {json.dumps(initial_revisions or {})};
                    window.KERNEL_ALIVE = {'true' if kernel_alive else 'false'};
                    window.KERNEL_STATE = {{
                        kernel: {{
                            can_run: {'true' if kernel_alive else 'false'},
                            is_alive: {'true' if kernel_alive else 'false'},
                            display_state: {'"connected"' if kernel_alive else '"disconnected"'}
                        }},
                        queue: {{
                            running_cell_id: null,
                            queued_cell_ids: []
                        }},
                        setup: {{ is_active: false }},
                        auth: {{ enabled: {'true' if colab_enabled else 'false'}, authenticated: {'true' if colab_authenticated else 'false'} }}
                    }};
                    window.HAS_CRAFT_CODE = {'true' if has_craft_code else 'false'};
                    window.AUTOSAVE_ENABLED = {'true' if config.autosave_enabled else 'false'};
                """),
                Script(f"document.addEventListener('DOMContentLoaded', () => connectWebSocket('{nb.id}'));"),
                Div(id="ephemeral"),  # Container for dialoghelper script injection (matches add_scr default)
                cls="container"
            ),
            cls="main-layout"
        ),
        Div(id="extension-panel-host", cls="extension-panel-host"),
        # New item modal (for file explorer)
        NewItemModal(),
        # Delete confirmation modal (for file explorer)
        DeleteConfirmModal(),
        # Kernel selection modal (overlay style, hidden by default)
        KernelModal(nb.id, nb.kernel_type, colab_authenticated, nb.colab_runtime_type, colab_account_email),
        # Auto-show kernel modal if no kernel is attached (runs after modal is in DOM)
        Script("if(!window.KERNEL_ALIVE){var o=document.getElementById('kernel-modal-overlay');if(o)o.classList.add('visible');}") if not kernel_alive else None,
        # Settings sidebar and overlay (outside main layout - overlay style)
        SettingsOverlay() if config else None,
        SettingsSidebar(config) if config else None,
        SkillCreatorModal() if config and config.skills_enabled and config.skills_creator_enabled else None,
        ExtensionManagerModal(),
    )
