"""Standalone file editor page and fragments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fasthtml.common import *

from .file_explorer import FileExplorerSidebar, NewItemModal, DeleteConfirmModal
from .icons import sprites as icon_sprites
from .settings import SettingsOverlay, SettingsSidebar
from dialeng.services.dialeng_config import DialengConfig
from dialeng.services.file_editor_service import FileOpenResult


def _display_name(rel_path: str) -> str:
    return Path(rel_path).name


def _file_icon(result: FileOpenResult) -> str:
    if result.status == "editable":
        return "file-code-2"
    if result.status == "noneditable":
        return "file"
    if result.status == "locked":
        return "lock"
    return "file-text"


def FileEditorPane(result: FileOpenResult):
    """Return the editor fragment or a non-editable/locked message."""
    title = _display_name(result.rel_path or (str(result.abs_path.name) if result.abs_path else "File"))

    if result.status == "editable":
        return Div(
            Input(type="hidden", id="file-editor-state", value="editable"),
            Input(type="hidden", id="file-editor-language", value=result.language),
            Input(type="hidden", id="file-editor-path", value=result.rel_path),
            Textarea(result.content, id="file-source", style="display:none;"),
            Div(id="file-monaco", cls="file-monaco"),
            id="file-editor-fragment",
            cls="file-editor-fragment",
        )

    if result.status == "locked":
        message = result.reason or "This file is already being edited by another Dialeng session."
    elif result.status == "noneditable":
        message = result.reason or "This file cannot be edited in Dialeng."
    elif result.status == "missing":
        message = result.reason or "File not found."
    else:
        message = "This resource cannot be opened in the file editor."

    return Div(
        Input(type="hidden", id="file-editor-state", value=result.status),
        Input(type="hidden", id="file-editor-path", value=result.rel_path),
        Div(
            icon_sprites(_file_icon(result), sz=18),
            H3(title, cls="file-editor-message-title"),
            P(message, cls="file-editor-message-body"),
            Button("Retry", cls="btn btn-sm", onclick="reloadFileEditorView()")
            if result.status == "locked" else None,
            cls=f"file-editor-message {result.status}"
        ),
        id="file-editor-fragment",
        cls="file-editor-fragment",
    )


def FileEditorPage(
    *,
    rel_path: str,
    root_dir: Path,
    config: Optional[DialengConfig] = None,
):
    """Render the standalone file editor page shell."""
    target_path = root_dir / rel_path
    current_dir = target_path.parent if target_path.parent.exists() else root_dir

    display_style = ""
    if config:
        notebook_width = config.display_notebook_width
        font_size = config.display_font_size
        btn_size = config.display_button_size
        btn_padding = {"compact": "3px 6px", "normal": "5px 10px", "large": "8px 14px"}.get(btn_size, "5px 10px")
        btn_font = {"compact": "0.75rem", "normal": "0.8rem", "large": "0.9rem"}.get(btn_size, "0.8rem")
        btn_sm_padding = {"compact": "2px 5px", "normal": "4px 8px", "large": "6px 12px"}.get(btn_size, "4px 8px")
        display_style = f":root {{ --notebook-width: {notebook_width}px; --base-font-size: {font_size}px; --btn-padding: {btn_padding}; --btn-font-size: {btn_font}; --btn-sm-padding: {btn_sm_padding}; }}"

    file_sidebar = FileExplorerSidebar(
        current_dir,
        root_dir,
        active_notebook_id="",
        active_file_relpath=rel_path,
        kernel_notebooks=set(),
    )

    return Titled(
        f"{_display_name(rel_path)} - Dialeng",
        Style(display_style) if display_style else None,
        icon_sprites,
        Div(
            file_sidebar,
            Div(
                Div(
                    Div(
                        Button(icon_sprites("panel-left-open", sz=16), cls="btn btn-sm file-explorer-toggle-btn",
                               onclick="toggleFileExplorer()", title="Toggle file explorer (Ctrl+Shift+E)"),
                        Span(icon_sprites("file-text", sz=16), _display_name(rel_path), cls="toolbar-title"),
                        cls="toolbar-left"
                    ),
                    Div(
                        Button(icon_sprites("settings", sz=16), cls="btn btn-sm settings-btn", id="settings-btn",
                               onclick="toggleSettings()", title="Settings"),
                        Button(icon_sprites("sun", sz=16), cls="theme-toggle", id="theme-toggle",
                               onclick="toggleTheme()", title="Toggle light/dark theme"),
                        Button(icon_sprites("save", sz=14), cls="btn btn-sm btn-save", id="save-btn",
                               onclick="saveOpenFile()", title="Save (Ctrl+S)", disabled=True),
                        cls="toolbar-right"
                    ),
                    cls="toolbar-container"
                ),
                Div(id="status", aria_live="polite", aria_atomic="true"),
                Div(
                    Div(
                        Div("Loading file…", cls="file-editor-loading"),
                        id="file-editor-container",
                        cls="file-editor-shell"
                    ),
                    cls="file-editor-page-body"
                ),
                Script(
                    f"""
                    window.DIALENG_PAGE_KIND = 'file';
                    window.DIALENG_FILE_PATH = {json.dumps(rel_path)};
                    document.addEventListener('DOMContentLoaded', () => initializeFileEditorPage({json.dumps(rel_path)}));
                    """
                ),
                Div(id="ephemeral"),
                cls="container"
            ),
            cls="main-layout"
        ),
        NewItemModal(),
        DeleteConfirmModal(),
        SettingsOverlay() if config else None,
        SettingsSidebar(config) if config else None,
    )
