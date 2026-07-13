"""Backend-rendered shell for the command palette."""
from fasthtml.common import Button, Div, H2, Input, P, Span

from .icons import sprites as icon_sprites


def CommandPaletteButton():
    return Button(
        icon_sprites("command", sz=16),
        cls="btn btn-sm command-palette-btn",
        id="command-palette-btn",
        type="button",
        onclick="openCommandPalette()",
        title="Command palette (Ctrl/Cmd+Shift+P)",
        aria_label="Open command palette",
    )


def CommandPaletteModal():
    return Div(
        Div(
            Div(
                Span(icon_sprites("command", sz=18), cls="command-palette-icon"),
                Input(
                    id="command-palette-search",
                    cls="command-palette-search",
                    placeholder="Search commands",
                    autocomplete="off",
                    role="combobox",
                    aria_controls="command-palette-list",
                    aria_expanded="false",
                    aria_autocomplete="list",
                    aria_label="Search commands",
                ),
                Button(
                    "x",
                    type="button",
                    cls="command-palette-close",
                    onclick="closeCommandPalette()",
                    aria_label="Close command palette",
                ),
                cls="command-palette-search-row",
            ),
            Div(
                Div(
                    H2("Command arguments", id="command-palette-args-title"),
                    P(id="command-palette-args-description"),
                    Div(id="command-palette-args-fields", cls="command-palette-args-fields"),
                    Div(
                        Button("Back", type="button", cls="btn btn-sm", id="command-palette-args-back"),
                        Button("Run", type="button", cls="btn btn-sm btn-primary", id="command-palette-args-run"),
                        cls="command-palette-args-actions",
                    ),
                    id="command-palette-args",
                    cls="command-palette-args",
                    hidden=True,
                ),
                Div(
                    Div(id="command-palette-list", cls="command-palette-list", role="listbox"),
                    Div("Loading commands...", id="command-palette-empty", cls="command-palette-empty"),
                    id="command-palette-results",
                ),
                Div(
                    Span("Enter to run · ↑↓ to navigate · Esc to close"),
                    Button(
                        "Cancel running command",
                        type="button",
                        cls="command-palette-cancel",
                        id="command-palette-cancel",
                        hidden=True,
                    ),
                    cls="command-palette-footer",
                ),
                cls="command-palette-body",
            ),
            cls="command-palette-dialog",
            role="dialog",
            aria_modal="true",
            aria_labelledby="command-palette-label",
        ),
        Span("Command palette", id="command-palette-label", cls="sr-only"),
        id="command-palette-overlay",
        cls="command-palette-overlay",
        onclick="if(event.target===this)closeCommandPalette()",
    )
