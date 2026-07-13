"""Extension Manager UI components."""
from fasthtml.common import Button, Div, H2, H3, Input, Label, Option, P, Pre, Select, Span

from .icons import sprites as icon_sprites


def ExtensionManagerButton():
    return Button(
        icon_sprites("blocks", sz=16),
        cls="btn btn-sm extension-manager-btn",
        id="extension-manager-btn",
        onclick="openExtensionManager()",
        title="Extensions",
        aria_label="Open Extension Manager",
    )


def ExtensionManagerModal():
    return Div(
        Div(
            Div(
                Div(
                    Span(icon_sprites("blocks", sz=20)),
                    Div(
                        H2("Extensions", id="extension-manager-title"),
                        P("Review, trust, enable, reload, and diagnose Dialeng extensions."),
                        cls="extension-manager-heading",
                    ),
                    cls="extension-manager-title",
                ),
                Button(
                    "Copy diagnostics",
                    type="button",
                    cls="btn btn-sm extension-diagnostics-copy",
                    onclick="copyExtensionDiagnostics()",
                    aria_label="Copy extension diagnostics",
                ),
                Button(
                    "x",
                    type="button",
                    cls="extension-manager-close",
                    onclick="closeExtensionManager()",
                    aria_label="Close Extension Manager",
                ),
                cls="extension-manager-header",
            ),
            Div(
                Div(
                    Div(
                        H3("Extension Studio"),
                        P("Scaffold a notebook-first extension, then build and validate it before trust."),
                    ),
                    Button(
                        "+ Create Extension",
                        type="button",
                        cls="btn btn-sm",
                        onclick="toggleExtensionStudio()",
                        id="extension-studio-toggle",
                    ),
                    cls="extension-studio-summary",
                ),
                Div(
                    Label("Name", Input(id="extension-studio-name", placeholder="notebook-audit")),
                    Label("Extension ID", Input(id="extension-studio-id", placeholder="local.notebook-audit")),
                    Label(
                        "Template",
                        Select(
                            Option("Basic", value="basic"),
                            Option("Tool", value="tool"),
                            Option("Cell", value="cell"),
                            Option("Lifecycle", value="lifecycle"),
                            Option("Provider", value="provider"),
                            Option("Package", value="package"),
                            id="extension-studio-template",
                        ),
                    ),
                    Button("Create notebook", type="button", cls="btn btn-primary", onclick="createExtensionFromStudio()"),
                    P(id="extension-studio-status", cls="extension-manager-status"),
                    Pre(id="extension-studio-output", cls="extension-studio-output"),
                    id="extension-studio-form",
                    cls="extension-studio-form",
                    hidden=True,
                ),
                Div("Loading extensions...", id="extension-manager-status", cls="extension-manager-status"),
                Div(id="extension-manager-list", cls="extension-manager-list"),
                Div(id="extension-manager-diagnostics", cls="extension-manager-diagnostics"),
                cls="extension-manager-body",
            ),
            cls="extension-manager-dialog",
            role="dialog",
            aria_modal="true",
            aria_labelledby="extension-manager-title",
        ),
        id="extension-manager-overlay",
        cls="extension-manager-overlay",
        onclick="if(event.target===this)closeExtensionManager()",
    )
