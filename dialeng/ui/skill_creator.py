"""Notebook-to-skill creator controls."""
from fasthtml.common import Button, Div, H2, H3, Input, Label, P, Pre, Span, Textarea

from .icons import sprites as icon_sprites


def SkillCreatorButton(enabled: bool = True):
    """Toolbar button for the staged skill creator."""
    if not enabled:
        return None
    return Button(
        icon_sprites("sparkles", sz=15),
        cls="btn btn-sm btn-icon skill-creator-btn",
        onclick="openSkillCreator()",
        title="Create a skill from notebooks",
        type="button",
    )


def SkillCreatorModal():
    """Render the project skill creation dialog; notebook choices load from the backend."""
    return Div(
        Div(
            Div(
                Div(
                    H2("Create Skill From Sources", id="skill-creator-title"),
                    P("Combine notebooks, text files, and web pages; review the draft, then install it explicitly."),
                    cls="skill-creator-heading",
                ),
                Button(
                    icon_sprites("x", sz=18),
                    cls="btn btn-sm btn-icon",
                    onclick="closeSkillCreator()",
                    title="Close",
                    type="button",
                ),
                cls="skill-creator-header",
            ),
            Div(
                Div(
                    Label("Skill name", For="skill-creator-name"),
                    Input(
                        id="skill-creator-name",
                        placeholder="data-analysis",
                        pattern="[a-z0-9][a-z0-9-]{0,63}",
                        autocomplete="off",
                    ),
                    Label("Trigger description", For="skill-creator-description"),
                    Textarea(
                        id="skill-creator-description",
                        placeholder="Use when the user needs to...",
                        rows="3",
                    ),
                    Label("Additional instructions", For="skill-creator-instructions"),
                    Textarea(
                        id="skill-creator-instructions",
                        placeholder="Optional workflow requirements",
                        rows="3",
                    ),
                    Div(
                        Label(Input(type="checkbox", id="skill-creator-use-llm", checked=True), " Refine with selected LLM"),
                        Label(Input(type="checkbox", id="skill-creator-outputs"), " Include bounded text outputs"),
                        cls="skill-creator-options",
                    ),
                    H3("Project sources"),
                    Div("Loading project sources...", id="skill-creator-sources", cls="skill-source-list"),
                    Label("Web pages", For="skill-creator-urls"),
                    Textarea(
                        id="skill-creator-urls",
                        placeholder="https://example.com/guide\nOne URL per line",
                        rows="3",
                    ),
                    cls="skill-creator-form",
                ),
                Div(
                    Div(
                        Span("Draft preview"),
                        Span(id="skill-creator-build-label", cls="skill-build-label"),
                        cls="skill-preview-heading",
                    ),
                    Div(id="skill-creator-files", cls="skill-preview-files"),
                    Pre("Generate a draft to inspect SKILL.md and its references.", id="skill-creator-preview"),
                    cls="skill-creator-preview",
                ),
                cls="skill-creator-body",
            ),
            Div(
                Div(id="skill-creator-status", cls="skill-creator-status", role="status", aria_live="polite"),
                Button("Cancel", cls="btn", type="button", onclick="closeSkillCreator()"),
                Button("Generate Draft", cls="btn", type="button", onclick="generateSkillDraft()", id="skill-generate-btn"),
                Button("Install Skill", cls="btn btn-primary", type="button", onclick="installSkillDraft()", id="skill-install-btn", disabled=True),
                cls="skill-creator-footer",
            ),
            cls="skill-creator-dialog",
            role="dialog",
            aria_modal="true",
            aria_labelledby="skill-creator-title",
        ),
        id="skill-creator-overlay",
        cls="skill-creator-overlay",
        onclick="if(event.target===this)closeSkillCreator()",
    )
