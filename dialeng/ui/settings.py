"""
Dialeng UI - Settings Panel Components

Settings sidebar for viewing and editing dialeng_config.json settings.
"""

from fasthtml.common import *
from typing import List, Tuple, Optional, Any, Dict
from .icons import sprites as ss
from dialeng.services.dialeng_config import DialengConfig


def _get_extension_settings_sections():
    """Get registered extension settings sections (lazy import to avoid circular deps)."""
    from dialeng.core.registry import registry
    return list(registry.settings_sections.values())


def SettingsGroup(title: str, *children, open: bool = False):
    """Collapsible settings section using <details> element.

    Args:
        title: Section title
        *children: Form controls to include in the section
        open: Whether the section starts expanded

    Returns:
        Details element with summary and content
    """
    return Details(
        Summary(title, cls="settings-group-title"),
        Div(*children, cls="settings-group-content"),
        cls="settings-group",
        open=open
    )


def SettingSelect(label: str, name: str, options: List[Tuple[str, str]], current: str,
                  help_text: str = None):
    """Dropdown select for settings.

    Args:
        label: Display label
        name: Form field name (should match config path, e.g., "modes.default")
        options: List of (value, display_label) tuples
        current: Currently selected value
        help_text: Optional help text shown below the control

    Returns:
        Div containing label and select element
    """
    return Div(
        Label(label, fr=name, cls="setting-label"),
        Select(
            *[Option(display, value=val, selected=current == val)
              for val, display in options],
            name=name,
            id=name.replace(".", "-"),
            cls="setting-select"
        ),
        Small(help_text, cls="setting-help") if help_text else None,
        cls="setting-row"
    )


def SettingToggle(label: str, name: str, current: bool = None, requires_restart: bool = False,
                  help_text: str = None, default: bool = False):
    """Toggle switch for boolean settings.

    Args:
        label: Display label
        name: Form field name (dot-notation, e.g. "colab.enabled" or "my_ext_enabled")
        current: Current boolean value. If None, auto-reads from config using `name`.
        requires_restart: If True, shows a restart warning
        help_text: Optional help text
        default: Default value when auto-reading from config and key doesn't exist

    Returns:
        Div containing toggle switch
    """
    if current is None:
        # Auto-resolve from config using the dotted field name
        from dialeng.services.dialeng_config import get_config
        config = get_config()
        # Traverse dotted path in raw_config (e.g. "colab.enabled" → config["colab"]["enabled"])
        value = config.raw_config
        for key in name.split('.'):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        current = bool(value) if value is not None else default
    restart_badge = Span(ss('triangle-alert', sz=12), " restart", cls="restart-badge") if requires_restart else None
    return Div(
        Div(
            Label(label, fr=name, cls="setting-label"),
            restart_badge,
            cls="setting-label-row"
        ),
        Label(
            # Hidden input ensures "off" is sent when checkbox is unchecked
            # (HTML checkboxes don't send anything when unchecked)
            Input(type="hidden", name=name, value="off"),
            Input(type="checkbox", name=name, id=name.replace(".", "-"),
                  checked=current, cls="toggle-input"),
            Span(cls="toggle-slider"),
            cls="toggle-switch"
        ),
        Small(help_text, cls="setting-help") if help_text else None,
        cls="setting-row setting-toggle-row"
    )


def SettingNumber(label: str, name: str, current: int, min_val: int = 0,
                  max_val: int = 100000, step: int = 1, help_text: str = None):
    """Number input for numeric settings.

    Args:
        label: Display label
        name: Form field name
        current: Current value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        step: Step increment
        help_text: Optional help text

    Returns:
        Div containing number input
    """
    return Div(
        Label(label, fr=name, cls="setting-label"),
        Input(type="number", name=name, id=name.replace(".", "-"),
              value=str(current), min=str(min_val), max=str(max_val),
              step=str(step), cls="setting-number"),
        Small(help_text, cls="setting-help") if help_text else None,
        cls="setting-row"
    )


def SettingRange(label: str, name: str, current: int, min_val: int = 1,
                 max_val: int = 10, help_text: str = None):
    """Slider range input for bounded numeric settings.

    Args:
        label: Display label
        name: Form field name
        current: Current value
        min_val: Minimum value
        max_val: Maximum value
        help_text: Optional help text

    Returns:
        Div containing slider with value display
    """
    return Div(
        Label(label, fr=name, cls="setting-label"),
        Div(
            Input(type="range", name=name, id=name.replace(".", "-"),
                  value=str(current), min=str(min_val), max=str(max_val),
                  cls="setting-range",
                  oninput=f"document.getElementById('{name.replace('.', '-')}-value').textContent = this.value"),
            Span(str(current), id=f"{name.replace('.', '-')}-value", cls="range-value"),
            cls="range-container"
        ),
        Small(help_text, cls="setting-help") if help_text else None,
        cls="setting-row"
    )


def SettingText(label: str, name: str, current: str, help_text: str = None):
    """Text input for string settings.

    Args:
        label: Display label
        name: Form field name
        current: Current value
        help_text: Optional help text

    Returns:
        Div containing text input
    """
    return Div(
        Label(label, fr=name, cls="setting-label"),
        Input(type="text", name=name, id=name.replace(".", "-"),
              value=current, cls="setting-text"),
        Small(help_text, cls="setting-help") if help_text else None,
        cls="setting-row"
    )


def SettingsSidebar(config: DialengConfig):
    """Main settings sidebar component.

    Renders all settings organized into collapsible groups.

    Args:
        config: Current DialengConfig instance

    Returns:
        Aside element containing the complete settings panel
    """
    # AWS region options
    aws_regions = [
        ("us-east-1", "US East (N. Virginia)"),
        ("us-west-2", "US West (Oregon)"),
        ("eu-west-1", "EU West (Ireland)"),
        ("eu-west-2", "EU West (London)"),
        ("ap-northeast-1", "Asia Pacific (Tokyo)"),
    ]

    # Dialog mode options
    mode_options = [
        ("mock", "Mock (No LLM)"),
        ("learning", "Learning"),
        ("concise", "Concise"),
        ("standard", "Standard"),
    ]

    colab_auth_options = [
        ("oauth_local", "Browser OAuth (local callback)"),
        ("oauth_remote", "Browser OAuth (copy/paste code)"),
        ("adc", "Application Default Credentials"),
    ]
    colab_runtime_options = [
        ("cpu", "CPU"),
        ("gpu", "GPU"),
        ("tpu", "TPU"),
    ]
    from dialeng.services.openai_codex_auth import get_openai_codex_auth_service
    codex_auth = get_openai_codex_auth_service().status()
    codex_label = (
        f"Connected ({codex_auth.get('account_id', '')[:12]}...)"
        if codex_auth.get("authenticated") else "Not connected"
    )

    return Aside(
        # Header with close button
        Div(
            Span(ss('settings', sz=16), " Settings", cls="settings-title"),
            Button("✕", cls="settings-close-btn", onclick="toggleSettings()",
                   title="Close settings"),
            cls="settings-header"
        ),

        # Scrollable content
        Div(
            Form(
                # Dialeng Display Settings (top of list for easy access)
                SettingsGroup(
                    "Dialeng Display Settings",
                    SettingNumber("Notebook Width (px)", "display.notebook_width",
                                  config.display_notebook_width, 600, 3000, 50,
                                  "Max width of the notebook container"),
                    SettingSelect("Button Size", "display.button_size",
                                  [("compact", "Compact"), ("normal", "Normal"), ("large", "Large")],
                                  config.display_button_size,
                                  "Size of all buttons (toolbar, cells, file explorer)"),
                    SettingNumber("Font Size (px)", "display.font_size",
                                  config.display_font_size, 10, 24, 1,
                                  "Base font size for the UI"),
                    SettingNumber("Reasoning Text Limit", "display.reasoning_truncate_chars",
                                  config.reasoning_truncate_chars, 0, 10000, 100,
                                  "Max characters for LLM reasoning (0 = no limit)"),
                    open=False
                ),

                # AWS Settings
                SettingsGroup(
                    "AWS Settings",
                    SettingSelect("Region", "aws.region", aws_regions, config.aws_region,
                                  "AWS region for Bedrock API calls"),
                    open=False
                ),

                # Model Defaults
                SettingsGroup(
                    "Model Defaults",
                    SettingSelect("Bedrock Model", "models.defaults.bedrock",
                                  config.get_model_choices("claudette:bedrock"),
                                  config.default_models.get("bedrock", "claude-haiku-4-5"),
                                  "Default model when using AWS Bedrock"),
                    SettingSelect("Claude Code Model", "models.defaults.claude_code_subscription",
                                  config.get_model_choices("claude_agent_sdk"),
                                  config.default_models.get("claude_code_subscription", "claude-sonnet-4-5"),
                                  "Default model when using Claude Code subscription"),
                    SettingSelect("OpenAI Codex Model", "models.defaults.openai_codex",
                                  config.get_model_choices("openai_codex"),
                                  config.default_models.get("openai_codex", "gpt-5.4"),
                                  "Default model for an OpenAI Codex subscription"),
                    SettingSelect("Default Dialog Mode", "modes.default", mode_options, config.default_mode,
                                  "Default mode when opening a notebook"),
                    open=True
                ),

                SettingsGroup(
                    "OpenAI Codex",
                    SettingToggle("Enable OpenAI Codex", "llm.openai_codex_enabled",
                                  config.openai_codex_enabled,
                                  help_text="Expose the Codex subscription provider in notebook controls."),
                    SettingSelect("Streaming transport", "llm.openai_codex_transport",
                                  [("auto", "WebSocket with SSE fallback"),
                                   ("websocket", "WebSocket only"),
                                   ("sse", "SSE only")],
                                  config.openai_codex_transport,
                                  "Auto falls back only before response output begins."),
                    Div(
                        Span(codex_label, id="openai-codex-auth-state", cls="setting-help"),
                        Div(
                            Button("Connect OpenAI", type="button", cls="btn btn-sm",
                                   onclick="startOpenAICodexLogin()",
                                   style="display:none;" if codex_auth.get("authenticated") else "",
                                   id="openai-codex-login"),
                            Button("Disconnect", type="button", cls="btn btn-sm",
                                   onclick="logoutOpenAICodex()",
                                   style="" if codex_auth.get("authenticated") else "display:none;",
                                   id="openai-codex-logout"),
                            cls="setting-actions",
                        ),
                        Small(
                            "Uses a Codex subscription OAuth session, not an OpenAI Platform API key.",
                            cls="setting-help",
                        ),
                        cls="setting-row",
                    ),
                    SettingToggle("Enable project skills", "skills.enabled",
                                  config.skills_enabled,
                                  help_text="Discover OpenAI Codex skills from .dialeng/skills."),
                    open=True,
                ),

                # Tool Settings
                SettingsGroup(
                    "Tool Settings",
                    SettingRange("Max Tool Steps", "tool_settings.max_steps",
                                 config.tool_max_steps, 1, 10,
                                 "Maximum tool calls per prompt (1-10)"),
                    SettingToggle("Require Confirmation", "tool_settings.require_confirmation",
                                  config.tool_require_confirmation,
                                  help_text="Prompt before file-modifying tools"),
                    SettingToggle("Enable Built-in Tools", "tool_settings.builtin_tools_enabled",
                                  config.tool_builtin_enabled,
                                  help_text="Enable file tools (view, rg, create, etc.)"),
                    open=False
                ),

                # Shell Settings
                SettingsGroup(
                    "Shell Settings",
                    SettingToggle("Enable Shell Cells", "shell.shell_cells_enabled",
                                  config.shell_cells_enabled, requires_restart=True,
                                  help_text="Show dedicated Shell cell type. If disabled, use %bash magic or !cmd in code cells."),
                    open=False
                ),

                # Google Colab Settings
                SettingsGroup(
                    "Google Colab",
                    SettingToggle("Enable Colab", "colab.enabled",
                                  config.colab_enabled,
                                  help_text="Enable Google Colab as a remote kernel. Save to reload — a 'Connect Google' button will appear in the toolbar."),
                    SettingSelect("Auth Provider", "colab.auth_provider",
                                  colab_auth_options, config.colab_auth_provider,
                                  "oauth_local is the default. oauth_remote works better for headless/remote servers. adc uses gcloud application-default credentials."),
                    SettingSelect("Default Runtime", "colab.default_runtime",
                                  colab_runtime_options, config.colab_default_runtime,
                                  "Runtime selected by default for new Colab notebooks."),
                    SettingText("Default Accelerator", "colab.default_accelerator",
                                config.colab_default_accelerator,
                                "Optional override such as T4, L4, A100, H100, V5E1, or V6E1."),
                    open=False
                ),

                # Advanced Settings
                SettingsGroup(
                    "Advanced",
                    SettingNumber("Thinking Max Tokens", "thinking.max_tokens",
                                  config.thinking_max_tokens, 0, 50000, 1000,
                                  "Extended thinking budget (0 = disable)"),
                    SettingToggle("Debug Mode", "llm.debug_mode",
                                  config.debug_mode,
                                  help_text="Enable debug logging to files"),
                    SettingText("Debug Log Directory", "llm.debug_log_dir",
                                config.debug_log_dir,
                                "Directory for debug log files"),
                    open=False
                ),

                # Extension settings sections
                *[reg.renderer(config)
                  for reg in sorted(
                      _get_extension_settings_sections(),
                      key=lambda r: r.order
                  )],

                id="settings-form",
                hx_post="/settings",
                hx_target="#settings-status",
                hx_swap="innerHTML"
            ),
            cls="settings-content"
        ),

        # Footer with save button and status
        Div(
            Button("Save Settings", type="submit", form="settings-form",
                   cls="btn settings-save-btn"),
            Div(id="settings-status", cls="settings-status"),
            cls="settings-footer"
        ),

        id="settings-sidebar",
        cls="settings-sidebar"
    )


def SettingsOverlay():
    """Semi-transparent overlay shown behind open sidebar.

    Returns:
        Div element that closes sidebar when clicked
    """
    return Div(
        id="settings-overlay",
        cls="settings-overlay",
        onclick="toggleSettings()"
    )
