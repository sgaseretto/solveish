"""
Dialeng UI - Kernel Selection Components

Kernel toolbar button and selection modal.
"""

from fasthtml.common import *
from .icons import sprites as icon_sprites
from dialeng.core.registry import registry


def KernelToolbarButton(nb, kernel_info: dict = None):
    """Compact kernel indicator for the toolbar. Clicking opens the kernel modal.

    Args:
        nb: Notebook instance
        kernel_info: Optional dict with keys like 'language', 'version', 'display_name'
    """
    kernel_type = nb.kernel_type
    reg = registry.kernels.get(kernel_type)

    if kernel_info:
        lang = kernel_info.get('language', 'Python')
        version = kernel_info.get('version', '')
        display = f"{lang} {version}".strip()
        icon_name = reg.icon if reg else 'microchip'
        btn_cls = "btn btn-sm kernel-toolbar-btn connected"
    else:
        display = reg.label if reg else kernel_type.title()
        icon_name = reg.icon if reg else 'house-plug'
        btn_cls = "btn btn-sm kernel-toolbar-btn"

    # Runtime badge for Colab
    runtime_badge = None
    if kernel_type == "colab":
        rt = nb.colab_runtime_type
        rt_icon = 'zap' if rt in ('gpu', 'tpu') else 'cpu'
        runtime_badge = Span(
            icon_sprites(rt_icon, sz=11),
            rt.upper(),
            cls="kernel-runtime-badge"
        )

    return Button(
        icon_sprites(icon_name, sz=14),
        Span(display, cls="kernel-btn-label"),
        runtime_badge,
        id="kernel-status-bar",
        type="button",
        cls=btn_cls,
        onclick=f"toggleKernelModal('{nb.id}')",
        title="Click to change kernel"
    )


def KernelModal(nb_id: str, current_kernel_type: str,
                colab_authenticated: bool = False,
                current_runtime: str = "cpu",
                colab_account_email: str = None):
    """Modal for selecting and switching kernels.

    Selection is client-side only until Apply is clicked.

    Args:
        nb_id: Notebook ID
        current_kernel_type: Currently active kernel type name
        colab_authenticated: Whether Colab is authenticated
        current_runtime: Current Colab runtime type
    """
    kernel_options = []
    for name, reg in sorted(registry.kernels.items(), key=lambda kv: kv[0]):
        is_active = name == current_kernel_type
        option_cls = "kernel-option active" if is_active else "kernel-option"

        # Check mark for active kernel
        status = icon_sprites('check', sz=14) if is_active else None

        # Auth note for Colab when not authenticated
        auth_note = None
        if reg.requires_auth and not colab_authenticated:
            auth_note = Span("Sign in required", cls="kernel-auth-note")
        elif name == "colab" and colab_authenticated and colab_account_email:
            auth_note = Span(f"Connected as {colab_account_email}", cls="kernel-auth-note")

        # Runtime picker (shown when this kernel is selected, only for kernels with runtime_options)
        runtime_picker = None
        if reg.runtime_options:
            runtime_picker = Div(
                *[Button(
                    opt.upper(),
                    type="button",
                    cls=f"btn btn-sm kernel-runtime-btn{'  active' if opt == current_runtime and is_active else ''}",
                    data_runtime=opt,
                    onclick=f"selectKernelRuntime(this, '{opt}')",
                ) for opt in reg.runtime_options],
                cls="kernel-runtime-options",
                style="" if is_active else "display:none;",
            )

        # Click selects this kernel visually (no server call yet)
        if reg.requires_auth and not colab_authenticated:
            click_action = f"window.open('/auth/google', '_blank', 'width=500,height=700')"
        else:
            click_action = f"selectKernelOption(this, '{name}')"

        kernel_options.append(
            Div(
                Div(
                    icon_sprites(reg.icon, sz=20),
                    Div(
                        Div(
                            Span(reg.label, cls="kernel-option-label"),
                            status,
                            cls="kernel-option-title-row"
                        ),
                        Small(reg.description, cls="kernel-option-desc") if reg.description else None,
                        auth_note,
                        cls="kernel-option-text"
                    ),
                    cls="kernel-option-main"
                ),
                runtime_picker,
                cls=option_cls,
                data_kernel=name,
                onclick=click_action,
            )
        )

    return Div(
        Div(
            Div(
                Span("Select Kernel", cls="kernel-modal-title"),
                Button("✕", type="button", cls="kernel-modal-close", onclick="toggleKernelModal()"),
                cls="kernel-modal-header"
            ),
            Div(*kernel_options, cls="kernel-modal-body"),
            # Hidden inputs to track selection
            Input(type="hidden", id="kernel-modal-selected-type", value=current_kernel_type),
            Input(type="hidden", id="kernel-modal-selected-runtime", value=current_runtime),
            Div(
                Button("Cancel", type="button", cls="btn btn-sm",
                       onclick="toggleKernelModal()"),
                Button("Apply", type="button", cls="btn btn-sm btn-primary",
                       id="kernel-apply-btn",
                       onclick=f"applyKernelSelection('{nb_id}')"),
                cls="kernel-modal-footer"
            ),
            cls="kernel-modal"
        ),
        id="kernel-modal-overlay",
        cls="kernel-modal-overlay",
        onclick="if(event.target===this) toggleKernelModal()"
    )
