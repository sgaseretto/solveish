"""
CRAFT Initialization Extension

Adds a toolbar button that lets users initialize a package-aware CRAFT.ipynb.
When clicked, it prompts for a package name and delegates to the
craft_init_service for the actual initialization logic.
"""

import logging

from dialeng.core.registry import register_action, register_toolbar_item_decorator

logger = logging.getLogger(__name__)


def _get_notebooks_dir():
    """Get NOTEBOOKS_DIR from app module (lazy import to avoid circular deps)."""
    from dialeng.app import NOTEBOOKS_DIR
    return NOTEBOOKS_DIR


@register_action("craft_init_detect")
def craft_init_detect(nb_id: str, **kwargs):
    """Detect existing package name from pyproject.toml for pre-filling the prompt."""
    from dialeng.services.craft_init_service import detect_pkg_name
    root_dir = _get_notebooks_dir()
    return {
        "lib_name": detect_pkg_name(root_dir),
        "craft_exists": (root_dir / "CRAFT.ipynb").exists(),
    }


@register_action("init_craft")
def init_craft_action(nb_id: str, pkg_name: str = "", **kwargs):
    """Create/update CRAFT.ipynb and pyproject.toml with the given package name."""
    if not pkg_name:
        return {"error": "Package name is required"}
    from dialeng.services.craft_init_service import init_craft
    return init_craft(_get_notebooks_dir(), pkg_name)


@register_toolbar_item_decorator("craft_init_button", position="right", order=85)
def render_craft_init_button(notebook, config):
    """Toolbar button to initialize a package-aware CRAFT.ipynb."""
    from fasthtml.common import Button, Script, Div, NotStr

    icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 7v10"/><path d="M11 7v10"/><path d="m15 7 2 10"/></svg>'

    js = """
    async function initCraftPackage() {
        // Detect existing config to pre-fill prompt
        const detectResp = await fetch(`${nbApiPath()}/ext/craft_init_detect`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: ''
        });
        const detected = await detectResp.json();

        if (detected.craft_exists) {
            if (!confirm('CRAFT.ipynb already exists. Re-running will update generated cells and preserve your custom cells. Continue?')) return;
        }

        const defaultName = detected.lib_name || '';
        const name = prompt('Enter package name (valid Python identifier):', defaultName);
        if (!name) return;

        const params = new URLSearchParams({pkg_name: name});
        const resp = await fetch(`${nbApiPath()}/ext/init_craft`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: params.toString()
        });
        const data = await resp.json();
        if (data.error) { alert('Error: ' + data.error); }
        else { alert('Created CRAFT.ipynb for package: ' + name); }
    }
    """
    return Div(
        Script(js),
        Button(
            NotStr(icon_svg),
            cls="btn btn-sm",
            title="Initialize Package / CRAFT.ipynb",
            onclick="initCraftPackage()",
        ),
    )


logger.info("CRAFT init extension loaded")
