"""
CRAFT Initialization Extension

Adds a toolbar button that lets users initialize a package-aware CRAFT.ipynb.
When clicked, it prompts for a package name and:
1. Creates/updates pyproject.toml with [tool.dialeng] lib_name
2. Creates CRAFT.ipynb with sys.path setup for the package folder
3. Creates the package directory with __init__.py
4. Merges any existing CRAFT.ipynb content into the new one
"""

import json
import logging
import re
from pathlib import Path

from dialeng.core.registry import register_action, register_toolbar_item_decorator

logger = logging.getLogger(__name__)


def _get_notebooks_dir() -> Path:
    """Get NOTEBOOKS_DIR from app module (lazy import to avoid circular deps)."""
    from dialeng.app import NOTEBOOKS_DIR
    return NOTEBOOKS_DIR


def _update_pyproject_toml(root_dir: Path, pkg_name: str) -> Path:
    """Create or update pyproject.toml with [tool.dialeng] lib_name."""
    pyproject_path = root_dir / "pyproject.toml"

    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        if "[tool.dialeng]" in content:
            # Try to update existing lib_name within the [tool.dialeng] section
            # Stop matching at the next section header to avoid cross-section edits
            updated = re.sub(
                r'(\[tool\.dialeng\][^\[]*?)lib_name\s*=\s*"[^"]*"',
                rf'\1lib_name = "{pkg_name}"',
                content,
            )
            if updated != content:
                content = updated
            else:
                # Section exists but lib_name key is missing — append it
                content = re.sub(
                    r'(\[tool\.dialeng\]\n)',
                    rf'\1lib_name = "{pkg_name}"\n',
                    content,
                )
        else:
            content = content.rstrip() + f'\n\n[tool.dialeng]\nlib_name = "{pkg_name}"\n'
    else:
        content = f'[tool.dialeng]\nlib_name = "{pkg_name}"\n'

    pyproject_path.write_text(content, encoding="utf-8")
    return pyproject_path


def _build_craft_cells(pkg_name: str) -> list:
    """Build the notebook cells for CRAFT.ipynb."""
    setup_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## Project Setup\n",
            "\n",
            f"This CRAFT notebook auto-executes when any notebook in this directory is opened.\n",
            "\n",
            f"It sets up the `{pkg_name}/` reuse workflow:\n",
            f"- Ensures `{pkg_name}/` is on `sys.path` so `from {pkg_name}.module import ...` works\n",
            "- Shows a status banner while loading\n",
            "\n",
            f"### The {pkg_name} Reuse Workflow\n",
            "\n",
            "1. Mark useful cells in any notebook with `#| export`\n",
            "2. Add `#| default_exp module_name` to set the module name\n",
            f"3. Save the notebook \u2192 `{pkg_name}/module_name.py` is auto-generated\n",
            f"4. Import from `{pkg_name}.module_name` in any other notebook",
        ],
    }

    setup_code = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# CRAFT auto-setup: ensure {pkg_name} is importable\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "_pkg_path = str(Path.cwd())\n",
            "if _pkg_path not in sys.path:\n",
            "    sys.path.insert(0, _pkg_path)\n",
            f"    print(f'[CRAFT] Added {{_pkg_path}} to sys.path')\n",
            "else:\n",
            f"    print(f'[CRAFT] sys.path already configured')\n",
            "\n",
            f"# Check if {pkg_name} exists and list available modules\n",
            f"_pkg_dir = Path.cwd() / '{pkg_name}'\n",
            "if _pkg_dir.exists():\n",
            "    modules = [f.stem for f in _pkg_dir.glob('*.py') if f.stem != '__init__']\n",
            "    if modules:\n",
            f"        print(f'[CRAFT] Available {pkg_name} modules: {{', '.join(modules)}}')\n",
            "    else:\n",
            f"        print('[CRAFT] {pkg_name}/ exists but has no modules yet')\n",
            "else:\n",
            f"    print('[CRAFT] No {pkg_name}/ directory yet \\u2014 save a notebook with #| default_exp to create one')",
        ],
        "outputs": [],
        "execution_count": None,
    }

    additional_note = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Additional functionalities\n",
            "\n",
            "Additional functionalities for this CRAFT.ipynb can be added below.",
        ],
    }

    return [setup_markdown, setup_code, additional_note]


def _build_notebook(cells: list) -> dict:
    """Build a standard .ipynb notebook dict from a list of cells."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


@register_action("init_craft")
def init_craft(nb_id: str, pkg_name: str = "", **kwargs):
    """Create/update CRAFT.ipynb and pyproject.toml with the given package name."""
    if not pkg_name:
        return {"error": "Package name is required"}

    pkg_name = pkg_name.strip()
    if not pkg_name.isidentifier():
        return {"error": f"'{pkg_name}' is not a valid Python identifier"}

    root_dir = _get_notebooks_dir()

    # 1. Update pyproject.toml
    pyproject_path = _update_pyproject_toml(root_dir, pkg_name)
    logger.info(f"[CRAFT-INIT] Updated {pyproject_path} with lib_name={pkg_name}")

    # 2. Build CRAFT.ipynb cells
    craft_cells = _build_craft_cells(pkg_name)

    # 3. Merge existing CRAFT.ipynb content if present (skip previously generated cells)
    craft_path = root_dir / "CRAFT.ipynb"
    merged_existing = False
    if craft_path.exists():
        try:
            existing = json.loads(craft_path.read_text(encoding="utf-8"))
            existing_cells = existing.get("cells", [])
            # Filter out cells generated by a previous init_craft run
            _generated_markers = ("# CRAFT auto-setup:", "## Project Setup", "## Additional functionalities")
            user_cells = []
            for cell in existing_cells:
                src = cell.get("source", [])
                if isinstance(src, list):
                    src = "".join(src)
                if not any(src.lstrip().startswith(m) for m in _generated_markers):
                    user_cells.append(cell)
            if user_cells:
                craft_cells.extend(user_cells)
                merged_existing = True
                logger.info(f"[CRAFT-INIT] Merged {len(user_cells)} existing user cells")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[CRAFT-INIT] Could not read existing CRAFT.ipynb: {e}")

    # 4. Write CRAFT.ipynb
    notebook = _build_notebook(craft_cells)
    craft_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[CRAFT-INIT] Created {craft_path}")

    # 5. Create package directory with __init__.py
    pkg_dir = root_dir / pkg_name
    pkg_dir.mkdir(exist_ok=True)
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    logger.info(f"[CRAFT-INIT] Ensured {pkg_dir}/ exists with __init__.py")

    return {
        "status": "ok",
        "pkg_name": pkg_name,
        "craft_path": str(craft_path),
        "pyproject_path": str(pyproject_path),
        "merged_existing": merged_existing,
    }


@register_toolbar_item_decorator("craft_init_button", position="right", order=85)
def render_craft_init_button(notebook, config):
    """Toolbar button to initialize a package-aware CRAFT.ipynb."""
    from fasthtml.common import Button, Script, Div, NotStr

    icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 7v10"/><path d="M11 7v10"/><path d="m15 7 2 10"/></svg>'

    js = """
    async function initCraftPackage() {
        const name = prompt('Enter package name (valid Python identifier):');
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
