"""Service for initializing package-aware CRAFT.ipynb files.

Shared by both the CLI (--init flag) and the toolbar extension.
Handles detection of existing nbdev/dialeng configuration, pyproject.toml
updates, CRAFT.ipynb creation/merge, and package directory setup.
"""
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_pkg_name(root_dir: Path) -> str:
    """Detect package name from existing pyproject.toml configuration.

    Checks [tool.dialeng] lib_name first, then [tool.nbdev] lib_name.
    Returns empty string if nothing is configured.
    """
    from dialeng.services.lib_export_service import get_lib_name, LIB_DIR_NAME
    lib_name = get_lib_name(root_dir)
    return lib_name if lib_name != LIB_DIR_NAME else ""


def derive_pkg_name(root_dir: Path) -> str:
    """Derive a valid Python package name from the directory name."""
    name = root_dir.resolve().name
    # Sanitize: replace hyphens/spaces with underscores, strip leading digits
    name = name.replace("-", "_").replace(" ", "_")
    name = name.lstrip("0123456789")
    if not name:
        name = "my_pkg"
    return name


def update_pyproject_toml(root_dir: Path, pkg_name: str) -> Path:
    """Create or update pyproject.toml with [tool.dialeng] lib_name."""
    pyproject_path = root_dir / "pyproject.toml"

    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        if "[tool.dialeng]" in content:
            # Update existing lib_name within [tool.dialeng] section
            # Stop matching at next section header to avoid cross-section edits
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


def build_craft_cells(pkg_name: str) -> list:
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

    transition_guide = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Transitioning to a Python Package\n",
            "\n",
            "When you're ready to distribute your code as an installable package, run:\n",
            "\n",
            "```bash\n",
            f"dialeng package init --name {pkg_name}\n",
            "```\n",
            "\n",
            "This extends your existing `pyproject.toml` with `[project]`, `[build-system]`, and\n",
            "`[tool.nbdev]` sections — no migration needed. Your `#| default_exp` and `#| export`\n",
            "directives are the same ones nbdev uses, so the transition is seamless.\n",
            "\n",
            "After scaffolding, you get access to the full nbdev workflow:\n",
            "\n",
            "| Command | Purpose |\n",
            "|---|---|\n",
            f"| `uv run nbdev_export` | Export cells into `{pkg_name}/` |\n",
            "| `uv run nbdev_test` | Run all notebook cells as tests |\n",
            "| `uv run nbdev_docs` | Generate documentation from notebooks |\n",
            "| `uv run nbdev_pypi` | Publish the package to PyPI |\n",
            "\n",
            f"Until then, the `{pkg_name}/` reuse workflow above is all you need.",
        ],
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

    return [setup_markdown, setup_code, transition_guide, additional_note]


def build_notebook(cells: list) -> dict:
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


# Markers used to identify generated cells (for idempotent merge)
GENERATED_MARKERS = (
    "# CRAFT auto-setup:",
    "## Project Setup",
    "## Transitioning to a Python Package",
    "## Additional functionalities",
)


def init_craft(root_dir: Path, pkg_name: str) -> dict:
    """Initialize CRAFT.ipynb and pyproject.toml for a given package name.

    This is the core logic shared by the CLI and the toolbar extension.
    Returns a dict with status and details.
    """
    pkg_name = pkg_name.strip()
    if not pkg_name.isidentifier():
        return {"error": f"'{pkg_name}' is not a valid Python identifier"}

    # 1. Update pyproject.toml
    pyproject_path = update_pyproject_toml(root_dir, pkg_name)
    logger.info(f"[CRAFT-INIT] Updated {pyproject_path} with lib_name={pkg_name}")

    # 2. Build CRAFT.ipynb cells
    craft_cells = build_craft_cells(pkg_name)

    # 3. Merge existing CRAFT.ipynb content if present (skip previously generated cells)
    craft_path = root_dir / "CRAFT.ipynb"
    merged_existing = False
    if craft_path.exists():
        try:
            existing = json.loads(craft_path.read_text(encoding="utf-8"))
            existing_cells = existing.get("cells", [])
            user_cells = []
            for cell in existing_cells:
                src = cell.get("source", [])
                if isinstance(src, list):
                    src = "".join(src)
                if not any(src.lstrip().startswith(m) for m in GENERATED_MARKERS):
                    user_cells.append(cell)
            if user_cells:
                craft_cells.extend(user_cells)
                merged_existing = True
                logger.info(f"[CRAFT-INIT] Merged {len(user_cells)} existing user cells")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[CRAFT-INIT] Could not read existing CRAFT.ipynb: {e}")

    # 4. Write CRAFT.ipynb
    notebook = build_notebook(craft_cells)
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


def resolve_and_init(root_dir: Path, pkg_name: str | None = None) -> dict:
    """Resolve package name and run init_craft with proper CLI output.

    Name resolution priority:
    1. Explicit pkg_name argument (from --init NAME)
    2. Existing [tool.dialeng] or [tool.nbdev] lib_name in pyproject.toml
    3. Directory name (sanitized to valid Python identifier)

    Returns the init_craft result dict.
    """
    # Detect existing config
    detected = detect_pkg_name(root_dir)
    craft_exists = (root_dir / "CRAFT.ipynb").exists()

    if pkg_name:
        # Explicit name provided
        if detected and detected != pkg_name:
            print(f"  Warning: pyproject.toml has lib_name='{detected}', overriding with '{pkg_name}'")
    elif detected:
        # Use detected name from existing config
        pkg_name = detected
        print(f"  Detected existing package name: {pkg_name}")
    else:
        # Derive from directory name
        pkg_name = derive_pkg_name(root_dir)
        print(f"  Using directory name as package: {pkg_name}")

    if craft_exists:
        print(f"  CRAFT.ipynb exists — updating generated cells, preserving custom cells")

    result = init_craft(root_dir, pkg_name)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return result

    print(f"\n  Initialized reuse workflow for '{pkg_name}':")
    print(f"    pyproject.toml  → [tool.dialeng] lib_name = \"{pkg_name}\"")
    print(f"    CRAFT.ipynb     → auto-setup for {pkg_name}/")
    print(f"    {pkg_name}/     → package directory with __init__.py")
    if result.get("merged_existing"):
        print(f"    (preserved existing user cells in CRAFT.ipynb)")
    print(f"\n  Notebooks with #| export will auto-extract to {pkg_name}/ on save.")

    return result
