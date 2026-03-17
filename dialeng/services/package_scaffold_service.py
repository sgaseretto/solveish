"""Scaffold an nbdev-compatible package from existing dialeng notebooks."""
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def scan_notebook_modules(root: Path) -> Dict[str, Path]:
    """Scan .ipynb files for #| default_exp directives."""
    from dialeng.services.lib_export_service import find_default_exp, get_lib_name

    modules = {}
    lib_name = get_lib_name(root)
    skip_dirs = {"AUTORUN", "_lib", ".autorun_modules", ".ipynb_checkpoints", lib_name}
    for nb_path in sorted(root.rglob("*.ipynb")):
        if any(part in skip_dirs for part in nb_path.parts):
            continue
        module_name = find_default_exp(nb_path)
        if module_name:
            modules[module_name] = nb_path
    return modules


def scaffold_package(root: Path, package_name: Optional[str] = None) -> Path:
    """Scaffold an nbdev-compatible project from existing notebooks.

    If pyproject.toml already exists (e.g. from CRAFT Init), merges the
    [project], [build-system], and [tool.nbdev] sections into it.
    """
    if package_name is None:
        package_name = root.resolve().name.replace("-", "_").replace(" ", "_")

    pyproject_path = root / "pyproject.toml"
    modules = scan_notebook_modules(root)
    module_list = ", ".join(f'"{m}"' for m in sorted(modules.keys()))

    nbdev_sections = f"""[project]
name = "{package_name}"
version = "0.0.1"
description = ""
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.nbdev]
lib_name = "{package_name}"
lib_path = "{package_name}"
nbs_path = "."
doc_path = "_docs"
# Discovered modules: {module_list}
"""

    if pyproject_path.exists():
        existing = pyproject_path.read_text(encoding="utf-8")
        # Check if already scaffolded
        if "[tool.nbdev]" in existing:
            print(f"\npyproject.toml already has [tool.nbdev] — skipping scaffold.")
            print(f"Edit {pyproject_path} manually if you need changes.")
            return pyproject_path
        # Merge: append the new sections to the existing content
        content = existing.rstrip() + "\n\n" + nbdev_sections
        pyproject_path.write_text(content, encoding="utf-8")
        print(f"\nExtended existing pyproject.toml with package sections.")
    else:
        pyproject_path.write_text(nbdev_sections, encoding="utf-8")
        print(f"\nPackage '{package_name}' scaffolded!")

    pkg_dir = root / package_name
    pkg_dir.mkdir(exist_ok=True)
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    print(f"  pyproject.toml updated with [tool.nbdev] config")
    print(f"  {pkg_dir}/ directory created")
    if modules:
        print(f"\n  Found {len(modules)} notebook module(s):")
        for mod, nb in sorted(modules.items()):
            print(f"    {mod} ← {nb.name}")
    print(f"\n  Next steps:")
    print(f"    1. Run: nbdev_install_hooks    (git-friendly notebooks)")
    print(f"    2. Run: nbdev_export           (generate {package_name}/ from notebooks)")
    print(f"    3. Run: nbdev_test             (run notebook tests)")
    print(f"    4. Run: nbdev_pypi             (publish to PyPI)")

    return pyproject_path
