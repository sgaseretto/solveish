"""Idempotent Dialeng project initialization."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dialeng.services.craft_init_service import resolve_and_init
from dialeng.services.dialeng_config import load_config


SKILLS_README = """# Dialeng project skills

Create one directory per skill and put a `SKILL.md` file in it:

```text
.dialeng/skills/my-skill/SKILL.md
```

Each `SKILL.md` needs YAML frontmatter with `name` and `description`. Project
skills are discovered only for the OpenAI Codex provider and can reference
additional files inside their own directory.

Dialeng also ships a built-in `skill-creator`. Use the toolbar workflow or
`dialeng skill create` to produce a staged, reviewable skill from notebooks.
"""


def _ensure_recovery_ignored(root: Path) -> bool:
    """Ignore private runtime context without ignoring installed project skills."""
    gitignore = root / ".gitignore"
    entries = (
        ".dialeng/recovery/",
        ".dialeng/skill-builds/",
        ".dialeng/extension-builds/",
    )
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    existing_lines = {line.strip() for line in existing.splitlines()}
    missing = [entry for entry in entries if entry not in existing_lines]
    if not missing:
        return False
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    gitignore.write_text(
        f"{existing}{prefix}\n# Dialeng private runtime context\n"
        + "\n".join(missing)
        + "\n",
        encoding="utf-8",
    )
    return True


def initialize_project(root: Path, name: Optional[str] = None) -> dict:
    """Initialize CRAFT reuse, config, and the project skill directory."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    craft_result = resolve_and_init(root, pkg_name=name)
    if craft_result.get("error"):
        return craft_result

    config_path = root / "dialeng_config.json"
    config_created = not config_path.exists()
    load_config(config_path=config_path, force_reload=True, create_if_missing=True)

    skills_dir = root / ".dialeng" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skills_readme = skills_dir / "README.md"
    skills_readme_created = not skills_readme.exists()
    if skills_readme_created:
        skills_readme.write_text(SKILLS_README, encoding="utf-8")
    gitignore_updated = _ensure_recovery_ignored(root)

    return {
        **craft_result,
        "root": str(root),
        "config_path": str(config_path),
        "config_created": config_created,
        "skills_dir": str(skills_dir),
        "skills_readme_created": skills_readme_created,
        "gitignore_updated": gitignore_updated,
    }
