"""Project-scoped Agent Skills discovery for OpenAI Codex prompts."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

MAX_SKILL_FILE_BYTES = 256 * 1024
MAX_RESOURCE_BYTES = 1024 * 1024
SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
CREATED_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    file_path: Path
    content_hash: str
    disable_model_invocation: bool = False
    source: str = "project"


@dataclass(frozen=True)
class SkillDiagnostic:
    path: str
    message: str


class SkillService:
    """Discover packaged and project-owned skills with project precedence."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        requested_root = self.project_root / ".dialeng" / "skills"
        self.skills_root = requested_root.resolve()
        self.builtin_skills_root = (Path(__file__).parent.parent / "builtin_skills").resolve()
        self._root_error = (
            "skills root resolves outside the project"
            if not self.skills_root.is_relative_to(self.project_root) else None
        )
        self._skills: Dict[str, Skill] = {}
        self._diagnostics: List[SkillDiagnostic] = []
        self._signature: Optional[tuple] = None

    @property
    def skills(self) -> List[Skill]:
        """Return project-owned skills for backwards-compatible project APIs."""
        self.refresh()
        return [skill for skill in self._skills.values() if skill.source == "project"]

    @property
    def all_skills(self) -> List[Skill]:
        """Return packaged built-ins plus project skills after override resolution."""
        self.refresh()
        return list(self._skills.values())

    @property
    def model_skills(self) -> List[Skill]:
        return [skill for skill in self.all_skills if not skill.disable_model_invocation]

    @property
    def diagnostics(self) -> List[SkillDiagnostic]:
        self.refresh()
        return list(self._diagnostics)

    def _directory_signature(self) -> tuple:
        entries = []
        roots = [("builtin", self.builtin_skills_root)]
        if not self._root_error:
            roots.append(("project", self.skills_root))
        for source, root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("SKILL.md")):
                try:
                    stat = path.stat()
                    entries.append((source, str(root), str(path), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    continue
        return tuple(entries)

    def refresh(self, force: bool = False) -> None:
        signature = self._directory_signature()
        if not force and signature == self._signature:
            return
        self._signature = signature
        self._skills = {}
        self._diagnostics = []
        if self._root_error:
            self._diagnostics.append(SkillDiagnostic(str(self.skills_root), self._root_error))
            logger.warning("Skills ignored: %s", self._root_error)

        skill_sources: Dict[str, str] = {}
        for source, root_str, path_str, _, size in signature:
            path = Path(path_str)
            try:
                resolved = path.resolve(strict=True)
                allowed_root = Path(root_str).resolve()
                if not resolved.is_relative_to(allowed_root):
                    raise ValueError(f"skill resolves outside {source} skills root")
                if size > MAX_SKILL_FILE_BYTES:
                    raise ValueError(f"SKILL.md exceeds {MAX_SKILL_FILE_BYTES} bytes")
                skill = validate_skill_file(resolved, allowed_root=allowed_root, source=source)
                name = skill.name
                if name in self._skills:
                    previous_source = skill_sources[name]
                    if previous_source == "builtin" and source == "project":
                        logger.info("Project skill overrides built-in skill: %s", name)
                    else:
                        raise ValueError(f"duplicate skill name '{name}'")
                self._skills[name] = skill
                skill_sources[name] = source
            except Exception as exc:
                diagnostic = SkillDiagnostic(str(path), str(exc))
                self._diagnostics.append(diagnostic)
                logger.warning("Skill ignored (path=%s, reason=%s)", path, exc)

        if self._skills:
            logger.info("Loaded skills: %s", ", ".join(sorted(self._skills)))

    @staticmethod
    def _parse_frontmatter(raw: str) -> tuple[dict, str]:
        if not raw.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter")
        parts = raw.split("---", 2)
        if len(parts) != 3:
            raise ValueError("SKILL.md frontmatter is not closed")
        metadata = yaml.safe_load(parts[1]) or {}
        if not isinstance(metadata, dict):
            raise ValueError("SKILL.md frontmatter must be a mapping")
        return metadata, parts[2]

    def augment_system_prompt(self, system_prompt: str, *, include_creator: bool = True) -> str:
        visible = self.model_skills
        if not include_creator:
            visible = [skill for skill in visible if skill.name != "skill-creator"]
        if not visible:
            return system_prompt
        lines = [
            "<skills>",
            "Use read_skill(name, path) only when a skill description matches the task.",
        ]
        for skill in visible:
            name = self._xml_escape(skill.name)
            description = self._xml_escape(skill.description)
            lines.append(f'  <skill name="{name}"><description>{description}</description></skill>')
        lines.append("</skills>")
        return f"{system_prompt.rstrip()}\n\n" + "\n".join(lines)

    @staticmethod
    def _xml_escape(value: str) -> str:
        return (value.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def read_skill(self, name: str, path: str = "SKILL.md") -> str:
        """Read a registered skill file or resource below its skill directory.

        Args:
            name: Registered skill name.
            path: Relative resource path within that skill directory.
        """
        self.refresh()
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Unknown skill '{name}'")
        if skill.disable_model_invocation:
            raise ValueError(f"Skill '{name}' disables model invocation")
        target = (skill.file_path.parent / path).resolve()
        if not target.is_relative_to(skill.file_path.parent):
            raise ValueError("Skill resource path escapes the skill directory")
        if not target.is_file():
            raise ValueError(f"Skill resource '{path}' does not exist")
        if target.stat().st_size > MAX_RESOURCE_BYTES:
            raise ValueError(f"Skill resource exceeds {MAX_RESOURCE_BYTES} bytes")
        return target.read_text(encoding="utf-8", errors="replace")

    def execution_metadata(self, *, include_creator: bool = True) -> List[dict]:
        return [
            {"name": skill.name, "sha256": skill.content_hash, "source": skill.source}
            for skill in self.all_skills
            if not skill.disable_model_invocation
            and (include_creator or skill.name != "skill-creator")
        ]


def validate_skill_name(name: str) -> str:
    """Validate names for newly generated skills without narrowing legacy discovery."""
    normalized = str(name or "").strip()
    if not CREATED_SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError("Skill name must use lowercase letters, numbers, and hyphens (64 characters max)")
    return normalized


def validate_skill_file(path: Path, *, allowed_root: Path, source: str = "project") -> Skill:
    """Validate one SKILL.md and return its runtime representation."""
    path = Path(path).resolve(strict=True)
    allowed_root = Path(allowed_root).resolve(strict=True)
    if not path.is_relative_to(allowed_root):
        raise ValueError("skill resolves outside its allowed root")
    if path.name != "SKILL.md":
        raise ValueError("skill entrypoint must be named SKILL.md")
    if path.stat().st_size > MAX_SKILL_FILE_BYTES:
        raise ValueError(f"SKILL.md exceeds {MAX_SKILL_FILE_BYTES} bytes")
    raw = path.read_text(encoding="utf-8")
    metadata, body = SkillService._parse_frontmatter(raw)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not SKILL_NAME_RE.fullmatch(name):
        raise ValueError("skill name is missing or invalid")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("skill description is required")
    if not body.strip():
        raise ValueError("skill instructions are required")
    if body.count("\n") + 1 > 500:
        raise ValueError("skill instructions exceed the 500-line limit")
    return Skill(
        name=name,
        description=description.strip(),
        content=body.strip(),
        file_path=path,
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        disable_model_invocation=bool(
            metadata.get("disable-model-invocation", metadata.get("disable_model_invocation", False))
        ),
        source=source,
    )


_skill_service: Optional[SkillService] = None


def set_skill_root(project_root: Path) -> SkillService:
    global _skill_service
    _skill_service = SkillService(project_root)
    from dialeng.services.tool_registry import get_tool_registry
    from dialeng.services.skill_creation_service import (
        build_skill_from_notebooks,
        commit_skill_build,
        inspect_skill_build,
        refresh_skill_build_sources,
        set_skill_creation_root,
        update_skill_build,
    )
    try:
        from dialeng.services.dialeng_config import get_config
        max_source_bytes = get_config().skills_max_source_bytes
    except Exception:
        max_source_bytes = 2 * 1024 * 1024
    set_skill_creation_root(project_root, max_source_bytes=max_source_bytes)
    tool_registry = get_tool_registry()
    tool_registry.register_builtin(_skill_service.read_skill, name="read_skill")
    tool_registry.register_builtin(build_skill_from_notebooks, name="build_skill_from_notebooks")
    tool_registry.register_builtin(inspect_skill_build, name="inspect_skill_build")
    tool_registry.register_builtin(update_skill_build, name="update_skill_build")
    tool_registry.register_builtin(refresh_skill_build_sources, name="refresh_skill_build_sources")
    tool_registry.register_builtin(commit_skill_build, name="commit_skill_build")
    return _skill_service


def get_skill_service() -> SkillService:
    global _skill_service
    if _skill_service is None:
        _skill_service = SkillService(Path.cwd())
    return _skill_service
