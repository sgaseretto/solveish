"""Staged, reviewable creation of project skills from Dialeng notebooks."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .notebook_context_service import NotebookContextBundle, NotebookContextService, reference_name

logger = logging.getLogger(__name__)
_INSTALL_LOCK = threading.RLock()


@dataclass(frozen=True)
class SkillBuild:
    build_id: str
    name: str
    description: str
    sources: tuple[str, ...]
    include_outputs: bool
    build_hash: str
    refined_by_llm: bool = False
    provider: str = ""
    model: str = ""
    warnings: tuple[str, ...] = ()
    install_root: str = "dialeng"
    link_other_root: bool = False


class SkillCreationService:
    """Create skills in staging and install them only after explicit confirmation."""

    def __init__(self, project_root: Path, *, max_source_bytes: int = 2 * 1024 * 1024) -> None:
        self.project_root = Path(project_root).resolve()
        self.skills_root = self.project_root / ".dialeng" / "skills"
        self.builds_root = self.project_root / ".dialeng" / "skill-builds"
        self.context_service = NotebookContextService(
            self.project_root, max_source_bytes=max_source_bytes
        )

    def create_build(
        self,
        name: str,
        description: str,
        sources: list[str],
        *,
        instructions: str = "",
        include_outputs: bool = False,
        install_root: Optional[str] = None,
        link_other_root: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Extract notebooks and stage a deterministic skill draft for review.

        Args:
            name: Lowercase skill name using letters, numbers, and hyphens.
            description: Specific description of when the skill should be used.
            sources: Project-relative notebooks/text files, glob patterns, or web URLs.
            instructions: Optional additional workflow requirements for the generated skill.
            include_outputs: Include bounded text outputs. Disabled by default.
            install_root: Original's folder: dialeng or agents. Defaults to project settings.
            link_other_root: Link from the other skills folder. Defaults to project settings.
        """
        from .skill_service import validate_skill_name
        from .dialeng_config import get_config

        name = validate_skill_name(name)
        config = get_config()
        install_root = config.skills_install_root if install_root is None else install_root
        link_other_root = config.skills_link_other_root if link_other_root is None else link_other_root
        self._validate_install_options(install_root, link_other_root)
        description = str(description or "").strip()
        if not description:
            raise ValueError("Skill description is required")
        bundle = self.context_service.extract(sources, include_outputs=include_outputs)
        build_id = uuid.uuid4().hex[:12]
        build_dir = self._build_dir(build_id)
        build_dir.mkdir(parents=True, exist_ok=False)

        references: list[str] = []
        redaction_count = 0
        for document in bundle.documents:
            filename = reference_name(document.source)
            references.append(filename)
            self._write_text(build_dir / "references" / filename, document.markdown)
            redaction_count += document.redaction_count

        skill_md = self._compose_skill_md(
            name,
            description,
            bundle,
            references,
            instructions=instructions,
        )
        self._write_text(build_dir / "SKILL.md", skill_md)
        provenance = {
            "format": 1,
            "sources": [
                {
                    "path": document.source,
                    "sha256": document.sha256,
                    "cells": document.cell_count,
                    "kind": document.kind,
                    "fetched_at": document.fetched_at or None,
                }
                for document in bundle.documents
            ],
            "include_outputs": include_outputs,
        }
        self._write_text(
            build_dir / ".dialeng-sources.json",
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        warnings = []
        if redaction_count:
            warnings.append(f"Redacted {redaction_count} likely secret value(s) from notebook context")
        build = SkillBuild(
            build_id=build_id,
            name=name,
            description=description,
            sources=tuple(document.source for document in bundle.documents),
            include_outputs=include_outputs,
            build_hash=self._directory_hash(build_dir),
            warnings=tuple(warnings),
            install_root=install_root,
            link_other_root=link_other_root,
        )
        self._write_manifest(build_dir, build)
        logger.info(
            "Staged source-derived skill (build=%s, skill=%s, sources=%d, bytes=%d)",
            build_id,
            name,
            len(bundle.documents),
            bundle.total_bytes,
        )
        return self.inspect_build(build_id)

    async def refine_build(
        self,
        build_id: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        instructions: str = "",
    ) -> dict[str, Any]:
        """Use a Dialeng LLM provider to improve only the staged SKILL.md body."""
        from dialeng.services.dialeng_config import get_config
        from dialeng.services.llm import llm_service

        build = self._load_build(build_id)
        build_dir = self._build_dir(build_id)
        provider_id = llm_service.resolve_provider_id(provider)
        config = get_config()
        selected_model = model or config.get_default_model(provider_id)
        references = [
            path.read_text(encoding="utf-8")
            for path in sorted((build_dir / "references").glob("*.md"))
        ]
        source_context = "\n\n".join(references)
        prompt = (
            "Create concise, actionable instructions for a Dialeng Agent Skill from the "
            "source references below. Return JSON only with one string field named "
            "instructions. Do not include YAML frontmatter. Keep the instructions below "
            "500 lines, tell the agent when to read the listed references, preserve important "
            "workflows and constraints, and do not claim capabilities absent from the sources.\n\n"
            f"Skill name: {build.name}\nDescription: {build.description}\n"
            f"Additional requirements: {instructions or 'None'}\n\n"
            f"<notebook_context>\n{source_context}\n</notebook_context>"
        )
        chunks: list[str] = []
        errors: list[str] = []
        async for event in llm_service.stream_response(
            prompt,
            [],
            "concise",
            selected_model,
            provider_id=provider_id,
        ):
            if event.get("type") == "chunk":
                chunks.append(str(event.get("content", "")))
            elif event.get("type") == "error":
                errors.append(str(event.get("content", "LLM refinement failed")))
        if errors:
            raise RuntimeError(errors[-1])
        payload = self._parse_json_response("".join(chunks))
        body = str(payload.get("instructions", "")).strip()
        if not body:
            raise ValueError("LLM refinement returned empty skill instructions")
        self.update_build(build_id, body)
        updated = self._load_build(build_id)
        updated = SkillBuild(
            **{
                **asdict(updated),
                "refined_by_llm": True,
                "provider": provider_id,
                "model": selected_model,
                "build_hash": self._directory_hash(build_dir),
            }
        )
        self._write_manifest(build_dir, updated)
        logger.info(
            "Refined notebook-derived skill (build=%s, provider=%s, model=%s)",
            build_id,
            provider_id,
            selected_model,
        )
        return self.inspect_build(build_id)

    def update_build(self, build_id: str, instructions: str) -> dict[str, Any]:
        """Replace staged skill instructions while preserving trusted frontmatter.

        Args:
            build_id: Staged build identifier.
            instructions: Complete Markdown body for SKILL.md, without frontmatter.
        """
        build = self._load_build(build_id)
        body = str(instructions or "").strip()
        if not body:
            raise ValueError("Skill instructions cannot be empty")
        if body.count("\n") + 1 > 500:
            raise ValueError("Skill instructions exceed the 500-line limit")
        build_dir = self._build_dir(build_id)
        self._write_text(
            build_dir / "SKILL.md",
            self._frontmatter(build.name, build.description) + body + "\n",
        )
        updated = SkillBuild(
            **{**asdict(build), "build_hash": self._directory_hash(build_dir)}
        )
        self._write_manifest(build_dir, updated)
        return self.inspect_build(build_id)

    def refresh_build_sources(self, build_id: str) -> dict[str, Any]:
        """Re-extract staged local and web sources while preserving reviewed instructions."""
        build = self._load_build(build_id)
        build_dir = self._build_dir(build_id)
        bundle = self.context_service.extract(
            build.sources, include_outputs=build.include_outputs,
        )
        references_dir = build_dir / "references"
        shutil.rmtree(references_dir, ignore_errors=True)
        for document in bundle.documents:
            self._write_text(
                references_dir / reference_name(document.source), document.markdown,
            )
        provenance = {
            "format": 1,
            "sources": [
                {
                    "path": document.source,
                    "sha256": document.sha256,
                    "cells": document.cell_count,
                    "kind": document.kind,
                    "fetched_at": document.fetched_at or None,
                }
                for document in bundle.documents
            ],
            "include_outputs": build.include_outputs,
        }
        self._write_text(
            build_dir / ".dialeng-sources.json",
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        updated = SkillBuild(
            **{**asdict(build), "build_hash": self._directory_hash(build_dir)}
        )
        self._write_manifest(build_dir, updated)
        logger.info("Refreshed staged skill sources (build=%s, sources=%d)", build_id, len(bundle.documents))
        return self.inspect_build(build_id)

    def inspect_build(self, build_id: str) -> dict[str, Any]:
        """Return a staged skill preview, validation state, and content hash."""
        build = self._load_build(build_id)
        build_dir = self._build_dir(build_id)
        files = {}
        for path in sorted(build_dir.rglob("*")):
            if path.is_file() and path.name != "build.json":
                relative = path.relative_to(build_dir).as_posix()
                files[relative] = path.read_text(encoding="utf-8", errors="replace")
        errors = self.validate_build(build_id)
        current_hash = self._preview_hash(build_dir, build)
        other_root = "agents" if build.install_root == "dialeng" else "dialeng"
        return {
            **asdict(build),
            "build_hash": current_hash,
            "files": files,
            "validation_errors": errors,
            "valid": not errors,
            "target": f".{build.install_root}/skills/{build.name}",
            "link_target": f".{other_root}/skills/{build.name}" if build.link_other_root else None,
        }

    def validate_build(self, build_id: str) -> list[str]:
        """Validate a staged build using the same rules as runtime discovery."""
        from .skill_service import validate_skill_file

        build_dir = self._build_dir(build_id)
        errors: list[str] = []
        try:
            validate_skill_file(build_dir / "SKILL.md", allowed_root=build_dir)
        except Exception as exc:
            errors.append(str(exc))
        for path in build_dir.rglob("*"):
            if path.is_symlink():
                errors.append(f"Symlinks are not allowed in generated skills: {path.name}")
            elif path.is_file() and not path.resolve().is_relative_to(build_dir.resolve()):
                errors.append(f"Generated file escapes build directory: {path.name}")
        return errors

    def commit_build(
        self,
        build_id: str,
        expected_hash: str = "",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Install a validated staged skill into the project.

        Args:
            build_id: Staged build identifier.
            expected_hash: Preview hash used to reject stale confirmations.
            overwrite: Replace an existing project skill. Disabled by default.
        """
        from .skill_service import get_skill_service

        preview = self.inspect_build(build_id)
        if preview["validation_errors"]:
            raise ValueError("Skill build is invalid: " + "; ".join(preview["validation_errors"]))
        if expected_hash and expected_hash != preview["build_hash"]:
            raise ValueError("Skill build changed after preview; inspect it again before installing")
        build = self._load_build(build_id)
        source = self._build_dir(build_id)
        with _INSTALL_LOCK:
            target, link = self._install_files(build, source, overwrite=overwrite)

        get_skill_service().refresh(force=True)
        shutil.rmtree(source, ignore_errors=True)
        logger.info(
            "Installed notebook-derived skill (build=%s, target=%s, sources=%d)",
            build.name,
            target,
            len(build.sources),
        )
        return {
            "ok": True,
            "name": build.name,
            "path": str(target.relative_to(self.project_root)),
            "link_path": str(link.relative_to(self.project_root)) if link else None,
            "sources": list(build.sources),
        }

    @staticmethod
    def _validate_install_options(install_root: str, link_other_root: bool) -> None:
        if install_root not in ("dialeng", "agents"):
            raise ValueError("install_root must be 'dialeng' or 'agents'")
        if not isinstance(link_other_root, bool):
            raise ValueError("link_other_root must be a boolean")

    def _install_files(self, build: SkillBuild, source: Path, *, overwrite: bool) -> tuple[Path, Optional[Path]]:
        """Install the original and optional relative alias, restoring the original on failure."""
        root = self.project_root / f".{build.install_root}" / "skills"
        other = "agents" if build.install_root == "dialeng" else "dialeng"
        link_root = self.project_root / f".{other}" / "skills"
        for directory in [root] + ([link_root] if build.link_other_root else []):
            if not directory.resolve().is_relative_to(self.project_root):
                raise ValueError("Skill install root escapes the project")
        if build.link_other_root and root.resolve() == link_root.resolve():
            raise ValueError("Skill original and link folders resolve to the same directory")

        target = root / build.name
        link = link_root / build.name if build.link_other_root else None
        if target.is_symlink():
            raise ValueError("Skill original is a symlink; select its original install_root to update it")
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Skill '{build.name}' already exists")
            if not target.is_dir():
                raise ValueError("Skill original must be a directory")
        # Never replace an independent skill or an unrelated/dangling link in the other root.
        if link and (link.exists() or link.is_symlink()):
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise FileExistsError(f"Skill link path '{link.relative_to(self.project_root)}' is already occupied")

        root.mkdir(parents=True, exist_ok=True)
        if link:
            link_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{build.name}-", dir=root))
        backup = root / f".{build.name}.backup-{uuid.uuid4().hex[:8]}"
        installed = False
        linked = False
        try:
            for path in source.rglob("*"):
                if path.name == "build.json":
                    continue
                relative = path.relative_to(source)
                destination = temp_dir / relative
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif path.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
            if target.exists():
                os.replace(target, backup)
            os.replace(temp_dir, target)
            installed = True
            if link and not link.is_symlink():
                link.symlink_to(os.path.relpath(target, start=link.parent), target_is_directory=True)
                linked = True
        except Exception:
            if linked:
                link.unlink()
            if installed:
                shutil.rmtree(target)
            if backup.exists():
                os.replace(backup, target)
            raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return target, link

    def list_builds(self) -> list[dict[str, Any]]:
        if not self.builds_root.exists():
            return []
        builds = []
        for path in sorted(self.builds_root.iterdir()):
            if path.is_dir() and (path / "build.json").is_file():
                try:
                    builds.append(asdict(self._load_build(path.name)))
                except Exception:
                    logger.warning("Ignoring malformed skill build: %s", path)
        return builds

    def _compose_skill_md(
        self,
        name: str,
        description: str,
        bundle: NotebookContextBundle,
        references: list[str],
        *,
        instructions: str,
    ) -> str:
        title = name.replace("-", " ").title()
        source_list = "\n".join(f"- `references/{item}`" for item in references)
        requirements = instructions.strip() or (
            "Use the source references as authoritative project context. Extract the applicable "
            "workflow, preserve demonstrated constraints, and verify assumptions against the references."
        )
        body = (
            f"# {title}\n\n"
            "## Workflow\n\n"
            f"{requirements}\n\n"
            "1. Identify which source reference matches the task.\n"
            "2. Read only the relevant reference file before acting.\n"
            "3. Follow the demonstrated workflow and adapt names or paths to the current project.\n"
            "4. State when the sources do not cover a required decision.\n\n"
            "## Source References\n\n"
            f"{source_list}\n"
        )
        return self._frontmatter(name, description) + body

    @staticmethod
    def _frontmatter(name: str, description: str) -> str:
        metadata = yaml.safe_dump(
            {"name": name, "description": description},
            sort_keys=False,
            allow_unicode=True,
        ).strip()
        return f"---\n{metadata}\n---\n"

    def _build_dir(self, build_id: str) -> Path:
        if not build_id or any(char not in "0123456789abcdef" for char in build_id):
            raise ValueError("Invalid skill build identifier")
        path = (self.builds_root / build_id).resolve()
        if not path.is_relative_to(self.builds_root.resolve()):
            raise ValueError("Skill build path escapes .dialeng/skill-builds")
        return path

    def _load_build(self, build_id: str) -> SkillBuild:
        path = self._build_dir(build_id) / "build.json"
        if not path.is_file():
            raise ValueError(f"Unknown skill build '{build_id}'")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sources"] = tuple(data.get("sources", ()))
        data["warnings"] = tuple(data.get("warnings", ()))
        build = SkillBuild(**data)
        from .skill_service import validate_skill_name
        validate_skill_name(build.name)
        self._validate_install_options(build.install_root, build.link_other_root)
        return build

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    def _write_manifest(self, build_dir: Path, build: SkillBuild) -> None:
        self._write_text(
            build_dir / "build.json",
            json.dumps({**asdict(build), "build_hash": self._preview_hash(build_dir, build)}, indent=2, sort_keys=True) + "\n",
        )

    def _preview_hash(self, directory: Path, build: SkillBuild) -> str:
        plan = {"name": build.name, "install_root": build.install_root, "link_other_root": build.link_other_root}
        return hashlib.sha256(
            (self._directory_hash(directory) + json.dumps(plan, sort_keys=True)).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _directory_hash(directory: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name == "build.json":
                continue
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _parse_json_response(value: str) -> dict[str, Any]:
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM refinement did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("LLM refinement response must be a JSON object")
        return result


_skill_creation_service: Optional[SkillCreationService] = None


def set_skill_creation_root(project_root: Path, *, max_source_bytes: int = 2 * 1024 * 1024) -> SkillCreationService:
    global _skill_creation_service
    _skill_creation_service = SkillCreationService(project_root, max_source_bytes=max_source_bytes)
    return _skill_creation_service


def get_skill_creation_service() -> SkillCreationService:
    global _skill_creation_service
    if _skill_creation_service is None:
        _skill_creation_service = SkillCreationService(Path.cwd())
    return _skill_creation_service


def build_skill_from_notebooks(
    name: str,
    description: str,
    sources: list[str],
    instructions: str = "",
    include_outputs: bool = False,
    install_root: Optional[str] = None,
    link_other_root: Optional[bool] = None,
) -> dict[str, Any]:
    """Stage a reviewable project skill from one or more notebooks.

    Args:
        name: Lowercase skill name.
        description: Specific trigger description for the skill.
        sources: Project-relative notebook files, directories, or glob patterns.
        instructions: Optional workflow requirements.
        include_outputs: Include bounded text outputs; normally leave false.
        install_root: Original's folder: dialeng or agents; omit to use project settings.
        link_other_root: Create a relative symlink in the other folder; omit to use project settings.
    """
    return get_skill_creation_service().create_build(
        name,
        description,
        sources,
        instructions=instructions,
        include_outputs=include_outputs,
        install_root=install_root,
        link_other_root=link_other_root,
    )


def inspect_skill_build(build_id: str) -> dict[str, Any]:
    """Inspect all files and validation results for a staged skill build.

    Args:
        build_id: Identifier returned by build_skill_from_notebooks.
    """
    return get_skill_creation_service().inspect_build(build_id)


def update_skill_build(build_id: str, instructions: str) -> dict[str, Any]:
    """Replace the Markdown instructions in a staged skill before installation.

    Args:
        build_id: Identifier returned by build_skill_from_notebooks.
        instructions: Complete SKILL.md body without YAML frontmatter.
    """
    return get_skill_creation_service().update_build(build_id, instructions)


def refresh_skill_build_sources(build_id: str) -> dict[str, Any]:
    """Refresh local files and web pages for a staged skill build.

    Args:
        build_id: Identifier returned by build_skill_from_notebooks.
    """
    return get_skill_creation_service().refresh_build_sources(build_id)


def commit_skill_build(build_id: str, expected_hash: str) -> dict[str, Any]:
    """Install a skill at its reviewed target and optionally link it in the other root.

    Args:
        build_id: Identifier returned by build_skill_from_notebooks.
        expected_hash: Hash from the latest inspect_skill_build result.
    """
    return get_skill_creation_service().commit_build(build_id, expected_hash)
