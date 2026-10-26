"""Backend-authoritative routes for project skill creation and installation."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from starlette.responses import JSONResponse


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def register_skill_routes(
    rt,
    *,
    project_root: Callable[[], Path],
    config_provider: Callable[[], object],
) -> None:
    """Register staged skill APIs against dynamic project/config providers."""

    @rt("/dialeng/skills")
    def get():
        from dialeng.services.skill_creation_service import get_skill_creation_service
        from dialeng.services.skill_service import get_skill_service

        root = Path(project_root()).resolve()
        service = get_skill_service()
        return {
            "enabled": bool(config_provider().skills_creator_enabled),
            "install_root": config_provider().skills_install_root,
            "link_other_root": config_provider().skills_link_other_root,
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "path": str(skill.file_path),
                }
                for skill in service.all_skills
            ],
            "diagnostics": [diagnostic.__dict__ for diagnostic in service.diagnostics],
            "notebooks": [
                path.relative_to(root).as_posix()
                for path in sorted(root.rglob("*.ipynb"))
                if ".ipynb_checkpoints" not in path.parts and ".dialeng" not in path.parts
            ],
            "source_files": [
                path.relative_to(root).as_posix()
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".css", ".html", ".sh"}
                and ".dialeng" not in path.parts
            ],
            "builds": get_skill_creation_service().list_builds(),
        }

    @rt("/dialeng/skills/build")
    async def post(request):
        from dialeng.services.skill_creation_service import get_skill_creation_service

        if not config_provider().skills_creator_enabled:
            return _error("Skill creation is disabled", 403)
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            sources = payload.get("sources", [])
            if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
                raise ValueError("sources must be a list of project-relative paths or HTTP(S) URLs")
            service = get_skill_creation_service()
            preview = service.create_build(
                str(payload.get("name", "")),
                str(payload.get("description", "")),
                sources,
                instructions=str(payload.get("instructions", "")),
                include_outputs=bool(payload.get("include_outputs", False)),
                install_root=payload.get("install_root", config_provider().skills_install_root),
                link_other_root=payload.get("link_other_root", config_provider().skills_link_other_root),
            )
            if payload.get("use_llm", True):
                try:
                    preview = await service.refine_build(
                        preview["build_id"],
                        provider=str(payload.get("provider") or "") or None,
                        model=str(payload.get("model") or "") or None,
                        instructions=str(payload.get("instructions", "")),
                    )
                except Exception as exc:
                    preview["refinement_error"] = str(exc)
            return preview
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            return _error(str(exc), 422)
        except Exception as exc:
            return _error(str(exc), 500)

    @rt("/dialeng/skills/build/{build_id}")
    def get(build_id: str):  # noqa: F811
        from dialeng.services.skill_creation_service import get_skill_creation_service

        try:
            return get_skill_creation_service().inspect_build(build_id)
        except ValueError as exc:
            return _error(str(exc), 404)

    @rt("/dialeng/skills/build/{build_id}/update")
    async def post(build_id: str, request):  # noqa: F811
        from dialeng.services.skill_creation_service import get_skill_creation_service

        try:
            payload = await request.json()
            return get_skill_creation_service().update_build(
                build_id, str(payload.get("instructions", ""))
            )
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(str(exc), 422)

    @rt("/dialeng/skills/build/{build_id}/refresh")
    def post(build_id: str):  # noqa: F811
        from dialeng.services.skill_creation_service import get_skill_creation_service

        try:
            return get_skill_creation_service().refresh_build_sources(build_id)
        except (ValueError, OSError) as exc:
            return _error(str(exc), 422)

    @rt("/dialeng/skills/build/{build_id}/commit")
    async def post(build_id: str, request):  # noqa: F811
        from dialeng.services.skill_creation_service import get_skill_creation_service

        if not config_provider().skills_creator_enabled:
            return _error("Skill creation is disabled", 403)
        try:
            payload = await request.json()
            return get_skill_creation_service().commit_build(
                build_id,
                str(payload.get("expected_hash", "")),
                overwrite=bool(payload.get("overwrite", False)),
            )
        except FileExistsError as exc:
            return _error(str(exc), 409)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            return _error(str(exc), 422)
