"""Backend-authoritative extension management and invocation routes."""
# ruff: noqa: F811
from __future__ import annotations

import asyncio
import inspect
import logging
import mimetypes
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.responses import FileResponse, JSONResponse, Response

from dialeng.extensions.manager import ExtensionManager, ExtensionNotTrustedError
from dialeng.extensions.resources import actions, commands, panels, status_items
from dialeng.services.extension_project_service import ExtensionProjectService

logger = logging.getLogger(__name__)


def _error(message: str, status_code: int = 400, *, kind: str = "extension_error") -> JSONResponse:
    from dialeng.extensions.diagnostics import redact_text
    return JSONResponse(
        {"ok": False, "error": redact_text(message), "type": kind},
        status_code=status_code,
    )


async def _payload(request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            value = await request.json()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    try:
        return dict(await request.form())
    except Exception:
        return {}


async def _invoke(handler, *args, **kwargs):
    result = handler(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _validate_object(payload: dict, schema: dict, *, label: str) -> None:
    if not schema:
        return
    if schema.get("type", "object") != "object":
        raise ValueError(f"{label} schema root must be an object")
    required = schema.get("required", [])
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Missing required {label} field(s): {', '.join(missing)}")
    python_types = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "object": dict, "array": list,
    }
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(payload) - set(properties))
        if unexpected:
            raise ValueError(f"Unexpected {label} field(s): {', '.join(unexpected)}")
    for name, definition in properties.items():
        if name not in payload or not isinstance(definition, dict):
            continue
        expected = python_types.get(definition.get("type"))
        value = payload[name]
        numeric_bool = definition.get("type") in {"integer", "number"} and isinstance(value, bool)
        if expected and (not isinstance(value, expected) or numeric_bool):
            raise ValueError(f"{label} field {name!r} must be {definition['type']}")
        if "enum" in definition and value not in definition["enum"]:
            raise ValueError(f"{label} field {name!r} must be one of {definition['enum']}")
        if isinstance(value, str):
            if len(value) < int(definition.get("minLength", 0)):
                raise ValueError(f"{label} field {name!r} is too short")
            if "maxLength" in definition and len(value) > int(definition["maxLength"]):
                raise ValueError(f"{label} field {name!r} is too long")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in definition and value < definition["minimum"]:
                raise ValueError(f"{label} field {name!r} is below its minimum")
            if "maximum" in definition and value > definition["maximum"]:
                raise ValueError(f"{label} field {name!r} exceeds its maximum")
        if isinstance(value, list) and isinstance(definition.get("items"), dict):
            item_type = python_types.get(definition["items"].get("type"))
            if item_type and any(not isinstance(item, item_type) for item in value):
                raise ValueError(f"{label} field {name!r} contains an invalid item")


def register_extension_routes(
    rt,
    *,
    manager_provider: Callable[[], ExtensionManager],
    broadcast_all_json: Callable[[dict], Awaitable[None]],
    command_available: Callable[[str, str], bool] | None = None,
) -> None:
    active_commands: dict[str, tuple[asyncio.Task, asyncio.Event, str]] = {}

    async def cancel_owner_commands(extension_id: str) -> None:
        pending = []
        for task, cancel_event, owner in tuple(active_commands.values()):
            if owner != extension_id:
                continue
            cancel_event.set()
            task.cancel()
            pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def bind_manager_hooks(manager: ExtensionManager) -> None:
        manager.context_services["cancel_extension_commands"] = cancel_owner_commands

    bind_manager_hooks(manager_provider())

    def active_owner(manager: ExtensionManager, extension_id: str) -> bool:
        return extension_id in manager.active or extension_id in manager.store.owners

    async def changed(manager: ExtensionManager) -> dict:
        snapshot = manager.snapshot()
        await broadcast_all_json({"type": "extension_snapshot", **snapshot})
        return snapshot

    @rt("/dialeng/extensions")
    async def get():
        manager = manager_provider()
        manager.refresh()
        return JSONResponse({"ok": True, **manager.snapshot()})

    @rt("/dialeng/extensions/doctor")
    async def get():
        manager = manager_provider()
        return JSONResponse(manager.doctor())

    @rt("/dialeng/extensions/contracts")
    async def get():
        from dialeng.extensions.contracts import contract_snapshot

        return JSONResponse({"ok": True, **contract_snapshot()})

    @rt("/dialeng/extensions/{extension_id}/trust")
    async def post(extension_id: str, request):
        manager = manager_provider()
        manager.refresh()
        try:
            source = manager._source(extension_id)
            body = await _payload(request)
            mode = str(body.get("mode") or "version")
            manager.trust_store.trust(source, mode)
            from dialeng.extensions.events import event_bus
            await event_bus.emit("project.trust_changed", {
                "project_root": str(manager.project_root),
                "extension_id": extension_id,
                "mode": mode,
                "trusted": True,
            })
            return JSONResponse({"ok": True, **await changed(manager)})
        except (KeyError, ValueError) as exc:
            return _error(str(exc), 404 if isinstance(exc, KeyError) else 400)

    @rt("/dialeng/extensions/{extension_id}/enable")
    async def post(extension_id: str):
        manager = manager_provider()
        try:
            await manager.enable(extension_id)
            return JSONResponse({"ok": True, **await changed(manager)})
        except ExtensionNotTrustedError as exc:
            return _error(str(exc), 403, kind="extension_not_trusted")
        except KeyError as exc:
            return _error(str(exc), 404)
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 200, kind="extension_activation_failed")

    @rt("/dialeng/extensions/{extension_id}/disable")
    async def post(extension_id: str):
        manager = manager_provider()
        await manager.disable_persisted(extension_id)
        return JSONResponse({"ok": True, **await changed(manager)})

    @rt("/dialeng/extensions/{extension_id}/reload")
    async def post(extension_id: str):
        manager = manager_provider()
        try:
            await manager.activate(extension_id, require_trust=True)
            return JSONResponse({"ok": True, **await changed(manager)})
        except ExtensionNotTrustedError as exc:
            return _error(str(exc), 403, kind="extension_not_trusted")
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 200, kind="extension_reload_failed")

    @rt("/dialeng/extensions/{extension_id}/resources/{kind}/{action}")
    async def post(extension_id: str, kind: str, action: str):
        if action not in {"enable", "disable"}:
            return _error("Resource action must be enable or disable", 400)
        manager = manager_provider()
        try:
            manager.set_resource_enabled(extension_id, kind, action == "enable")
            return JSONResponse({"ok": True, **await changed(manager)})
        except KeyError as exc:
            return _error(str(exc), 404)
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 422, kind="extension_resource_failed")

    @rt("/dialeng/extensions/studio/templates")
    async def get():
        return JSONResponse({"ok": True, "templates": ExtensionProjectService.templates()})

    @rt("/dialeng/extensions/studio/create")
    async def post(request):
        manager = manager_provider()
        body = await _payload(request)
        service = ExtensionProjectService(manager.project_root)
        try:
            result = service.create(
                str(body.get("name") or ""),
                template=str(body.get("template") or "basic"),
                extension_id=str(body.get("extension_id") or "") or None,
            )
            manager.refresh()
            return JSONResponse({"ok": True, "build": result, **await changed(manager)})
        except FileExistsError as exc:
            return _error(str(exc), 409, kind="extension_exists")
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 422, kind="extension_create_failed")

    @rt("/dialeng/extensions/studio/{extension_id}/build")
    async def post(extension_id: str):
        manager = manager_provider()
        service = ExtensionProjectService(manager.project_root)
        try:
            root = service.resolve_extension(extension_id)
            result = service.build(root)
            manager.refresh()
            return JSONResponse({"ok": True, "build": result, **await changed(manager)})
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 422, kind="extension_build_failed")

    @rt("/dialeng/extensions/studio/{extension_id}/prepare")
    async def post(extension_id: str):
        manager = manager_provider()
        service = ExtensionProjectService(manager.project_root)
        try:
            result = service.prepare(service.resolve_extension(extension_id))
            manager.refresh()
            return JSONResponse({"ok": bool(result.get("ok")), "result": result, **await changed(manager)})
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 422, kind="extension_prepare_failed")

    @rt("/dialeng/extensions/studio/{extension_id}/diagnostics")
    async def get(extension_id: str):
        manager = manager_provider()
        service = ExtensionProjectService(manager.project_root)
        try:
            return JSONResponse({"ok": True, **service.diagnostics(extension_id)})
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 404, kind="extension_diagnostics_failed")

    @rt("/dialeng/{nb_id}/extensions/{extension_id}/actions/{action_name}")
    async def post(nb_id: str, extension_id: str, action_name: str, request):
        manager = manager_provider()
        if not active_owner(manager, extension_id):
            return _error("Extension is not active", 409)
        from dialeng.core.registry import registry
        registration = actions.get(f"{extension_id}:{action_name}")
        legacy_handler = registry.actions.get(f"{extension_id}:{action_name}")
        if registration is None and legacy_handler is None:
            return _error("Unknown extension action", 404)
        operation_id = f"ext-action-{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        try:
            body = await _payload(request)
            if registration:
                _validate_object(body, registration.request_schema, label="request")
                result = await _invoke(registration.handler, nb_id=nb_id, **body)
                if isinstance(result, dict):
                    _validate_object(result, registration.response_schema, label="response")
            else:
                result = await _invoke(legacy_handler, nb_id=nb_id, **body)
            logger.info(
                "Extension action completed (operation_id=%s, extension=%s, action=%s, duration_ms=%.1f)",
                operation_id, extension_id, action_name, (time.perf_counter() - started) * 1000,
            )
            payload = result if isinstance(result, dict) else {"result": result}
            return JSONResponse(payload, headers={"X-Dialeng-Operation-Id": operation_id})
        except ValueError as exc:
            return _error(str(exc), 422, kind="extension_action_validation_failed")
        except Exception as exc:
            logger.exception(
                "Extension action failed (operation_id=%s, extension=%s, action=%s)",
                operation_id, extension_id, action_name,
            )
            return _error(f"{type(exc).__name__}: {exc}", 500, kind="extension_action_failed")

    @rt("/dialeng/{nb_id}/extensions/commands/{extension_id}/{command_name}")
    async def post(nb_id: str, extension_id: str, command_name: str, request):
        manager = manager_provider()
        bind_manager_hooks(manager)
        if not active_owner(manager, extension_id):
            return _error("Extension is not active", 409)
        command = commands.get(f"{extension_id}:{command_name}")
        if command is None:
            return _error("Unknown extension command", 404)
        if command_available and not command_available(command.availability, nb_id):
            return _error(
                f"Command is unavailable while {command.availability!r} is not satisfied",
                409,
                kind="extension_command_unavailable",
            )
        requested_operation_id = request.headers.get("x-dialeng-operation-id", "")
        operation_id = (
            requested_operation_id
            if re.fullmatch(r"ext-command-[A-Za-z0-9-]{8,48}", requested_operation_id)
            and requested_operation_id not in active_commands
            else f"ext-command-{uuid.uuid4().hex[:10]}"
        )
        cancel_event = asyncio.Event()

        async def report_progress(
            message: str,
            *,
            current: float | None = None,
            total: float | None = None,
        ) -> None:
            await broadcast_all_json({
                "type": "extension_command_progress",
                "operation_id": operation_id,
                "extension_id": extension_id,
                "command": command_name,
                "notebook_id": nb_id,
                "message": str(message),
                "current": current,
                "total": total,
            })

        context = {
            "operation_id": operation_id,
            "notebook_id": nb_id,
            "project_root": str(manager.project_root),
            "args": await _payload(request),
            "cancel_event": cancel_event,
            "report_progress": report_progress,
        }
        started = time.perf_counter()
        try:
            _validate_object(context["args"], command.args_schema, label="argument")
            task = asyncio.create_task(
                _invoke(command.handler, context),
                name=f"extension-command:{extension_id}:{command_name}:{operation_id}",
            )
            active_commands[operation_id] = (task, cancel_event, extension_id)
            await report_progress("Started")
            result = await task
            await report_progress("Completed")
            logger.info(
                "Extension command completed (operation_id=%s, extension=%s, command=%s, duration_ms=%.1f)",
                operation_id, extension_id, command_name, (time.perf_counter() - started) * 1000,
            )
            payload = result if isinstance(result, dict) else {"result": result}
            return JSONResponse(
                payload,
                headers={"X-Dialeng-Operation-Id": operation_id},
            )
        except asyncio.CancelledError:
            await report_progress("Cancelled")
            return _error("Command was cancelled", 200, kind="extension_command_cancelled")
        except ValueError as exc:
            return _error(str(exc), 422, kind="extension_command_validation_failed")
        except Exception as exc:
            logger.exception(
                "Extension command failed (operation_id=%s, extension=%s, command=%s)",
                operation_id, extension_id, command_name,
            )
            return _error(f"{type(exc).__name__}: {exc}", 500, kind="extension_command_failed")
        finally:
            active_commands.pop(operation_id, None)

    @rt("/dialeng/extensions/commands/{operation_id}/cancel")
    async def post(operation_id: str):
        active = active_commands.get(operation_id)
        if active is None:
            return _error("Extension command is not active", 404, kind="extension_command_not_active")
        task, cancel_event, _extension_id = active
        cancel_event.set()
        task.cancel()
        return JSONResponse({"ok": True, "operation_id": operation_id, "cancelled": True})

    @rt("/dialeng/{nb_id}/extensions/surfaces/{kind}/{extension_id}/{surface_name}")
    async def get(nb_id: str, kind: str, extension_id: str, surface_name: str):
        manager = manager_provider()
        if not active_owner(manager, extension_id):
            return Response(status_code=204)
        collection = panels if kind == "panel" else status_items if kind == "status" else None
        if collection is None:
            return _error("Unknown extension surface", 404)
        registration = collection.get(f"{extension_id}:{surface_name}")
        if registration is None:
            return Response(status_code=204)
        context = {
            "notebook_id": nb_id,
            "project_root": str(manager.project_root),
            "extension_id": extension_id,
        }
        try:
            return await _invoke(registration.renderer, context)
        except Exception as exc:
            return _error(f"{type(exc).__name__}: {exc}", 500, kind="extension_surface_failed")

    @rt("/dialeng/extensions/{extension_id}/assets/{asset_path:path}")
    async def get(extension_id: str, asset_path: str):
        manager = manager_provider()
        active = manager.active.get(extension_id)
        if active is None:
            return _error("Extension is not active", 404)
        declared = {
            item
            for values in active.source.manifest.assets.values()
            for item in values
        }
        if asset_path not in declared:
            return _error("Asset is not declared by the extension", 404)
        candidate = (active.source.root / asset_path).resolve()
        try:
            candidate.relative_to(active.source.root.resolve())
        except ValueError:
            return _error("Asset path escapes extension root", 400)
        if not candidate.is_file():
            return _error("Extension asset not found", 404)
        media_type = mimetypes.guess_type(candidate.name)[0]
        return FileResponse(
            candidate,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @rt("/dialeng/extensions/{extension_id}/asset")
    async def get(extension_id: str, path: str = ""):
        """Serve a declared asset through a query path FastHTML handles portably."""
        manager = manager_provider()
        active = manager.active.get(extension_id)
        if active is None:
            return _error("Extension is not active", 404)
        declared = {
            item for values in active.source.manifest.assets.values() for item in values
        }
        if path not in declared:
            return _error("Asset is not declared by the extension", 404)
        candidate = (active.source.root / path).resolve()
        if not candidate.is_relative_to(active.source.root.resolve()) or not candidate.is_file():
            return _error("Extension asset not found", 404)
        return FileResponse(
            candidate,
            media_type=mimetypes.guess_type(candidate.name)[0],
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
