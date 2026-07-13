"""Notebook-first extension scaffolding, export, and isolated validation."""
from __future__ import annotations

import json
import hashlib
import ast
import os
import re
import subprocess
import sys
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dialeng.core.extensions import extract_extension
from dialeng.extensions.discovery import discover_extensions
from dialeng.extensions.manifest import load_extension_manifest
from dialeng.extensions.manifest import extension_source_hash
from dialeng.extensions.contracts import (
    API_POLICY,
    LEGACY_IMPORT_DIAGNOSTICS,
    PRIVILEGED_IN_PROCESS_API_METHODS,
    RPC_SAFE_API_METHODS,
)


_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_TEMPLATES = {"basic", "tool", "cell", "lifecycle", "provider", "package"}
_EDITABLE_SUFFIXES = {".ipynb", ".py", ".md", ".toml", ".json", ".css", ".js"}


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist validation evidence without exposing partial records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ExtensionBuildResult:
    extension_id: str
    root: str
    manifest: str
    notebook: str
    module: str
    exported_cells: int
    validation: dict[str, Any]
    notebook_name: str = ""


def _setup_source(template: str, extension_id: str) -> str:
    if template == "tool":
        return '''#| export
def inspect_value(value: str) -> dict:
    """Return a deterministic inspection of one value."""
    return {{"value": value, "length": len(value)}}

def setup(api, ctx):
    api.register_tool("inspect-value", inspect_value)
'''
    if template == "lifecycle":
        return '''#| export
def setup(api, ctx):
    def on_saved(payload):
        ctx.logger.info("Notebook saved: %s", payload.get("notebook_id"))
    api.on("notebook.saved", on_saved, name="log-save")
'''
    if template == "cell":
        return '''#| export
from dialeng.extensions import ExtensionCellType
from dialeng.document.cell import Cell
from fasthtml.common import Div, Pre

class ExampleCell(Cell):
    cell_type = "example-cell"

def render_example(cell, notebook_id):
    return Div(Pre(cell.source), id=f"cell-{cell.id}", cls="cell example-cell")

def serialize_example(cell):
    return {"cell_type": "raw", "source": cell.source, "metadata": {"dialeng_cell_type": "example-cell"}}

def deserialize_example(raw):
    if raw.get("metadata", {}).get("dialeng_cell_type") != "example-cell": return None
    source = raw.get("source", "")
    if isinstance(source, list): source = "".join(source)
    return ExampleCell(id=raw.get("id", "example"), source=source)

def example_to_messages(cell):
    return [{"role": "user", "content": f"[Example cell]\\n{cell.source}"}]

def setup(api, ctx):
    api.register_cell_type(
        ExtensionCellType(ExampleCell, "example-cell", "file-code", "Example"),
        renderer=render_example,
        serializer=serialize_example,
        deserializer=deserialize_example,
        llm_converter=example_to_messages,
    )
'''
    if template == "provider":
        return '''#| export
from dialeng.extensions import ExtensionProvider

class ExampleProvider:
    async def initialize(self): return None

def setup(api, ctx):
    api.register_provider(ExtensionProvider(
        name="example-provider", label="Example Provider", factory=ExampleProvider,
    ))
'''
    return f'''#| export
def setup(api, ctx):
    async def say_hello(command_context):
        greeting = ctx.state.get("greeting", "Hello")
        return {{"message": f"{{greeting}} from {extension_id}"}}
    api.register_command(
        "say-hello",
        title="Say hello",
        description="Show a deterministic extension greeting.",
        handler=say_hello,
    )
'''


def _notebook(name: str, template: str, extension_id: str, module_name: str = "extension") -> dict[str, Any]:
    source = _setup_source(template, extension_id)
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "extension-docs",
                "metadata": {},
                "source": [
                    f"# {name.replace('-', ' ').title()}\n",
                    "\nA Dialeng extension authored and tested from a notebook.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "default-export",
                "metadata": {},
                "outputs": [],
                "source": [f"#| default_exp {module_name}\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "extension-setup",
                "metadata": {"dialeng": {"is_exported": True}},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "extension-smoke-test",
                "metadata": {},
                "outputs": [],
                "source": ["assert callable(setup)\n"],
            },
        ],
        "metadata": {
            "dialeng_provider": "openai_codex",
            "solveit_model": "gpt-5.4",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class ExtensionProjectService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.extensions_root = self.project_root / ".dialeng" / "extensions"

    @staticmethod
    def validate_name(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not _NAME_RE.fullmatch(normalized):
            raise ValueError("Extension name must start with a letter and contain lowercase letters, numbers, or hyphens")
        return normalized

    def create(
        self,
        name: str,
        *,
        template: str = "basic",
        extension_id: str | None = None,
    ) -> dict[str, Any]:
        name = self.validate_name(name)
        if template not in _TEMPLATES:
            raise ValueError(f"Unknown extension template: {template}")
        extension_id = extension_id or f"local.{name}"
        module_name = "extension" if template != "package" else f"dialeng_ext_{name.replace('-', '_')}"
        root = self.extensions_root / name
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Extension already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        contributes = {
            "basic": "commands = true",
            "tool": "tools = true",
            "cell": "cell_types = true",
            "lifecycle": "events = true",
            "provider": "providers = true",
            "package": "commands = true\nskills = true\ntemplates = true",
        }[template]
        resources_block = (
            f'\n[resources]\nskills = ["resources/skills/{name}"]\n'
            'templates = ["resources/templates"]\n'
            if template == "package"
            else ""
        )
        (root / "extension.toml").write_text(
            f'''schema_version = 1
id = "{extension_id}"
name = "{name.replace('-', ' ').title()}"
version = "0.1.0"
description = "Dialeng {template} extension created from a notebook."
entrypoint = "{module_name}:setup"
dialeng_api = ">=1,<2"
scope = "project"

[permissions]
filesystem_read = ["project"]
filesystem_write = []
network = false
subprocess = false
browser_assets = false

[contributes]
{contributes}
{resources_block}
''',
            encoding="utf-8",
        )
        (root / "extension.ipynb").write_text(
            json.dumps(_notebook(name, template, extension_id, module_name), indent=1) + "\n",
            encoding="utf-8",
        )
        tests = root / "tests"
        tests.mkdir()
        (tests / "test_extension.py").write_text(
            f"from {module_name} import setup\n\n\ndef test_setup_is_callable():\n    assert callable(setup)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            f"# {name.replace('-', ' ').title()}\n\n"
            "Edit `extension.ipynb`, then run `dialeng extension prepare "
            f"{root}`.\n",
            encoding="utf-8",
        )
        if template == "package":
            package_name = f"dialeng-extension-{name}"
            skill_dir = root / "resources" / "skills" / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Use the {name} extension workflow.\n"
                "disable-model-invocation: true\n---\n"
                "This resource is disabled by default until installed explicitly.\n",
                encoding="utf-8",
            )
            template_dir = root / "resources" / "templates"
            template_dir.mkdir(parents=True)
            (template_dir / "README.md").write_text(
                "# Extension notebook template\n\nBundled independently from runtime activation.\n",
                encoding="utf-8",
            )
            (root / "pyproject.toml").write_text(
                f'''[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "0.1.0"
description = "Dialeng extension package for {name}."
requires-python = ">=3.11"
dependencies = ["dialeng>=0.11"]

[dependency-groups]
dev = ["nbdev", "pytest"]

[project.entry-points."dialeng.extensions"]
"{extension_id}" = "{module_name}:setup"

[tool.setuptools]
py-modules = ["{module_name}"]

[tool.setuptools.data-files]
"dialeng-extension-manifests/{name}" = ["extension.toml"]
"dialeng-extension-manifests/{name}/resources/skills/{name}" = ["resources/skills/{name}/SKILL.md"]
"dialeng-extension-manifests/{name}/resources/templates" = ["resources/templates/README.md"]

[tool.nbdev]
nbs_path = "."
lib_path = "."
recursive = false
clean_ids = true
''',
                encoding="utf-8",
            )
            workflow = root / ".github" / "workflows"
            workflow.mkdir(parents=True)
            (workflow / "extension-ci.yml").write_text(
                '''name: Extension CI
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - run: uv run nbdev-export --path extension.ipynb
      - run: uv run nbdev-test --path extension.ipynb --n_workers 1
      - run: uv run pytest -q
      - run: git diff --exit-code
''',
                encoding="utf-8",
            )
        return self.build(root)

    @staticmethod
    def templates() -> list[dict[str, str]]:
        descriptions = {
            "basic": "Command and toolbar-ready extension skeleton.",
            "tool": "Typed LLM tool with provider-neutral schema.",
            "cell": "Custom cell model, renderer, persistence, and LLM conversion.",
            "lifecycle": "Ordered lifecycle events and managed cleanup.",
            "provider": "Deterministic LLM provider skeleton.",
            "package": "Installable multi-resource package starting point.",
        }
        return [{"name": name, "description": descriptions[name]} for name in sorted(_TEMPLATES)]

    def resolve_extension(self, extension_id_or_name: str) -> Path:
        """Resolve one project extension without allowing path traversal."""
        query = str(extension_id_or_name or "").strip()
        if not query:
            raise ValueError("Extension ID or name is required")
        self.extensions_root.mkdir(parents=True, exist_ok=True)
        for root in sorted(self.extensions_root.iterdir()):
            if not root.is_dir():
                continue
            try:
                manifest = load_extension_manifest(root / "extension.toml")
            except Exception:
                continue
            if query in {manifest.id, root.name}:
                return root.resolve()
        raise KeyError(f"Unknown project extension: {query}")

    def _safe_file(self, extension_id: str, relative_path: str, *, must_exist: bool = True) -> Path:
        root = self.resolve_extension(extension_id)
        relative = Path(str(relative_path or ""))
        if relative.is_absolute() or not relative.parts:
            raise ValueError("Extension file path must be relative")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Extension file path escapes the extension root")
        if target.suffix.lower() not in _EDITABLE_SUFFIXES:
            raise ValueError(f"Unsupported extension file type: {target.suffix or '(none)'}")
        if must_exist and not target.is_file():
            raise FileNotFoundError(f"Extension file not found: {relative_path}")
        return target

    def read_file(self, extension_id: str, path: str) -> dict[str, Any]:
        target = self._safe_file(extension_id, path)
        raw = target.read_text(encoding="utf-8")
        return {
            "extension_id": load_extension_manifest(self.resolve_extension(extension_id) / "extension.toml").id,
            "path": target.relative_to(self.resolve_extension(extension_id)).as_posix(),
            "content": raw,
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }

    def update_file(
        self,
        extension_id: str,
        path: str,
        content: str,
        *,
        expected_sha256: str = "",
    ) -> dict[str, Any]:
        """Update a project extension file with optional stale-write protection."""
        target = self._safe_file(extension_id, path, must_exist=False)
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        current_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
        if expected_sha256 and expected_sha256 != current_hash:
            raise ValueError("Extension file changed since it was inspected")
        if target.name == "extension.toml" and target.exists():
            raise ValueError("Use the Extension Studio manifest editor for manifest changes")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return self.read_file(extension_id, target.relative_to(self.resolve_extension(extension_id)).as_posix())

    def build(self, path: Path) -> dict[str, Any]:
        root = Path(path).resolve()
        manifest = load_extension_manifest(root / "extension.toml")
        notebook = root / "extension.ipynb"
        if not notebook.is_file():
            raise FileNotFoundError(f"Extension notebook not found: {notebook}")
        module_name = manifest.entrypoint.split(":", 1)[0]
        module = root / (module_name.replace(".", "/") + ".py")
        exported = extract_extension(notebook, module, marker="#| export")
        if exported < 1:
            raise ValueError("Extension notebook has no #| export cells")
        validation = self.validate(root, isolated=False)
        notebook_name = ""
        try:
            notebook_name = notebook.relative_to(self.project_root).with_suffix("").as_posix()
        except ValueError:
            pass
        return ExtensionBuildResult(
            extension_id=manifest.id,
            root=str(root),
            manifest=str(root / "extension.toml"),
            notebook=str(notebook),
            module=str(module),
            exported_cells=exported,
            validation=validation,
            notebook_name=notebook_name,
        ).__dict__

    def validate(self, path: Path, *, isolated: bool = True) -> dict[str, Any]:
        root = Path(path).resolve()
        manifest = load_extension_manifest(root / "extension.toml")
        module_name = manifest.entrypoint.split(":", 1)[0]
        module = root / (module_name.replace(".", "/") + ".py")
        if not module.is_file():
            raise FileNotFoundError(f"Generated extension module not found: {module}")
        compile(module.read_text(encoding="utf-8"), str(module), "exec")
        result: dict[str, Any] = {
            "ok": True,
            "extension_id": manifest.id,
            "module": str(module),
            "isolated": isolated,
        }
        if isolated:
            script = '''import asyncio, json, sys
from pathlib import Path
from dialeng.extensions.manager import ExtensionManager
root = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
manager = ExtensionManager(project, user_root=project / ".no-user", development_paths=(root,))
async def run():
    manager.refresh()
    active = await manager.activate(manager.discovery.active.popitem()[0], require_trust=False)
    snapshot = manager.snapshot()
    await manager.shutdown()
    print(json.dumps({"ok": True, "module": active.module_name, "snapshot": snapshot}))
asyncio.run(run())
'''
            completed = subprocess.run(
                [sys.executable, "-c", script, str(root), str(self.project_root)],
                cwd=self.project_root,
                text=True,
                capture_output=True,
                timeout=20,
            )
            if completed.returncode != 0:
                return {
                    **result,
                    "ok": False,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            result["subprocess"] = json.loads(completed.stdout.strip().splitlines()[-1])
        return result

    def check(self, path: Path) -> dict[str, Any]:
        """Check compatibility and isolation readiness without importing extension code."""

        root = Path(path).expanduser().resolve()
        diagnostics: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        result: dict[str, Any] = {
            "ok": False,
            "root": str(root),
            "api_policy": API_POLICY.public(),
            "imported": False,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "rpc_safe_methods": list(RPC_SAFE_API_METHODS),
            "privileged_in_process_methods": list(PRIVILEGED_IN_PROCESS_API_METHODS),
        }
        try:
            manifest = load_extension_manifest(root / "extension.toml")
        except Exception as exc:
            diagnostics.append({
                "code": "manifest_invalid",
                "message": f"{type(exc).__name__}: {exc}",
                "path": str(root / "extension.toml"),
            })
            return result

        result.update({
            "extension_id": manifest.id,
            "version": manifest.version,
            "dialeng_api": manifest.dialeng_api,
            "manifest_schema": manifest.schema_version,
            "manifest": manifest.public(),
            "source_hash": extension_source_hash(root),
            "compatible": True,
        })
        module_name = manifest.entrypoint.split(":", 1)[0]
        module = root / (module_name.replace(".", "/") + ".py")
        if not module.is_file():
            diagnostics.append({
                "code": "entrypoint_missing",
                "message": f"Entrypoint module is missing: {module_name}",
                "path": str(module),
            })
            return result

        source = module.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(module))
        except SyntaxError as exc:
            diagnostics.append({
                "code": "syntax_error",
                "message": str(exc),
                "path": str(module),
                "line": str(exc.lineno or 0),
            })
            return result

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        for imported, message in LEGACY_IMPORT_DIAGNOSTICS.items():
            if any(name == imported or name.startswith(imported + ".") for name in imported_modules):
                warnings.append({
                    "code": "privileged_import",
                    "message": message,
                    "path": str(module),
                    "import": imported,
                })

        privileged_contributions = sorted(
            key for key in ("callbacks", "cell_types", "kernels", "panels", "providers", "services", "status")
            if manifest.contributes.get(key)
        )
        result["runtime_mode"] = (
            "privileged_in_process"
            if privileged_contributions or any(item["code"] == "privileged_import" for item in warnings)
            else "isolatable"
        )
        result["privileged_contributions"] = privileged_contributions
        result["module"] = str(module)
        result["ok"] = not diagnostics
        return result

    def test(self, path: Path, *, browser: bool = False) -> dict[str, Any]:
        root = Path(path).resolve()
        command = [sys.executable, "-m", "pytest", "-q", str(root / "tests")]
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "browser_requested": browser,
        }

    def prepare(self, path: Path) -> dict[str, Any]:
        build = self.build(path)
        validation = self.validate(Path(build["root"]), isolated=True)
        tests = self.test(Path(build["root"]))
        nbdev = self.nbdev_verify(Path(build["root"]))
        result = {
            **build,
            "validation": validation,
            "tests": tests,
            "nbdev": nbdev,
            "ok": bool(validation.get("ok") and tests.get("ok") and nbdev.get("ok", True)),
        }
        if result["ok"]:
            build_root = self.project_root / ".dialeng" / "extension-builds"
            build_root.mkdir(parents=True, exist_ok=True)
            record = {
                "extension_id": build["extension_id"],
                "source_hash": extension_source_hash(Path(build["root"])),
                "validated": True,
                "tests": tests.get("ok", False),
            }
            record_path = build_root / f"{hashlib.sha256(build['extension_id'].encode()).hexdigest()}.json"
            _write_private_json(record_path, record)
            result["build_record"] = str(record_path)
            result["source_hash"] = record["source_hash"]
        return result

    def nbdev_verify(self, path: Path) -> dict[str, Any]:
        root = Path(path).resolve()
        config = root / "pyproject.toml"
        if not config.is_file() or "[tool.nbdev]" not in config.read_text(encoding="utf-8"):
            return {"ok": True, "enabled": False, "commands": []}
        executable_dir = Path(sys.executable).resolve().parent
        commands = []
        for script, args in (
            ("nbdev-export", ["--path", "extension.ipynb"]),
            ("nbdev-test", ["--path", "extension.ipynb", "--n_workers", "1"]),
            ("nbdev-clean", ["--fname", "extension.ipynb"]),
        ):
            executable = shutil.which(script) or str(executable_dir / script)
            command = [executable, *args]
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=120,
            )
            commands.append({
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
            if completed.returncode != 0:
                return {"ok": False, "enabled": True, "commands": commands}
        # The Dialeng exporter remains canonical for the runtime artifact.
        self.build(root)
        return {"ok": True, "enabled": True, "commands": commands}

    def package(self, path: Path) -> dict[str, Any]:
        root = Path(path).resolve()
        prepared = self.prepare(root)
        if not prepared.get("ok"):
            return {"ok": False, "prepare": prepared}
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError("Package template requires pyproject.toml")
        completed = subprocess.run(
            ["uv", "build", str(root), "--out-dir", str(root / "dist")],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            timeout=120,
        )
        artifacts = [str(path) for path in sorted((root / "dist").glob("*"))]
        return {
            "ok": completed.returncode == 0,
            "prepare": prepared,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "artifacts": artifacts,
        }

    def install_local(self, source: Path) -> dict[str, Any]:
        source = Path(source).expanduser().resolve()
        manifest = load_extension_manifest(source / "extension.toml")
        target = self.extensions_root / source.name
        if target.exists():
            raise FileExistsError(f"Project extension already exists: {target}")
        self.extensions_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", "dist", "build", ".git",
        ))
        installed = load_extension_manifest(target / "extension.toml")
        if installed.id != manifest.id:
            shutil.rmtree(target, ignore_errors=True)
            raise ValueError("Installed extension manifest changed during copy")
        return {"ok": True, "extension_id": installed.id, "target": str(target)}

    def remove_local(self, extension_id: str) -> dict[str, Any]:
        root = self.resolve_extension(extension_id)
        if not root.is_relative_to(self.extensions_root.resolve()):
            raise ValueError("Only project-local extensions can be removed")
        shutil.rmtree(root)
        return {"ok": True, "extension_id": extension_id, "removed": str(root)}

    def migrate_autorun(self, notebook_path: Path) -> dict[str, Any]:
        """Copy a legacy AUTORUN extension notebook into the managed layout."""
        notebook_path = Path(notebook_path).resolve()
        if notebook_path.suffix != ".ipynb" or not notebook_path.is_file():
            raise ValueError("AUTORUN migration requires an existing .ipynb file")
        try:
            notebook_path.relative_to(self.project_root / "AUTORUN")
        except ValueError as exc:
            raise ValueError("Only notebooks inside the project's AUTORUN directory can be migrated") from exc
        name = self.validate_name(notebook_path.stem.lower().replace("_", "-"))
        created = self.create(name, template="basic", extension_id=f"legacy.autorun-{name}")
        target = Path(created["root"])
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook.setdefault("cells", [])
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "id": "dialeng-managed-setup",
            "metadata": {"dialeng": {"is_exported": True}},
            "outputs": [],
            "source": [
                "#| export\n",
                "\n",
                "def setup(api, ctx):\n",
                "    # Legacy decorators are captured during staged module import.\n",
                "    return None\n",
            ],
        })
        (target / "extension.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        result = self.build(target)
        result["legacy_source"] = str(notebook_path)
        result["message"] = "Original AUTORUN notebook was not modified. Review and test the managed copy before enabling it."
        return result

    def list(self) -> dict[str, Any]:
        return discover_extensions(
            self.project_root,
            user_root=self.project_root / ".no-user-extensions",
        ).public()

    def diagnostics(self, extension_id: str) -> dict[str, Any]:
        root = self.resolve_extension(extension_id)
        result: dict[str, Any] = {"extension_id": extension_id, "root": str(root), "files": []}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result["files"].append({
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                })
        try:
            manifest = load_extension_manifest(root / "extension.toml")
            result["extension_id"] = manifest.id
            result["manifest"] = manifest.public()
            result["validation"] = self.validate(root, isolated=False)
        except Exception as exc:
            result["validation"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return result


_extension_project_service: ExtensionProjectService | None = None


def set_extension_project_root(project_root: Path) -> ExtensionProjectService:
    global _extension_project_service
    _extension_project_service = ExtensionProjectService(project_root)
    return _extension_project_service


def get_extension_project_service() -> ExtensionProjectService:
    global _extension_project_service
    if _extension_project_service is None:
        _extension_project_service = ExtensionProjectService(Path.cwd())
    return _extension_project_service


def inspect_extension_api(topic: str = "") -> dict[str, Any]:
    """Inspect the stable Dialeng extension API and authoring workflow.

    Args:
        topic: Optional contribution topic such as tools, commands, events, cells, or lifecycle.
    """
    methods = {
        "lifecycle": ["setup(api, ctx)", "ctx.create_task", "ctx.add_cleanup", "api.on"],
        "commands": ["api.register_command", "api.register_shortcut", "api.register_toolbar_item"],
        "tools": ["api.register_tool"],
        "cells": ["api.register_cell_type"],
        "ui": ["api.register_panel", "api.register_status_item", "api.register_asset"],
        "settings": ["api.register_setting", "api.register_settings_section"],
        "runtime": [
            "ExtensionKernel", "ExtensionProvider", "ExtensionCellType",
            "api.register_callback", "api.register_kernel", "api.register_provider",
        ],
    }
    selected = str(topic or "").strip().lower()
    return {
        "dialeng_api": API_POLICY.version,
        "compatibility": API_POLICY.public(),
        "rpc_safe_methods": list(RPC_SAFE_API_METHODS),
        "privileged_in_process_methods": list(PRIVILEGED_IN_PROCESS_API_METHODS),
        "topic": selected or "all",
        "methods": methods.get(selected, methods),
        "rules": [
            "Use only ExtensionAPI and ExtensionContext public methods.",
            "Keep extension.ipynb as the source of truth for generated Python.",
            "Register all long-lived resources through the context cleanup lifecycle.",
            "Activation always requires an explicit user trust decision.",
        ],
    }


def list_extension_templates() -> list[dict[str, str]]:
    """List notebook-first extension templates available in this Dialeng project."""
    return ExtensionProjectService.templates()


def create_extension(name: str, template: str = "basic", extension_id: str = "") -> dict[str, Any]:
    """Create and build a project extension notebook without activating it.

    Args:
        name: Lowercase project extension directory name.
        template: One of basic, tool, cell, lifecycle, provider, or package.
        extension_id: Optional globally namespaced manifest ID.
    """
    return get_extension_project_service().create(
        name, template=template, extension_id=extension_id or None,
    )


def read_extension(extension_id: str, path: str = "extension.ipynb") -> dict[str, Any]:
    """Read one text file within a project extension directory."""
    return get_extension_project_service().read_file(extension_id, path)


def update_extension_file(
    extension_id: str,
    path: str,
    content: str,
    expected_sha256: str = "",
) -> dict[str, Any]:
    """Replace one constrained project extension file after optional hash validation."""
    return get_extension_project_service().update_file(
        extension_id, path, content, expected_sha256=expected_sha256,
    )


def build_extension(extension_id: str) -> dict[str, Any]:
    """Export a project extension notebook and run static validation."""
    service = get_extension_project_service()
    return service.build(service.resolve_extension(extension_id))


def validate_extension(extension_id: str) -> dict[str, Any]:
    """Validate a project extension in an isolated Python subprocess."""
    service = get_extension_project_service()
    return service.validate(service.resolve_extension(extension_id), isolated=True)


def run_extension_tests(extension_id: str, level: str = "unit") -> dict[str, Any]:
    """Run extension-local tests; browser mode is reported for Studio review."""
    if level not in {"unit", "browser"}:
        raise ValueError("Extension test level must be unit or browser")
    service = get_extension_project_service()
    return service.test(service.resolve_extension(extension_id), browser=level == "browser")


def show_extension_diagnostics(extension_id: str) -> dict[str, Any]:
    """Show manifest, file, and validation diagnostics for one project extension."""
    return get_extension_project_service().diagnostics(extension_id)


def request_extension_activation(extension_id: str) -> dict[str, Any]:
    """Create a user-review request without trusting or activating extension code."""
    service = get_extension_project_service()
    root = service.resolve_extension(extension_id)
    manifest = load_extension_manifest(root / "extension.toml")
    return {
        "ok": True,
        "extension_id": manifest.id,
        "source_hash": extension_source_hash(root),
        "requires_user_confirmation": True,
        "next_step": "Open Extensions, review permissions and source, then choose Trust and Enable.",
    }
