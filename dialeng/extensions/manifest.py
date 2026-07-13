"""Manifest parsing and source hashing for Dialeng extensions."""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


EXTENSION_API_VERSION = "1.0.0"
_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_SCOPES = {"builtin", "user", "project", "development", "package"}
_HASH_EXCLUDES = {
    "__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache",
    ".dialeng-extension-build", "dist", "build",
}


class ExtensionManifestError(ValueError):
    """Raised when an extension manifest is malformed or incompatible."""


@dataclass(frozen=True)
class ExtensionManifest:
    schema_version: int
    id: str
    name: str
    version: str
    description: str
    entrypoint: str
    dialeng_api: str
    scope: str = "project"
    permissions: dict[str, Any] = field(default_factory=dict)
    contributes: dict[str, bool] = field(default_factory=dict)
    assets: dict[str, list[str]] = field(default_factory=dict)
    author: dict[str, str] = field(default_factory=dict)
    resources: dict[str, list[str]] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "dialeng_api": self.dialeng_api,
            "scope": self.scope,
            "permissions": self.permissions,
            "contributes": self.contributes,
            "assets": self.assets,
            "author": self.author,
            "resources": self.resources,
        }

    def assert_compatible(self, api_version: str = EXTENSION_API_VERSION) -> None:
        try:
            compatible = Version(api_version) in SpecifierSet(self.dialeng_api)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise ExtensionManifestError(f"Invalid Dialeng API constraint: {exc}") from exc
        if not compatible:
            raise ExtensionManifestError(
                f"Extension requires Dialeng API {self.dialeng_api}; running {api_version}"
            )


@dataclass(frozen=True)
class ExtensionSource:
    root: Path
    manifest_path: Path
    manifest: ExtensionManifest
    source_hash: str
    origin: str
    precedence: int
    package_entrypoint: str = ""

    @property
    def identity(self) -> str:
        return f"{self.manifest.id}@{self.manifest.version}:{self.source_hash}"

    def public(self) -> dict[str, Any]:
        return {
            **self.manifest.public(),
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "source_hash": self.source_hash,
            "origin": self.origin,
            "precedence": self.precedence,
            "package_entrypoint": self.package_entrypoint,
        }


def _string(raw: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ExtensionManifestError(f"Manifest field {key!r} must be a non-empty string")
    return value.strip()


def _normalize_assets(root: Path, raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExtensionManifestError("Manifest assets must be a table")
    result: dict[str, list[str]] = {}
    for kind, paths in raw.items():
        if kind not in {"css", "js"} or not isinstance(paths, list):
            raise ExtensionManifestError(f"Invalid asset declaration: {kind}")
        normalized: list[str] = []
        for item in paths:
            if not isinstance(item, str) or not item:
                raise ExtensionManifestError(f"Invalid {kind} asset path")
            candidate = (root / item).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError as exc:
                raise ExtensionManifestError(f"Asset escapes extension root: {item}") from exc
            if not candidate.is_file():
                raise ExtensionManifestError(f"Declared asset does not exist: {item}")
            normalized.append(candidate.relative_to(root.resolve()).as_posix())
        result[kind] = normalized
    return result


def _normalize_resources(root: Path, raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExtensionManifestError("Manifest resources must be a table")
    allowed = {"skills", "templates", "craft", "notebook_templates"}
    result: dict[str, list[str]] = {}
    for kind, paths in raw.items():
        if kind not in allowed or not isinstance(paths, list):
            raise ExtensionManifestError(f"Invalid resource declaration: {kind}")
        normalized = []
        for item in paths:
            if not isinstance(item, str) or not item:
                raise ExtensionManifestError(f"Invalid {kind} resource path")
            candidate = (root / item).resolve()
            if not candidate.is_relative_to(root.resolve()):
                raise ExtensionManifestError(f"Resource escapes extension root: {item}")
            if not candidate.exists():
                raise ExtensionManifestError(f"Declared resource does not exist: {item}")
            normalized.append(candidate.relative_to(root.resolve()).as_posix())
        result[kind] = normalized
    return result


def load_extension_manifest(path: Path) -> ExtensionManifest:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ExtensionManifestError(f"Could not read extension manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExtensionManifestError("Extension manifest root must be a table")

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ExtensionManifestError("Only extension manifest schema_version = 1 is supported")
    extension_id = _string(raw, "id")
    if not _ID_RE.fullmatch(extension_id):
        raise ExtensionManifestError(
            "Extension id must be lowercase and namespaced with '.', '_' or '-'"
        )
    version = _string(raw, "version")
    try:
        Version(version)
    except InvalidVersion as exc:
        raise ExtensionManifestError(f"Invalid extension version {version!r}") from exc
    scope = _string(raw, "scope", default="project")
    if scope not in _SCOPES:
        raise ExtensionManifestError(f"Unsupported extension scope: {scope}")
    entrypoint = _string(raw, "entrypoint")
    if ":" not in entrypoint:
        raise ExtensionManifestError("Entrypoint must use module:function syntax")

    permissions = raw.get("permissions") or {}
    contributes = raw.get("contributes") or {}
    author = raw.get("author") or {}
    if not isinstance(permissions, dict) or not isinstance(contributes, dict):
        raise ExtensionManifestError("permissions and contributes must be tables")
    if not isinstance(author, dict):
        raise ExtensionManifestError("author must be a table")

    manifest = ExtensionManifest(
        schema_version=1,
        id=extension_id,
        name=_string(raw, "name"),
        version=version,
        description=_string(raw, "description"),
        entrypoint=entrypoint,
        dialeng_api=_string(raw, "dialeng_api"),
        scope=scope,
        permissions=dict(permissions),
        contributes={str(key): bool(value) for key, value in contributes.items()},
        assets=_normalize_assets(path.parent, raw.get("assets")),
        author={str(key): str(value) for key, value in author.items()},
        resources=_normalize_resources(path.parent, raw.get("resources")),
    )
    manifest.assert_compatible()
    return manifest


def extension_source_hash(root: Path) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in _HASH_EXCLUDES or part.endswith(".egg-info") for part in path.parts
        ):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def permission_fingerprint(manifest: ExtensionManifest) -> str:
    payload = json.dumps(manifest.permissions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
