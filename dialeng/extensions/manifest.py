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

from .contracts import (
    EXTENSION_API_VERSION,
    MANIFEST_CONTRIBUTION_FIELDS,
    MANIFEST_PERMISSION_BOOL_FIELDS,
    MANIFEST_PERMISSION_LIST_FIELDS,
    MANIFEST_SCHEMA_VERSION,
)


_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_SCOPES = {"builtin", "user", "project", "development", "package"}
_TOP_LEVEL_FIELDS = {
    "schema_version", "id", "name", "version", "description", "entrypoint",
    "dialeng_api", "scope", "permissions", "contributes", "assets", "author",
    "resources",
}
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


def _normalize_permissions(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExtensionManifestError("permissions must be a table")
    list_fields = set(MANIFEST_PERMISSION_LIST_FIELDS)
    bool_fields = set(MANIFEST_PERMISSION_BOOL_FIELDS)
    unknown = sorted(set(raw) - list_fields - bool_fields)
    if unknown:
        raise ExtensionManifestError(f"Unknown permission declaration(s): {', '.join(unknown)}")
    result: dict[str, Any] = {}
    for key in MANIFEST_PERMISSION_LIST_FIELDS:
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ExtensionManifestError(f"Permission {key} must be a list of non-empty strings")
        result[key] = list(value)
    for key in MANIFEST_PERMISSION_BOOL_FIELDS:
        if key not in raw:
            continue
        if not isinstance(raw[key], bool):
            raise ExtensionManifestError(f"Permission {key} must be true or false")
        result[key] = raw[key]
    return result


def _normalize_contributions(raw: Any) -> dict[str, bool]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExtensionManifestError("contributes must be a table")
    unknown = sorted(set(raw) - set(MANIFEST_CONTRIBUTION_FIELDS))
    if unknown:
        raise ExtensionManifestError(f"Unknown contribution declaration(s): {', '.join(unknown)}")
    invalid = sorted(key for key, value in raw.items() if not isinstance(value, bool))
    if invalid:
        raise ExtensionManifestError(
            f"Contribution declaration(s) must be true or false: {', '.join(invalid)}"
        )
    return dict(raw)


def load_extension_manifest(path: Path) -> ExtensionManifest:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ExtensionManifestError(f"Could not read extension manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExtensionManifestError("Extension manifest root must be a table")
    unknown_fields = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        raise ExtensionManifestError(
            f"Unknown manifest field(s) for schema 1: {', '.join(unknown_fields)}"
        )

    schema_version = raw.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ExtensionManifestError(
            f"Only extension manifest schema_version = {MANIFEST_SCHEMA_VERSION} is supported"
        )
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

    author = raw.get("author") or {}
    if not isinstance(author, dict):
        raise ExtensionManifestError("author must be a table")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in author.items()):
        raise ExtensionManifestError("author values must be strings")

    manifest = ExtensionManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        id=extension_id,
        name=_string(raw, "name"),
        version=version,
        description=_string(raw, "description"),
        entrypoint=entrypoint,
        dialeng_api=_string(raw, "dialeng_api"),
        scope=scope,
        permissions=_normalize_permissions(raw.get("permissions")),
        contributes=_normalize_contributions(raw.get("contributes")),
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
