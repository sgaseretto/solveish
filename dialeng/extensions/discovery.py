"""Discover extension metadata without importing extension code."""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Iterable
import hashlib

from .manifest import (
    ExtensionManifestError,
    ExtensionSource,
    extension_source_hash,
    load_extension_manifest,
)


_PRECEDENCE = {
    "builtin": 10,
    "package": 20,
    "user": 30,
    "project": 40,
    "development": 50,
}


def _distribution_hash(distribution) -> str:
    digest = hashlib.sha256()
    for item in sorted(distribution.files or (), key=str):
        path = Path(distribution.locate_file(item)).resolve()
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class ExtensionDiagnostic:
    path: str
    error: str


@dataclass
class DiscoveryResult:
    active: dict[str, ExtensionSource] = field(default_factory=dict)
    shadowed: dict[str, list[ExtensionSource]] = field(default_factory=dict)
    diagnostics: list[ExtensionDiagnostic] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "active": [source.public() for source in self.active.values()],
            "shadowed": {
                key: [source.public() for source in values]
                for key, values in self.shadowed.items()
            },
            "diagnostics": [diagnostic.__dict__ for diagnostic in self.diagnostics],
        }


def _manifest_paths(path: Path) -> Iterable[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name == "extension.toml":
        yield path
        return
    if not path.is_dir():
        return
    direct = path / "extension.toml"
    if direct.is_file():
        yield direct
        return
    for candidate in sorted(path.glob("*/extension.toml")):
        if candidate.is_file():
            yield candidate


def discover_extensions(
    project_root: Path,
    *,
    user_root: Path | None = None,
    development_paths: Iterable[Path] = (),
    builtin_root: Path | None = None,
    include_entry_points: bool = True,
) -> DiscoveryResult:
    """Discover manifests and resolve precedence without importing entrypoints."""
    project_root = Path(project_root).resolve()
    candidates: list[tuple[str, Path]] = []
    if builtin_root is not None:
        candidates.append(("builtin", Path(builtin_root)))
    if user_root is None:
        user_root = Path.home() / ".dialeng" / "extensions"
    candidates.append(("user", Path(user_root)))
    candidates.append(("project", project_root / ".dialeng" / "extensions"))
    candidates.extend(("development", Path(path)) for path in development_paths)

    result = DiscoveryResult()
    discovered: dict[str, list[ExtensionSource]] = {}
    seen_paths: set[Path] = set()
    for origin, root in candidates:
        for manifest_path in _manifest_paths(root):
            manifest_path = manifest_path.resolve()
            if manifest_path in seen_paths:
                continue
            seen_paths.add(manifest_path)
            try:
                manifest = load_extension_manifest(manifest_path)
                source = ExtensionSource(
                    root=manifest_path.parent,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    source_hash=extension_source_hash(manifest_path.parent),
                    origin=origin,
                    precedence=_PRECEDENCE[origin],
                )
                discovered.setdefault(manifest.id, []).append(source)
            except ExtensionManifestError as exc:
                result.diagnostics.append(ExtensionDiagnostic(str(manifest_path), str(exc)))

    if include_entry_points:
        try:
            entry_points = metadata.entry_points(group="dialeng.extensions")
        except TypeError:  # Python/importlib compatibility
            entry_points = metadata.entry_points().select(group="dialeng.extensions")
        except Exception as exc:
            result.diagnostics.append(ExtensionDiagnostic("dialeng.extensions", str(exc)))
            entry_points = ()
        for entry_point in entry_points:
            try:
                distribution = entry_point.dist
                candidates = [
                    Path(distribution.locate_file(item)).resolve()
                    for item in (distribution.files or ())
                    if Path(str(item)).name in {"extension.toml", "dialeng-extension.toml"}
                ]
                if not candidates:
                    raise ExtensionManifestError(
                        f"Installed entry point {entry_point.name!r} has no packaged extension.toml"
                    )
                matching = []
                for manifest_path in candidates:
                    try:
                        manifest = load_extension_manifest(manifest_path)
                    except ExtensionManifestError:
                        continue
                    if manifest.id == entry_point.name or len(candidates) == 1:
                        matching.append((manifest_path, manifest))
                if len(matching) != 1:
                    raise ExtensionManifestError(
                        f"Could not uniquely match manifest for installed entry point {entry_point.name!r}"
                    )
                manifest_path, manifest = matching[0]
                manifest = replace(manifest, entrypoint=entry_point.value, scope="package")
                source = ExtensionSource(
                    root=manifest_path.parent,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    source_hash=_distribution_hash(distribution),
                    origin="package",
                    precedence=_PRECEDENCE["package"],
                    package_entrypoint=entry_point.name,
                )
                discovered.setdefault(manifest.id, []).append(source)
            except Exception as exc:
                result.diagnostics.append(ExtensionDiagnostic(
                    f"entry-point:{getattr(entry_point, 'name', 'unknown')}", str(exc),
                ))

    for extension_id, sources in discovered.items():
        ordered = sorted(sources, key=lambda item: item.precedence, reverse=True)
        highest = ordered[0].precedence
        conflicts = [item for item in ordered if item.precedence == highest]
        if len(conflicts) > 1:
            result.diagnostics.append(ExtensionDiagnostic(
                extension_id,
                "Multiple extensions with the same id have equal precedence: "
                + ", ".join(str(item.root) for item in conflicts),
            ))
            result.shadowed[extension_id] = ordered
            continue
        result.active[extension_id] = ordered[0]
        if len(ordered) > 1:
            result.shadowed[extension_id] = ordered[1:]
    return result
