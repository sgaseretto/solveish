"""Deterministic, repository-safe extension environment lockfiles."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from .contracts import EXTENSION_API_VERSION
from .discovery import discover_extensions
from .manifest import ExtensionSource, extension_source_hash, permission_fingerprint
from .state import ExtensionStateStore


LOCKFILE_VERSION = 1
LOCKFILE_NAME = "extensions.lock"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dialeng_version() -> str:
    try:
        return metadata.version("dialeng")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, timeout=20,
    )
    if completed.returncode:
        raise ValueError(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Serialize lock/sync operations without placing state in the lockfile."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ExtensionLockService:
    """Resolve, verify, and restore one project's enabled extension set."""

    def __init__(
        self,
        project_root: Path,
        *,
        state_store: ExtensionStateStore | None = None,
        user_root: Path | None = None,
        development_paths: tuple[Path, ...] = (),
        cache_root: Path | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.dialeng_root = self.project_root / ".dialeng"
        self.path = self.dialeng_root / LOCKFILE_NAME
        self.guard_path = self.dialeng_root / f".{LOCKFILE_NAME}.guard"
        self.state_store = state_store or ExtensionStateStore()
        self.user_root = user_root
        self.development_paths = tuple(Path(path).resolve() for path in development_paths)
        default_cache = Path.home() / ".cache" / "dialeng" / "extensions"
        self.cache_root = Path(
            cache_root or os.environ.get("DIALENG_EXTENSION_CACHE", default_cache)
        ).expanduser().resolve()

    def _discover(self):
        return discover_extensions(
            self.project_root,
            user_root=self.user_root,
            development_paths=self.development_paths,
        )

    def _environment(self) -> dict[str, Any]:
        uv_lock = self.project_root / "uv.lock"
        if not uv_lock.is_file():
            return {"uv_lock": None}
        return {"uv_lock": {"path": "uv.lock", "sha256": _sha256(uv_lock)}}

    def _source_coordinate(self, source: ExtensionSource) -> dict[str, Any]:
        root = source.root.resolve()
        try:
            relative = root.relative_to(self.project_root).as_posix()
        except ValueError:
            relative = ""
        if relative:
            return {"type": "project_path", "path": relative}
        if source.origin == "package":
            package_name = source.package_name or source.package_entrypoint
            if not package_name:
                raise ValueError(f"Installed extension {source.manifest.id} has no package identity")
            wheel = self._wheel_coordinate(package_name)
            if wheel:
                return wheel
            return {
                "type": "installed_package",
                "package": package_name,
                "package_version": source.package_version or source.manifest.version,
            }
        try:
            repository = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
            remote = _run_git(repository, "remote", "get-url", "origin")
            commit = _run_git(repository, "rev-parse", "HEAD")
            subdirectory = root.relative_to(repository).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Extension {source.manifest.id} comes from a machine-specific path. "
                "Move it into the project, install it as a package, or use an immutable Git source."
            ) from exc
        return {
            "type": "git",
            "url": remote,
            "commit": commit,
            "subdirectory": "" if subdirectory == "." else subdirectory,
        }

    def _wheel_coordinate(self, package_name: str) -> dict[str, Any] | None:
        try:
            distribution = metadata.distribution(package_name)
            direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        except (metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
            return None
        url = str(direct_url.get("url") or "")
        parsed = urllib.parse.urlparse(url)
        if not parsed.path.lower().endswith(".whl"):
            return None
        archive_hash = str((direct_url.get("archive_info") or {}).get("hash") or "")
        algorithm, _, digest = archive_hash.partition("=")
        wheel_path = None
        if parsed.scheme == "file":
            wheel_path = Path(urllib.request.url2pathname(parsed.path)).resolve()
            if not wheel_path.is_file():
                raise ValueError(f"Installed extension wheel is unavailable: {wheel_path.name}")
            actual = _sha256(wheel_path)
        else:
            actual = digest if algorithm == "sha256" else ""
        if not actual:
            raise ValueError(f"Installed extension wheel has no immutable SHA-256: {package_name}")
        coordinate: dict[str, Any] = {
            "type": "wheel", "filename": Path(parsed.path).name, "sha256": actual,
        }
        if wheel_path is not None:
            try:
                coordinate["path"] = wheel_path.relative_to(self.project_root).as_posix()
            except ValueError:
                pass
            self._cache_wheel(wheel_path, actual)
        else:
            coordinate["url"] = url
        return coordinate

    def _entry(self, source: ExtensionSource) -> dict[str, Any]:
        manifest = source.manifest
        return {
            "id": manifest.id,
            "version": manifest.version,
            "source": self._source_coordinate(source),
            "artifact_hash": source.source_hash,
            "manifest_hash": _sha256(source.manifest_path),
            "source_hash": source.source_hash,
            "dialeng_api": manifest.dialeng_api,
            "permissions": manifest.permissions,
            "permission_hash": permission_fingerprint(manifest),
            "resources": manifest.resources,
            "enabled_resources": sorted(
                self.state_store.enabled_resources(self.project_root, manifest.id)
            ),
        }

    def _source_cache_path(self, source_hash: str) -> Path:
        return self.cache_root / "sources" / source_hash

    def _wheel_cache_path(self, digest: str, filename: str) -> Path:
        return self.cache_root / "wheels" / digest / filename

    def _cache_source(self, source: ExtensionSource) -> Path:
        return self._cache_directory(source.root, source.source_hash, source.manifest.id)

    def _cache_directory(self, root: Path, source_hash: str, extension_id: str) -> Path:
        target = self._source_cache_path(source_hash)
        guard = self.cache_root / ".guards" / f"source-{source_hash}.lock"
        with _exclusive_lock(guard):
            if target.is_dir() and extension_source_hash(target) == source_hash:
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{source_hash[:12]}-", dir=target.parent))
            candidate = temporary / "source"
            try:
                shutil.copytree(
                    root, candidate,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
                )
                if extension_source_hash(candidate) != source_hash:
                    raise ValueError(f"Could not create a byte-identical cache for {extension_id}")
                if target.exists():
                    shutil.rmtree(target)
                os.replace(candidate, target)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return target

    def _cache_wheel(self, wheel: Path, digest: str, *, filename: str | None = None) -> Path:
        if _sha256(wheel) != digest:
            raise ValueError(f"Wheel SHA-256 mismatch: {wheel.name}")
        artifact_name = filename or wheel.name
        target = self._wheel_cache_path(digest, artifact_name)
        guard = self.cache_root / ".guards" / f"wheel-{digest}.lock"
        with _exclusive_lock(guard):
            if target.is_file() and _sha256(target) == digest:
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{artifact_name}.", dir=target.parent)
            os.close(fd)
            try:
                shutil.copy2(wheel, temporary)
                if _sha256(Path(temporary)) != digest:
                    raise ValueError(f"Cached wheel SHA-256 mismatch: {artifact_name}")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return target

    def _install_cached_source(self, entry: dict[str, Any], target: Path) -> None:
        cached = self._source_cache_path(str(entry["source_hash"]))
        if not cached.is_dir() or extension_source_hash(cached) != entry["source_hash"]:
            raise FileNotFoundError(
                f"Locked extension artifact is unavailable in cache: {entry['id']} ({entry['source_hash']})"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=f".{entry['id']}.sync-", dir=target.parent))
        candidate = temporary_root / "source"
        backup = temporary_root / "previous"
        try:
            shutil.copytree(cached, candidate)
            if extension_source_hash(candidate) != entry["source_hash"]:
                raise ValueError(f"Cached source hash mismatch for {entry['id']}")
            had_previous = target.exists()
            if had_previous:
                os.replace(target, backup)
            try:
                os.replace(candidate, target)
            except Exception:
                if had_previous and backup.exists():
                    os.replace(backup, target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def resolve(self) -> dict[str, Any]:
        discovery = self._discover()
        enabled = self.state_store.enabled(self.project_root)
        missing = sorted(enabled - set(discovery.active))
        errors = [item.__dict__ for item in discovery.diagnostics]
        errors.extend({"extension_id": item, "error": "Enabled extension was not discovered"} for item in missing)
        entries = []
        for extension_id in sorted(enabled & set(discovery.active)):
            try:
                entries.append(self._entry(discovery.active[extension_id]))
            except ValueError as exc:
                errors.append({"extension_id": extension_id, "error": str(exc)})
        payload = {
            "lock_version": LOCKFILE_VERSION,
            "dialeng_version": _dialeng_version(),
            "extension_api": EXTENSION_API_VERSION,
            "environment": self._environment(),
            "extensions": entries,
        }
        return {"ok": not errors, "lock": payload, "diagnostics": errors}

    def write(self) -> dict[str, Any]:
        with _exclusive_lock(self.guard_path):
            result = self.resolve()
            if not result["ok"]:
                return result
            discovery = self._discover()
            for entry in result["lock"]["extensions"]:
                source = discovery.active.get(entry["id"])
                if source is not None and entry["source"].get("type") in {"project_path", "git"}:
                    self._cache_source(source)
            self.dialeng_root.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".extensions.lock.", dir=self.dialeng_root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(result["lock"], handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return {**result, "path": str(self.path)}

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(f"Extension lockfile does not exist: {self.path}")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read extension lockfile: {exc}") from exc
        if payload.get("lock_version") != LOCKFILE_VERSION or not isinstance(payload.get("extensions"), list):
            raise ValueError(f"Unsupported extension lockfile version: {payload.get('lock_version')}")
        return payload

    @staticmethod
    def _entry_drift(expected: dict[str, Any], source: ExtensionSource | None) -> list[dict[str, Any]]:
        extension_id = expected.get("id", "unknown")
        if source is None:
            return [{"extension_id": extension_id, "field": "source", "expected": "present", "actual": "missing"}]
        actual = {
            "version": source.manifest.version,
            "source_hash": source.source_hash,
            "manifest_hash": _sha256(source.manifest_path),
            "permission_hash": permission_fingerprint(source.manifest),
            "dialeng_api": source.manifest.dialeng_api,
            "resources": source.manifest.resources,
        }
        drift = []
        for field, value in actual.items():
            if expected.get(field) != value:
                drift.append({
                    "extension_id": extension_id,
                    "field": field,
                    "expected": expected.get(field),
                    "actual": value,
                })
        return drift

    def check(self) -> dict[str, Any]:
        try:
            lock = self.read()
        except (FileNotFoundError, ValueError) as exc:
            return {"ok": False, "path": str(self.path), "drift": [{"field": "lockfile", "error": str(exc)}]}
        discovery = self._discover()
        expected_ids = {str(item.get("id")) for item in lock["extensions"]}
        drift = []
        for item in lock["extensions"]:
            drift.extend(self._entry_drift(item, discovery.active.get(str(item.get("id")))))
        enabled = self.state_store.enabled(self.project_root)
        if enabled != expected_ids:
            drift.append({
                "field": "enabled_extensions",
                "expected": sorted(expected_ids),
                "actual": sorted(enabled),
            })
        expected_environment = lock.get("environment", {})
        if expected_environment != self._environment():
            drift.append({
                "field": "environment",
                "expected": expected_environment,
                "actual": self._environment(),
            })
        return {
            "ok": not drift and not discovery.diagnostics,
            "path": str(self.path),
            "drift": drift,
            "diagnostics": [item.__dict__ for item in discovery.diagnostics],
        }

    def _restore_git(self, entry: dict[str, Any], *, offline: bool = False) -> None:
        source = entry["source"]
        target = self.project_root / ".dialeng" / "extensions" / entry["id"]
        if target.exists() and extension_source_hash(target) == entry["source_hash"]:
            return
        cached = self._source_cache_path(entry["source_hash"])
        if cached.is_dir() and extension_source_hash(cached) == entry["source_hash"]:
            self._install_cached_source(entry, target)
            return
        if offline:
            raise FileNotFoundError(
                f"Locked Git artifact is unavailable for offline sync: {entry['id']}"
            )
        temporary_root = Path(tempfile.mkdtemp(prefix=".extension-sync-", dir=self.dialeng_root))
        try:
            repository = temporary_root / "repository"
            subprocess.run(
                ["git", "clone", "--no-checkout", source["url"], str(repository)],
                check=True, text=True, capture_output=True, timeout=120,
            )
            subprocess.run(
                ["git", "-C", str(repository), "checkout", "--detach", source["commit"]],
                check=True, text=True, capture_output=True, timeout=60,
            )
            extension_root = repository / source.get("subdirectory", "")
            candidate = temporary_root / "extension"
            shutil.copytree(extension_root, candidate, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            if extension_source_hash(candidate) != entry["source_hash"]:
                raise ValueError(f"Git source hash mismatch for {entry['id']}")
            self._cache_directory(candidate, entry["source_hash"], entry["id"])
            self._install_cached_source(entry, target)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _restore_package(self, entry: dict[str, Any], *, offline: bool = False) -> None:
        source = entry["source"]
        requirement = f"{source['package']}=={source['package_version']}"
        command = ["uv", "pip", "install", *self._uv_python_arguments()]
        if offline:
            command.append("--offline")
        command.append(requirement)
        completed = subprocess.run(
            command,
            cwd=self.project_root, text=True, capture_output=True, timeout=180,
        )
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or f"Could not install {requirement}")

    def _restore_wheel(self, entry: dict[str, Any], *, offline: bool = False) -> None:
        source = entry["source"]
        digest = str(source["sha256"])
        filename = str(source["filename"])
        wheel = self._wheel_cache_path(digest, filename)
        project_path = source.get("path")
        if project_path:
            candidate = (self.project_root / str(project_path)).resolve()
            if not candidate.is_relative_to(self.project_root):
                raise ValueError(f"Locked wheel path escapes the project: {entry['id']}")
            if candidate.is_file():
                wheel = self._cache_wheel(candidate, digest)
        if not wheel.is_file() and not offline and source.get("url"):
            wheel.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=wheel.parent)
            os.close(fd)
            try:
                urllib.request.urlretrieve(str(source["url"]), temporary)
                wheel = self._cache_wheel(Path(temporary), digest, filename=filename)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if not wheel.is_file() or _sha256(wheel) != digest:
            mode = "offline " if offline else ""
            raise FileNotFoundError(f"Locked wheel artifact is unavailable for {mode}sync: {filename}")
        command = ["uv", "pip", "install", *self._uv_python_arguments()]
        if offline:
            command.append("--offline")
        command.append(str(wheel))
        completed = subprocess.run(
            command,
            cwd=self.project_root, text=True, capture_output=True, timeout=180,
        )
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or f"Could not install {filename}")

    def _uv_python_arguments(self) -> list[str]:
        candidates = (
            self.project_root / ".venv" / "bin" / "python",
            self.project_root / ".venv" / "Scripts" / "python.exe",
        )
        python = next((path for path in candidates if path.is_file()), None)
        return ["--python", str(python)] if python else []

    def _restore_project_path(self, entry: dict[str, Any]) -> None:
        candidate = (self.project_root / entry["source"]["path"]).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise ValueError(f"Locked path escapes the project: {entry['id']}")
        if candidate.is_dir() and extension_source_hash(candidate) == entry["source_hash"]:
            return
        self._install_cached_source(entry, candidate)

    def sync(self, *, offline: bool = False) -> dict[str, Any]:
        with _exclusive_lock(self.guard_path):
            try:
                lock = self.read()
                for entry in lock["extensions"]:
                    source_type = entry.get("source", {}).get("type")
                    if source_type == "project_path":
                        self._restore_project_path(entry)
                    elif source_type == "git":
                        self._restore_git(entry, offline=offline)
                    elif source_type == "wheel":
                        self._restore_wheel(entry, offline=offline)
                    elif source_type == "installed_package":
                        self._restore_package(entry, offline=offline)
                    else:
                        raise ValueError(f"Unsupported locked source type: {source_type}")
                expected = {str(item["id"]) for item in lock["extensions"]}
                current = self.state_store.enabled(self.project_root)
                for extension_id in current | expected:
                    self.state_store.set_enabled(self.project_root, extension_id, extension_id in expected)
                result = self.check()
                result["enabled"] = sorted(expected)
                result["trust_granted"] = False
                result["offline"] = offline
                return result
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return {"ok": False, "path": str(self.path), "error": str(exc)}

    def update(self, extension_id: str = "") -> dict[str, Any]:
        previous = self.read() if self.path.is_file() else {"extensions": []}
        old = {item["id"]: item for item in previous["extensions"]}
        resolved = self.resolve()
        if not resolved["ok"]:
            return resolved
        current = {item["id"]: item for item in resolved["lock"]["extensions"]}
        if extension_id and extension_id not in current:
            return {"ok": False, "error": f"Enabled extension was not found: {extension_id}"}
        changes = []
        ids = [extension_id] if extension_id else sorted(set(old) | set(current))
        for item_id in ids:
            before, after = old.get(item_id), current.get(item_id)
            for field in ("version", "source_hash", "dialeng_api", "permissions", "resources"):
                before_value = before.get(field) if before else None
                after_value = after.get(field) if after else None
                if before_value != after_value:
                    changes.append({
                        "extension_id": item_id,
                        "field": field,
                        "before": before_value,
                        "after": after_value,
                        "requires_trust_review": field in {"source_hash", "permissions"},
                    })
        written = self.write()
        return {**written, "changes": changes}
