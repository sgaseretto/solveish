"""Helpers for plain-text file editing outside notebook mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mimetypes
import time


FILE_LOCK_TIMEOUT_SECONDS = 30.0

_TEXT_EXTENSIONS = {
    ".css": "css",
    ".csv": "plaintext",
    ".env": "shell",
    ".gitignore": "plaintext",
    ".html": "html",
    ".ini": "ini",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".sql": "sql",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".txt": "plaintext",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass
class FileLease:
    client_id: str
    encoding: str
    mtime_ns: int
    size: int
    acquired_at: float
    last_seen_at: float


@dataclass
class FileOpenResult:
    status: str
    rel_path: str
    abs_path: Path | None = None
    content: str = ""
    language: str = "plaintext"
    encoding: str = "utf-8"
    mtime_ns: int = 0
    size: int = 0
    reason: str = ""

    @property
    def editable(self) -> bool:
        return self.status == "editable"


class FileEditorService:
    """Backend-authoritative file inspection and single-writer lease tracking."""

    def __init__(self, root: Path):
        self._root = root
        self._leases: dict[str, FileLease] = {}

    def set_root(self, root: Path) -> None:
        self._root = root
        self._cleanup_expired_leases()

    def resolve_rel_path(self, rel_path: str) -> Path | None:
        rel = Path(rel_path)
        target = self._root / rel
        try:
            resolved = target.resolve()
        except FileNotFoundError:
            resolved = target.resolve(strict=False)
        if not resolved.is_relative_to(self._root.resolve()):
            return None
        return resolved

    def rel_path_for(self, path: Path) -> str:
        return str(path.resolve().relative_to(self._root.resolve()))

    def open_file(self, rel_path: str, client_id: str) -> FileOpenResult:
        self._cleanup_expired_leases()
        path = self.resolve_rel_path(rel_path)
        if path is None or not path.exists() or not path.is_file():
            return FileOpenResult(status="missing", rel_path=rel_path, reason="File not found.")
        if path.suffix == ".ipynb":
            return FileOpenResult(status="notebook", rel_path=self.rel_path_for(path), abs_path=path)

        canonical_rel = self.rel_path_for(path)
        inspection = self._inspect_file(path)
        if inspection.status != "editable":
            inspection.rel_path = canonical_rel
            inspection.abs_path = path
            return inspection

        lease = self._leases.get(canonical_rel)
        if lease and lease.client_id != client_id:
            return FileOpenResult(
                status="locked",
                rel_path=canonical_rel,
                abs_path=path,
                reason="This file is currently being edited by another Dialeng session.",
            )

        now = time.monotonic()
        self._leases[canonical_rel] = FileLease(
            client_id=client_id,
            encoding=inspection.encoding,
            mtime_ns=inspection.mtime_ns,
            size=inspection.size,
            acquired_at=lease.acquired_at if lease else now,
            last_seen_at=now,
        )
        inspection.rel_path = canonical_rel
        inspection.abs_path = path
        return inspection

    def heartbeat(self, rel_path: str, client_id: str) -> bool:
        self._cleanup_expired_leases()
        canonical_rel = self._canonical_rel_path(rel_path)
        if not canonical_rel:
            return False
        lease = self._leases.get(canonical_rel)
        if not lease or lease.client_id != client_id:
            return False
        lease.last_seen_at = time.monotonic()
        return True

    def release(self, rel_path: str, client_id: str) -> None:
        canonical_rel = self._canonical_rel_path(rel_path)
        if not canonical_rel:
            return
        lease = self._leases.get(canonical_rel)
        if lease and lease.client_id == client_id:
            self._leases.pop(canonical_rel, None)

    def save_file(
        self,
        rel_path: str,
        client_id: str,
        content: str,
        *,
        expected_mtime_ns: int | None = None,
        expected_size: int | None = None,
    ) -> FileOpenResult:
        self._cleanup_expired_leases()
        path = self.resolve_rel_path(rel_path)
        if path is None or not path.exists() or not path.is_file():
            return FileOpenResult(status="missing", rel_path=rel_path, reason="File not found.")

        canonical_rel = self.rel_path_for(path)
        lease = self._leases.get(canonical_rel)
        if not lease or lease.client_id != client_id:
            return FileOpenResult(
                status="locked",
                rel_path=canonical_rel,
                abs_path=path,
                reason="This file is currently being edited by another Dialeng session.",
            )

        current_mtime_ns, current_size = self._file_signature(path)
        expected_mtime = expected_mtime_ns if expected_mtime_ns is not None and expected_mtime_ns >= 0 else lease.mtime_ns
        expected_file_size = expected_size if expected_size is not None and expected_size >= 0 else lease.size
        if (current_mtime_ns, current_size) != (expected_mtime, expected_file_size):
            return FileOpenResult(
                status="conflict",
                rel_path=canonical_rel,
                abs_path=path,
                reason="This file changed on disk since it was opened. Reload before saving.",
                mtime_ns=current_mtime_ns,
                size=current_size,
            )

        path.write_text(content, encoding=lease.encoding)
        lease.last_seen_at = time.monotonic()
        inspection = self._inspect_file(path)
        inspection.rel_path = canonical_rel
        inspection.abs_path = path
        lease.encoding = inspection.encoding
        lease.mtime_ns = inspection.mtime_ns
        lease.size = inspection.size
        return inspection

    def _canonical_rel_path(self, rel_path: str) -> str | None:
        path = self.resolve_rel_path(rel_path)
        if path is None:
            return None
        return self.rel_path_for(path)

    def _cleanup_expired_leases(self) -> None:
        now = time.monotonic()
        expired = [
            rel_path
            for rel_path, lease in self._leases.items()
            if now - lease.last_seen_at > FILE_LOCK_TIMEOUT_SECONDS
        ]
        for rel_path in expired:
            self._leases.pop(rel_path, None)

    def _inspect_file(self, path: Path) -> FileOpenResult:
        raw = path.read_bytes()
        mtime_ns, size = self._file_signature(path)
        encoding = self._detect_encoding(raw)
        if encoding in {"utf-16", "utf-32"}:
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                return FileOpenResult(
                    status="noneditable",
                    rel_path="",
                    abs_path=path,
                    reason="This file could not be decoded as text.",
                    language="plaintext",
                    mtime_ns=mtime_ns,
                    size=size,
                )
            return FileOpenResult(
                status="editable",
                rel_path="",
                abs_path=path,
                content=text,
                language=self._language_for_path(path),
                encoding=encoding,
                mtime_ns=mtime_ns,
                size=size,
            )

        if self._is_binary(raw, path):
            return FileOpenResult(
                status="noneditable",
                rel_path="",
                abs_path=path,
                reason="This file is not plain text and cannot be edited in Dialeng.",
                language="plaintext",
                mtime_ns=mtime_ns,
                size=size,
            )

        text = None
        chosen_encoding = encoding
        for candidate in dict.fromkeys([encoding, "utf-8-sig", "utf-8", "latin-1"]):
            try:
                text = raw.decode(candidate)
                chosen_encoding = candidate
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            return FileOpenResult(
                status="noneditable",
                rel_path="",
                abs_path=path,
                reason="This file could not be decoded as text.",
                language="plaintext",
                mtime_ns=mtime_ns,
                size=size,
            )

        return FileOpenResult(
            status="editable",
            rel_path="",
            abs_path=path,
            content=text,
            language=self._language_for_path(path),
            encoding=chosen_encoding,
            mtime_ns=mtime_ns,
            size=size,
        )

    def _file_signature(self, path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _language_for_path(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in _TEXT_EXTENSIONS:
            return _TEXT_EXTENSIONS[suffix]
        return "plaintext"

    def _detect_encoding(self, raw: bytes) -> str:
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            return "utf-16"
        if raw.startswith(b"\xff\xfe\x00\x00") or raw.startswith(b"\x00\x00\xfe\xff"):
            return "utf-32"
        return "utf-8"

    def _is_binary(self, raw: bytes, path: Path) -> bool:
        if not raw:
            return False
        if b"\x00" in raw:
            return True

        suffix = path.suffix.lower()
        mime, _ = mimetypes.guess_type(path.name)
        if suffix in _TEXT_EXTENSIONS:
            return False
        if mime and mime.startswith("text/"):
            return False

        sample = raw[:2048]
        non_text = sum(
            1
            for byte in sample
            if byte not in (9, 10, 13) and not (32 <= byte <= 126)
        )
        return (non_text / max(len(sample), 1)) > 0.30
