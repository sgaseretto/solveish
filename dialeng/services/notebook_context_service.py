"""Deterministic, bounded notebook extraction for skill authoring."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx

from dialeng.document.cell import CellType
from dialeng.document.serialization import load_notebook


DEFAULT_MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARS = 4000
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*(['\"]?)[^\s'\"]+\2"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)
TEXT_SOURCE_SUFFIXES = {
    ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".css", ".html", ".htm", ".sh",
}
WEB2MD_API_URL = "https://web2md.answer.ai/api"


@dataclass(frozen=True)
class NotebookContextDocument:
    source: str
    markdown: str
    sha256: str
    cell_count: int
    redaction_count: int = 0
    kind: str = "notebook"
    fetched_at: str = ""


@dataclass(frozen=True)
class NotebookContextBundle:
    documents: tuple[NotebookContextDocument, ...]
    total_bytes: int

    def as_xml(self) -> str:
        """Return nbs2ctx-style XML while preserving Dialeng cell semantics."""
        from xml.sax.saxutils import escape

        items = []
        for index, document in enumerate(self.documents, start=1):
            items.append(
                f'<document index="{index}">\n'
                f'<source>{escape(document.source)}</source>\n'
                f'<document_content>\n{escape(document.markdown)}\n</document_content>\n'
                f'</document>'
            )
        return "<documents>\n" + "\n".join(items) + "\n</documents>"


class NotebookContextService:
    """Resolve project notebooks and render safe LLM/skill reference documents."""

    def __init__(self, project_root: Path, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES) -> None:
        self.project_root = Path(project_root).resolve()
        self.max_source_bytes = max(1024, int(max_source_bytes))

    def resolve_sources(self, sources: Iterable[str]) -> list[Path]:
        """Resolve supported local files, directories, and glob patterns safely."""
        resolved: dict[str, Path] = {}
        for raw_source in sources:
            source = str(raw_source or "").strip()
            if not source:
                continue
            if self._is_url(source):
                continue
            candidate = (self.project_root / source).resolve()
            matches: list[Path]
            if any(char in source for char in "*?["):
                matches = [path.resolve() for path in self.project_root.glob(source)]
            elif candidate.is_dir():
                matches = [
                    path.resolve() for path in candidate.rglob("*")
                    if path.is_file() and (path.suffix == ".ipynb" or path.suffix.lower() in TEXT_SOURCE_SUFFIXES)
                ]
            else:
                matches = [candidate]

            for path in matches:
                if not path.is_relative_to(self.project_root):
                    raise ValueError(f"Notebook source escapes the project: {source}")
                if ".ipynb_checkpoints" in path.parts or ".dialeng" in path.parts:
                    continue
                if not path.is_file() or not (
                    path.suffix == ".ipynb" or path.suffix.lower() in TEXT_SOURCE_SUFFIXES
                ):
                    raise ValueError(f"Supported source does not exist: {source}")
                relative = path.relative_to(self.project_root).as_posix()
                resolved[relative] = path

        if not resolved:
            if not any(self._is_url(str(source or "").strip()) for source in sources):
                raise ValueError("Select at least one notebook, text file, or URL source")
        return [resolved[key] for key in sorted(resolved)]

    def extract(self, sources: Iterable[str], *, include_outputs: bool = False) -> NotebookContextBundle:
        """Extract selected notebooks into deterministic Markdown documents."""
        documents: list[NotebookContextDocument] = []
        total_bytes = 0
        source_values = [str(source or "").strip() for source in sources if str(source or "").strip()]
        for path in self.resolve_sources(source_values):
            document = (
                self._render_notebook(path, include_outputs=include_outputs)
                if path.suffix == ".ipynb" else self._render_text_file(path)
            )
            total_bytes += len(document.markdown.encode("utf-8"))
            if total_bytes > self.max_source_bytes:
                raise ValueError(
                    f"Notebook context exceeds the {self.max_source_bytes}-byte project limit"
                )
            documents.append(document)
        for url in sorted(dict.fromkeys(source for source in source_values if self._is_url(source))):
            document = self._render_url(url)
            total_bytes += len(document.markdown.encode("utf-8"))
            if total_bytes > self.max_source_bytes:
                raise ValueError(
                    f"Skill source context exceeds the {self.max_source_bytes}-byte project limit"
                )
            documents.append(document)
        if not documents:
            raise ValueError("Select at least one notebook, text file, or URL source")
        return NotebookContextBundle(tuple(documents), total_bytes)

    def _render_text_file(self, path: Path) -> NotebookContextDocument:
        source = path.relative_to(self.project_root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, redactions = self._redact(raw)
        if path.suffix.lower() == ".md":
            markdown = f"# File: {source}\n\n{text.strip()}\n"
        else:
            language = path.suffix.lower().lstrip(".") or "text"
            markdown = f"# File: {source}\n\n```{language}\n{text.rstrip()}\n```\n"
        return NotebookContextDocument(
            source=source,
            markdown=markdown,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cell_count=0,
            redaction_count=redactions,
            kind="file",
        )

    def _render_url(self, url: str) -> NotebookContextDocument:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(f"Invalid web source URL: {url}")
        endpoint = os.environ.get("DIALENG_WEB2MD_URL", WEB2MD_API_URL)
        try:
            response = httpx.post(
                endpoint,
                data={"url": url, "extractor": "traf"},
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"Could not convert web source with web2md: {url}: {exc}") from exc
        text, redactions = self._redact(response.text.strip())
        if not text:
            raise ValueError(f"web2md returned no readable content for: {url}")
        markdown = f"# Web source: {url}\n\n{text}\n"
        return NotebookContextDocument(
            source=url,
            markdown=markdown,
            sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            cell_count=0,
            redaction_count=redactions,
            kind="url",
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _is_url(source: str) -> bool:
        return source.startswith("https://") or source.startswith("http://")

    def _render_notebook(self, path: Path, *, include_outputs: bool) -> NotebookContextDocument:
        notebook = load_notebook(path)
        source = path.relative_to(self.project_root).as_posix()
        parts = [f"# Notebook: {source}"]
        included_cells = 0
        redactions = 0

        for index, cell in enumerate(notebook.cells, start=1):
            if cell.skipped:
                continue
            cell_type = cell.cell_type.value if isinstance(cell.cell_type, CellType) else str(cell.cell_type)
            text, count = self._redact(cell.source or "")
            redactions += count
            included_cells += 1
            metadata = []
            if cell.pinned:
                metadata.append("pinned")
            if cell.is_exported:
                metadata.append("exported")
            suffix = f" ({', '.join(metadata)})" if metadata else ""

            if cell_type == CellType.NOTE.value:
                parts.extend([f"## Note cell {index}{suffix}", text])
            elif cell_type == CellType.PROMPT.value:
                response, count = self._redact(cell.output or "")
                redactions += count
                parts.extend([f"## Prompt cell {index}{suffix}", "### User", text])
                if response:
                    parts.extend(["### Assistant", response])
            else:
                parts.extend([f"## {cell_type.title()} cell {index}{suffix}", f"```python\n{text}\n```"])
                if include_outputs and cell.output:
                    output, count = self._redact(cell.output[:MAX_OUTPUT_CHARS])
                    redactions += count
                    if len(cell.output) > MAX_OUTPUT_CHARS:
                        output += "\n... (output truncated)"
                    parts.extend(["### Text output", f"```text\n{output}\n```"])

        markdown = "\n\n".join(part for part in parts if part is not None).strip() + "\n"
        return NotebookContextDocument(
            source=source,
            markdown=markdown,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cell_count=included_cells,
            redaction_count=redactions,
            kind="notebook",
        )

    @staticmethod
    def _redact(value: str) -> tuple[str, int]:
        redactions = 0
        result = value
        for pattern in _SECRET_PATTERNS:
            result, count = pattern.subn("[REDACTED]", result)
            redactions += count
        return result, redactions


def reference_name(source: str) -> str:
    """Return a stable reference filename for a project-relative notebook path."""
    stem = source[:-6] if source.endswith(".ipynb") else source
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem.replace("/", "-")).strip("-.").lower()[:80]
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'notebook'}-{digest}.md"
