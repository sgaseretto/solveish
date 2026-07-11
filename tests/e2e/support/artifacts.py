"""Failure artifact collection and secret redaction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token)\s*[:=]\s*)[^\s,\"']+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def redact(value: str) -> str:
    for pattern in TOKEN_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1) if match.lastindex else ''}[REDACTED]", value)
    return value


def safe_test_name(nodeid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("_")


def collect_failure_artifacts(
    *,
    nodeid: str,
    context,
    pages: list,
    console_events: list[dict[str, Any]],
    page_errors: list[str],
    server_log: Path,
    project_root: Path,
    base_url: str,
) -> Path:
    artifact_dir = Path("test-results") / safe_test_name(nodeid)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for index, page in enumerate(pages):
        try:
            page.screenshot(path=str(artifact_dir / f"page-{index}.png"), full_page=True)
            (artifact_dir / f"page-{index}.html").write_text(
                redact(page.content()), encoding="utf-8"
            )
        except Exception as exc:
            (artifact_dir / f"page-{index}-capture-error.txt").write_text(str(exc))

    (artifact_dir / "console.json").write_text(
        redact(json.dumps(console_events, indent=2, default=str)), encoding="utf-8"
    )
    (artifact_dir / "page-errors.json").write_text(
        redact(json.dumps(page_errors, indent=2)), encoding="utf-8"
    )
    if server_log.exists():
        (artifact_dir / "server.log").write_text(
            redact(server_log.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
    for notebook in project_root.rglob("*.ipynb"):
        destination = artifact_dir / "notebooks" / notebook.relative_to(project_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            redact(notebook.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )

    try:
        import httpx

        for page in pages:
            notebook_id = page.evaluate("() => window.NOTEBOOK_ID || null")
            if not notebook_id:
                continue
            response = httpx.get(f"{base_url}/dialeng/{notebook_id}/diagnostics", timeout=3)
            (artifact_dir / f"diagnostics-{notebook_id}.json").write_text(
                redact(response.text), encoding="utf-8"
            )
    except Exception as exc:
        (artifact_dir / "diagnostics-error.txt").write_text(str(exc))
    return artifact_dir
