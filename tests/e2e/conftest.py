"""Pytest fixtures for real-server Playwright E2E tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import sync_playwright

from .support.artifacts import collect_failure_artifacts
from .support.project_factory import E2EProject


def pytest_runtest_makereport(item, call):
    setattr(item, f"rep_{call.when}", call)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def e2e_project(tmp_path: Path) -> E2EProject:
    project = E2EProject(tmp_path / "project")
    project.create_scenarios()
    return project


@pytest.fixture
def external_codex_project(tmp_path: Path) -> E2EProject:
    project = E2EProject(tmp_path / "codex-project")
    project.create_codex_scenario()
    return project


@pytest.fixture
def e2e_server(e2e_project: E2EProject, tmp_path: Path):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "dialeng-server.log"
    env = os.environ.copy()
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "DIALENG_OPENAI_CODEX_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        env.pop(name, None)
    env.update(
        {
            "DIALENG_E2E": "1",
            "DIALENG_NOTEBOOKS_DIR": str(e2e_project.root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dialeng.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 25
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=0.5)
            if response.status_code == 200 and response.json() == {"status": "ok", "e2e": True}:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=5)
        log_handle.close()
        pytest.fail(f"Dialeng did not become healthy: {last_error}\n{log_path.read_text()}")

    if process.poll() is not None:
        log_handle.close()
        pytest.fail(f"Dialeng exited during startup:\n{log_path.read_text()}")

    server = {
        "base_url": base_url,
        "log_path": log_path,
        "process": process,
        "project": e2e_project,
    }
    yield server

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    log_handle.close()


@pytest.fixture
def external_codex_server(external_codex_project: E2EProject, tmp_path: Path):
    """Start a normal Dialeng server with real Codex authentication enabled."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "dialeng-codex-server.log"
    env = os.environ.copy()
    env.pop("DIALENG_E2E", None)
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        env.pop(name, None)
    env.update({
        "DIALENG_NOTEBOOKS_DIR": str(external_codex_project.root),
        "PYTHONUNBUFFERED": "1",
    })
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dialeng.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=0.5)
            if response.status_code == 200:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=5)
        log_handle.close()
        pytest.fail(f"Codex E2E server did not become healthy: {last_error}\n{log_path.read_text()}")

    if process.poll() is not None:
        log_handle.close()
        pytest.fail(f"Codex E2E server exited during startup:\n{log_path.read_text()}")

    server = {
        "base_url": base_url,
        "log_path": log_path,
        "process": process,
        "project": external_codex_project,
    }
    yield server

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    log_handle.close()


@pytest.fixture(scope="package")
def playwright_runtime():
    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="package")
def browser(playwright_runtime):
    browser = playwright_runtime.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture
def browser_context(browser, e2e_server, request):
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.set_default_timeout(10_000)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    console_events: list[dict] = []
    page_errors: list[str] = []
    pages: list = []

    def register_page(page):
        pages.append(page)
        page.on(
            "console",
            lambda message: console_events.append(
                {"type": message.type, "text": message.text, "url": page.url}
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))

    context.on("page", register_page)
    context._dialeng_pages = pages
    context._dialeng_console_events = console_events
    context._dialeng_page_errors = page_errors
    yield context

    unexpected_console = [
        event
        for event in console_events
        if event["type"] == "error"
    ]
    failed = bool(getattr(request.node, "rep_call", None) and request.node.rep_call.excinfo)
    failed = failed or bool(unexpected_console or page_errors)
    if failed:
        artifact_dir = collect_failure_artifacts(
            nodeid=request.node.nodeid,
            context=context,
            pages=pages,
            console_events=console_events,
            page_errors=page_errors,
            server_log=e2e_server["log_path"],
            project_root=e2e_server["project"].root,
            base_url=e2e_server["base_url"],
        )
        context.tracing.stop(path=str(artifact_dir / "trace.zip"))
    else:
        context.tracing.stop()
    context.close()

    assert not page_errors, f"Unexpected browser page errors: {page_errors}"
    assert not unexpected_console, f"Unexpected browser console errors: {unexpected_console}"


@pytest.fixture
def page(browser_context):
    return browser_context.new_page()


@pytest.fixture
def external_codex_context(browser, external_codex_server, request):
    """Capture the same diagnostics as deterministic browser tests."""
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.set_default_timeout(20_000)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    console_events: list[dict] = []
    page_errors: list[str] = []
    pages: list = []

    def register_page(page):
        pages.append(page)
        page.on(
            "console",
            lambda message: console_events.append(
                {"type": message.type, "text": message.text, "url": page.url}
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))

    context.on("page", register_page)
    yield context

    unexpected_console = [event for event in console_events if event["type"] == "error"]
    failed = bool(getattr(request.node, "rep_call", None) and request.node.rep_call.excinfo)
    failed = failed or bool(unexpected_console or page_errors)
    if failed:
        artifact_dir = collect_failure_artifacts(
            nodeid=request.node.nodeid,
            context=context,
            pages=pages,
            console_events=console_events,
            page_errors=page_errors,
            server_log=external_codex_server["log_path"],
            project_root=external_codex_server["project"].root,
            base_url=external_codex_server["base_url"],
        )
        context.tracing.stop(path=str(artifact_dir / "trace.zip"))
    else:
        context.tracing.stop()
    context.close()

    assert not page_errors, f"Unexpected browser page errors: {page_errors}"
    assert not unexpected_console, f"Unexpected browser console errors: {unexpected_console}"


@pytest.fixture
def external_codex_page(external_codex_context):
    return external_codex_context.new_page()
