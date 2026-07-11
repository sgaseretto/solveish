"""Opt-in browser validation against the real OpenAI Codex transport."""

from __future__ import annotations

import os
import re

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, open_notebook


pytestmark = [
    pytest.mark.external_openai,
    pytest.mark.skipif(
        os.environ.get("DIALENG_RUN_EXTERNAL_OPENAI") != "1",
        reason="set DIALENG_RUN_EXTERNAL_OPENAI=1 to use stored Codex authentication",
    ),
]


def _open_codex(page, server, name: str):
    auth = page.request.get(f"{server['base_url']}/auth/openai/status")
    assert auth.ok
    assert auth.json().get("authenticated"), (
        "OpenAI Codex is not authenticated. Connect OAuth or set "
        "DIALENG_OPENAI_CODEX_ACCESS_TOKEN before running external tests."
    )
    open_notebook(page, server["base_url"], name, close_kernel_modal=False)
    attach_local_kernel(page)
    expect(page.locator("#provider-select")).to_have_value("openai_codex")
    expect(page.locator("#model-select")).to_have_value("gpt-5.4")


def _run_prompt(page, cell_id: str, expected) -> dict:
    cell = page.locator(f"#cell-{cell_id}")
    with page.expect_response(
        lambda response: response.url.endswith(f"/cell/{cell_id}/run"),
        timeout=120_000,
    ) as completed:
        page.evaluate(
            "cellId => clickCellRunButton(document.getElementById(`cell-${cellId}`), {advance: false})",
            cell_id,
        )
    assert completed.value.ok, completed.value.text()
    expect(cell.locator(".ai-preview")).to_contain_text(expected, timeout=120_000)
    expect(cell.locator(".btn-run")).to_be_visible(timeout=120_000)
    expect(cell).not_to_have_class(re.compile(r"\bstreaming\b"), timeout=120_000)
    page.wait_for_timeout(150)
    return completed.value.headers


def _run_code(page, cell_id: str, expected_output: str) -> None:
    cell = page.locator(f"#cell-{cell_id}")
    cell.locator(".btn-run").click()
    expect(cell.locator(".cell-output")).to_contain_text(expected_output, timeout=60_000)


def test_real_codex_uses_current_notebook_context(external_codex_page, external_codex_server):
    page = external_codex_page
    _open_codex(page, external_codex_server, "codex-context")
    _run_prompt(page, "cdxprm01", "Jane Doe")
    expect(page.locator("#cell-cdxprm01 .ai-preview")).not_to_contain_text("Joe Doe")


def test_real_codex_receives_code_cell_images(external_codex_page, external_codex_server):
    page = external_codex_page
    _open_codex(page, external_codex_server, "codex-image")
    page.locator("#cell-cdximg01 .btn-run").click()
    expect(page.locator("#cell-cdximg01 img.mime-image")).to_be_visible()
    _run_prompt(page, "cdxprm02", re.compile("red", re.IGNORECASE))
    expect(page.locator("#cell-cdxprm02 .ai-preview")).to_contain_text(
        re.compile("blue", re.IGNORECASE)
    )


def test_real_codex_calls_kernel_tools(external_codex_page, external_codex_server):
    page = external_codex_page
    _open_codex(page, external_codex_server, "codex-tool")
    _run_code(page, "cdxtol01", "DIALENG_TOOL_READY")
    _run_prompt(page, "cdxprm03", "DIALENG_TOOL_OK:notebook")


def test_real_codex_loads_project_skills(external_codex_page, external_codex_server):
    page = external_codex_page
    _open_codex(page, external_codex_server, "codex-skill")
    _run_prompt(page, "cdxprm04", "DIALENG_SKILL_OK")


def test_real_codex_output_saves_and_reloads(external_codex_page, external_codex_server):
    page = external_codex_page
    _open_codex(page, external_codex_server, "codex-persistence")
    _run_prompt(page, "cdxprm05", "DIALENG_PERSIST_OK")
    expect(page.locator("#cell-cdxprm05 .prompt-diagnostics")).to_contain_text("gpt-5.4")
    expect(page.locator("#notebook-llm-usage")).not_to_have_text("0 tok")
    expect(page.locator("#notebook-save-state")).to_have_class(
        "notebook-save-state modified", timeout=20_000
    )
    with page.expect_response(lambda response: response.url.endswith("/save")) as saved:
        page.locator("#save-btn").click()
    assert saved.value.ok, saved.value.text()
    expect(page.locator("#notebook-save-state")).to_have_class(
        "notebook-save-state saved", timeout=20_000
    )
    page.reload()
    expect(page.locator("#cell-cdxprm05 .ai-preview")).to_contain_text(
        "DIALENG_PERSIST_OK"
    )
    expect(page.locator("#cell-cdxprm05 .prompt-diagnostics")).to_contain_text("gpt-5.4")
