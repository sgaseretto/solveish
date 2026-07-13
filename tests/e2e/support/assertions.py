"""Semantic browser assertions and interactions."""

from __future__ import annotations

import re
from urllib.parse import quote

from playwright.sync_api import Page, expect


def open_notebook(page: Page, base_url: str, name: str, *, close_kernel_modal: bool = True) -> None:
    page.goto(f"{base_url}/dialeng/?name={quote(name)}")
    expect(page.locator("#cells")).to_be_visible()
    expect(page.locator("#cells > .cell").first).to_be_visible()
    if close_kernel_modal:
        modal = page.locator("#kernel-modal-overlay")
        if modal.count() and "visible" in (modal.get_attribute("class") or ""):
            modal.get_by_role("button", name="Cancel").click()
            expect(modal).not_to_have_class("visible")


def cell_ids(page: Page) -> list[str]:
    return page.locator("#cells > .cell").evaluate_all(
        "cells => cells.map(cell => cell.id.replace('cell-', ''))"
    )


def expect_cell_count(page: Page, count: int) -> None:
    expect(page.locator("#cells > .cell")).to_have_count(count)


def attach_local_kernel(page: Page) -> None:
    modal = page.locator("#kernel-modal-overlay")
    if "visible" not in (modal.get_attribute("class") or ""):
        page.locator("#kernel-status-bar").click()
        expect(modal).to_have_class(re.compile(r"\bvisible\b"))
    modal.locator('.kernel-option[data-kernel="local"]').click()
    modal.get_by_role("button", name="Apply").click()
    expect(modal).not_to_have_class("visible")
    # A clean environment may need to build Matplotlib's font cache before the
    # first local kernel becomes ready. The UI must remain initializing until
    # notebook setup has actually completed rather than exposing a false idle.
    expect(page.locator("#kernel-dot")).to_have_attribute("title", "Kernel: idle", timeout=45_000)


def run_prompt(page: Page, cell_id: str) -> None:
    cell = page.locator(f"#cell-{cell_id}")
    version = cell.locator(f"#version-{cell_id}").input_value()
    with page.expect_response(
        lambda response: response.url.endswith(f"/cell/{cell_id}/run"),
        timeout=120_000,
    ) as completed:
        cell.locator(".btn-run").click()
    assert completed.value.ok, completed.value.text()
    expect(cell.locator(f"#version-{cell_id}")).not_to_have_value(version, timeout=20_000)
    expect(cell.locator(".btn-run")).to_be_visible(timeout=20_000)


def edit_prompt_input(page: Page, cell_id: str, value: str) -> None:
    cell = page.locator(f"#cell-{cell_id}")
    textarea = cell.locator('textarea[name="prompt_source"]')
    if not textarea.is_visible():
        cell.locator(".prompt-preview").dblclick()
    expect(textarea).to_be_visible()
    textarea.fill(value)
    with page.expect_response(lambda response: response.url.endswith(f"/{cell_id}/source")):
        textarea.blur()
    preview = cell.locator(".prompt-preview")
    if preview.count():
        expect(preview).to_contain_text(value)
    else:
        expect(textarea).to_have_value(value)


def edit_prompt_output(page: Page, cell_id: str, value: str) -> None:
    cell = page.locator(f"#cell-{cell_id}")
    textarea = cell.locator('textarea[name="output"]')
    cell.locator(".ai-preview").dblclick()
    expect(textarea).to_be_visible()
    textarea.fill(value)
    with page.expect_response(lambda response: response.url.endswith(f"/{cell_id}/output")):
        textarea.blur()
    expect(cell.locator(".ai-preview")).to_contain_text(value)
