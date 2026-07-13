"""Real-browser acceptance coverage for the backend command catalog."""
from __future__ import annotations

import re
import sys

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, expect_cell_count, open_notebook


pytestmark = pytest.mark.e2e


def _palette_shortcut() -> str:
    return "Meta+Shift+P" if sys.platform == "darwin" else "Control+Shift+P"


def _open_manager(page):
    page.locator("#extension-manager-btn").click()
    expect(page.locator("#extension-manager-overlay")).to_have_class(re.compile(r"\bvisible\b"))


def _trust_hello(page):
    extension_id = "dev.dialeng.hello-workspace"
    card = page.locator(f'.extension-manager-card[data-extension-id="{extension_id}"]')
    page.once("dialog", lambda dialog: dialog.accept())
    card.get_by_role("button", name="Review & trust").click()
    expect(card).to_have_class(re.compile(r"\bactive\b"), timeout=20_000)
    page.locator(".extension-manager-close").click()


def _trust_extension(page, extension_id: str):
    card = page.locator(f'.extension-manager-card[data-extension-id="{extension_id}"]')
    page.once("dialog", lambda dialog: dialog.accept())
    card.get_by_role("button", name="Review & trust").click()
    expect(card).to_have_class(re.compile(r"\bactive\b"), timeout=20_000)


def _search(page, value: str):
    search = page.locator("#command-palette-search")
    search.fill(value)
    return page.locator("#command-palette-list .command-palette-item")


def test_command_palette_core_extension_arguments_availability_and_focus(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "extensions")
    open_notebook(second, e2e_server["base_url"], "extensions")

    first.locator("#settings-btn").focus()
    first.keyboard.press(_palette_shortcut())
    palette = first.locator("#command-palette-overlay")
    expect(palette).to_have_class(re.compile(r"\bvisible\b"))
    expect(first.locator("#command-palette-search")).to_be_focused()
    items = _search(first, "Open Settings")
    expect(items).to_have_count(1)
    first.keyboard.press("Enter")
    expect(first.locator("#settings-sidebar")).to_have_class(re.compile(r"\bopen\b"))
    expect(palette).not_to_have_class(re.compile(r"\bvisible\b"))
    first.locator(".settings-close-btn").click()

    first.locator("#command-palette-btn").click()
    items = _search(first, "Add Cell")
    expect(items).to_have_count(1)
    first.keyboard.press("Enter")
    expect(first.locator("#command-palette-args")).to_be_visible()
    first.locator('[data-command-arg="cell_type"]').select_option("note")
    before = first.locator("#cells > .cell").count()
    first.locator("#command-palette-args-run").click()
    expect_cell_count(first, before + 1)
    expect(palette).not_to_have_class(re.compile(r"\bvisible\b"))

    _open_manager(first)
    _trust_hello(first)
    first.locator("#command-palette-btn").click()
    items = _search(first, "Kernel only")
    expect(items).to_have_count(1)
    expect(items.first).to_have_attribute("aria-disabled", "true")
    expect(items.first).to_contain_text("Attach a running kernel")
    attach_local_kernel(second)
    expect(items.first).to_have_attribute("aria-disabled", "false", timeout=20_000)
    first.keyboard.press("Escape")

    second.locator("#command-palette-btn").click()
    items = _search(second, "Say hello")
    expect(items).to_have_count(1)
    second.keyboard.press("Enter")
    expect(second.locator("#status")).to_contain_text("Hello from Hello Workspace", timeout=20_000)
    expect(second.locator("#command-palette-overlay")).not_to_have_class(re.compile(r"\bvisible\b"))

    second.locator("#command-palette-btn").click()
    items = _search(second, "Slow greeting")
    expect(items).to_have_count(1)
    second.keyboard.press("Enter")
    cancel = second.locator("#command-palette-cancel")
    expect(cancel).to_be_visible()
    operation = second.evaluate(
        "({active: !!commandPaletteState.activeInvocation, operationId: commandPaletteState.activeInvocation?.operationId || '', cancellable: typeof commandPaletteState.activeInvocation?.cancel === 'function'})"
    )
    assert operation["active"] and operation["cancellable"] and operation["operationId"], operation
    cancel.click()
    expect(second.locator("#status")).to_contain_text("cancel", timeout=20_000)
    expect(cancel).to_be_hidden()
    second.keyboard.press("Escape")

    second.locator("#command-palette-btn").click()
    items = _search(second, "Say hello")
    expect(items).to_have_count(1)
    _open_manager(first)
    card = first.locator('.extension-manager-card[data-extension-id="dev.dialeng.hello-workspace"]')
    card.get_by_role("button", name="Disable").click()
    expect(items).to_have_count(0, timeout=20_000)
    first.locator(".extension-manager-close").click()
    second.keyboard.press("Escape")

    second.locator("#command-palette-btn").focus()
    second.keyboard.press(_palette_shortcut())
    expect(second.locator("#command-palette-search")).to_be_focused()
    second.keyboard.press("Escape")
    expect(second.locator("#command-palette-btn")).to_be_focused()


def test_command_palette_thousand_commands_search_conflicts_and_safe_text(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "extensions")
    _open_manager(page)
    _trust_extension(page, "dev.e2e.command-stress")
    page.locator(".extension-manager-close").click()

    started = page.evaluate("performance.now()")
    page.locator("#command-palette-btn").click()
    expect(page.locator("#command-palette-list .command-palette-item")).to_have_count(1007, timeout=20_000)
    opened_ms = page.evaluate("performance.now()") - started
    assert opened_ms < 3000

    started = page.evaluate("performance.now()")
    items = _search(page, "Synthetic command 0999")
    expect(items).to_have_count(1)
    searched_ms = page.evaluate("performance.now()") - started
    assert searched_ms < 1000

    items = _search(page, "Unsafe title")
    expect(items).to_have_count(1)
    expect(items.first).to_contain_text("<img src=x onerror=alert(1)> Unsafe title")
    expect(items.first.locator("img")).to_have_count(0)
    expect(items.first.locator(".command-palette-shortcut.conflict")).to_have_count(1)
