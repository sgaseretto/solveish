"""Backend-authoritative settings behavior in one and two tabs."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import open_notebook


pytestmark = pytest.mark.e2e


def _open_settings(page):
    with page.expect_response(
        lambda response: response.url.endswith("/settings") and response.request.method == "GET"
    ) as loaded:
        page.locator("#settings-btn").click()
    assert loaded.value.status == 200
    assert "fields" in loaded.value.json()
    expect(page.locator("#settings-sidebar")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#settings-status")).to_contain_text("Settings are current", timeout=10_000)


def test_settings_apply_live_validate_and_sync_dirty_edits(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "providers")
    open_notebook(second, e2e_server["base_url"], "providers")

    _open_settings(first)
    _open_settings(second)
    first.locator('.settings-group').filter(has_text="Dialeng Display Settings").locator('summary').click()
    second.locator('.settings-group').filter(has_text="Dialeng Display Settings").locator('summary').click()

    first_font = first.locator('#settings-form input[name="display.font_size"]')
    second_font = second.locator('#settings-form input[name="display.font_size"]')
    first_font.fill("17")
    with first.expect_response(lambda response: response.url.endswith("/settings") and response.request.method == "POST") as saved:
        first.locator("#settings-save-btn").click()
    assert saved.value.status == 202
    expect(first.locator("html")).to_have_attribute("style", re.compile(r"--base-font-size: 17px"))
    expect(second_font).to_have_value("17")

    # Keep a local dirty edit while another tab saves. The dirty value remains,
    # but the revision and untouched fields advance to the backend snapshot.
    second_font.fill("18")
    first_font.fill("19")
    with first.expect_response(lambda response: response.url.endswith("/settings") and response.request.method == "POST"):
        first.locator("#settings-save-btn").click()
    expect(second_font).to_have_value("18")
    expect(second.locator("#settings-change-summary")).to_contain_text("1 unsaved change")

    with second.expect_response(lambda response: response.url.endswith("/settings") and response.request.method == "POST") as merged:
        second.locator("#settings-save-btn").click()
    assert merged.value.status == 202
    expect(first_font).to_have_value("18")
