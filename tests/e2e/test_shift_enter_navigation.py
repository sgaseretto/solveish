"""Shift+Enter focus and final-cell creation in long notebooks."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, expect_cell_count, open_notebook


pytestmark = pytest.mark.e2e


def test_shift_enter_keeps_position_and_creates_trailing_cell(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "long", close_kernel_modal=False)
    attach_local_kernel(page)
    expect_cell_count(page, 25)

    current = page.locator("#cell-lng00018")
    following = page.locator("#cell-lng00019")
    current.scroll_into_view_if_needed()
    current.locator(".cell-header").click()
    page.keyboard.press("Shift+Enter")
    expect(following).to_have_class(re.compile(r"\bfocused\b"), timeout=10_000)
    assert following.evaluate("el => Math.abs(el.getBoundingClientRect().top)") < 900

    final_prompt = page.locator("#cell-lng99999")
    final_prompt.scroll_into_view_if_needed()
    final_prompt.locator('textarea[name="prompt_source"]').focus()
    page.keyboard.press("Shift+Enter")
    expect_cell_count(page, 26)
    trailing = page.locator("#cells > .cell").last
    expect(trailing).to_have_attribute("data-type", "code")
    expect(trailing).to_have_class(re.compile(r"\bfocused\b"))
    assert trailing.evaluate("el => Math.abs(el.getBoundingClientRect().top)") < 1000
