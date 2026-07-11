"""Graceful recovery when an open notebook is deleted."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import open_notebook


pytestmark = pytest.mark.e2e


def test_open_notebook_deletion_recovers_all_tabs(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "deletion")
    open_notebook(second, e2e_server["base_url"], "deletion")

    first.locator("button[title='Delete deletion']").click()
    expect(first.locator("#delete-confirm-modal")).to_have_class(re.compile(r"\bvisible\b"))
    first.locator("#delete-confirm-modal").get_by_role("button", name="Delete").click()

    deleted_url = re.compile(r"/dialeng/workspace\?deleted=1$")
    expect(first).to_have_url(deleted_url, timeout=15_000)
    expect(second).to_have_url(deleted_url, timeout=15_000)
    expect(first.get_by_text("Notebook deleted")).to_be_visible()
    expect(second.get_by_text("Notebook deleted")).to_be_visible()
    assert not (e2e_server["project"].root / "deletion.ipynb").exists()
