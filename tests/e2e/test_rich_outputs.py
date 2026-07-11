"""Jupyter-compatible rich output rendering in the browser."""

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, open_notebook


pytestmark = pytest.mark.e2e


def test_html_json_and_image_outputs_render(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "rich", close_kernel_modal=False)
    attach_local_kernel(page)
    cell = page.locator("#cell-rch00001")
    cell.locator(".btn-run").click()

    expect(cell.locator("#e2e-rich-html")).to_have_text("Rich HTML", timeout=30_000)
    expect(cell.locator(".mime-json")).to_contain_text('"answer": 42')
    expect(cell.locator("img.mime-image")).to_be_visible()
    expect(cell.locator(".cell-meta")).to_contain_text("[1]")
