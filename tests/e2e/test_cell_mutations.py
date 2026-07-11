"""Backend-authoritative cell structure operations."""

import pytest
from playwright.sync_api import expect

from .support.assertions import (
    attach_local_kernel,
    cell_ids,
    expect_cell_count,
    open_notebook,
)


pytestmark = pytest.mark.e2e


def test_add_convert_move_duplicate_merge_and_delete(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "mutations", close_kernel_modal=False)
    attach_local_kernel(page)
    expect_cell_count(page, 3)

    first = page.locator("#cell-mut00001")
    first.locator(".btn-run").click()
    expect(first.locator(".cell-output")).to_contain_text("", timeout=20_000)
    expect(first.locator(".cell-meta")).to_contain_text("[1]", timeout=20_000)

    with page.expect_response(lambda response: response.url.endswith("/type")):
        first.locator(".type-select").select_option("note")
    expect(first).to_have_attribute("data-type", "note")
    expect(first).to_contain_text("first = 1")

    with page.expect_response(lambda response: response.url.endswith("/type")):
        first.locator(".type-select").select_option("prompt")
    expect(first).to_have_attribute("data-type", "prompt")
    expect(first.locator('textarea[name="prompt_source"]')).to_have_value("first = 1")

    page.locator("#cells > .add-row").last.get_by_role("button", name="Code").click()
    expect_cell_count(page, 4)
    added_id = cell_ids(page)[-1]

    before_move = cell_ids(page)
    page.locator("#cell-mut00002 button[title='Move down']").click()
    page.wait_for_function(
        "before => [...document.querySelectorAll('#cells > .cell')].map(c => c.id).join(',') !== before",
        arg=",".join(f"cell-{cell_id}" for cell_id in before_move),
    )
    assert cell_ids(page).index("mut00002") == before_move.index("mut00002") + 1

    duplicate_source_id = cell_ids(page)[0]
    page.locator(f"#cell-{duplicate_source_id} .cell-header").click()
    page.keyboard.press("q")
    expect_cell_count(page, 5)

    merge_id = cell_ids(page)[1]
    page.locator(f"#cell-{merge_id} .cell-header").click()
    page.keyboard.press("Shift+M")
    expect_cell_count(page, 4)

    page.locator(f"#cell-{added_id} button[title^='Delete']").click()
    expect_cell_count(page, 3)
