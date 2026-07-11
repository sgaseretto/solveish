"""Two-tab source, output, and structure synchronization."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import (
    attach_local_kernel,
    edit_prompt_input,
    edit_prompt_output,
    expect_cell_count,
    open_notebook,
    run_prompt,
)


pytestmark = pytest.mark.e2e


def test_two_tabs_share_source_output_and_structure(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "collaboration", close_kernel_modal=False)
    attach_local_kernel(first)
    open_notebook(second, e2e_server["base_url"], "collaboration")

    edit_prompt_input(first, "col00001", "Shared source from tab one")
    expect(second.locator('#cell-col00001 textarea[name="prompt_source"]')).to_have_value(
        "Shared source from tab one"
    )

    run_prompt(first, "col00001")
    expect(second.locator("#cell-col00001 .ai-preview")).to_contain_text("demo response")
    expect(second.locator("#cell-col00001 .prompt-diagnostics")).to_contain_text(
        "completed"
    )
    expect(second.locator("#cell-col00001")).not_to_have_class(
        re.compile(r"\bstreaming\b")
    )
    second.wait_for_timeout(150)

    second.bring_to_front()
    edit_prompt_output(second, "col00001", "Edited in tab two")
    expect(first.locator("#cell-col00001 .ai-preview")).to_contain_text("Edited in tab two")

    previous_count = first.locator("#cells > .cell").count()
    first.locator("#cells > .add-row").last.get_by_role("button", name="Code").click()
    expect_cell_count(first, previous_count + 1)
    expect_cell_count(second, previous_count + 1)


def test_prompt_continues_when_initiating_tab_closes(browser_context, e2e_server):
    first = browser_context.new_page()
    observer = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "collaboration", close_kernel_modal=False)
    attach_local_kernel(first)
    open_notebook(observer, e2e_server["base_url"], "collaboration")

    with first.expect_response(lambda response: response.url.endswith("/mode")):
        first.locator("#mode-select").select_option("concise")
    with first.expect_response(lambda response: response.url.endswith("/model")):
        first.locator("#model-select").select_option("e2e-slow")
        first.locator("#model-select").dispatch_event("change")
    expect(observer.locator("#model-select")).to_have_value("e2e-slow")

    first.evaluate(
        "() => clickCellRunButton(document.getElementById('cell-col00001'), {advance: false})"
    )
    expect(observer.locator("#cell-col00001")).to_have_class(
        re.compile(r"\bstreaming\b"), timeout=10_000
    )
    first.close()

    expect(observer.locator("#cell-col00001 .ai-preview")).to_contain_text(
        "E2E response: Shared source", timeout=20_000
    )
    expect(observer.locator("#cell-col00001 .btn-run")).to_be_visible()
    expect(observer.locator("#cell-col00001 .prompt-diagnostics")).to_contain_text(
        "completed"
    )
