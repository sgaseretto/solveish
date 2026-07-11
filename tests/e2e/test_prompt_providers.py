"""Prompt execution across deterministic modes and models."""

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, open_notebook, run_prompt


pytestmark = pytest.mark.e2e


def test_mock_then_deterministic_provider_rerun(page, e2e_server):
    open_notebook(
        page, e2e_server["base_url"], "providers", close_kernel_modal=False
    )
    attach_local_kernel(page)
    cell = page.locator("#cell-pro00001")

    run_prompt(page, "pro00001")
    expect(cell.locator(".ai-preview")).to_contain_text("demo response", timeout=20_000)

    with page.expect_response(lambda response: response.url.endswith("/mode")):
        page.locator("#mode-select").select_option("concise")
    expect(page.locator("#provider-select")).to_be_visible()
    with page.expect_response(lambda response: response.url.endswith("/provider")):
        page.locator("#provider-select").select_option("e2e_fake")
    expect(page.locator("#model-select")).to_have_value("e2e-fast")
    with page.expect_response(lambda response: response.url.endswith("/model")):
        page.locator("#model-select").select_option("e2e-slow")
        page.locator("#model-select").dispatch_event("change")
    expect(page.locator("#model-select")).to_have_value("e2e-slow")

    run_prompt(page, "pro00001")
    expect(cell.locator(".ai-preview")).to_contain_text(
        "E2E response: Describe deterministic testing.", timeout=20_000
    )
    expect(cell.locator(".prompt-diagnostics")).to_contain_text("e2e-slow")
    diagnostics = cell.locator(".prompt-diagnostics")
    expect(diagnostics).not_to_have_attribute("open", "")
    diagnostics.locator("summary").click()
    expect(diagnostics).to_have_attribute("open", "")
    expect(diagnostics.locator(".prompt-diagnostic-grid")).to_be_visible()
    expect(page.locator("#notebook-llm-usage")).not_to_have_text("0 tok")

    with page.expect_response(lambda response: response.url.endswith("/model")):
        page.locator("#model-select").select_option("e2e-error")
        page.locator("#model-select").dispatch_event("change")
    with page.expect_response(
        lambda response: response.url.endswith("/cell/pro00001/run"), timeout=20_000,
    ) as failed:
        page.evaluate(
            "() => clickCellRunButton(document.getElementById('cell-pro00001'), {advance: false})"
        )
    assert failed.value.ok
    expect(cell.locator(".ai-preview")).to_contain_text(
        "E2E response: Describe deterministic testing."
    )
    expect(cell.locator(".ai-preview")).not_to_contain_text("provider error")
    expect(cell.locator(".prompt-diagnostics")).to_contain_text("error")
    expect(cell.locator(".prompt-diagnostics")).not_to_have_attribute("open", "")
    expect(page.locator("#status .status-action")).to_have_text("Retry")
