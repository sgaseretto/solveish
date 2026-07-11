"""Repeated editing of both sides of a prompt cell."""

import pytest
from playwright.sync_api import expect

from .support.assertions import (
    attach_local_kernel,
    edit_prompt_input,
    edit_prompt_output,
    open_notebook,
    run_prompt,
)


pytestmark = pytest.mark.e2e


def test_prompt_input_and_output_survive_repeated_edits(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "editing", close_kernel_modal=False)
    attach_local_kernel(page)
    run_prompt(page, "edt00001")

    for index in range(3):
        edit_prompt_input(page, "edt00001", f"My name is Joe {index}.")
        edit_prompt_output(page, "edt00001", f"Nice to meet you, Joe {index}.")

    page.locator("#save-btn").click()
    expect(page.locator("#notebook-save-state")).to_have_class("notebook-save-state saved")
    page.reload()
    expect(page.locator("#cell-edt00001 .prompt-preview")).to_contain_text("My name is Joe 2.")
    expect(page.locator("#cell-edt00001 .ai-preview")).to_contain_text(
        "Nice to meet you, Joe 2."
    )
