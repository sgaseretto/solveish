"""Review-first notebook-to-skill creation in a real browser."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import open_notebook


pytestmark = pytest.mark.e2e


def test_create_preview_and_install_skill_from_notebook(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "providers")

    with page.expect_response(
        lambda response: response.url.endswith("/dialeng/skills")
        and response.request.method == "GET"
    ) as loaded:
        page.locator(".skill-creator-btn").click()
    assert loaded.value.status == 200
    expect(page.locator("#skill-creator-overlay")).to_have_class(
        re.compile(r"\bvisible\b")
    )

    page.get_by_label("Skill name", exact=True).fill("e2e-notebook-skill")
    page.get_by_label("Trigger description", exact=True).fill(
        "Use when validating the deterministic E2E notebook workflow."
    )
    page.get_by_label("Refine with selected LLM", exact=True).uncheck()
    page.get_by_label("providers.ipynb", exact=True).check()

    with page.expect_response(
        lambda response: response.url.endswith("/dialeng/skills/build")
        and response.request.method == "POST"
    ) as staged:
        page.locator("#skill-generate-btn").click()
    assert staged.value.status == 200
    expect(page.locator("#skill-creator-status")).to_contain_text("Draft validated")
    expect(page.locator("#skill-install-btn")).to_be_enabled()
    expect(page.locator("#skill-creator-preview")).to_contain_text("e2e-notebook-skill")
    assert page.locator(".skill-preview-file").count() == 3

    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_response(
        lambda response: "/commit" in response.url
        and response.request.method == "POST"
    ) as committed:
        page.locator("#skill-install-btn").click()
    assert committed.value.status == 200
    expect(page.locator("#skill-creator-status")).to_contain_text(
        "Installed e2e-notebook-skill"
    )
    assert (
        e2e_server["project"].root
        / ".dialeng"
        / "skills"
        / "e2e-notebook-skill"
        / "SKILL.md"
    ).is_file()
