"""Review-first notebook-to-skill creation in a real browser."""

import re

import pytest
from playwright.sync_api import expect

from .support.assertions import open_notebook


pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("install_root", ["dialeng", "agents"])
def test_create_preview_and_install_skill_from_notebook(page, e2e_server, install_root):
    linked = install_root == "agents"
    if linked:
        snapshot = page.request.get(f"{e2e_server['base_url']}/settings").json()
        response = page.request.post(f"{e2e_server['base_url']}/settings", data={
            "revision": snapshot["revision"],
            "changes": {"skills.install_root": "agents", "skills.link_other_root": True},
        })
        assert response.status == 202
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
    expect(page.get_by_label("Original skill location", exact=True)).to_have_value(install_root)
    expect(page.locator("#skill-creator-link-other-root")).to_be_checked(checked=linked)

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
    expect(page.locator("#skill-creator-build-label")).to_contain_text(f".{install_root}/skills/e2e-notebook-skill")
    if linked:
        expect(page.locator("#skill-creator-build-label")).to_contain_text("Link: .dialeng/skills/e2e-notebook-skill")
        # Editing the next draft's controls must not redirect the reviewed build.
        page.get_by_label("Original skill location", exact=True).select_option("dialeng")
        page.locator("#skill-creator-link-other-root").uncheck()

    def accept_reviewed_destination(dialog):
        assert f".{install_root}/skills/e2e-notebook-skill" in dialog.message
        if linked:
            assert "link it at .dialeng/skills/e2e-notebook-skill" in dialog.message
        dialog.accept()

    page.once("dialog", accept_reviewed_destination)
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
        / f".{install_root}"
        / "skills"
        / "e2e-notebook-skill"
        / "SKILL.md"
    ).is_file()
    if linked:
        original = e2e_server["project"].root / ".agents/skills/e2e-notebook-skill"
        alias = e2e_server["project"].root / ".dialeng/skills/e2e-notebook-skill"
        assert not original.is_symlink()
        assert alias.is_symlink() and alias.resolve() == original
