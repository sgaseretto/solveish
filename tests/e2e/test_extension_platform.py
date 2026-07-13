"""End-to-end contract tests for the Dialeng extension platform."""

from pathlib import Path
import re

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, open_notebook


pytestmark = pytest.mark.e2e


def _card(page, extension_id: str):
    return page.locator(f'.extension-manager-card[data-extension-id="{extension_id}"]')


def _open_manager(page):
    page.locator("#extension-manager-btn").click()
    expect(page.locator("#extension-manager-overlay")).to_have_class(
        re.compile(r"\bvisible\b")
    )
    expect(page.locator("#extension-manager-list")).not_to_contain_text("Loading extensions")


def _trust_and_enable(page, extension_id: str):
    card = _card(page, extension_id)
    expect(card).to_be_visible()
    page.once("dialog", lambda dialog: dialog.accept())
    card.get_by_role("button", name="Review & trust").click()
    expect(card).to_have_class(re.compile(r"\bactive\b"), timeout=20_000)


def test_hello_workspace_trust_settings_commands_reload_and_two_tabs(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "extensions")
    open_notebook(second, e2e_server["base_url"], "extensions")

    _open_manager(first)
    extension_id = "dev.dialeng.hello-workspace"
    _trust_and_enable(first, extension_id)
    expect(first.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)
    expect(second.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)

    first.locator(".extension-manager-close").click()
    first.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]').click()
    expect(first.locator("#status")).to_contain_text("Hello from Hello Workspace (1)")

    first.locator("#settings-btn").click()
    group = first.locator(".settings-group").filter(has_text="Hello Workspace")
    expect(group).to_be_visible()
    group.locator("summary").click()
    greeting = group.locator('input[name="extensions.dev.dialeng.hello-workspace.greeting"]')
    greeting.fill("Hola")
    with first.expect_response(lambda response: response.url.endswith("/settings") and response.request.method == "POST"):
        first.locator("#settings-save-btn").click()
    expect(first.locator("#settings-status")).to_contain_text(
        re.compile(r"Settings (saved|are active)"), timeout=20_000,
    )
    first.locator(".settings-close-btn").click()
    first.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]').click()
    expect(first.locator("#status")).to_contain_text("Hola from Hello Workspace (2)")

    _open_manager(first)
    _card(first, extension_id).get_by_role("button", name="Reload").click()
    expect(first.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)
    expect(second.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)
    _card(first, extension_id).get_by_role("button", name="Disable").click()
    expect(first.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(0)
    expect(second.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(0)


def test_execution_guard_blocks_kernel_and_cleans_up(browser_context, e2e_server):
    page = browser_context.new_page()
    open_notebook(page, e2e_server["base_url"], "extensions", close_kernel_modal=False)
    attach_local_kernel(page)
    _open_manager(page)
    extension_id = "dev.dialeng.execution-guard"
    _trust_and_enable(page, extension_id)
    page.locator(".extension-manager-close").click()

    expect(page.locator("#cell-ext00001 .monaco-editor")).to_be_visible(timeout=20_000)
    page.evaluate("monacoEditors['ext00001'].setValue(\"# BLOCK\\nprint('should not run')\")")
    page.locator("#cell-ext00001 .btn-run").click()
    expect(page.locator("#cell-ext00001 .cell-output")).to_contain_text(
        "Execution Guard blocked marker", timeout=20_000,
    )
    expect(page.locator("#header-ext00001")).not_to_contain_text("[1]")
    expect(page.locator("#cell-ext00001 .btn-run")).to_be_enabled()

    _open_manager(page)
    _card(page, extension_id).get_by_role("button", name="Disable").click()
    page.locator(".extension-manager-close").click()
    page.evaluate("monacoEditors['ext00001'].setValue(\"print('allowed after disable')\")")
    page.locator("#cell-ext00001 .btn-run").click()
    expect(page.locator("#cell-ext00001 .cell-output")).to_contain_text(
        "allowed after disable", timeout=20_000,
    )


def test_diagram_cell_round_trip_assets_collaboration_and_disabled_fallback(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server["base_url"], "extensions")
    open_notebook(second, e2e_server["base_url"], "extensions")
    _open_manager(first)
    extension_id = "dev.dialeng.diagram-cell"
    _trust_and_enable(first, extension_id)
    expect(first.locator('link[data-dialeng-extension-asset*="diagram.css"]')).to_have_count(1)
    expect(first.locator('script[data-dialeng-extension-asset*="diagram.js"]')).to_have_count(1)
    first.wait_for_function("() => window.__dialengExtensionMetrics?.diagramListeners === 1")
    first.evaluate("""() => {
        ToolUI.showToolCall('ext00001', 'diagram-tool-1', 'diagram-preview', {source: 'A --> B'});
        ToolUI.showToolResult('ext00001', 'diagram-tool-1', 'diagram-preview', {content: 'rendered'}, 'success');
    }""")
    expect(first.locator("#tool-call-diagram-tool-1.diagram-tool-renderer")).to_contain_text("A --> B")
    expect(first.locator("#tool-call-diagram-tool-1 .diagram-tool-result")).to_contain_text("success: rendered")
    first.locator(".extension-manager-close").click()

    with first.expect_response(lambda response: response.url.endswith("/type")):
        first.locator("#cell-ext00002 .type-select").select_option("diagram")
    expect(first.locator("#cell-ext00002")).to_have_attribute("data-cell-type", "diagram")
    source = first.locator("#cell-ext00002 .diagram-source")
    source.fill("graph TD\nA --> B")
    with first.expect_response(lambda response: response.url.endswith("/source")):
        source.blur()
    expect(first.locator("#cell-ext00002 .diagram-preview")).to_contain_text("A --> B")
    expect(second.locator("#cell-ext00002")).to_contain_text("A --> B")

    first.locator("#save-btn").click()
    first.reload()
    modal = first.locator("#kernel-modal-overlay")
    if "visible" in (modal.get_attribute("class") or ""):
        modal.get_by_role("button", name="Cancel").click()
    expect(first.locator("#cell-ext00002 .diagram-source")).to_have_value("graph TD\nA --> B")

    _open_manager(first)
    _card(first, extension_id).get_by_role("button", name="Disable").click()
    expect(first.locator('link[data-dialeng-extension-asset*="diagram.css"]')).to_have_count(0)
    expect(first.locator('script[data-dialeng-extension-asset*="diagram.js"]')).to_have_count(0)
    first.wait_for_function("() => window.__dialengExtensionMetrics?.diagramListeners === 0")
    first.wait_for_function("() => window.__dialengExtensionMetrics?.diagramToolRenderers === 0")
    first.reload()
    expect(first.locator("#cell-ext00002")).to_have_class(
        re.compile(r"\bunsupported-extension-cell\b")
    )
    expect(first.locator("#cell-ext00002")).to_contain_text("A --> B")


def test_project_inspector_panel_and_independent_skill_resource(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "extensions")
    _open_manager(page)
    extension_id = "dev.dialeng.project-inspector"
    _trust_and_enable(page, extension_id)
    card = _card(page, extension_id)
    card.get_by_role("button", name="Enable skills").click()
    expect(card.get_by_role("button", name="Disable skills")).to_be_visible()
    expect(page.locator('[data-extension-surface="dev.dialeng.project-inspector:project"]')).to_contain_text(
        "Project Inspector", timeout=20_000,
    )
    page.locator(".extension-manager-close").click()
    page.locator('[data-extension-toolbar="dev.dialeng.project-inspector:inspect"]').click()
    expect(page.locator("#status")).to_contain_text("notebooks")


def test_extension_studio_create_build_and_isolated_prepare(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "extensions")
    _open_manager(page)
    page.locator("#extension-studio-toggle").click()
    page.locator("#extension-studio-name").fill("browser-created")
    page.locator("#extension-studio-id").fill("dev.e2e.browser-created")
    page.locator("#extension-studio-template").select_option("basic")
    page.once("dialog", lambda dialog: dialog.dismiss())
    page.get_by_role("button", name="Create notebook").click()
    expect(page.locator("#extension-studio-status")).to_contain_text("Created and statically validated", timeout=20_000)
    card = _card(page, "dev.e2e.browser-created")
    expect(card).to_have_class(re.compile(r"\bblocked\b"))
    card.get_by_role("button", name="Build").click()
    expect(page.locator("#extension-studio-status")).to_contain_text("static validation passed", timeout=20_000)
    card.get_by_role("button", name="Validate").click()
    expect(page.locator("#extension-studio-status")).to_contain_text("Isolated validation", timeout=30_000)
    expect(card).to_have_class(re.compile(r"\bblocked\b"))


def test_failed_extension_update_rolls_back_then_corrected_update_activates(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "extensions")
    _open_manager(page)
    extension_id = "dev.dialeng.hello-workspace"
    _trust_and_enable(page, extension_id)
    source = (
        e2e_server["project"].root
        / ".dialeng" / "extensions" / "hello-workspace" / "extension.py"
    )
    original = source.read_text(encoding="utf-8")
    source.write_text(original + "\nthis is invalid python !!!\n", encoding="utf-8")

    page.evaluate("window.DialengExtensions.refresh()")
    card = _card(page, extension_id)
    expect(card).to_contain_text("Source changed")
    page.once("dialog", lambda dialog: dialog.accept())
    card.get_by_role("button", name="Review update").click()
    expect(page.locator("#extension-manager-status")).to_contain_text("SyntaxError", timeout=20_000)
    expect(page.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)

    source.write_text(original.replace("Hello Workspace", "Hello Workspace Updated", 1), encoding="utf-8")
    page.evaluate("window.DialengExtensions.refresh()")
    page.once("dialog", lambda dialog: dialog.accept())
    _card(page, extension_id).get_by_role("button", name="Review update").click()
    expect(_card(page, extension_id)).to_have_class(re.compile(r"\bactive\b"), timeout=20_000)
    expect(page.locator('[data-extension-toolbar="dev.dialeng.hello-workspace:hello"]')).to_have_count(1)
