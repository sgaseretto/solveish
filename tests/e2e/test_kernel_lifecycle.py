"""Local-kernel and browser-connection lifecycle behavior."""

import pytest
from playwright.sync_api import expect

from .support.assertions import attach_local_kernel, open_notebook


pytestmark = pytest.mark.e2e


def test_kernel_attach_restart_terminate_and_reconnect(page, e2e_server):
    open_notebook(page, e2e_server["base_url"], "kernel", close_kernel_modal=False)
    attach_local_kernel(page)
    expect(page.locator("#kernel-dot")).to_have_attribute("title", "Kernel: idle")

    generation = page.evaluate(
        "async () => (await (await fetch('/dialeng/kernel/diagnostics')).json()).runtime.kernel_generation"
    )
    with page.expect_response(lambda response: response.url.endswith("/kernel/restart")):
        page.locator("button[title='Restart kernel']").click()
    page.wait_for_function(
        """async generation => {
            const response = await fetch('/dialeng/kernel/diagnostics');
            const data = await response.json();
            return data.runtime.kernel_generation > generation && data.kernel.is_alive;
        }""",
        arg=generation,
        timeout=20_000,
    )
    expect(page.locator("#kernel-dot")).to_have_attribute("title", "Kernel: idle", timeout=20_000)

    response = page.request.post(
        f"{e2e_server['base_url']}/dialeng/kernel/__e2e__/websocket/disconnect"
    )
    assert response.ok
    page.wait_for_function(
        """async () => {
            const response = await fetch('/dialeng/kernel/diagnostics');
            const data = await response.json();
            return data.runtime.websocket_clients === 1;
        }""",
        timeout=20_000,
    )

    response = page.request.post(
        f"{e2e_server['base_url']}/dialeng/kernel/__e2e__/kernel/terminate"
    )
    assert response.ok and response.json()["ok"]
    expect(page.locator("#kernel-dot")).to_have_attribute(
        "title", "Kernel: disconnected", timeout=10_000
    )

    with page.expect_response(lambda response: response.url.endswith("/kernel/restart")):
        page.locator("button[title='Restart kernel']").click()
    expect(page.locator("#kernel-dot")).to_have_attribute("title", "Kernel: idle", timeout=20_000)
