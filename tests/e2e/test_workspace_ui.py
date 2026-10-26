"""Workspace view state survives partial updates without controlling document state."""

from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import expect

from .support.assertions import open_notebook
from .support.project_factory import CellSpec


pytestmark = pytest.mark.e2e


def files_toggle(page):
    return page.locator('[data-workspace-toggle="files"]')


def open_outline(page):
    page.locator('#outline-toggle-btn').click()
    expect(page.locator('#outline-toggle-btn')).to_have_attribute('aria-expanded', 'true')
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')


def navigate_folder(page, name):
    page.locator('#file-list-content').get_by_role('button', name=name, exact=True).click()
    expect(page.locator('#current-explorer-path')).to_have_value(name)
    expect(page.locator('#file-list-content')).to_have_count(1)


def test_panel_folder_and_outline_state_survive_reload_and_document_navigation(page, e2e_server):
    root = e2e_server['project'].root
    (root / 'nested').mkdir()
    (root / 'nested' / 'notes.md').write_text('# A plain file\n')
    open_notebook(page, e2e_server['base_url'], 'deletion')
    navigate_folder(page, 'nested')
    open_outline(page)
    section = page.locator('[data-outline-section="variables"]')
    section.locator('summary').click()
    expect(section).not_to_have_attribute('open', '')
    page.evaluate('refreshOutline()')
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')
    expect(section).not_to_have_attribute('open', '')

    open_notebook(page, e2e_server['base_url'], 'deletion')
    expect(page.locator('#outline-toggle-btn')).to_have_attribute('aria-expanded', 'true')
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')
    expect(section).not_to_have_attribute('open', '')
    files_toggle(page).click()
    expect(page.locator('#current-explorer-path')).to_have_value('nested')
    page.get_by_role('link', name='notes.md', exact=True).click()
    expect(page.locator('#file-editor-container')).to_be_visible()
    expect(page.locator('#current-explorer-path')).to_have_value('nested')
    files_toggle(page).click()
    open_notebook(page, e2e_server['base_url'], 'deletion')
    expect(files_toggle(page)).to_have_attribute('aria-expanded', 'false')
    expect(page.locator('#outline-toggle-btn')).to_have_attribute('aria-expanded', 'false')


def test_closed_files_revalidate_and_missing_folder_recovers(page, e2e_server):
    root = e2e_server['project'].root
    folder = root / 'temporary'
    folder.mkdir()
    open_notebook(page, e2e_server['base_url'], 'deletion')
    navigate_folder(page, 'temporary')
    files_toggle(page).click()
    (folder / 'new.md').write_text('# External change\n')
    files_toggle(page).click()
    expect(page.get_by_role('link', name='new.md', exact=True)).to_be_visible()
    files_toggle(page).click()
    (folder / 'new.md').unlink()
    folder.rmdir()
    files_toggle(page).click()
    expect(page.locator('#current-explorer-path')).to_have_value('')
    expect(page.get_by_role('link', name='deletion', exact=True)).to_be_visible()


def test_late_folder_response_cannot_overwrite_newer_navigation(page, e2e_server):
    root = e2e_server['project'].root
    for name in ('first', 'second'):
        (root / name).mkdir()
        (root / name / f'{name}.md').write_text(name)
    # Exercise the generation guard even if abort cannot stop delivery.
    page.add_init_script('''
        const Controller = window.AbortController;
        window.AbortController = class extends Controller { abort() {} };
    ''')
    open_notebook(page, e2e_server['base_url'], 'deletion')
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('aria-busy', 'false')
    held = []

    def hold_first(route):
        if parse_qs(urlparse(route.request.url).query).get('path') == ['first']:
            held.append((route, route.fetch()))
        else:
            route.continue_()

    page.route('**/files?*', hold_first)
    page.get_by_role('button', name='first', exact=True).click()
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('aria-busy', 'true')
    navigate_folder(page, 'second')
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    # Flush a task after delivery, then assert canonical view and unique root.
    page.evaluate('() => new Promise(resolve => setTimeout(resolve, 50))')
    expect(page.locator('#current-explorer-path')).to_have_value('second')
    expect(page.get_by_role('link', name='second.md', exact=True)).to_be_visible()
    expect(page.locator('#file-list-content')).to_have_count(1)


def test_outline_refresh_preserves_live_choices_and_cannot_reopen_closed_panel(page, e2e_server):
    page.add_init_script('''
        const Controller = window.AbortController;
        window.AbortController = class extends Controller { abort() {} };
    ''')
    open_notebook(page, e2e_server['base_url'], 'deletion')
    open_outline(page)
    page.evaluate("document.getElementById('outline-sidebar').auditShell = true")
    held = []
    page.route('**/dialeng/deletion/outline', lambda route: held.append((route, route.fetch())))
    page.evaluate('refreshOutline()')
    section = page.locator('[data-outline-section="variables"]')
    section.locator('summary').click()
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')
    expect(section).not_to_have_attribute('open', '')
    expect(section.locator('summary')).to_be_focused()
    assert page.evaluate("document.getElementById('outline-sidebar').auditShell") is True

    page.evaluate('refreshOutline()')
    page.get_by_role('button', name='Close outline').click()
    files_toggle(page).click()
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    page.evaluate('() => new Promise(resolve => setTimeout(resolve, 50))')
    expect(page.locator('#outline-toggle-btn')).to_have_attribute('aria-expanded', 'false')
    expect(page.locator('#outline-sidebar')).to_have_attribute('inert', '')
    expect(files_toggle(page)).to_have_attribute('aria-expanded', 'true')


def test_view_state_is_tab_local_and_bad_storage_does_not_break_controls(browser_context, e2e_server):
    first = browser_context.new_page()
    second = browser_context.new_page()
    open_notebook(first, e2e_server['base_url'], 'deletion')
    open_outline(first)
    open_notebook(second, e2e_server['base_url'], 'deletion')
    expect(files_toggle(second)).to_have_attribute('aria-expanded', 'true')
    # Corrupt after pagehide saves the old page, before the new controller reads.
    first.add_init_script('''
        for (const key of Object.keys(sessionStorage)) {
            if (key.startsWith('dialeng.workspace-ui.')) sessionStorage.setItem(key, 'not-json');
        }
    ''')
    open_notebook(first, e2e_server['base_url'], 'deletion')
    expect(files_toggle(first)).to_have_attribute('aria-expanded', 'true')
    first.evaluate('''() => {
        const setItem = Storage.prototype.setItem;
        Storage.prototype.setItem = function(key, value) {
            if (key.startsWith('dialeng.workspace-ui.')) throw new Error('Storage full');
            return setItem.call(this, key, value);
        };
    }''')
    open_outline(first)

    second.add_init_script('''
        const getItem = Storage.prototype.getItem;
        Storage.prototype.getItem = function(key) {
            if (key.startsWith('dialeng.workspace-ui.')) throw new Error('Storage disabled');
            return getItem.call(this, key);
        };
    ''')
    open_notebook(second, e2e_server['base_url'], 'deletion')
    open_outline(second)


def test_keyboard_focus_settings_and_mobile_drawer(page, e2e_server):
    (e2e_server['project'].root / 'keyboard-folder').mkdir()
    open_notebook(page, e2e_server['base_url'], 'deletion')
    page.evaluate("setFocusedCell('del00001')")
    folder = page.get_by_role('button', name='keyboard-folder', exact=True)
    folder.focus()
    folder.press('Enter')
    expect(page.locator('#current-explorer-path')).to_have_value('keyboard-folder')
    page.keyboard.press('Escape')
    expect(files_toggle(page)).to_be_focused()
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('inert', '')
    page.locator('#settings-btn').click()
    expect(page.locator('#settings-search')).to_be_focused()
    page.keyboard.press('Escape')
    expect(page.locator('#settings-btn')).to_be_focused()
    expect(page.locator('#settings-sidebar')).to_have_attribute('inert', '')

    page.set_viewport_size({'width': 390, 'height': 844})
    content_width = page.locator('.main-layout > .container').bounding_box()['width']
    files_toggle(page).click()
    expect(page.locator('.workspace-sidebar-backdrop')).to_be_visible()
    expect(page.locator('.main-layout > .container')).to_have_attribute('inert', '')
    assert page.locator('#file-explorer-sidebar').bounding_box()['width'] <= 358
    assert page.locator('.main-layout > .container').bounding_box()['width'] == content_width
    page.keyboard.press('Escape')
    expect(files_toggle(page)).to_be_focused()
    expect(page.locator('.workspace-sidebar-backdrop')).not_to_be_visible()
    expect(page.locator('.main-layout > .container')).not_to_have_attribute('inert', '')
    open_outline(page)
    page.get_by_role('link', name='Delete me', exact=True).click()
    expect(page.locator('#outline-toggle-btn')).to_have_attribute('aria-expanded', 'false')
    expect(page.locator('.main-layout > .container')).not_to_have_attribute('inert', '')


def test_refresh_failure_keeps_previous_view_and_retry_recovers(page, e2e_server):
    open_notebook(page, e2e_server['base_url'], 'deletion')
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('aria-busy', 'false')
    page.evaluate('''() => {
        const fetch = window.fetch;
        let fail = true;
        window.fetch = (url, options) => {
            if (String(url).startsWith('/files?') && fail) {
                fail = false;
                return Promise.reject(new Error('Offline'));
            }
            return fetch(url, options);
        };
    }''')
    page.get_by_role('button', name='Refresh file explorer').click()
    expect(page.locator('#files-panel-status')).to_contain_text('Could not update')
    expect(page.get_by_role('link', name='deletion', exact=True)).to_be_visible()
    page.locator('#files-panel-status').get_by_role('button', name='Retry').click()
    expect(page.locator('#files-panel-status')).not_to_be_visible()
    expect(page.locator('#file-list-content')).to_have_count(1)


def test_panel_scroll_is_restored_after_reload_and_refresh(page, e2e_server):
    project = e2e_server['project']
    folder = project.root / 'many-files'
    folder.mkdir()
    for index in range(80):
        (folder / f'file-{index:03}.md').write_text('Example')
    project.write_notebook('outline-scroll', [
        CellSpec('note', '\n'.join(f'## Heading {index}' for index in range(80)), 'scroll01'),
    ])
    open_notebook(page, e2e_server['base_url'], 'outline-scroll')
    navigate_folder(page, 'many-files')
    page.locator('.file-explorer-content').evaluate('element => { element.scrollTop = 500; }')
    open_notebook(page, e2e_server['base_url'], 'outline-scroll')
    expect(page.locator('#current-explorer-path')).to_have_value('many-files')
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('aria-busy', 'false')
    assert page.locator('.file-explorer-content').evaluate('element => element.scrollTop') == 500
    open_outline(page)
    page.locator('#outline-content').evaluate('element => { element.scrollTop = 400; }')
    page.evaluate('refreshOutline()')
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')
    assert page.locator('#outline-content').evaluate('element => element.scrollTop') == 400
    open_notebook(page, e2e_server['base_url'], 'outline-scroll')
    expect(page.locator('#outline-sidebar')).to_have_attribute('aria-busy', 'false')
    assert page.locator('#outline-content').evaluate('element => element.scrollTop') == 400


def test_folder_creation_completion_cannot_replace_newer_directory(page, e2e_server):
    root = e2e_server['project'].root
    (root / 'destination').mkdir()
    (root / 'destination' / 'keep.md').write_text('Stay in this folder')
    open_notebook(page, e2e_server['base_url'], 'deletion')
    expect(page.locator('#file-explorer-sidebar')).to_have_attribute('aria-busy', 'false')
    held = []
    page.route('**/files/new-folder', lambda route: held.append((route, route.fetch())))
    page.get_by_role('button', name='New notebook or folder').click()
    page.locator('#new-item-name').fill('created')
    page.locator('#new-item-type-folder').click()
    page.locator('#new-item-modal').get_by_role('button', name='Create', exact=True).click()
    navigate_folder(page, 'destination')
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    page.evaluate('() => new Promise(resolve => setTimeout(resolve, 50))')
    assert (root / 'created').is_dir()
    expect(page.locator('#current-explorer-path')).to_have_value('destination')
    expect(page.get_by_role('link', name='keep.md', exact=True)).to_be_visible()
    expect(page.locator('#file-list-content')).to_have_count(1)


def test_settings_surface_refresh_keeps_shell_visibility_and_inflight_edits(page, e2e_server):
    open_notebook(page, e2e_server['base_url'], 'deletion')
    page.locator('#settings-btn').click()
    expect(page.locator('#settings-status')).to_contain_text('Settings are current')
    page.wait_for_function('!extensionClientState.settingsPendingSignature')
    page.evaluate("document.getElementById('settings-sidebar').auditShell = true")
    held = []
    page.route('**/settings/view', lambda route: held.append((route, route.fetch())))

    def refresh_surface(name):
        page.evaluate('''name => {
            void refreshExtensionSettingsSurface({extensions: [{id: 'panel-test',
                contributions: [{kind: 'settings.definition', name}]}]});
        }''', name)

    refresh_surface('first')
    page.locator('.settings-close-btn').click()
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    page.wait_for_function('!extensionClientState.settingsPendingSignature')
    expect(page.locator('#settings-sidebar')).to_have_attribute('inert', '')
    expect(page.locator('#settings-btn')).to_have_attribute('aria-expanded', 'false')
    assert page.evaluate("document.getElementById('settings-sidebar').auditShell") is True

    page.locator('#settings-btn').click()
    expect(page.locator('#settings-status')).to_contain_text('Settings are current')
    page.locator('.settings-group').filter(has_text='Dialeng Display Settings').locator('summary').click()
    refresh_surface('second')
    font = page.locator('input[name="display.font_size"]')
    font.fill('17')
    assert held
    route, response = held.pop()
    route.fulfill(response=response)
    page.wait_for_function('!extensionClientState.settingsPendingSignature')
    expect(font).to_have_value('17')
    expect(page.locator('#settings-change-summary')).to_contain_text('1 unsaved change')
    expect(page.locator('#settings-sidebar')).not_to_have_attribute('inert', '')
    assert page.evaluate("document.getElementById('settings-sidebar').auditShell") is True
