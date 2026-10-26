"""Server panel contracts: data fragments, safe paths, and workspace identity."""

import json

from bs4 import BeautifulSoup
from fasthtml.common import to_xml
from starlette.testclient import TestClient

import dialeng.app as appmod
from dialeng.ui.file_explorer import FileExplorerSidebar


def test_explorer_identity_is_workspace_scoped_and_controls_are_semantic(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (first / 'folder & space').mkdir()
    html = BeautifulSoup(to_xml(FileExplorerSidebar(first, first, 'demo')), 'html.parser')
    other = BeautifulSoup(to_xml(FileExplorerSidebar(second, second, 'demo')), 'html.parser')
    assert html.aside['data-workspace-id'] != other.aside['data-workspace-id']
    assert str(tmp_path) not in html.aside['data-workspace-id']
    folder = html.select_one('[data-explorer-path="folder & space"]')
    assert folder.name == 'button'
    assert folder['type'] == 'button'
    assert len(html.select('#file-list-content')) == 1


def test_panel_routes_return_only_data_fragments_and_recover_missing_folder(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    (root / 'panel.ipynb').write_text(json.dumps({
        'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
        'cells': [{'cell_type': 'markdown', 'source': ['# Panel heading'], 'metadata': {}}],
    }))
    previous = appmod.NOTEBOOKS_DIR
    try:
        appmod.set_root_dir(root)
        client = TestClient(appmod.app)
        headers = {'HX-Request': 'true'}
        outline = client.get('/dialeng/panel/outline', headers=headers)
        assert outline.status_code == 200
        dom = BeautifulSoup(outline.text, 'html.parser')
        assert len(dom.select('#outline-content')) == 1
        assert not dom.select('#outline-sidebar')
        assert 'Panel heading' in dom.get_text()
        for path in ('deleted-folder', '../outside', str(tmp_path)):
            files = client.get('/files', params={'path': path}, headers=headers)
            assert files.status_code == 200
            dom = BeautifulSoup(files.text, 'html.parser')
            assert len(dom.select('#file-list-content')) == 1
            assert dom.select_one('#current-explorer-path')['value'] == ''
            assert dom.select_one('a[href="/dialeng/?name=panel"]')
    finally:
        appmod.notebooks.pop('panel', None)
        appmod.set_root_dir(previous)
