"""Regression tests for prompt-response draft and backend synchronization."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_node(source: str) -> dict:
    result = subprocess.run(
        ["node", "-e", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_response_draft_is_not_replaced_by_remote_output_while_editing():
    result = _run_node("""
require('./dialeng/static/js/prompt_response_editor.js');
const editor = new DialengPromptResponseEditor();
editor.register('prompt-1', 'Generated answer');
const opened = editor.begin('prompt-1', '');
editor.input('prompt-1', 'Locally edited answer');
const remote = editor.receive('prompt-1', '');
console.log(JSON.stringify({opened, remote, session: editor.sessions.get('prompt-1')}));
""")

    assert result["opened"] == "Generated answer"
    assert result["remote"] == {
        "apply": False,
        "safeVersion": False,
        "value": "Locally edited answer",
    }
    assert result["session"]["pendingRemote"] == ""
    assert result["session"]["draft"] == "Locally edited answer"


def test_unchanged_remote_output_can_advance_version_during_edit():
    result = _run_node("""
require('./dialeng/static/js/prompt_response_editor.js');
const editor = new DialengPromptResponseEditor();
editor.register('prompt-1', 'Generated answer');
editor.begin('prompt-1', 'Generated answer');
const remote = editor.receive('prompt-1', 'Generated answer');
console.log(JSON.stringify(remote));
""")

    assert result == {
        "apply": False,
        "safeVersion": True,
        "value": "Generated answer",
    }


def test_intentional_empty_response_is_preserved_after_commit():
    result = _run_node("""
require('./dialeng/static/js/prompt_response_editor.js');
const editor = new DialengPromptResponseEditor();
editor.register('prompt-1', 'Generated answer');
editor.begin('prompt-1', 'Generated answer');
editor.input('prompt-1', '');
editor.finish('prompt-1', '');
editor.commit('prompt-1', '');
console.log(JSON.stringify(editor.sessions.get('prompt-1')));
""")

    assert result["authoritative"] == ""
    assert result["draft"] == ""
    assert result["dirty"] is False
    assert result["editing"] is False


def test_response_save_is_idempotent_and_accepts_its_own_websocket_commit():
    result = _run_node("""
require('./dialeng/static/js/prompt_response_editor.js');
const editor = new DialengPromptResponseEditor();
editor.register('prompt-1', 'Generated answer');
editor.begin('prompt-1', 'Generated answer');
editor.input('prompt-1', 'Edited answer');
editor.finish('prompt-1', 'Edited answer');
const firstSave = editor.startSave('prompt-1');
const duplicateSave = editor.startSave('prompt-1');
const websocketCommit = editor.receive('prompt-1', 'Edited answer');
console.log(JSON.stringify({
  firstSave,
  duplicateSave,
  websocketCommit,
  session: editor.sessions.get('prompt-1'),
}));
""")

    assert result["firstSave"] == "Edited answer"
    assert result["duplicateSave"] is None
    assert result["websocketCommit"] == {
        "apply": True,
        "safeVersion": True,
        "value": "Edited answer",
    }
    assert result["session"]["authoritative"] == "Edited answer"
    assert result["session"]["saving"] is False
