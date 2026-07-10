"""Independent checks for extracted browser structure and batching primitives."""

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


def test_structure_controller_rejects_duplicates_and_missing_units():
    result = _run_node("""
require('./dialeng/static/js/structure_controller.js');
const units = new Map([
  ['a', {cellEl: {}, addRowEl: {}}],
  ['b', {cellEl: {}, addRowEl: {}}],
]);
const resolve = id => units.get(id);
console.log(JSON.stringify({
  empty: dialengStructure.plan([], resolve),
  valid: dialengStructure.plan(['b', 'a'], resolve),
  duplicate: dialengStructure.plan(['a', 'a'], resolve),
  missing: dialengStructure.plan(['a', 'c'], resolve),
}));
""")

    assert result["empty"]["valid"] is True
    assert result["empty"]["units"] == []
    assert result["valid"]["valid"] is True
    assert len(result["valid"]["units"]) == 2
    assert result["duplicate"]["reason"] == "duplicate_id"
    assert result["missing"]["reason"] == "missing_unit"


def test_frame_batcher_coalesces_scheduled_chunks_and_supports_manual_flush():
    result = _run_node("""
require('./dialeng/static/js/execution_ui.js');
const callbacks = [];
const cancelled = [];
const flushed = [];
const batcher = new DialengFrameBatcher({
  onFlush: (key, chunk, options) => flushed.push({key, chunk, options}),
  maxBytes: 8,
  schedule: callback => { callbacks.push(callback); return callbacks.length - 1; },
  cancel: handle => cancelled.push(handle),
});
batcher.append('one', 'ab');
batcher.append('one', 'cd');
callbacks[0]();
batcher.append('two', '1234');
batcher.append('two', '5678');
batcher.append('three', 'x');
batcher.flush('three', {render: false});
console.log(JSON.stringify({flushed, cancelled}));
""")

    assert [item["chunk"] for item in result["flushed"]] == ["abcd", "12345678", "x"]
    assert result["flushed"][2]["options"] == {"render": False}
    assert result["cancelled"] == [0, 1, 2]


def test_cell_run_controls_unlock_after_every_terminal_state():
    result = _run_node("""
require('./dialeng/static/js/execution_ui.js');
const classes = new Set();
const runButton = {disabled: false, style: {display: ''}};
const cancelButton = {style: {display: 'none'}};
const cell = {
  classList: {
    toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
  },
  querySelector: selector => selector === '.btn-run' ? runButton : cancelButton,
};
DialengSetCellRunControlState(cell, true);
const running = {
  classes: [...classes],
  runDisabled: runButton.disabled,
  runDisplay: runButton.style.display,
  cancelDisplay: cancelButton.style.display,
};
DialengSetCellRunControlState(cell, false);
const finished = {
  classes: [...classes],
  runDisabled: runButton.disabled,
  runDisplay: runButton.style.display,
  cancelDisplay: cancelButton.style.display,
};
console.log(JSON.stringify({running, finished}));
""")

    assert result["running"] == {
        "classes": ["streaming"],
        "runDisabled": True,
        "runDisplay": "none",
        "cancelDisplay": "",
    }
    assert result["finished"] == {
        "classes": [],
        "runDisabled": False,
        "runDisplay": "",
        "cancelDisplay": "none",
    }
