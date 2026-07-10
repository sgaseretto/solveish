"""Direct tests for the framework-independent browser revision tracker."""

import json
import subprocess
from pathlib import Path


MODULE = Path(__file__).parents[1] / "dialeng" / "static" / "js" / "notebook_client.js"


def _run_tracker(script: str) -> dict:
    program = f"""
global.window = globalThis;
require({json.dumps(str(MODULE))});
const tracker = new DialengRevisionTracker(['notebook', 'kernel', 'output']);
{script}
"""
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_kernel_snapshot_does_not_advance_notebook_revision():
    result = _run_tracker("""
tracker.recordSnapshot({revisions: {notebook: 1}}, ['notebook']);
tracker.recordSnapshot({revisions: {notebook: 2, kernel: 3}}, ['kernel']);
const accepted = tracker.accept({
  type: 'cell_delete', stream: 'notebook', revision: 2, event_id: 'delete'
});
console.log(JSON.stringify({revisions: tracker.revisions, accepted}));
""")

    assert result["revisions"]["kernel"] == 3
    assert result["revisions"]["notebook"] == 2
    assert result["accepted"]["accepted"] is True


def test_gap_and_duplicate_detection_are_explicit():
    result = _run_tracker("""
const gap = tracker.accept({
  type: 'cell_add', stream: 'notebook', revision: 3, event_id: 'gap'
});
const first = tracker.accept({
  type: 'cell_add', stream: 'notebook', revision: 1, event_id: 'one'
});
const duplicate = tracker.accept({
  type: 'cell_add', stream: 'notebook', revision: 1, event_id: 'one'
});
console.log(JSON.stringify({gap, first, duplicate, revisions: tracker.revisions}));
""")

    assert result["gap"]["gap"] == {"stream": "notebook", "from": 0, "to": 3}
    assert result["first"]["accepted"] is True
    assert result["duplicate"]["duplicate"] is True
    assert result["revisions"]["notebook"] == 1


def test_document_revisions_do_not_overwrite_event_stream_sequence():
    result = _run_tracker("""
const documents = new DialengDocumentRevisionTracker();
tracker.recordSnapshot({revisions: {notebook: 4}}, ['notebook']);
documents.recordNotebook({revision: 20, structure_revision: 9});
documents.recordEvent({notebook_revision: 21, structure_revision: 10});
const accepted = tracker.accept({
  type: 'cell_delete', stream: 'notebook', revision: 5, event_id: 'delete'
});
console.log(JSON.stringify({
  streamRevisions: tracker.revisions,
  documentRevisions: documents.revisions,
  accepted,
}));
""")

    assert result["streamRevisions"]["notebook"] == 5
    assert result["documentRevisions"] == {"notebook": 21, "structure": 10}
    assert result["accepted"]["accepted"] is True


def test_run_intent_is_consumed_once_and_defaults_to_advancing():
    program = f"""
global.window = globalThis;
require({json.dumps(str(MODULE))});
const tracker = new DialengRunIntentTracker();
const button = {{dataset: {{}}}};
const defaultAdvance = tracker.consume(button);
tracker.mark(button, {{advance: false}});
const stay = tracker.consume(button);
const resetToDefault = tracker.consume(button);
tracker.mark(button, {{advance: true}});
const advance = tracker.consume(button);
console.log(JSON.stringify({{defaultAdvance, stay, resetToDefault, advance, dataset: button.dataset}}));
"""
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)

    assert data == {
        "defaultAdvance": True,
        "stay": False,
        "resetToDefault": True,
        "advance": True,
        "dataset": {},
    }
