"""Direct tests for the standalone browser notification module."""

import json
import subprocess
from pathlib import Path


MODULE = Path(__file__).parents[1] / "dialeng" / "static" / "js" / "notifications.js"


def test_notifications_escape_content_and_bound_history():
    program = f"""
const status = {{innerHTML: '', textContent: '', querySelector: () => null}};
global.window = globalThis;
global.document = {{getElementById: id => id === 'status' ? status : null}};
require({json.dumps(str(MODULE))});
for (let index = 0; index < 55; index++) {{
  DialengNotifications.show(`<unsafe ${{index}}>`, 'warning', 0);
}}
console.log(JSON.stringify({{html: status.innerHTML, history: DialengNotifications.history}}));
"""
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["html"] == '<div class="status warning">&lt;unsafe 54&gt;</div>'
    assert len(payload["history"]) == 50
    assert payload["history"][0]["message"] == "<unsafe 5>"
