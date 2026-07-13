"""Public Extension API compatibility, schemas, and doctor contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dialeng.extensions.context import RegistrationScope, registration_scope
from dialeng.extensions.contracts import (
    API_POLICY,
    ExtensionCellType,
    ExtensionKernel,
    ExtensionProvider,
    ExtensionSetting,
    extension_contract_schema,
)
from dialeng.extensions.diagnostics import build_extension_doctor
from dialeng.extensions.discovery import discover_extensions
from dialeng.extensions.state import ExtensionStateStore
from dialeng.extensions.trust import ExtensionTrustStore
from dialeng.services.extension_project_service import ExtensionProjectService


def _manifest(extension_id: str, *, dialeng_api: str = ">=1,<2") -> str:
    return f'''schema_version = 1
id = "{extension_id}"
name = "Contract fixture"
version = "1.0.0"
description = "Independent API compatibility fixture."
entrypoint = "extension:setup"
dialeng_api = "{dialeng_api}"
scope = "project"

[permissions]
network = false

[contributes]
commands = true
'''


def test_generated_contract_schema_matches_packaged_catalog():
    path = Path(__file__).parents[1] / "dialeng" / "schemas" / "extension-api-v1.schema.json"

    assert json.loads(path.read_text(encoding="utf-8")) == extension_contract_schema()
    assert API_POLICY.compatible_range == ">=1,<2"


def test_public_declaration_types_capture_without_core_registry_imports(tmp_path):
    from dialeng.extensions import ExtensionAPI, ExtensionContext

    class ExampleCell:
        cell_type = "contract-cell"

    context = ExtensionContext("fixture.contract", tmp_path)
    api = ExtensionAPI(context)
    scope = RegistrationScope("fixture.contract", "1.0.0")
    with registration_scope(scope):
        api.register_setting(ExtensionSetting("enabled", "bool", True))
        api.register_kernel(ExtensionKernel("contract", "Contract", "box", lambda: None))
        api.register_provider(ExtensionProvider("contract", "Contract", lambda: None))
        api.register_cell_type(ExtensionCellType(ExampleCell, "contract-cell"))

    assert context.public_identity()["api_version"] == "1.0.0"
    assert {item.kind for item in scope.contributions} == {
        "settings.definition", "registry.kernel", "registry.provider", "registry.cell_type",
    }


def test_extension_check_rejects_incompatible_manifest_without_import(tmp_path):
    extension = tmp_path / "incompatible"
    extension.mkdir()
    marker = tmp_path / "imported.txt"
    (extension / "extension.toml").write_text(
        _manifest("fixture.incompatible", dialeng_api=">=2,<3"), encoding="utf-8",
    )
    (extension / "extension.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\ndef setup(api, ctx): pass\n",
        encoding="utf-8",
    )

    result = ExtensionProjectService(tmp_path).check(extension)

    assert result["ok"] is False
    assert result["imported"] is False
    assert result["diagnostics"][0]["code"] == "manifest_invalid"
    assert not marker.exists()


def test_extension_check_classifies_privileged_imports(tmp_path):
    extension = tmp_path / "privileged"
    extension.mkdir()
    (extension / "extension.toml").write_text(
        _manifest("fixture.privileged"), encoding="utf-8",
    )
    (extension / "extension.py").write_text(
        "from dialeng.core.registry import registry\n\ndef setup(api, ctx): pass\n",
        encoding="utf-8",
    )

    result = ExtensionProjectService(tmp_path).check(extension)

    assert result["ok"] is True
    assert result["runtime_mode"] == "privileged_in_process"
    assert result["warnings"][0]["code"] == "privileged_import"


def test_manifest_schema_rejects_unknown_and_coerced_declarations(tmp_path):
    extension = tmp_path / "malformed"
    extension.mkdir()
    manifest = _manifest("fixture.malformed").replace(
        "commands = true", "commands = \"yes\"\nunknown_surface = true",
    )
    (extension / "extension.toml").write_text(manifest, encoding="utf-8")
    (extension / "extension.py").write_text("def setup(api, ctx): pass\n", encoding="utf-8")

    result = ExtensionProjectService(tmp_path).check(extension)

    assert result["ok"] is False
    assert "Unknown contribution declaration" in result["diagnostics"][0]["message"]


def test_malformed_command_schema_is_rejected_during_registration(tmp_path):
    from dialeng.extensions import ExtensionAPI, ExtensionContext

    api = ExtensionAPI(ExtensionContext("fixture.schemas", tmp_path))
    with pytest.raises(ValueError, match="schema root must be an object"):
        api.register_command(
            "broken", title="Broken", description="Broken schema", handler=lambda context: None,
            args_schema={"type": "array"},
        )


def test_extension_doctor_reports_reproducibility_and_runtime_fields(tmp_path):
    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "contract"
    extension.mkdir(parents=True)
    (extension / "extension.toml").write_text(_manifest("fixture.contract"), encoding="utf-8")
    (extension / "extension.py").write_text("def setup(api, ctx): pass\n", encoding="utf-8")
    discovery = discover_extensions(
        project,
        user_root=tmp_path / "no-user",
        include_entry_points=False,
    )
    trust = ExtensionTrustStore(tmp_path / "trust.json")
    state = ExtensionStateStore(tmp_path / "state.json")
    source = discovery.active["fixture.contract"]
    trust.trust(source, "version")
    state.set_enabled(project, "fixture.contract", True)

    result = build_extension_doctor(project, discovery, trust, state)

    item = result["extensions"][0]
    assert result["ok"] is True
    assert item["origin"] == "project"
    assert item["trusted"] is True
    assert item["enabled"] is True
    assert item["registrations"] == []
    assert item["managed_tasks"] == 0
    assert item["runtime_state"] == "not_running"
