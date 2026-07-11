import json

from dialeng.services.project_init_service import initialize_project


def test_project_init_is_idempotent_and_preserves_custom_craft_cells(tmp_path):
    project = tmp_path / "sample-project"
    first = initialize_project(project, name="sample_pkg")
    assert first["status"] == "ok"
    assert (project / "dialeng_config.json").exists()
    assert (project / ".dialeng" / "skills" / "README.md").exists()
    assert ".dialeng/recovery/" in (project / ".gitignore").read_text(encoding="utf-8")

    craft_path = project / "CRAFT.ipynb"
    notebook = json.loads(craft_path.read_text(encoding="utf-8"))
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["Custom content"]})
    craft_path.write_text(json.dumps(notebook), encoding="utf-8")
    config_before = (project / "dialeng_config.json").read_text(encoding="utf-8")

    second = initialize_project(project)
    rerun = json.loads(craft_path.read_text(encoding="utf-8"))
    assert second["config_created"] is False
    assert sum("Custom content" in "".join(cell.get("source", [])) for cell in rerun["cells"]) == 1
    assert (project / "dialeng_config.json").read_text(encoding="utf-8") == config_before
    assert (project / ".gitignore").read_text(encoding="utf-8").count(".dialeng/recovery/") == 1


def test_cli_init_parsing():
    from dialeng.cli import parse_args

    args = parse_args(["init", "project", "--name", "pkg"])
    assert args.subcommand == "init"
    assert args.path == "project"
    assert args.project_name == "pkg"
