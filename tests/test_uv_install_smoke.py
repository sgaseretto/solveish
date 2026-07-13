import os
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("DIALENG_RUN_UV_INSTALL_SMOKE") != "1",
    reason="set DIALENG_RUN_UV_INSTALL_SMOKE=1 to run the isolated dependency install",
)
def test_uv_wheel_install_and_project_init_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    env = tmp_path / "venv"
    project = tmp_path / "project"

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("dialeng-*.whl"))
    subprocess.run(["uv", "venv", str(env)], check=True, capture_output=True, text=True)
    python = env / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    dialeng = env / "bin" / "dialeng"
    subprocess.run(
        [str(dialeng), "init", str(project), "--name", "smoke_pkg"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(dialeng), "init", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    schema = tmp_path / "extension-api-v1.schema.json"
    completed = subprocess.run(
        [str(dialeng), "extension", "schema", "--output", str(schema)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (project / "CRAFT.ipynb").exists()
    assert (project / "dialeng_config.json").exists()
    assert (project / ".dialeng" / "skills" / "README.md").exists()
    assert json.loads(schema.read_text(encoding="utf-8"))["title"] == "Dialeng Extension API 1 Contracts"
    assert json.loads(completed.stdout)["policy"]["version"] == "1.0.0"


@pytest.mark.skipif(
    os.environ.get("DIALENG_RUN_UV_INSTALL_SMOKE") != "1",
    reason="set DIALENG_RUN_UV_INSTALL_SMOKE=1 to run Git and project-local installs",
)
def test_uv_git_tool_and_project_local_install_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    requirement = f"dialeng @ git+file://{repo}@{revision}"
    tool_root = tmp_path / "tools"
    tool_bin = tmp_path / "bin"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tool_root),
        "UV_TOOL_BIN_DIR": str(tool_bin),
    }
    subprocess.run(
        ["uv", "tool", "install", "--force", requirement],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    tool_schema = tmp_path / "tool-schema.json"
    subprocess.run(
        [str(tool_bin / "dialeng"), "extension", "schema", "--output", str(tool_schema)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    project = tmp_path / "local-project"
    subprocess.run(["uv", "init", "--bare", str(project)], check=True, capture_output=True, text=True)
    subprocess.run(
        ["uv", "add", requirement],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    project_schema = tmp_path / "project-schema.json"
    subprocess.run(
        ["uv", "run", "dialeng", "extension", "schema", "--output", str(project_schema)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(tool_schema.read_text(encoding="utf-8"))["title"].startswith("Dialeng")
    assert json.loads(project_schema.read_text(encoding="utf-8"))["title"].startswith("Dialeng")
