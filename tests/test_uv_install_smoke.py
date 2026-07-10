import os
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

    assert (project / "CRAFT.ipynb").exists()
    assert (project / "dialeng_config.json").exists()
    assert (project / ".dialeng" / "skills" / "README.md").exists()
