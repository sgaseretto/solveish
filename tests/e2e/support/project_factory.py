"""Disposable project fixtures for browser tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from dialeng.document.cell import Cell, CellType
from dialeng.document.notebook import Notebook


@dataclass(frozen=True)
class CellSpec:
    cell_type: str
    source: str = ""
    cell_id: str | None = None
    output: str = ""


class E2EProject:
    """Build notebooks and configuration under an isolated project root."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_config(self) -> Path:
        config = {
            "models": {
                "available": [
                    {
                        "id": model,
                        "name": model,
                        "providers": ["e2e_fake"],
                        "capabilities": ["streaming", "thinking", "usage"],
                    }
                    for model in ("e2e-fast", "e2e-slow", "e2e-error")
                ],
                "defaults": {
                    "e2e_fake": "e2e-fast",
                },
            },
            "modes": {"default": "mock"},
            "llm": {"openai_codex_enabled": False},
            "skills": {"enabled": True, "creator_enabled": True},
            "tool_settings": {
                "max_steps": 3,
                "require_confirmation": False,
                "builtin_tools_enabled": False,
            },
            "persistence": {"autosave": {"enabled": False}},
            "colab": {"enabled": False},
        }
        path = self.root / "dialeng_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path

    def write_codex_config(self) -> Path:
        """Write an isolated configuration for opt-in real Codex tests."""
        config = {
            "models": {
                "available": [
                    {
                        "id": model,
                        "name": model,
                        "providers": ["openai_codex"],
                        "capabilities": ["tools", "images", "thinking", "skills", "usage"],
                    }
                    for model in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex")
                ],
                "defaults": {"openai_codex": "gpt-5.4"},
                "openai_codex_map": {
                    "gpt-5.4": "gpt-5.4",
                    "gpt-5.4-mini": "gpt-5.4-mini",
                    "gpt-5.3-codex": "gpt-5.3-codex",
                },
            },
            "modes": {"default": "concise"},
            "llm": {"openai_codex_enabled": True, "openai_codex_transport": "auto"},
            "skills": {"enabled": True, "creator_enabled": True},
            "tool_settings": {
                "max_steps": 5,
                "require_confirmation": False,
                "builtin_tools_enabled": True,
            },
            "persistence": {"autosave": {"enabled": False}},
            "colab": {"enabled": False},
        }
        path = self.root / "dialeng_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path

    def create_codex_scenario(self) -> None:
        """Create isolated real-provider notebooks with one call per scenario."""
        self.write_codex_config()
        self.write_notebook(
            "codex-context",
            [
                CellSpec("note", "My name is Jane Doe.", "cdxnote1"),
                CellSpec("prompt", "What is my name? Reply with only the current name.", "cdxprm01", "Joe Doe"),
            ],
            mode="concise", provider="openai_codex", model="gpt-5.4",
        )
        self.write_notebook(
            "codex-image",
            [
                CellSpec(
                    "code",
                    "from PIL import Image, ImageDraw\n"
                    "img = Image.new('RGB', (160, 120), 'red')\n"
                    "ImageDraw.Draw(img).rectangle((50, 30, 110, 90), fill='blue')\n"
                    "display(img)",
                    "cdximg01",
                ),
                CellSpec(
                    "prompt",
                    "Describe the two dominant colors in the image immediately above.",
                    "cdxprm02",
                ),
            ],
            mode="concise", provider="openai_codex", model="gpt-5.4",
        )
        self.write_notebook(
            "codex-tool",
            [
                CellSpec(
                    "code",
                    "def codex_e2e_tool(value: str) -> str:\n"
                    "    \"Return a deterministic marker for tool validation.\"\n"
                    "    return f'DIALENG_TOOL_OK:{value}'\n"
                    "print('DIALENG_TOOL_READY')",
                    "cdxtol01",
                ),
                CellSpec(
                    "prompt",
                    "Call &`codex_e2e_tool` with value notebook and report its result exactly.",
                    "cdxprm03",
                ),
            ],
            mode="concise", provider="openai_codex", model="gpt-5.4",
        )
        self.write_notebook(
            "codex-skill",
            [
                CellSpec(
                    "prompt",
                    "Use the codex-e2e skill and report its unique validation marker exactly.",
                    "cdxprm04",
                ),
            ],
            mode="concise", provider="openai_codex", model="gpt-5.4",
        )
        self.write_notebook(
            "codex-persistence",
            [CellSpec("prompt", "Reply with exactly DIALENG_PERSIST_OK.", "cdxprm05")],
            mode="concise", provider="openai_codex", model="gpt-5.4",
        )
        skill_dir = self.root / ".dialeng" / "skills" / "codex-e2e"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: codex-e2e\n"
            "description: Use when a test asks for the Codex E2E validation marker.\n"
            "---\n\n"
            "The unique validation marker is DIALENG_SKILL_OK.\n",
            encoding="utf-8",
        )

    def write_notebook(
        self,
        name: str,
        cells: list[CellSpec],
        *,
        mode: str = "mock",
        provider: str = "e2e_fake",
        model: str = "e2e-fast",
    ) -> Path:
        notebook_cells = []
        for index, spec in enumerate(cells):
            cell = Cell(
                id=spec.cell_id or f"cell{index:04d}",
                cell_type=CellType(spec.cell_type),
                source=spec.source,
            )
            if spec.output:
                cell.output = spec.output
            notebook_cells.append(cell)
        notebook = Notebook(
            id=name.replace("/", "~"),
            title=Path(name).name,
            dialog_mode=mode,
            llm_provider=provider,
            model=model,
            cells=notebook_cells,
        )
        path = self.root / f"{name}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook.save(path, atomic=False)
        return path

    def create_scenarios(self) -> None:
        self.write_config()
        self.write_notebook(
            "mutations",
            [
                CellSpec("code", "first = 1", "mut00001"),
                CellSpec("note", "## Middle", "mut00002"),
                CellSpec("code", "third = 3", "mut00003"),
            ],
        )
        self.write_notebook(
            "extensions",
            [
                CellSpec("code", "print('extension baseline')", "ext00001"),
                CellSpec("note", "## Extension playground", "ext00002"),
            ],
        )
        fixture_root = Path(__file__).parents[3] / "examples" / "extensions"
        destination = self.root / ".dialeng" / "extensions"
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "hello-workspace", "execution-guard", "diagram-cell",
            "project-inspector-package",
        ):
            shutil.copytree(
                fixture_root / name,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "dist", "build"),
            )
        stress = destination / "command-stress"
        stress.mkdir()
        (stress / "extension.toml").write_text(
            '''schema_version = 1
id = "dev.e2e.command-stress"
name = "Command Stress"
version = "0.1.0"
description = "One thousand command palette fixtures."
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "project"
[permissions]
network = false
[contributes]
commands = true
''',
            encoding="utf-8",
        )
        (stress / "extension.py").write_text(
            '''def setup(api, ctx):
    for index in range(1000):
        title = "<img src=x onerror=alert(1)> Unsafe title" if index == 0 else f"Synthetic command {index:04d}"
        api.register_command(
            f"synthetic-{index:04d}",
            title=title,
            description=f"Palette scale fixture {index}.",
            category="Stress",
            keywords=("synthetic", "scale"),
            default_shortcut="Meta+S" if index == 0 else "",
            handler=lambda context, value=index: {"result": value},
        )
''',
            encoding="utf-8",
        )
        self.write_notebook(
            "providers",
            [CellSpec("prompt", "Describe deterministic testing.", "pro00001")],
        )
        self.write_notebook(
            "long",
            [
                CellSpec("note", f"## Section {index}\n\n{'content ' * 18}", f"lng{index:05d}")
                for index in range(24)
            ]
            + [CellSpec("prompt", "Finish the long notebook.", "lng99999")],
        )
        self.write_notebook(
            "editing",
            [CellSpec("prompt", "My name is Joe.", "edt00001")],
        )
        self.write_notebook(
            "collaboration",
            [CellSpec("prompt", "Shared source", "col00001")],
        )
        self.write_notebook(
            "kernel",
            [CellSpec("code", "print('kernel ready')", "ker00001")],
        )
        self.write_notebook(
            "deletion",
            [CellSpec("note", "# Delete me", "del00001")],
        )
        self.write_notebook(
            "rich",
            [
                CellSpec(
                    "code",
                    """from IPython.display import HTML, JSON, display
display(HTML("<strong id='e2e-rich-html'>Rich HTML</strong>"))
display(JSON({'answer': 42, 'kind': 'json'}))
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1])
plt.title('E2E plot')
plt.show()""",
                    "rch00001",
                )
            ],
        )
