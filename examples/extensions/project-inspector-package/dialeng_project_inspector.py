from pathlib import Path

from fasthtml.common import Div, H3, P


def _stats(root: Path) -> dict:
    files = 0
    notebooks = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file() or any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        files += 1
        notebooks += path.suffix == ".ipynb"
        total_bytes += min(path.stat().st_size, 10_000_000)
        if files >= 10_000:
            break
    return {"files": files, "notebooks": notebooks, "bounded_bytes": total_bytes}


def setup(api, ctx):
    def inspect_project() -> dict:
        """Return bounded file and notebook statistics for the current project."""
        return _stats(ctx.project_root)

    async def inspect_command(_command_context):
        stats = inspect_project()
        return {"message": f"{stats['files']} files, {stats['notebooks']} notebooks"}

    def panel(_surface_context):
        stats = inspect_project()
        return Div(
            H3("Project Inspector"),
            P(f"Files: {stats['files']}"),
            P(f"Notebooks: {stats['notebooks']}"),
            P(f"Bounded bytes: {stats['bounded_bytes']}"),
            cls="project-inspector-panel",
        )

    api.register_tool("inspect-project", inspect_project)
    api.register_command(
        "inspect-project", title="Inspect project",
        description="Count project files and notebooks without reading file contents.",
        handler=inspect_command,
    )
    api.register_toolbar_item(
        "inspect", command="inspect-project", title="Inspect", icon="scan-search", order=20,
    )
    api.register_panel("project", title="Project Inspector", renderer=panel, location="right")
