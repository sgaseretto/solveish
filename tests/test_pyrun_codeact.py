"""Tests for CodeAct-style prompt tool access inside pyrun."""

import asyncio

from dialeng.services.builtin_tools import DialengRunPython, pyrun
from dialeng.services.tool_registry import ToolRegistry, get_tool_registry


class _FakeKernel:
    def __init__(self):
        self.calls = []

    async def execute_tool(self, name: str, kwargs: dict, timeout: float = 60.0) -> dict:
        self.calls.append((name, dict(kwargs)))
        if name == "obj.method":
            return {"status": "success", "result": {"type": "text", "content": '{"value": 7}'}}
        if name == "tool_a":
            return {"status": "success", "result": {"type": "text", "content": "42"}}
        return {"status": "success", "result": {"type": "text", "content": repr(kwargs)}}


def test_dialeng_pyrun_allows_prompt_scoped_sync_tools():
    tool = DialengRunPython(ok_dests=["."])
    token = tool.push_tool_context(
        tool_globals={"hello": lambda: "world"},
        allowed_names={"hello"},
    )
    try:
        result = asyncio.run(tool("hello()"))
        assert result == "world"
    finally:
        tool.reset_tool_context(token)


def test_registry_pyrun_context_exposes_dynamic_tools_with_await():
    registry = ToolRegistry()
    kernel = _FakeKernel()
    token = registry.push_pyrun_context(
        notebook_id="demo",
        kernel=kernel,
        func_names=["tool_a"],
        include_builtins=False,
    )
    try:
        result = asyncio.run(pyrun("await tool_a()"))
        assert result == 42
        assert kernel.calls == [("tool_a", {})]
    finally:
        registry.reset_pyrun_context(token)


def test_registry_pyrun_context_exposes_dotted_tools_via_namespace():
    registry = ToolRegistry()
    kernel = _FakeKernel()
    token = registry.push_pyrun_context(
        notebook_id="demo",
        kernel=kernel,
        func_names=["obj.method"],
        include_builtins=False,
    )
    try:
        result = asyncio.run(pyrun("result_ = await obj.method(path='demo.txt')\nresult_"))
        assert result == {"value": 7}
        assert kernel.calls == [("obj.method", {"path": "demo.txt"})]
    finally:
        registry.reset_pyrun_context(token)


def test_registry_pyrun_context_exposes_builtin_tools(tmp_path):
    registry = get_tool_registry()
    kernel = _FakeKernel()
    target = tmp_path / "sample.txt"
    token = registry.push_pyrun_context(
        notebook_id="demo",
        kernel=kernel,
        func_names=[],
        include_builtins=True,
    )
    try:
        result = asyncio.run(
            pyrun(f"create({str(target)!r}, 'hello', overwrite=True)")
        )
        assert "Created file" in result
        assert target.read_text(encoding="utf-8") == "hello"
    finally:
        registry.reset_pyrun_context(token)
