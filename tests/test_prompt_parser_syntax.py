"""Tests for expression injection and richer tool syntax parsing."""

import asyncio

from dialeng.services.prompt_parser import (
    get_function_schemas,
    parse_prompt,
    strip_special_syntax,
    substitute_variables,
    tool_target_to_api_name,
)
from dialeng.services.tool_registry import ToolRegistry


class _FakeKernel:
    async def evaluate_expression(self, expression: str, timeout: float = 5.0):
        values = {
            "user_name": {"exists": True, "repr": "'Joe Doe'", "var_type": "str"},
            "len(items)": {"exists": True, "repr": "3", "var_type": "int"},
            "config.theme.name": {"exists": True, "repr": "'solarized'", "var_type": "str"},
        }
        return {"expression": expression, "name": expression, **values.get(expression, {"exists": False, "error": "missing"})}

    async def introspect_function(self, name: str, timeout: float = 5.0):
        return {
            "name": name,
            "exists": True,
            "is_callable": True,
            "docstring": f"Call {name}",
            "parameters": {"path": {"type": "str", "description": "path"}},
            "return_type": "str",
        }


def test_parse_prompt_supports_expressions_and_rich_tool_refs():
    expressions, funcs = parse_prompt(
        "Use $`len(items)` and $`config.theme.name` with &`obj.method` and &`[tool_a, tool_b, obj.method]`."
    )

    assert expressions == ["len(items)", "config.theme.name"]
    assert funcs == ["obj.method", "tool_a", "tool_b"]


def test_strip_special_syntax_handles_lists_and_expressions():
    text = "Run &`[tool_a, obj.method]` after checking $`len(items)`."
    assert strip_special_syntax(text) == "Run tool_a, obj.method after checking len(items)."


def test_tool_target_to_api_name_preserves_simple_names_and_sanitizes_dotted_names():
    assert tool_target_to_api_name("tool_a") == "tool_a"
    sanitized = tool_target_to_api_name("obj.method")
    assert sanitized.startswith("obj-method--")
    assert sanitized != "obj.method"


def test_substitute_variables_evaluates_expressions():
    kernel = _FakeKernel()
    result, info = asyncio.run(
        substitute_variables(
            "Hello $`user_name`, theme=$`config.theme.name`, count=$`len(items)`",
            kernel,
            "demo",
        )
    )

    assert result == "Hello 'Joe Doe', theme='solarized', count=3"
    assert info["len(items)"]["repr"] == "3"


def test_registry_tracks_safe_tool_aliases_for_dotted_names():
    registry = ToolRegistry()
    kernel = _FakeKernel()

    tools = asyncio.run(registry.get_tools_for_prompt(["obj.method"], kernel, "demo", include_builtins=False))

    assert len(tools) == 1
    safe_name = tools[0]["name"]
    assert safe_name.startswith("obj-method--")
    assert registry.resolve_dynamic_tool_name("demo", safe_name) == "obj.method"
    assert registry.resolve_tool_display_name("demo", safe_name) == "obj.method"


def test_get_function_schemas_expands_rich_targets_with_safe_api_names():
    kernel = _FakeKernel()

    schemas, info = asyncio.run(get_function_schemas(kernel, "demo", ["obj.method", "tool_a"]))

    assert info["obj.method"]["exists"] is True
    assert schemas[0]["dialeng_target_name"] == "obj.method"
    assert schemas[0]["dialeng_display_name"] == "obj.method"
    assert schemas[0]["name"].startswith("obj-method--")
    assert schemas[1]["name"] == "tool_a"
