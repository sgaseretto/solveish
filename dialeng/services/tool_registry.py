"""
Tool registry for AI-callable functions.

This module manages:
1. Built-in tools (view, rg, create, str_replace, insert) - always available
2. Dynamic tools from kernel (via &`function` syntax) - per-prompt

Tools are converted to Anthropic tool schema format for Claude.
"""
import inspect
import logging
from typing import Dict, List, Callable, Any, Optional

logger = logging.getLogger(__name__)

# Python type to JSON Schema type mapping
TYPE_MAP = {
    int: 'integer',
    float: 'number',
    str: 'string',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
    type(None): 'null',
}

# String type names to JSON Schema types
TYPE_NAME_MAP = {
    'int': 'integer',
    'float': 'number',
    'str': 'string',
    'bool': 'boolean',
    'list': 'array',
    'List': 'array',
    'dict': 'object',
    'Dict': 'object',
    'None': 'null',
    'NoneType': 'null',
    'any': 'string',
    'Any': 'string',
    'Optional': 'string',
}

# Tools that modify files (require confirmation when enabled)
FILE_MODIFYING_TOOLS = {'create', 'str_replace', 'insert', 'sed'}


def python_type_to_json_schema(py_type) -> Dict[str, Any]:
    """
    Convert a Python type annotation to JSON Schema.

    Handles basic types, Optional, List, Dict, etc.

    Args:
        py_type: Python type annotation

    Returns:
        JSON Schema type definition
    """
    import typing

    # Handle None
    if py_type is None or py_type is type(None):
        return {'type': 'null'}

    # Handle basic types
    if py_type in TYPE_MAP:
        return {'type': TYPE_MAP[py_type]}

    # Handle string type names
    if isinstance(py_type, str):
        return {'type': TYPE_NAME_MAP.get(py_type, 'string')}

    # Handle typing generics
    origin = getattr(py_type, '__origin__', None)
    args = getattr(py_type, '__args__', ())

    if origin is not None:
        # Optional[X] is Union[X, None]
        if origin is type(None):
            return {'type': 'null'}

        # Handle Union (including Optional)
        try:
            import typing
            if origin is typing.Union:
                # Filter out NoneType for Optional
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    return python_type_to_json_schema(non_none_args[0])
                # Multiple types - use first non-None
                if non_none_args:
                    return python_type_to_json_schema(non_none_args[0])
        except Exception:
            pass

        # Handle List[X]
        if origin is list:
            schema = {'type': 'array'}
            if args:
                schema['items'] = python_type_to_json_schema(args[0])
            return schema

        # Handle Dict[K, V]
        if origin is dict:
            return {'type': 'object'}

    # Handle class with __name__
    if hasattr(py_type, '__name__'):
        type_name = py_type.__name__
        return {'type': TYPE_NAME_MAP.get(type_name, 'string')}

    # Default to string
    return {'type': 'string'}


def function_to_tool_schema(func: Callable, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a Python function to an Anthropic tool definition.

    Args:
        func: Python function to convert
        name: Optional override for function name

    Returns:
        Anthropic tool definition dict
    """
    func_name = name or func.__name__
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or f"Call the {func_name} function"

    # Build properties from parameters
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # Get type schema
        if param.annotation != inspect.Parameter.empty:
            prop = python_type_to_json_schema(param.annotation)
        else:
            prop = {'type': 'string'}

        # Add description from docstring parsing
        prop['description'] = _extract_param_description(docstring, param_name)

        properties[param_name] = prop

        # Mark as required if no default
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        'name': func_name,
        'description': docstring,
        'input_schema': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }


def _extract_param_description(docstring: str, param_name: str) -> str:
    """Extract parameter description from docstring (Google/numpy style)."""
    import re

    if not docstring:
        return param_name

    lines = docstring.split('\n')
    in_params_section = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for section headers
        if stripped.lower() in ('args:', 'arguments:', 'parameters:', 'params:'):
            in_params_section = True
            continue
        elif stripped.lower() in ('returns:', 'return:', 'raises:', 'examples:', 'example:'):
            in_params_section = False
            continue

        if in_params_section:
            # Google style: "param_name: description" or "param_name (type): description"
            google_match = re.match(
                rf'^\s*{re.escape(param_name)}\s*(?:\([^)]*\))?\s*:\s*(.+)',
                line
            )
            if google_match:
                return google_match.group(1).strip()

            # Numpy style: "param_name : type" followed by description
            numpy_match = re.match(rf'^\s*{re.escape(param_name)}\s*:', line)
            if numpy_match and i + 1 < len(lines):
                return lines[i + 1].strip()

    return param_name


def is_file_modifying_tool(tool_name: str) -> bool:
    """Check if a tool modifies files (for confirmation flow)."""
    return tool_name in FILE_MODIFYING_TOOLS


class ToolRegistry:
    """
    Registry for AI-callable tools.

    Manages:
    - Built-in tools (always available)
    - Dynamic tools from kernel introspection

    Usage:
        registry = ToolRegistry()
        registry.register_builtin(view_func)

        # Get all tools for a prompt
        tools = await registry.get_tools_for_prompt(
            func_names=['my_func'],
            kernel=kernel,
            notebook_id='notebook1'
        )
    """

    def __init__(self):
        self._builtin_tools: Dict[str, Callable] = {}
        self._builtin_schemas: Dict[str, dict] = {}

    def register_builtin(self, func: Callable, name: Optional[str] = None):
        """
        Register a built-in tool.

        Args:
            func: Function to register
            name: Optional name override
        """
        tool_name = name or func.__name__
        self._builtin_tools[tool_name] = func
        self._builtin_schemas[tool_name] = function_to_tool_schema(func, tool_name)
        logger.debug(f"Registered built-in tool: {tool_name}")

    def get_builtin_tool(self, name: str) -> Optional[Callable]:
        """Get a built-in tool function by name."""
        return self._builtin_tools.get(name)

    def get_builtin_schemas(self) -> List[dict]:
        """Get Anthropic tool schemas for all built-in tools."""
        return list(self._builtin_schemas.values())

    def get_builtin_names(self) -> List[str]:
        """Get names of all registered built-in tools."""
        return list(self._builtin_tools.keys())

    def is_builtin(self, name: str) -> bool:
        """Check if a tool is a built-in."""
        return name in self._builtin_tools

    async def execute_builtin(self, name: str, kwargs: dict) -> dict:
        """
        Execute a built-in tool.

        Args:
            name: Tool name
            kwargs: Arguments to pass

        Returns:
            Result dict with 'status' and 'result' or 'error'
        """
        if name not in self._builtin_tools:
            return {
                'status': 'error',
                'error': f"Built-in tool '{name}' not found"
            }

        try:
            func = self._builtin_tools[name]
            result = func(**kwargs)

            # Format result for LLM
            if isinstance(result, str):
                return {
                    'status': 'success',
                    'result': {'type': 'text', 'content': result}
                }
            else:
                return {
                    'status': 'success',
                    'result': {'type': 'text', 'content': repr(result)}
                }

        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    async def get_tools_for_prompt(
        self,
        func_names: List[str],
        kernel,
        notebook_id: str,
        include_builtins: bool = True
    ) -> List[dict]:
        """
        Get all tool schemas for a prompt.

        Combines built-in tools with dynamically discovered kernel functions.

        Args:
            func_names: Function names from &`func` syntax
            kernel: SubprocessKernel instance
            notebook_id: Notebook identifier
            include_builtins: Whether to include built-in tools

        Returns:
            List of Anthropic tool schema dicts
        """
        from .prompt_parser import get_function_schemas

        tools = []

        # Add built-in tools first
        if include_builtins:
            tools.extend(self.get_builtin_schemas())

        # Add dynamic tools from kernel
        if func_names:
            dynamic_schemas, _ = await get_function_schemas(
                kernel, notebook_id, func_names
            )
            tools.extend(dynamic_schemas)

        return tools


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry, creating if needed."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        # Register built-in tools
        _register_builtin_tools(_registry)
    return _registry


def _register_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools."""
    try:
        from .builtin_tools import BUILTIN_TOOLS
        for func in BUILTIN_TOOLS:
            registry.register_builtin(func)
        logger.info(f"Registered {len(BUILTIN_TOOLS)} built-in tools")
    except ImportError as e:
        logger.warning(f"Could not import built-in tools: {e}")
