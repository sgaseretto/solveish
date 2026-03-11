"""
Prompt parser for ai-jup-style syntax.

This module parses special syntax in prompts and note cells:
- $`variable` - Reference kernel variable (value substituted into prompt)
- &`function` - Expose Python function as AI-callable tool

Both syntaxes work in prompt cells and note (markdown) cells.
"""
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Pattern for variable references: $`variable_name`
VAR_PATTERN = re.compile(r'\$`([a-zA-Z_][a-zA-Z0-9_]*)`')

# Pattern for function/tool references: &`function_name`
FUNC_PATTERN = re.compile(r'&`([a-zA-Z_][a-zA-Z0-9_]*)`')


def parse_prompt(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract variable and function names from text.

    Args:
        text: Prompt or note cell content

    Returns:
        Tuple of (variable_names, function_names) - both as unique lists
    """
    variables = VAR_PATTERN.findall(text)
    functions = FUNC_PATTERN.findall(text)

    # Return unique names preserving order
    seen_vars = set()
    unique_vars = []
    for v in variables:
        if v not in seen_vars:
            seen_vars.add(v)
            unique_vars.append(v)

    seen_funcs = set()
    unique_funcs = []
    for f in functions:
        if f not in seen_funcs:
            seen_funcs.add(f)
            unique_funcs.append(f)

    return unique_vars, unique_funcs


def has_special_syntax(text: str) -> bool:
    """
    Check if text contains $` or &` syntax.

    Args:
        text: Text to check

    Returns:
        True if special syntax is found
    """
    return bool(VAR_PATTERN.search(text) or FUNC_PATTERN.search(text))


def has_variable_syntax(text: str) -> bool:
    """Check if text contains $`variable` syntax."""
    return bool(VAR_PATTERN.search(text))


def has_function_syntax(text: str) -> bool:
    """Check if text contains &`function` syntax."""
    return bool(FUNC_PATTERN.search(text))


async def substitute_variables(
    text: str,
    kernel,
    notebook_id: str,
    var_names: Optional[List[str]] = None
) -> Tuple[str, dict]:
    """
    Replace $`var` syntax with actual variable values from kernel.

    Args:
        text: Text containing $`var` syntax
        kernel: SubprocessKernel instance
        notebook_id: Notebook identifier
        var_names: Optional pre-parsed variable names (avoids re-parsing)

    Returns:
        Tuple of (substituted_text, variable_info_dict)
        variable_info_dict maps var_name -> {type, repr, exists, error?}
    """
    if var_names is None:
        var_names, _ = parse_prompt(text)

    if not var_names:
        return text, {}

    variable_info = {}
    result = text

    for var_name in var_names:
        # Query kernel for variable value
        info = await kernel.introspect_variable(var_name)
        variable_info[var_name] = info

        if info.get('exists'):
            # Substitute $`var_name` with the repr value
            pattern = re.compile(rf'\$`{re.escape(var_name)}`')
            var_repr = info.get('repr', '<unknown>')
            # Use a replacer function to avoid issues with $ in replacement
            result = pattern.sub(lambda m: var_repr, result)
            logger.debug(f"Substituted $`{var_name}` with: {var_repr[:50]}...")
        else:
            # Variable not found - leave syntax but add error marker
            error = info.get('error', 'Variable not found')
            logger.warning(f"Variable $`{var_name}` not found: {error}")
            # Keep the original syntax so the AI knows what was referenced
            pattern = re.compile(rf'\$`{re.escape(var_name)}`')
            result = pattern.sub(f"$`{var_name}` (ERROR: {error})", result)

    return result, variable_info


async def get_function_schemas(
    kernel,
    notebook_id: str,
    func_names: List[str]
) -> Tuple[List[dict], dict]:
    """
    Get Anthropic tool schemas for referenced functions.

    Args:
        kernel: SubprocessKernel instance
        notebook_id: Notebook identifier
        func_names: List of function names to introspect

    Returns:
        Tuple of (tool_schemas, function_info_dict)
        tool_schemas is a list of Anthropic tool definitions
        function_info_dict maps func_name -> introspection result
    """
    if not func_names:
        return [], {}

    tool_schemas = []
    function_info = {}

    for func_name in func_names:
        # Query kernel for function metadata
        info = await kernel.introspect_function(func_name)
        function_info[func_name] = info

        if info.get('exists') and info.get('is_callable'):
            # Build Anthropic tool schema
            schema = _build_tool_schema(func_name, info)
            tool_schemas.append(schema)
            logger.debug(f"Built tool schema for &`{func_name}`")
        else:
            error = info.get('error', 'Function not found or not callable')
            logger.warning(f"Function &`{func_name}` not available: {error}")

    return tool_schemas, function_info


def _build_tool_schema(func_name: str, func_info: dict) -> dict:
    """
    Build an Anthropic tool schema from function introspection info.

    Args:
        func_name: Function name
        func_info: Dict from kernel introspection with signature, docstring, parameters

    Returns:
        Anthropic tool definition dict
    """
    # Map Python types to JSON Schema types
    type_map = {
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
        'any': 'string',  # Default to string for untyped params
    }

    # Build properties from parameters
    properties = {}
    required = []

    params = func_info.get('parameters', {})
    for param_name, param_info in params.items():
        py_type = param_info.get('type', 'any')

        # Handle generic types like List[int]
        base_type = py_type.split('[')[0] if '[' in py_type else py_type
        json_type = type_map.get(base_type, 'string')

        prop = {
            'type': json_type,
            'description': param_info.get('description', param_name)
        }

        # Add items type for arrays
        if json_type == 'array' and '[' in py_type:
            # Extract inner type from List[int] -> int
            inner_match = re.search(r'\[([^\]]+)\]', py_type)
            if inner_match:
                inner_type = inner_match.group(1).strip()
                inner_json = type_map.get(inner_type, 'string')
                prop['items'] = {'type': inner_json}

        properties[param_name] = prop

        # Mark as required if no default
        if 'default' not in param_info:
            required.append(param_name)

    # Build the tool schema
    docstring = func_info.get('docstring', '')
    if not docstring:
        docstring = f"Call the {func_name} function"

    return {
        'name': func_name,
        'description': docstring,
        'input_schema': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }


def strip_special_syntax(text: str) -> str:
    """
    Remove $`var` and &`func` syntax from text, leaving just the names.

    Useful for displaying cleaned text after processing.

    Args:
        text: Text with special syntax

    Returns:
        Text with syntax markers removed (just variable/function names remain)
    """
    # Replace $`name` with just name
    result = VAR_PATTERN.sub(r'\1', text)
    # Replace &`name` with just name
    result = FUNC_PATTERN.sub(r'\1', result)
    return result
