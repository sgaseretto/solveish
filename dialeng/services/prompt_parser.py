"""
Prompt parser for ai-jup-style syntax.

This module parses special syntax in prompts and note cells:
- $`variable` / $`expression` - Reference kernel values (evaluated fresh)
- &`function` - Expose Python function as AI-callable tool
- &`obj.method` - Expose an object method as a tool
- &`[tool_a, tool_b]` - Expose multiple tools with one reference

Both syntaxes work in prompt cells and note (markdown) cells.
"""
import hashlib
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Pattern for variable/expression references: $`...`
VAR_PATTERN = re.compile(r'\$`([^`]+)`')

# Pattern for function/tool references: &`...`
FUNC_PATTERN = re.compile(r'&`([^`]+)`')

SIMPLE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Return unique strings preserving first-seen order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _expand_function_reference(raw_ref: str) -> List[str]:
    """Expand a raw &`...` reference into one or more tool targets."""
    ref = raw_ref.strip()
    if not ref:
        return []
    if ref.startswith('[') and ref.endswith(']'):
        inner = ref[1:-1]
        parts = [part.strip() for part in inner.split(',')]
        return [part for part in parts if part]
    return [ref]


def is_simple_identifier(expr: str) -> bool:
    """Whether an expression is a plain identifier name."""
    return bool(SIMPLE_IDENTIFIER_PATTERN.fullmatch(expr.strip()))


def tool_target_to_api_name(target: str) -> str:
    """Convert a raw tool target into a provider-safe API tool name."""
    cleaned = target.strip()
    if SIMPLE_IDENTIFIER_PATTERN.fullmatch(cleaned):
        return cleaned

    base = re.sub(r'[^a-zA-Z0-9_]+', '-', cleaned).strip('-') or 'tool'
    digest = hashlib.sha1(cleaned.encode('utf-8')).hexdigest()[:8]
    return f"{base}--{digest}"


def parse_prompt(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract variable and function names from text.

    Args:
        text: Prompt or note cell content

    Returns:
        Tuple of (expressions, function_targets) - both as unique lists
    """
    expressions = [expr.strip() for expr in VAR_PATTERN.findall(text) if expr.strip()]
    functions: List[str] = []
    for raw_ref in FUNC_PATTERN.findall(text):
        functions.extend(_expand_function_reference(raw_ref))

    return _unique_preserve_order(expressions), _unique_preserve_order(functions)


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
    """Check if text contains $`...` syntax."""
    return bool(VAR_PATTERN.search(text))


def has_function_syntax(text: str) -> bool:
    """Check if text contains &`...` syntax."""
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
        text: Text containing $`...` syntax
        kernel: SubprocessKernel instance
        notebook_id: Notebook identifier
        var_names: Optional pre-parsed variable names (avoids re-parsing)

    Returns:
        Tuple of (substituted_text, variable_info_dict)
        variable_info_dict maps expression -> {type, repr, exists, error?}
    """
    if var_names is None:
        var_names, _ = parse_prompt(text)

    if not var_names:
        return text, {}

    variable_info = {}
    result = text

    for var_name in var_names:
        # Query kernel for variable or expression value
        if hasattr(kernel, "evaluate_expression"):
            info = await kernel.evaluate_expression(var_name)
        else:
            info = await kernel.introspect_variable(var_name)
        variable_info[var_name] = info

        if info.get('exists'):
            # Substitute $`expr` with the repr value
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
        function_info_dict maps raw tool target -> introspection result
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

    api_name = tool_target_to_api_name(func_name)

    return {
        'name': api_name,
        'description': docstring,
        'input_schema': {
            'type': 'object',
            'properties': properties,
            'required': required
        },
        'dialeng_target_name': func_name,
        'dialeng_display_name': func_name,
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
    def _strip_tool(match):
        expanded = _expand_function_reference(match.group(1))
        return ', '.join(expanded) if expanded else ''

    result = VAR_PATTERN.sub(lambda m: m.group(1).strip(), text)
    result = FUNC_PATTERN.sub(_strip_tool, result)
    return result
