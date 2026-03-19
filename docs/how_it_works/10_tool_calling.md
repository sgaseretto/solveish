# Tool Calling in Dialeng

Dialeng supports AI-powered tool calling, allowing Claude to execute Python functions
and interact with your codebase directly from prompt cells.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Prompt Cell                               │
│  "Use &`analyze_data` to process $`df` and show me the results" │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt Parser                               │
│  1. Extract $`df` → query kernel for value                      │
│  2. Extract &`analyze_data` → build tool schema                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Service                                │
│  Send prompt to Claude with:                                     │
│  - Variable values substituted                                   │
│  - Tool definitions (JSON schema)                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Tool Loop                                  │
│  While Claude returns tool_use blocks:                          │
│  1. Execute tool in kernel                                       │
│  2. Send result back to Claude                                   │
│  3. Continue until max_steps or no more tool calls              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Response                               │
│  Claude's response with tool results incorporated                │
└─────────────────────────────────────────────────────────────────┘
```

## Syntax Reference

### Variable References: `$`variable``

Reference any Python variable from the kernel namespace:

```python
# Code cell
data = [1, 2, 3, 4, 5]
stats = {"mean": 3.0, "std": 1.41}
```

```
# Prompt cell
What is the mean of $`data`? The stats show $`stats`.
```

The AI sees: "What is the mean of [1, 2, 3, 4, 5]? The stats show {'mean': 3.0, 'std': 1.41}."

### Function Tools: `&`function``

Expose Python functions as tools the AI can call:

```python
# Code cell
def calculate_stats(numbers: list) -> dict:
    """Calculate mean and standard deviation of a list of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        Dictionary with 'mean' and 'std' keys
    """
    import statistics
    return {"mean": statistics.mean(numbers), "std": statistics.stdev(numbers)}
```

```
# Prompt cell
Use &`calculate_stats` to analyze $`data`.
```

The AI receives a tool definition and can call `calculate_stats` with appropriate arguments.

### Using Syntax in Note Cells

Both `$`variable`` and `&`function`` syntax work in **note (markdown) cells**, not just prompt cells. This allows you to:

1. **Document with live values**:
   ```markdown
   ## Current State
   - Training accuracy: $`train_acc`
   - Validation accuracy: $`val_acc`
   ```

2. **Declare available tools upfront**:
   ```markdown
   ## Tools for this analysis
   - &`load_data`: Load CSV files
   - &`clean_data`: Handle missing values
   - &`plot_histogram`: Visualize distributions
   ```

When a prompt cell is executed, it scans all preceding note cells for `$`var`` and `&`func`` syntax and makes those variables/functions available.

## Built-in Tools

These tools are always available (no `&` prefix needed):

| Tool | Description | Example |
|------|-------------|---------|
| `view(path, start_line, num_lines)` | View file contents with line numbers | `view("app.py", 1, 50)` |
| `rg(pattern, path, file_type)` | Search files with ripgrep | `rg("def main", ".", "py")` |
| `create(path, content)` | Create a new file | `create("test.py", "print('hi')")` |
| `str_replace(file, old, new)` | Replace string in file | `str_replace("app.py", "old", "new")` |
| `insert(file, line, content)` | Insert at line number | `insert("app.py", 10, "# comment")` |
| `pyrun(code)` | Safe sandboxed Python execution | `pyrun("sum(range(100))")` |

### Built-in Tool Details

#### `view(path, start_line=1, num_lines=100, show_line_numbers=True)`

View file or directory contents with optional line numbers.

- For files: Shows contents with line numbers (like `cat -n`)
- For directories: Shows listing (like `ls -la`)

```python
# Examples
view("app.py")           # View first 100 lines
view("app.py", 50, 20)   # View lines 50-69
view("src/")             # List directory contents
```

#### `rg(pattern, path=".", file_type=None, max_results=50, context_lines=0)`

Search files using ripgrep (rg) for fast pattern matching.

```python
# Examples
rg("def main", "src/")        # Find 'def main' in src/
rg("TODO", ".", "py")         # Find TODOs in Python files
rg("class.*Controller", ".", context_lines=2)  # With context
```

#### `create(path, content, overwrite=False)`

Create a new file with the given content.

```python
# Examples
create("test.py", "print('hello')")
create("config.json", '{"key": "value"}')
```

#### `str_replace(file, old_str, new_str)`

Replace the first occurrence of an exact string in a file.

```python
# Examples
str_replace("app.py", "debug = True", "debug = False")
str_replace("config.json", '"port": 3000', '"port": 8080')
```

#### `insert(file, line, content)`

Insert content at a specific line number (1-indexed, inserts BEFORE the line).

```python
# Examples
insert("app.py", 1, "# New header comment")
insert("main.py", 10, "    # Debug line\n    print(x)")
```

#### `pyrun(code, concise=True)`

Safe sandboxed Python execution via [safepyrun](https://github.com/AnswerDotAI/safepyrun). Runs code with access to a curated subset of the standard library while blocking dangerous operations (filesystem writes, process spawning, system modification).

```python
# Examples
pyrun("sum(range(100))")
pyrun("import math; math.sqrt(144)")
pyrun("[x**2 for x in range(10)]")
```

Key features:
- **Allowlist-based**: Only permitted callables are accessible (re, json, math, pathlib, etc.)
- **State persistence**: Variables/functions ending with `_` persist across calls (non-`_` names are discarded)
- **Write policies**: The Dialeng instance allows writes relative to cwd via `ok_dests=['.']`
- **Async-native**: Supports `await`, `async for`, `async with`

Limitations:
- **`_` suffix required**: Only names ending with `_` persist across calls and are callable. `def hello(x): ...` won't be available in subsequent calls — use `def hello_(x): ...` instead.
- **No recursive functions**: Functions defined inside `pyrun` cannot call themselves recursively, even with the `_` suffix. This is a Python `exec()` limitation, not a safepyrun design choice. Use iterative implementations instead.

See `notebooks/safepyrun_demo.ipynb` and `docs/how_it_works/18_safepyrun_integration.md` for details.

## Tool Loop

The AI can call multiple tools in sequence (up to `max_steps`, default 5):

```
# Prompt
Use view to read app.py, then use rg to find all function definitions.
```

Flow:
1. AI calls `view("app.py")`
2. Result returned to AI
3. AI calls `rg("def ", "app.py")`
4. Result returned to AI
5. AI generates final response

## Configuration

Edit `dialeng_config.json`:

```json
{
  "tool_settings": {
    "max_steps": 5,
    "require_confirmation": false,
    "builtin_tools_enabled": true
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_steps` | 5 | Maximum tool calls per prompt (1-10) |
| `require_confirmation` | false | Prompt before file-modifying tools |
| `builtin_tools_enabled` | true | Enable built-in file tools |

## Confirmation Mode

When `require_confirmation: true`, file-modifying tools (create, str_replace, insert) show a dialog before execution:

```
┌─────────────────────────────────────┐
│ ⚠️ Confirm: str_replace            │
│ ─────────────────────────────────── │
│ {                                   │
│   "file": "app.py",                 │
│   "old_str": "debug = True",        │
│   "new_str": "debug = False"        │
│ }                                   │
│ ─────────────────────────────────── │
│         [Cancel]  [Execute]         │
└─────────────────────────────────────┘
```

## Implementation Details

### Files Involved

| File | Purpose |
|------|---------|
| `/services/prompt_parser.py` | Parses `$`var`` and `&`func`` syntax |
| `/services/tool_registry.py` | Tool schema generation and registry |
| `/services/builtin_tools.py` | Built-in file tools implementation |
| `/services/llm_service.py` | Tool loop and LLM integration |
| `/services/kernel/kernel_worker.py` | Kernel introspection and tool execution |
| `/services/kernel/subprocess_kernel.py` | Async kernel interface |

### Kernel Introspection

When you reference `$`variable``, the system queries the kernel:

```python
# Introspect variable: returns type, repr (truncated to 500 chars)
result = await kernel.introspect_variable("my_var")
# {'exists': True, 'var_type': 'list', 'repr': '[1, 2, 3]'}
```

When you reference `&`function``, the system introspects the function:

```python
# Introspect function: returns signature, docstring, parameters
result = await kernel.introspect_function("my_func")
# {'exists': True, 'is_callable': True, 'signature': '(x: int, y: str = "default")',
#  'docstring': '...', 'parameters': {...}}
```

### Tool Schema Generation

Functions are converted to Anthropic tool schema format:

```python
def greet(name: str, greeting: str = "Hello") -> str:
    """Greet a person.

    Args:
        name: Person's name
        greeting: Greeting word to use

    Returns:
        The greeting message
    """
    return f"{greeting}, {name}!"
```

Becomes:

```json
{
  "name": "greet",
  "description": "Greet a person.\n\nArgs:\n    name: Person's name\n    greeting: Greeting word to use\n\nReturns:\n    The greeting message",
  "input_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string", "description": "Person's name"},
      "greeting": {"type": "string", "description": "Greeting word to use"}
    },
    "required": ["name"]
  }
}
```

## Provider Support

Tool calling works with both LLM providers:

### claudette (Direct API/Bedrock)

Uses the native Anthropic tool calling API.

### claude-agent-sdk (Claude Code Subscription)

Uses claude-agent-sdk directly for tool support via MCP servers.

## Error Handling

- **Invalid variable**: Shows error marker in prompt and continues
- **Missing function**: Tool not available, AI informed
- **Tool execution error**: Error returned to AI, it may retry or explain
- **Timeout**: Tools have a 60-second execution timeout

## Examples

### Example 1: Variable Substitution

```python
# Code cell
data = [10, 20, 30, 40, 50]
```

```
# Prompt cell
Calculate the average of $`data` and explain the calculation.
```

### Example 2: Function as Tool

```python
# Code cell
def analyze(numbers: list) -> dict:
    """Analyze a list of numbers."""
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers)
    }
```

```
# Prompt cell
Use &`analyze` to analyze [1, 2, 3, 4, 5] and interpret the results.
```

### Example 3: Multi-Tool Workflow

```
# Prompt cell
Use view to read README.md, then use rg to find all TODO comments
in the codebase, and summarize what needs to be done.
```

### Example 4: Note Cell Tool Declaration

```markdown
# Note cell
## Available Analysis Tools

Use these tools for data analysis:
- &`load_csv` - Load data from CSV
- &`clean_data` - Remove null values
- &`plot_chart` - Create visualizations

Current dataset: $`df` with $`df.shape[0]` rows.
```

```
# Prompt cell
Analyze the current dataset and create a summary visualization.
```

## Troubleshooting

### "Variable not found"

The variable doesn't exist in the kernel namespace. Make sure:
1. You've run the code cell that defines the variable
2. The variable name is spelled correctly
3. The variable is in global scope (not inside a function)

### "Function not callable"

The referenced name exists but isn't a function. Check:
1. The name points to a function, not a variable
2. The function is defined in global scope

### Tools not appearing

1. Check `tool_settings.builtin_tools_enabled` in config
2. For dynamic tools, ensure the function has proper type hints
3. The function must be in the kernel namespace (run the definition cell)

### Confirmation dialog not showing

1. Enable `tool_settings.require_confirmation` in config
2. Only file-modifying tools (create, str_replace, insert) trigger confirmation
