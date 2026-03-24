# safepyrun Integration

Dialeng integrates [safepyrun](https://github.com/AnswerDotAI/safepyrun) as a built-in LLM tool, providing the AI with safe sandboxed Python execution during prompt responses.

## Architecture

```mermaid
sequenceDiagram
    participant User
    participant LLM as Claude (LLM)
    participant Registry as ToolRegistry
    participant PyRun as DialengRunPython
    participant Core as safepyrun core

    User->>LLM: Prompt (e.g., "calculate fibonacci")
    LLM->>Registry: tool_use: pyrun(code="...")
    Note over Registry: Current prompt tools already parsed from built-ins + &`tool` refs
    Registry->>PyRun: await pyrun(code)
    PyRun->>Core: _run_python(...) with prompt-scoped globals + allowlist
    Note over Core: RestrictedPython compiles AST<br/>Allowlist checks each callable<br/>Captures stdout/stderr/result
    Core-->>PyRun: {"result": ..., "stdout": ...}
    PyRun-->>Registry: {"result": ..., "stdout": ...}
    Registry-->>LLM: Tool result
    LLM-->>User: Response with computed result
```

## How It Works

safepyrun uses [RestrictedPython](https://restrictedpython.readthedocs.io/) to compile Python source into a modified AST where every attribute access, item access, and iteration is routed through hook functions. These hooks check an allowlist of permitted callables before allowing execution.

### Threat Model

The sandbox is designed for an LLM — a well-meaning but occasionally clumsy collaborator. It prevents accidental damage (hallucinated cleanup steps, misunderstood requests), not deliberate escape attempts.

### Allowlist Tiers

1. **Curated stdlib subset**: `re`, `json`, `math`, `pathlib`, `datetime`, `collections`, `itertools`, `functools`, read-only `httpx.get`, and many more
2. **User-registered functions**: Via `allow()` — e.g., `allow('numpy.array', 'numpy.ndarray.sum')`
3. **LLM self-service**: Any symbol created with a trailing `_` (like `helper_`) is automatically available in subsequent calls
4. **Prompt-scoped Dialeng tools**: During a tool-enabled prompt run, Dialeng temporarily injects the current prompt's built-in tools plus explicit `&` references into `pyrun`

## The `_` Suffix Convention

This is safepyrun's core mechanism for cross-call state management:

### What it does

- **Persistence**: After each `pyrun` call, only variables/functions whose names end with `_` (and don't start with `_`) are exported to the persistent globals dict (`g`). Everything else is discarded.
- **Callability**: Only callables ending with `_`, or explicitly registered via `allow()`, are permitted to be called. Non-`_` callables are wrapped in `_Uncallable` and raise `PermissionError` if invoked.

### Why it exists

This is a deliberate security design. Without it, an LLM could define arbitrary functions and call them across sandbox invocations, effectively bypassing the allowlist. The `_` suffix acts as an explicit opt-in: the user (or LLM) must intentionally mark something as persistent.

### How it works internally

```mermaid
flowchart TD
    A["pyrun('def hello_(x): ...')"] --> B["Code executes in restricted env"]
    B --> C["hello_ defined in loc (locals)"]
    C --> D{"Name ends with _?"}
    D -->|Yes| E["_export() copies to g (globals)"]
    D -->|No| F["Discarded after call"]
    E --> G["Next pyrun call"]
    G --> H["tools dict built from g.items()"]
    H --> I{"_callable_ok?"}
    I -->|"Ends with _ → Yes"| J["Available as-is"]
    I -->|"No _ suffix"| K["Wrapped in _Uncallable"]
```

### Examples

```python
# These PERSIST across calls (names end with _):
await pyrun("x_ = 10")
await pyrun("x_")              # → 10

await pyrun("def greet_(name): return f'hello {name}'")
await pyrun("greet_('Joe')")   # → 'hello Joe'

# These DO NOT persist (no _ suffix):
await pyrun("x = 10")
await pyrun("x")               # → NameError

await pyrun("def greet(name): return f'hello {name}'")
await pyrun("greet('Joe')")    # → NameError
```

## Registration

In `dialeng/services/builtin_tools.py`:

```python
class DialengRunPython:
    async def __call__(self, code: str, concise: bool = True):
        return await _run_python(...)

pyrun = DialengRunPython(ok_dests=['.'])

BUILTIN_TOOLS = [view, rg, create, str_replace, insert, pyrun]
```

The wrapper keeps safepyrun's persistence model, but Dialeng now layers prompt-scoped tool injection on top:

```mermaid
flowchart TD
    A["Prompt parsed"] --> B["Built-ins enabled?"]
    A --> C["Explicit &`tool` refs found"]
    B --> D["ToolRegistry.push_pyrun_context(...)"]
    C --> D
    D --> E["DialengRunPython stores task-scoped globals + allowed names"]
    E --> F["pyrun(code)"]
    F --> G["Temporary tool globals merged into sandbox state"]
    G --> H["Temporary allowlist merged into __llmtools__"]
    H --> I["safepyrun.core._run_python(...)"]
    I --> J["Allowlist + tool globals restored"]
```

Only the current prompt's allowed tools are injected. Notebook functions that were not exposed via `&` syntax remain unavailable inside `pyrun`.

## Write Permissions

The Dialeng instance uses `ok_dests=['.']`, allowing the AI to write files relative to the current working directory. Path traversal attempts (`../`, `subdir/../../`) are detected and blocked.

## Limitations

### No recursive functions *defined inside pyrun*

Functions defined inside `pyrun` **cannot call themselves recursively**. This is not a safepyrun design choice — it's a fundamental consequence of how Python's `exec(code, globals, locals)` works with separate globals and locals dicts.

**Exception:** Functions defined in normal Python and registered via `allow()` CAN be recursive, because their `__globals__` is the real module globals dict (which contains themselves):

```python
from safepyrun import allow

def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)
allow('fib')

await pyrun('fib(10)')  # → 55 ✅
```

**Root cause:**

```mermaid
sequenceDiagram
    participant Code as pyrun("def fib_(n): ... fib_(n-1)")
    participant Exec as exec(compiled, rg, loc)
    participant Func as fib_ function object

    Code->>Exec: Execute definition
    Exec->>Func: fib_ created in loc (locals dict)
    Note over Func: fib_.__globals__ = rg (globals dict)
    Note over Func: But fib_ is in loc, NOT in rg
    Func->>Func: Body calls fib_(n-1)
    Note over Func: Python looks up "fib_" in __globals__ (rg)
    Note over Func: fib_ not found in rg → NameError
```

When Python defines a function inside `exec(code, globals_dict, locals_dict)`:
1. The function object is stored in `locals_dict`
2. The function's `__globals__` is set to `globals_dict`
3. When the function tries to call itself, Python looks up the name in `__globals__`
4. Since the function is in `locals_dict` (not `globals_dict`), the name lookup fails

This happens even with `_` suffix, even in the same `pyrun` call, and even across calls (because the function retains the `rg` dict from the call where it was defined, and that dict never contained the function).

**Workaround**: Use iterative implementations instead of recursion:

```python
# This FAILS (recursive):
await pyrun("def fib_(n): return n if n <= 1 else fib_(n-1) + fib_(n-2)")

# This WORKS (iterative):
await pyrun("""
def fib_(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""")
```

### No private names

Names starting with `_` are excluded from both export and the `tools` dict. Names with `__` (dunders) are further restricted by `SafeTransformer`.

### Module method access requires registration

You can `import` any module, but calling its methods requires them to be in the allowlist. For example, `json.loads()` works (registered), but `os.system()` does not.

### Prompt-scoped tools inside `pyrun`

When the current prompt exposes tools, `pyrun` can use them directly:

```python
# built-in tool inside pyrun
create("notes.txt", "hello", overwrite=True)

# explicit notebook tool inside pyrun
await analyze_data(df_name="sales")

# dotted tool reference inside pyrun
await dialog.read_msg(id="abc123")
```

Rules:
- Built-in tools are available inside `pyrun` only when built-ins are enabled for the prompt
- Notebook/dialog tools are available inside `pyrun` only if they were explicitly exposed via `&` syntax
- Explicit notebook/dialog tools are async inside the sandbox and should be called with `await`
- Tool results are coerced back to Python values when possible (`42`, `{"a": 1}`, lists, etc.); richer outputs fall back to structured metadata

## Using pyrun in Notebooks

### As an LLM tool (primary use case)

In a **prompt cell**, the AI automatically has access to `pyrun`. It knows the `_` suffix convention from the tool's docstring:

```
Calculate the first 20 prime numbers using pyrun
```

The AI will call `pyrun` with appropriate code, using `_` suffixed names for any state it needs to persist. When the same prompt also exposes tools, the model can choose a CodeAct-style flow and use those tools from within the same `pyrun` call instead of spending multiple tool loop turns.

### Direct use in code cells

You can also use `pyrun` directly in **code cells** for testing or scripting:

```python
from dialeng.services.builtin_tools import pyrun

await pyrun("x_ = 42")
await pyrun("x_ * 2")  # → 84
```

Note that this `pyrun` instance shares state with the LLM's tool calls — `_` suffixed variables set by the AI are visible in code cells and vice versa.

## Key Files

| File | Purpose |
|------|---------|
| `dialeng/services/builtin_tools.py` | Dialeng `pyrun` wrapper, task-scoped tool injection, result coercion |
| `dialeng/services/tool_registry.py` | Prompt-scoped `pyrun` context construction and dotted tool namespaces |
| `notebooks/safepyrun_demo.ipynb` | Interactive demo notebook |
