"""
Built-in file tools for AI tool calling.

Inspired by fastcore.tools, these tools are always available to the AI
without requiring &` prefix. They provide file exploration and editing
capabilities.

Tools:
- view: View file/directory contents with line numbers
- rg: Search files using ripgrep
- create: Create a new file
- str_replace: Replace exact string in file
- strs_replace: Replace multiple exact strings in a file
- insert: Insert content at line number
- replace_lines: Replace a line range in a file
- file_insert_line: Insert one line at a given location
- file_del_lines: Delete a line range from a file
- pyrun: Safe sandboxed Python execution via safepyrun
"""
import asyncio
import ast
import contextvars
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any, Dict, Set
import logging
from fastcore.imports import __llmtools__
from safepyrun.core import _run_python

logger = logging.getLogger(__name__)

# Maximum file size to read (prevent memory issues)
MAX_FILE_SIZE = 1024 * 1024  # 1MB

# Maximum lines to return
MAX_LINES = 500


@dataclass
class PyRunExecutionContext:
    """Temporary prompt-scoped tools made available inside pyrun."""
    tool_globals: Dict[str, Any] = field(default_factory=dict)
    allowed_names: Set[str] = field(default_factory=set)


_pyrun_context: contextvars.ContextVar[Optional[PyRunExecutionContext]] = contextvars.ContextVar(
    "dialeng_pyrun_context",
    default=None,
)


class DialengRunPython:
    """Safe Python tool with prompt-scoped CodeAct-style tool injection."""

    def __init__(self, g: Optional[dict] = None, ok_dests: Optional[list[str]] = None):
        self.g = g if g is not None else {}
        self.ok_dests = ok_dests
        self.__name__ = "pyrun"
        self._lock = asyncio.Lock()

    @property
    def __doc__(self):
        return (
            "Execute restricted Python with access to Dialeng's safe sandbox. "
            "Values or helper functions ending with `_` persist across calls. "
            "Built-in tools enabled for the current prompt, plus any explicit &`tool` or "
            "&`obj.method` references from the notebook context, are also available inside "
            "this sandbox for CodeAct-style multi-step work. "
            "Notebook or dialog tools exposed this way are async and should be called with "
            "`await` inside pyrun, for example `await read_msgid(id='abc')` or "
            "`await obj.method(x=1)`. Tools that were not exposed for the current prompt are "
            "not available inside pyrun."
        )

    def push_tool_context(self, *, tool_globals: Dict[str, Any], allowed_names: Set[str]):
        """Install prompt-scoped tools for the current async task."""
        return _pyrun_context.set(
            PyRunExecutionContext(
                tool_globals=dict(tool_globals),
                allowed_names=set(allowed_names),
            )
        )

    def reset_tool_context(self, token) -> None:
        """Remove a previously-installed prompt-scoped tool context."""
        _pyrun_context.reset(token)

    @staticmethod
    def _coerce_text_result(content: str):
        """Recover JSON/Python-literal values when possible for CodeAct flows."""
        stripped = content.strip()
        if not stripped:
            return content
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(stripped)
            except Exception:
                continue
        return content

    @classmethod
    def coerce_tool_result(cls, result: dict):
        """Convert Dialeng tool execution payloads to ergonomic pyrun values."""
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "Tool execution failed"))

        payload = result.get("result", {})
        if not isinstance(payload, dict):
            return payload

        result_type = payload.get("type", "text")
        content = payload.get("content")

        if result_type == "text":
            return cls._coerce_text_result(content if isinstance(content, str) else repr(content))

        if result_type in {"html", "image"}:
            return payload

        return content

    async def __call__(self, code: str, concise: bool = True):
        ctx = _pyrun_context.get()
        tool_globals = ctx.tool_globals if ctx else {}
        allowed_names = ctx.allowed_names if ctx else set()

        async with self._lock:
            previous_values = {
                name: self.g[name]
                for name in tool_globals
                if name in self.g
            }
            inserted_names = [name for name in tool_globals if name not in previous_values]
            previous_allowed = set(__llmtools__)

            try:
                self.g.update(tool_globals)
                __llmtools__.update(allowed_names)
                return await _run_python(code, g=self.g, ok_dests=self.ok_dests, concise=concise)
            finally:
                __llmtools__.clear()
                __llmtools__.update(previous_allowed)
                for name in inserted_names:
                    self.g.pop(name, None)
                for name, value in previous_values.items():
                    self.g[name] = value


def view(
    path: str,
    start_line: int = 1,
    num_lines: int = 100,
    show_line_numbers: bool = True
) -> str:
    """
    View file or directory contents with optional line numbers.

    For files: Shows contents with line numbers (like cat -n).
    For directories: Shows listing (like ls -la).

    Args:
        path: Path to file or directory
        start_line: Line number to start from (1-indexed, files only)
        num_lines: Number of lines to show (files only)
        show_line_numbers: Whether to prefix lines with numbers

    Returns:
        File contents with line numbers, or directory listing

    Examples:
        view("app.py")  # View first 100 lines
        view("app.py", 50, 20)  # View lines 50-69
        view("src/")  # List directory contents
    """
    p = Path(path).expanduser()

    if not p.exists():
        return f"Error: Path '{path}' does not exist"

    if p.is_dir():
        # Directory listing
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            lines = []
            for entry in entries[:MAX_LINES]:
                if entry.is_dir():
                    lines.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    lines.append(f"  {entry.name} ({_format_size(size)})")
            result = f"Directory: {path}\n" + "\n".join(lines)
            if len(entries) > MAX_LINES:
                result += f"\n... ({len(entries) - MAX_LINES} more entries)"
            return result
        except PermissionError:
            return f"Error: Permission denied reading directory '{path}'"

    # File reading
    try:
        # Check file size
        if p.stat().st_size > MAX_FILE_SIZE:
            return f"Error: File '{path}' is too large (>{MAX_FILE_SIZE} bytes)"

        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        # Handle line range (1-indexed)
        start_idx = max(0, start_line - 1)
        end_idx = min(start_idx + num_lines, total_lines)

        if start_idx >= total_lines:
            return f"Error: start_line {start_line} is beyond file length ({total_lines} lines)"

        selected_lines = all_lines[start_idx:end_idx]

        # Format with line numbers
        if show_line_numbers:
            width = len(str(end_idx))
            formatted = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted.append(f"{i:>{width}}│ {line.rstrip()}")
            result = "\n".join(formatted)
        else:
            result = "".join(selected_lines).rstrip()

        # Add info header
        if start_idx > 0 or end_idx < total_lines:
            header = f"File: {path} (lines {start_idx + 1}-{end_idx} of {total_lines})\n"
        else:
            header = f"File: {path} ({total_lines} lines)\n"

        return header + result

    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied reading '{path}'"
    except Exception as e:
        return f"Error reading '{path}': {str(e)}"


def rg(
    pattern: str,
    path: str = ".",
    file_type: Optional[str] = None,
    max_results: int = 50,
    context_lines: int = 0
) -> str:
    """
    Search files using ripgrep (rg) for fast pattern matching.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_type: Filter by file type (e.g., 'py', 'js', 'md')
        max_results: Maximum number of matches to return
        context_lines: Number of context lines before/after match

    Returns:
        Search results with file paths and matching lines

    Examples:
        rg("def main", "src/")  # Find 'def main' in src/
        rg("TODO", ".", "py")  # Find TODOs in Python files
        rg("class.*Controller", ".", context_lines=2)
    """
    # Build ripgrep command
    cmd = ['rg', '--line-number', '--no-heading']

    if context_lines > 0:
        cmd.extend(['-C', str(context_lines)])

    if file_type:
        cmd.extend(['-t', file_type])

    cmd.extend(['-m', str(max_results)])  # Max matches per file
    cmd.append(pattern)
    cmd.append(path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()
        if not output:
            if result.returncode == 1:
                return f"No matches found for pattern: {pattern}"
            elif result.returncode == 2:
                return f"Error: {result.stderr.strip()}"

        # Limit total output
        lines = output.split('\n')
        if len(lines) > MAX_LINES:
            output = '\n'.join(lines[:MAX_LINES])
            output += f"\n... ({len(lines) - MAX_LINES} more matches)"

        return f"Search results for '{pattern}':\n{output}"

    except FileNotFoundError:
        # ripgrep not installed, fall back to grep
        return _fallback_grep(pattern, path, file_type, max_results)
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error during search: {str(e)}"


def _fallback_grep(pattern: str, path: str, file_type: Optional[str], max_results: int) -> str:
    """Fallback to grep if ripgrep is not available."""
    cmd = ['grep', '-r', '-n', '-I']  # recursive, line numbers, ignore binary

    if file_type:
        cmd.extend(['--include', f'*.{file_type}'])

    cmd.append(pattern)
    cmd.append(path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern: {pattern}"

        lines = output.split('\n')[:max_results]
        return f"Search results for '{pattern}':\n" + '\n'.join(lines)

    except Exception as e:
        return f"Error during search: {str(e)}"


def create(path: str, content: str, overwrite: bool = False) -> str:
    """
    Create a new file with the given content.

    Args:
        path: Path where to create the file
        content: Content to write to the file
        overwrite: If True, overwrite existing file; if False, error if exists

    Returns:
        Success message or error

    Examples:
        create("test.py", "print('hello')")
        create("config.json", '{"key": "value"}')
    """
    p = Path(path).expanduser()

    # Check if file exists
    if p.exists() and not overwrite:
        return f"Error: File '{path}' already exists. Use overwrite=True to replace."

    try:
        # Create parent directories if needed
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"Created file: {path} ({len(content)} bytes)"

    except PermissionError:
        return f"Error: Permission denied creating '{path}'"
    except Exception as e:
        return f"Error creating '{path}': {str(e)}"


def str_replace(file: str, old_str: str, new_str: str) -> str:
    """
    Replace the first occurrence of a string in a file.

    The old_str must match exactly (including whitespace).
    Only the first occurrence is replaced.

    Args:
        file: Path to the file to modify
        old_str: Exact string to find and replace
        new_str: Replacement string

    Returns:
        Success message showing the replacement, or error

    Examples:
        str_replace("app.py", "debug = True", "debug = False")
        str_replace("config.json", '"port": 3000', '"port": 8080')
    """
    p = Path(file).expanduser()

    if not p.exists():
        return f"Error: File '{file}' does not exist"

    if not p.is_file():
        return f"Error: '{file}' is not a file"

    try:
        content = p.read_text(encoding='utf-8')

        if old_str not in content:
            return f"Error: String not found in '{file}':\n{old_str[:100]}..."

        # Find the position for context
        pos = content.find(old_str)
        line_num = content[:pos].count('\n') + 1

        # Replace first occurrence only
        new_content = content.replace(old_str, new_str, 1)

        p.write_text(new_content, encoding='utf-8')

        return f"Replaced in {file} at line {line_num}:\n- {old_str[:50]}...\n+ {new_str[:50]}..."

    except UnicodeDecodeError:
        return f"Error: '{file}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied modifying '{file}'"
    except Exception as e:
        return f"Error modifying '{file}': {str(e)}"


def insert(file: str, line: int, content: str) -> str:
    """
    Insert content at a specific line number in a file.

    The new content is inserted BEFORE the specified line.
    Line numbers are 1-indexed.

    Args:
        file: Path to the file to modify
        line: Line number where to insert (1-indexed)
        content: Content to insert (can be multiple lines)

    Returns:
        Success message or error

    Examples:
        insert("app.py", 1, "# New header comment")
        insert("main.py", 10, "    # Debug line\\n    print(x)")
    """
    p = Path(file).expanduser()

    if not p.exists():
        return f"Error: File '{file}' does not exist"

    if not p.is_file():
        return f"Error: '{file}' is not a file"

    if line < 1:
        return "Error: Line number must be >= 1"

    try:
        lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
        total_lines = len(lines)

        # Allow inserting at line total_lines + 1 (append)
        if line > total_lines + 1:
            return f"Error: Line {line} is beyond file length ({total_lines} lines)"

        # Ensure content ends with newline
        if content and not content.endswith('\n'):
            content += '\n'

        # Insert at position (0-indexed)
        insert_idx = line - 1
        lines.insert(insert_idx, content)

        p.write_text(''.join(lines), encoding='utf-8')

        content_preview = content.strip()[:50]
        return f"Inserted at {file}:{line}:\n{content_preview}..."

    except UnicodeDecodeError:
        return f"Error: '{file}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied modifying '{file}'"
    except Exception as e:
        return f"Error modifying '{file}': {str(e)}"


def strs_replace(file: str, old_strs: list[str], new_strs: list[str]) -> str:
    """
    Replace multiple exact strings in a file.

    Each old/new pair is applied once in order.
    """
    if len(old_strs) != len(new_strs):
        return "Error: old_strs and new_strs must have the same length"

    p = Path(file).expanduser()
    if not p.exists():
        return f"Error: File '{file}' does not exist"
    if not p.is_file():
        return f"Error: '{file}' is not a file"

    try:
        content = p.read_text(encoding='utf-8')
        replaced = 0
        for old_str, new_str in zip(old_strs, new_strs):
            if old_str in content:
                content = content.replace(old_str, new_str, 1)
                replaced += 1
        if replaced == 0:
            return f"Error: None of the requested strings were found in '{file}'"
        p.write_text(content, encoding='utf-8')
        return f"Applied {replaced} string replacement(s) in {file}"
    except UnicodeDecodeError:
        return f"Error: '{file}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied modifying '{file}'"
    except Exception as e:
        return f"Error modifying '{file}': {str(e)}"


def replace_lines(file: str, start_line: int, end_line: int, new_content: str) -> str:
    """
    Replace a 1-indexed inclusive line range in a text file.
    """
    p = Path(file).expanduser()
    if not p.exists():
        return f"Error: File '{file}' does not exist"
    if not p.is_file():
        return f"Error: '{file}' is not a file"
    if start_line < 1 or end_line < start_line:
        return "Error: Invalid line range"

    try:
        lines = p.read_text(encoding='utf-8').splitlines()
        total_lines = len(lines)
        if start_line > total_lines + 1:
            return f"Error: start_line {start_line} is beyond file length ({total_lines} lines)"

        replacement_lines = new_content.splitlines()
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)
        lines[start_idx:end_idx] = replacement_lines
        p.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        return f"Replaced lines {start_line}-{end_line} in {file}"
    except UnicodeDecodeError:
        return f"Error: '{file}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied modifying '{file}'"
    except Exception as e:
        return f"Error modifying '{file}': {str(e)}"


def file_insert_line(file: str, line: int, content: str) -> str:
    """
    Insert a single line before the given 1-indexed line number.
    """
    return insert(file, line, content)


def file_del_lines(file: str, start_line: int, end_line: int) -> str:
    """
    Delete a 1-indexed inclusive line range from a text file.
    """
    p = Path(file).expanduser()
    if not p.exists():
        return f"Error: File '{file}' does not exist"
    if not p.is_file():
        return f"Error: '{file}' is not a file"
    if start_line < 1 or end_line < start_line:
        return "Error: Invalid line range"

    try:
        lines = p.read_text(encoding='utf-8').splitlines()
        total_lines = len(lines)
        if start_line > total_lines:
            return f"Error: start_line {start_line} is beyond file length ({total_lines} lines)"
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)
        del lines[start_idx:end_idx]
        p.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        return f"Deleted lines {start_line}-{end_line} from {file}"
    except UnicodeDecodeError:
        return f"Error: '{file}' is not a text file"
    except PermissionError:
        return f"Error: Permission denied modifying '{file}'"
    except Exception as e:
        return f"Error modifying '{file}': {str(e)}"


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == 'B' else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# Safe Python sandbox — ok_dests=['.'] allows writes relative to cwd (matches Solveit)
pyrun = DialengRunPython(ok_dests=['.'])

# Export list of all built-in tools
BUILTIN_TOOLS: List = [
    view,
    rg,
    create,
    str_replace,
    strs_replace,
    insert,
    replace_lines,
    file_insert_line,
    file_del_lines,
    pyrun,
]
