"""
DialogHelper Service - Core logic for cell/message operations.

This module provides the shared logic used by both:
- HTTP endpoints (for dialoghelper library compatibility)
- LLM context building (reuses the same functions)

The dialoghelper library (https://github.com/AnswerDotAI/dialoghelper) allows
programmatic manipulation of notebook cells from within notebook code. This
service implements the server-side logic that dialoghelper's call_endp() calls.
"""
import re
from typing import List, Dict, Optional, Tuple, Any

MAX_CONTEXT_CELLS = 25

# ============================================================================
# Core Cell Query Functions (reused by endpoints AND context building)
# ============================================================================

def get_msg_idx(notebook, msgid: str) -> int:
    """
    Find cell index by ID. Returns -1 if not found.

    Used by:
    - msg_idx_ endpoint
    - All other endpoints that need to locate a cell by ID
    - build_context_messages()

    Args:
        notebook: Notebook object with cells list
        msgid: Cell ID to find

    Returns:
        Index of the cell, or -1 if not found
    """
    for i, c in enumerate(notebook.cells):
        if c.id == msgid:
            return i
    return -1


def find_msgs(
    notebook,
    re_pattern: str = "",
    msg_type: str = "",
    pinned_only: bool = False,
    skipped: Optional[bool] = None,  # None=include all, True=only skipped, False=only non-skipped
    limit: int = 100,
    before_idx: Optional[int] = None  # Only include cells before this index
) -> List[Tuple[int, Any]]:
    """
    Search cells by pattern, type, or properties.
    Returns list of (index, cell) tuples.

    Used by:
    - find_msgs_ endpoint
    - build_context_messages (to find pinned cells and window cells)

    Args:
        notebook: Notebook object with cells list
        re_pattern: Regex pattern to match against cell source
        msg_type: Filter by cell type (code, note, prompt)
        pinned_only: If True, only return pinned cells
        skipped: None=all, True=only skipped, False=only non-skipped
        limit: Maximum number of results
        before_idx: Only include cells before this index

    Returns:
        List of (index, cell) tuples matching the criteria
    """
    results = []
    cells = notebook.cells[:before_idx] if before_idx is not None else notebook.cells

    for i, c in enumerate(cells):
        # Filter by type
        if msg_type and c.cell_type != msg_type:
            continue
        # Filter by pattern
        if re_pattern and not re.search(re_pattern, c.source):
            continue
        # Filter by pinned
        if pinned_only and not c.pinned:
            continue
        # Filter by skipped
        if skipped is not None and c.skipped != skipped:
            continue

        results.append((i, c))
        if len(results) >= limit:
            break
    return results


def read_msg(
    notebook,
    n: int = 0,
    relative: bool = True,
    msgid: str = "",
    current_idx: int = 0,
    view_range: str = "",
    nums: bool = False
) -> Dict:
    """
    Read cell content by index or ID.

    Used by:
    - read_msg_ endpoint

    Args:
        notebook: Notebook object with cells list
        n: Offset (if relative) or absolute index (if not relative)
        relative: If True, n is relative to current_idx
        msgid: Find by ID instead of index (takes precedence)
        current_idx: Reference index for relative lookups
        view_range: Line range like "1:10" to extract subset
        nums: Include line numbers in output

    Returns:
        Dict with: id, idx, type, source, output, pinned, skipped
        Or dict with error key if not found
    """
    # Find target cell
    if msgid:
        idx = get_msg_idx(notebook, msgid)
        if idx == -1:
            return {"error": f"Message {msgid} not found"}
    elif relative:
        idx = current_idx + n
    else:
        idx = n

    if idx < 0 or idx >= len(notebook.cells):
        return {"error": f"Index {idx} out of range"}

    cell = notebook.cells[idx]
    content = cell.source

    # Apply view_range
    if view_range:
        lines = content.split('\n')
        parts = view_range.split(':')
        start = int(parts[0]) - 1 if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(lines)
        lines = lines[start:end]
        if nums:
            lines = [f"{i+start+1}: {line}" for i, line in enumerate(lines)]
        content = '\n'.join(lines)
    elif nums:
        lines = content.split('\n')
        content = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))

    return {
        "id": cell.id,
        "idx": idx,
        "type": cell.cell_type,
        "source": content,
        "output": cell.output,
        "pinned": cell.pinned,
        "skipped": cell.skipped
    }


def get_cells_before(notebook, msgid: str) -> List[Any]:
    """
    Get all cells before the given message ID.

    Args:
        notebook: Notebook object with cells list
        msgid: Cell ID to find

    Returns:
        List of cells before the specified cell
    """
    idx = get_msg_idx(notebook, msgid)
    if idx == -1:
        return []
    return notebook.cells[:idx]


def cell_to_dict(cell) -> Dict:
    """
    Convert cell to dictionary for JSON serialization.

    Used by:
    - curr_dialog_ endpoint (with_messages=True)

    Args:
        cell: Cell object

    Returns:
        Dictionary representation of the cell
    """
    return {
        "id": cell.id,
        "type": cell.cell_type,
        "source": cell.source,
        "output": cell.output,
        "pinned": cell.pinned,
        "skipped": cell.skipped,
        "collapsed": cell.collapsed,
        "input_collapse": cell.input_collapse,
        "output_collapse": cell.output_collapse,
        "execution_count": cell.execution_count,
        "time_run": cell.time_run
    }


# ============================================================================
# LLM Context Building (leverages the functions above)
# ============================================================================

def build_context_messages(notebook, current_cell_id: str) -> List[Dict]:
    """
    Build LLM context messages using dialoghelper functions.

    Strategy:
    1. Use find_msgs() to get pinned cells (always included first)
    2. Use find_msgs() to get the window of recent non-pinned cells
    3. Combine up to MAX_CONTEXT_CELLS total (pinned count towards limit)
    4. Convert to claudette-agent message format

    Args:
        notebook: Notebook object with cells list
        current_cell_id: ID of the current prompt cell being executed

    Returns:
        List of message dicts in claudette-agent format:
        [{"role": "user"/"assistant", "content": "..."}]
    """
    current_idx = get_msg_idx(notebook, current_cell_id)
    if current_idx == -1:
        return []

    # 1. Find pinned cells before current (using find_msgs)
    pinned_results = find_msgs(
        notebook,
        pinned_only=True,
        skipped=False,  # Exclude skipped cells
        before_idx=current_idx
    )
    pinned_cells = [cell for _, cell in pinned_results]

    # 2. Find non-pinned, non-skipped cells before current
    non_pinned_results = find_msgs(
        notebook,
        pinned_only=False,
        skipped=False,
        before_idx=current_idx,
        limit=1000  # Get all, we'll slice later
    )
    # Filter out pinned cells (already included)
    non_pinned_cells = [cell for _, cell in non_pinned_results if not cell.pinned]

    # 3. Calculate window size (pinned cells count towards the 25 limit)
    remaining_slots = MAX_CONTEXT_CELLS - len(pinned_cells)
    window_cells = non_pinned_cells[-remaining_slots:] if remaining_slots > 0 else []

    # 4. Combine: pinned first, then window
    all_cells = pinned_cells + window_cells

    # 5. Convert to messages
    messages = []
    for cell in all_cells:
        messages.extend(cell_to_messages(cell))

    return messages


def cell_to_messages(cell) -> List[Dict]:
    """
    Convert a cell to claudette-agent message format.

    Cell type mapping:
    - code: User message with python code block + output
    - note: User message with markdown content
    - prompt: User message (source) + Assistant message (output)

    Args:
        cell: Cell object

    Returns:
        List of message dicts with "role" and "content" keys
    """
    if cell.cell_type == "code":
        content = f"```python\n{cell.source}\n```"
        if cell.output:
            content += f"\nOutput:\n```\n{cell.output}\n```"
        return [{"role": "user", "content": content}]
    elif cell.cell_type == "note":
        return [{"role": "user", "content": cell.source}]
    elif cell.cell_type == "prompt":
        msgs = [{"role": "user", "content": cell.source}]
        if cell.output:
            msgs.append({"role": "assistant", "content": cell.output})
        return msgs
    return []
