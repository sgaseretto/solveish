"""
DialogHelper Service - Core logic for cell/message operations.

This module provides the shared logic used by both:
- HTTP endpoints (for dialoghelper library compatibility)
- LLM context building (reuses the same functions)

The dialoghelper library (https://github.com/AnswerDotAI/dialoghelper) allows
programmatic manipulation of notebook cells from within notebook code. This
service implements the server-side logic that dialoghelper's call_endp() calls.
"""
import os
import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from xml.sax.saxutils import escape as xml_escape

logger = logging.getLogger(__name__)

MAX_CONTEXT_CELLS = 25
MAX_TRUNC_LEN = 200  # Default truncation length for output/source

# Per-notebook clipboard storage for copy/paste operations
_clipboards: Dict[str, List[Dict]] = defaultdict(list)

# Per-notebook change logs for log_changed support
_change_logs: Dict[str, List[Dict]] = defaultdict(list)

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
    before_idx: Optional[int] = None,  # Only include cells before this index
    # New dialoghelper params
    use_case: bool = False,
    use_regex: bool = True,
    only_err: bool = False,
    only_exp: bool = False,
    only_chg: bool = False,
    ids: str = "",
    include_output: bool = True,
) -> List[Tuple[int, Any]]:
    """
    Search cells by pattern, type, or properties.
    Returns list of (index, cell) tuples.

    Used by:
    - find_msgs_ endpoint
    - build_context_messages (to find pinned cells and window cells)

    Args:
        notebook: Notebook object with cells list
        re_pattern: Regex/literal pattern to match against cell source
        msg_type: Filter by cell type (code, note, prompt, raw)
        pinned_only: If True, only return pinned cells
        skipped: None=all, True=only skipped, False=only non-skipped
        limit: Maximum number of results
        before_idx: Only include cells before this index
        use_case: If True, case-sensitive search
        use_regex: If True, use regex matching; otherwise literal
        only_err: If True, only cells with error outputs
        only_exp: If True, only exported cells
        only_chg: If True, only changed cells (version > 0)
        ids: Comma-separated cell IDs to filter by
        include_output: If True, also search in output text

    Returns:
        List of (index, cell) tuples matching the criteria
    """
    results = []
    cells = notebook.cells[:before_idx] if before_idx is not None else notebook.cells

    # Pre-parse ID filter
    id_set = {s.strip() for s in ids.split(',') if s.strip()} if ids else None

    # Compile search pattern
    re_flags = 0 if use_case else re.IGNORECASE
    compiled_pattern = None
    if re_pattern:
        if use_regex:
            try:
                compiled_pattern = re.compile(re_pattern, re_flags)
            except re.error:
                compiled_pattern = re.compile(re.escape(re_pattern), re_flags)
        else:
            compiled_pattern = re.compile(re.escape(re_pattern), re_flags)

    for i, c in enumerate(cells):
        # Filter by IDs
        if id_set and c.id not in id_set:
            continue
        # Filter by type
        if msg_type and c.cell_type != msg_type:
            continue
        # Filter by pattern (search source and optionally output)
        if compiled_pattern:
            source_match = compiled_pattern.search(c.source)
            output_match = include_output and c.output and compiled_pattern.search(c.output)
            if not source_match and not output_match:
                continue
        # Filter by pinned
        if pinned_only and not getattr(c, 'pinned', False):
            continue
        # Filter by skipped
        if skipped is not None and getattr(c, 'skipped', False) != skipped:
            continue
        # Filter by error
        if only_err:
            has_error = any(o.output_type == 'error' for o in getattr(c, 'outputs', []))
            if not has_error:
                continue
        # Filter by exported
        if only_exp and not getattr(c, 'is_exported', False):
            continue
        # Filter by changed
        if only_chg and getattr(c, 'version', 0) == 0:
            continue

        results.append((i, c))
        if limit and len(results) >= limit:
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

    # dialoghelper expects flat dict format - dict2obj converts this so
    # both result.content and result['content'] work
    ct = getattr(cell, 'cell_type', 'code')
    type_str = ct.value if hasattr(ct, 'value') else str(ct)
    return {
        "id": cell.id,
        "idx": idx,
        "type": type_str,
        "msg_type": type_str,
        "content": content,  # dialoghelper uses 'content', not 'source'
        "output": cell.output,
        "pinned": getattr(cell, 'pinned', False),
        "skipped": getattr(cell, 'skipped', False)
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
    ct = getattr(cell, 'cell_type', 'code')
    type_str = ct.value if hasattr(ct, 'value') else str(ct)
    ic = getattr(cell, 'input_collapse', 0)
    oc = getattr(cell, 'output_collapse', 0)
    return {
        "id": cell.id,
        "type": type_str,
        "msg_type": type_str,
        "source": cell.source,
        "output": cell.output,
        "pinned": getattr(cell, 'pinned', False),
        "skipped": getattr(cell, 'skipped', False),
        "collapsed": getattr(cell, 'collapsed', False),
        "input_collapse": ic.value if hasattr(ic, 'value') else int(ic),
        "output_collapse": oc.value if hasattr(oc, 'value') else int(oc),
        "heading_collapsed": getattr(cell, 'heading_collapsed', False),
        "bookmark": getattr(cell, 'bookmark', 0),
        "is_exported": getattr(cell, 'is_exported', False),
        "execution_count": getattr(cell, 'execution_count', None),
        "time_run": getattr(cell, 'time_run', None)
    }


# ============================================================================
# XML Formatting for find_msgs (as_xml=True)
# ============================================================================

def _truncate(text: str, max_len: int = MAX_TRUNC_LEN) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len] + "..."


def _add_line_nums(text: str) -> str:
    """Add line numbers to text."""
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))


def _extract_header_section(source: str, header: str) -> str:
    """Extract content under a specific markdown header."""
    lines = source.split('\n')
    in_section = False
    section_lines = []
    header_level = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            heading_text = stripped.lstrip('#').strip()
            if not in_section and header.lower() in heading_text.lower():
                in_section = True
                header_level = level
                section_lines.append(line)
                continue
            elif in_section and level <= header_level:
                break
        if in_section:
            section_lines.append(line)

    return '\n'.join(section_lines)


def _get_headers(source: str) -> str:
    """Extract only markdown header lines."""
    lines = source.split('\n')
    return '\n'.join(l for l in lines if l.lstrip().startswith('#'))


def format_msgs_as_xml(
    results: List[Tuple[int, Any]],
    include_output: bool = True,
    include_meta: bool = False,
    nums: bool = False,
    trunc_out: bool = True,
    trunc_in: bool = False,
    headers_only: bool = False,
    header_section: str = "",
) -> str:
    """
    Format find_msgs results as XML string.

    Args:
        results: List of (index, cell) tuples
        include_output: Include output in XML
        include_meta: Include metadata (pinned, skipped, etc.)
        nums: Add line numbers to source
        trunc_out: Truncate output text
        trunc_in: Truncate source text
        headers_only: Only include markdown headers from source
        header_section: Extract specific header section

    Returns:
        XML string representation
    """
    parts = ["<messages>"]

    for idx, cell in results:
        type_str = cell.cell_type.value if hasattr(cell.cell_type, 'value') else str(cell.cell_type)
        attrs = f'idx="{idx}" id="{xml_escape(cell.id)}" type="{xml_escape(type_str)}"'
        if include_meta:
            attrs += f' pinned="{getattr(cell, "pinned", False)}" skipped="{getattr(cell, "skipped", False)}"'
            attrs += f' exported="{getattr(cell, "is_exported", False)}" bookmark="{getattr(cell, "bookmark", 0)}"'
        parts.append(f"<msg {attrs}>")

        # Source content
        source = cell.source
        if header_section:
            source = _extract_header_section(source, header_section)
        elif headers_only:
            source = _get_headers(source)

        if nums:
            source = _add_line_nums(source)
        if trunc_in:
            source = _truncate(source)

        parts.append(f"<source>{xml_escape(source)}</source>")

        # Output
        if include_output and cell.output:
            output = _truncate(cell.output) if trunc_out else cell.output
            parts.append(f"<output>{xml_escape(output)}</output>")

        parts.append("</msg>")

    parts.append("</messages>")
    return '\n'.join(parts)


def format_msgs_as_json(
    results: List[Tuple[int, Any]],
    include_output: bool = True,
    include_meta: bool = False,
    nums: bool = False,
    trunc_out: bool = True,
    trunc_in: bool = False,
    headers_only: bool = False,
    header_section: str = "",
) -> List[Dict]:
    """
    Format find_msgs results as list of dicts (JSON-compatible).

    Args:
        results: List of (index, cell) tuples
        Same params as format_msgs_as_xml

    Returns:
        List of message dicts
    """
    msgs = []
    for idx, cell in results:
        source = cell.source
        if header_section:
            source = _extract_header_section(source, header_section)
        elif headers_only:
            source = _get_headers(source)

        if nums:
            source = _add_line_nums(source)
        if trunc_in:
            source = _truncate(source)

        type_str = cell.cell_type.value if hasattr(cell.cell_type, 'value') else str(cell.cell_type)
        msg = {
            "idx": idx,
            "id": cell.id,
            "type": type_str,
            "msg_type": type_str,
            "content": source,
        }

        if include_output:
            output = cell.output or ""
            if trunc_out:
                output = _truncate(output)
            msg["output"] = output

        if include_meta:
            msg["pinned"] = getattr(cell, 'pinned', False)
            msg["skipped"] = getattr(cell, 'skipped', False)
            msg["is_exported"] = getattr(cell, 'is_exported', False)
            msg["bookmark"] = getattr(cell, 'bookmark', 0)

        msgs.append(msg)

    return msgs


# ============================================================================
# Clipboard Operations
# ============================================================================

def clipboard_copy(notebook, notebook_id: str, cell_ids: List[str], cut: bool = False) -> Dict:
    """
    Copy (or cut) cells to the clipboard.

    Args:
        notebook: Notebook object
        notebook_id: Notebook identifier for clipboard storage
        cell_ids: List of cell IDs to copy
        cut: If True, also remove cells from notebook

    Returns:
        Dict with status and count
    """
    copied = []
    for cid in cell_ids:
        idx = get_msg_idx(notebook, cid)
        if idx >= 0:
            cell = notebook.cells[idx]
            copied.append(cell_to_dict(cell))

    _clipboards[notebook_id] = copied

    if cut:
        # Remove cells in reverse order to maintain indices
        for cid in reversed(cell_ids):
            idx = get_msg_idx(notebook, cid)
            if idx >= 0:
                notebook.cells.pop(idx)

    return {"status": "ok", "count": len(copied), "cmd": "cut" if cut else "copy"}


def clipboard_paste(notebook, notebook_id: str, ref_id: str = "", after: bool = True) -> List[str]:
    """
    Paste cells from clipboard into notebook.

    Args:
        notebook: Notebook object
        notebook_id: Notebook identifier for clipboard storage
        ref_id: Reference cell ID for insertion point
        after: If True, paste after ref_id; otherwise before

    Returns:
        List of new cell IDs
    """
    from dialeng.document.cell import Cell

    clipboard = _clipboards.get(notebook_id, [])
    if not clipboard:
        return []

    # Find insertion point
    if ref_id:
        ref_idx = get_msg_idx(notebook, ref_id)
        if ref_idx == -1:
            ref_idx = len(notebook.cells) - 1
        insert_idx = ref_idx + 1 if after else ref_idx
    else:
        insert_idx = len(notebook.cells)

    new_ids = []
    for i, cell_data in enumerate(clipboard):
        new_cell = Cell(
            cell_type=cell_data.get("type", "code"),
            source=cell_data.get("source", ""),
            pinned=cell_data.get("pinned", False),
            skipped=cell_data.get("skipped", False),
            is_exported=cell_data.get("is_exported", False),
        )
        if cell_data.get("output"):
            new_cell.output = cell_data["output"]
        notebook.cells.insert(insert_idx + i, new_cell)
        new_ids.append(new_cell.id)

    return new_ids


# ============================================================================
# Change Logging (for log_changed support)
# ============================================================================

def log_change(notebook_id: str, action: str, cell_id: str, details: Dict = None):
    """
    Log a change for audit trail.

    Args:
        notebook_id: Notebook identifier
        action: Type of change (update, delete, etc.)
        cell_id: ID of affected cell
        details: Additional details about the change
    """
    from datetime import datetime
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "cell_id": cell_id,
        "details": details or {}
    }
    _change_logs[notebook_id].append(entry)
    logger.info(f"Change logged: {action} on {cell_id} in {notebook_id}")


def get_change_log(notebook_id: str) -> List[Dict]:
    """Get the change log for a notebook."""
    return _change_logs.get(notebook_id, [])


# ============================================================================
# LLM Context Building (leverages the functions above)
# ============================================================================

def build_context_messages(notebook, current_cell_id: str) -> List[Dict]:
    """
    Build LLM context messages using dialoghelper functions.

    Strategy:
    1. Use find_msgs() to get pinned cells (always included)
    2. Use find_msgs() to get the window of recent non-pinned cells
    3. Combine up to MAX_CONTEXT_CELLS total (pinned count towards limit)
    4. Sort all cells by original index to maintain chronological order
    5. Convert to claudette-agent message format

    Args:
        notebook: Notebook object with cells list
        current_cell_id: ID of the current prompt cell being executed

    Returns:
        List of message dicts in claudette-agent format:
        [{"role": "user"/"assistant", "content": "..."}]
    """
    logger.info(f"build_context_messages: Building context for cell {current_cell_id}")

    current_idx = get_msg_idx(notebook, current_cell_id)
    if current_idx == -1:
        logger.warning(f"build_context_messages: Cell {current_cell_id} not found")
        return []

    logger.info(f"build_context_messages: Current cell is at index {current_idx}")

    # 1. Find pinned cells before current (using find_msgs)
    # Keep (index, cell) tuples to preserve order information
    pinned_results = find_msgs(
        notebook,
        pinned_only=True,
        skipped=False,  # Exclude skipped cells
        before_idx=current_idx
    )
    pinned_indices = {idx for idx, _ in pinned_results}
    logger.info(f"build_context_messages: Found {len(pinned_results)} pinned cells")

    # 2. Find non-pinned, non-skipped cells before current
    non_pinned_results = find_msgs(
        notebook,
        pinned_only=False,
        skipped=False,
        before_idx=current_idx,
        limit=1000  # Get all, we'll slice later
    )
    # Filter out pinned cells (already included) - keep (index, cell) tuples
    non_pinned_tuples = [(idx, cell) for idx, cell in non_pinned_results if idx not in pinned_indices]
    logger.info(f"build_context_messages: Found {len(non_pinned_tuples)} non-pinned cells")

    # 3. Calculate window size (pinned cells count towards the 25 limit)
    remaining_slots = MAX_CONTEXT_CELLS - len(pinned_results)
    window_tuples = non_pinned_tuples[-remaining_slots:] if remaining_slots > 0 else []

    # 4. Combine and sort by index to maintain chronological order
    # This is the key fix: cells must appear in notebook order, not pinned-first
    all_tuples = list(pinned_results) + window_tuples
    all_tuples.sort(key=lambda x: x[0])  # Sort by index

    logger.info(f"build_context_messages: Total {len(all_tuples)} cells in context")

    # 5. Convert to messages (in chronological order)
    messages = []
    for idx, cell in all_tuples:
        cell_messages = cell_to_messages(cell)
        messages.extend(cell_messages)
        # Log each cell being included
        logger.info(f"build_context_messages: Cell[{idx}] type={cell.cell_type} id={cell.id}")
        logger.info(f"  source: {cell.source[:80]}..." if len(cell.source) > 80 else f"  source: {cell.source}")
        logger.info(f"  output: {cell.output[:80]}..." if cell.output and len(cell.output) > 80 else f"  output: {cell.output}")
        logger.info(f"  -> {len(cell_messages)} messages")

    # 6. Prepend CRAFT.ipynb context (note/prompt cells from CRAFT files)
    notebook_path = getattr(notebook, 'path', None)
    if notebook_path:
        try:
            from dialeng.services.craft_service import find_craft_files, get_craft_context
            from pathlib import Path
            # Use NOTEBOOKS_DIR as root; fall back to parent of notebook
            root = Path(os.environ.get("DIALENG_NOTEBOOKS_DIR", "notebooks"))
            craft_paths = find_craft_files(notebook_path, root)
            if craft_paths:
                craft_messages = get_craft_context(craft_paths)
                if craft_messages:
                    logger.info(f"build_context_messages: Prepending {len(craft_messages)} CRAFT messages")
                    messages = craft_messages + messages
        except Exception as e:
            logger.error(f"build_context_messages: CRAFT context error: {e}")

    logger.info(f"build_context_messages: Final context has {len(messages)} messages")
    return messages


def cell_to_messages(cell) -> List[Dict]:
    """
    Convert a cell to claudette-agent message format.

    Uses the extensible dispatch system from dialeng.core.dispatch.
    Extensions can register custom converters for new cell types.

    Cell type mapping (defaults):
    - code: User message with python code block + output
    - note: User message with markdown content
    - prompt: User message (source) + Assistant message (output)

    Args:
        cell: Cell object

    Returns:
        List of message dicts with "role" and "content" keys
    """
    from dialeng.core.dispatch import cell_to_llm_messages
    return cell_to_llm_messages(cell)
