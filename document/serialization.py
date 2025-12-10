"""
Notebook serialization to/from .ipynb format.

Uses execnb.nbio for Jupyter notebook I/O, with custom handling
for Solveit-specific features (prompt cells, dialog mode, etc.).
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import uuid

from execnb.nbio import read_nb, write_nb, new_nb

from .cell import Cell, CellType, CellOutput, CollapseLevel
from .notebook import Notebook


# Solveit separator token for prompt cells (user prompt + AI response)
SOLVEIT_SEPARATOR = "##### Reply<!-- SOLVEIT_SEPARATOR_{id} -->"


def _cell_to_jupyter(cell: Cell) -> dict:
    """
    Convert internal Cell to Jupyter cell format.

    - CODE cells -> 'code' cells
    - NOTE cells -> 'markdown' cells
    - PROMPT cells -> 'markdown' cells with solveit_ai metadata
    """
    if cell.cell_type == CellType.CODE:
        # Code cell with outputs
        outputs = []
        for out in cell.outputs:
            if out.output_type == 'stream':
                outputs.append({
                    'output_type': 'stream',
                    'name': out.stream_name or 'stdout',
                    'text': [out.content] if isinstance(out.content, str) else out.content
                })
            elif out.output_type == 'execute_result':
                outputs.append({
                    'output_type': 'execute_result',
                    'data': {'text/plain': [str(out.content)]},
                    'metadata': out.metadata or {},
                    'execution_count': cell.execution_count
                })
            elif out.output_type == 'display_data':
                outputs.append({
                    'output_type': 'display_data',
                    'data': out.content if isinstance(out.content, dict) else {'text/plain': [str(out.content)]},
                    'metadata': out.metadata or {}
                })
            elif out.output_type == 'error':
                outputs.append({
                    'output_type': 'error',
                    'ename': out.ename or 'Error',
                    'evalue': out.evalue or '',
                    'traceback': out.traceback or []
                })

        return {
            'cell_type': 'code',
            'source': cell.source,
            'metadata': {
                'id': cell.id,
                'skipped': cell.skipped,
                'pinned': cell.pinned,
                'is_exported': cell.is_exported,
                'collapsed': cell.collapsed,
                'input_collapse': cell.input_collapse.value,
                'output_collapse': cell.output_collapse.value,
            },
            'outputs': outputs,
            'execution_count': cell.execution_count
        }

    elif cell.cell_type == CellType.NOTE:
        return {
            'cell_type': 'markdown',
            'source': cell.source,
            'metadata': {
                'id': cell.id,
                'solveit_note': True,
                'collapsed': cell.collapsed,
            }
        }

    elif cell.cell_type == CellType.PROMPT:
        # Combine user prompt + AI response with separator
        separator = SOLVEIT_SEPARATOR.replace('{id}', cell.id)
        source = cell.source
        if cell.output:
            source = f"{cell.source}\n\n{separator}\n\n{cell.output}"

        return {
            'cell_type': 'markdown',
            'source': source,
            'metadata': {
                'id': cell.id,
                'solveit_ai': True,
                'use_thinking': cell.use_thinking,
                'collapsed': cell.collapsed,
                'pinned': cell.pinned,
            }
        }

    # Fallback
    return {
        'cell_type': 'code',
        'source': cell.source,
        'metadata': {'id': cell.id},
        'outputs': [],
        'execution_count': None
    }


def _jupyter_to_cell(jcell: dict, index: int = 0) -> Cell:
    """
    Convert Jupyter cell format to internal Cell.

    Detects cell type from metadata flags (solveit_ai, solveit_note).
    """
    cell_type_str = jcell.get('cell_type', 'code')
    metadata = jcell.get('metadata', {})
    source = jcell.get('source', '')

    # Handle source as list or string
    if isinstance(source, list):
        source = ''.join(source)

    # Determine cell type
    if cell_type_str == 'code':
        cell_type = CellType.CODE
    elif metadata.get('solveit_ai'):
        cell_type = CellType.PROMPT
    elif metadata.get('solveit_note'):
        cell_type = CellType.NOTE
    else:
        # Default markdown cells to NOTE
        cell_type = CellType.NOTE if cell_type_str == 'markdown' else CellType.CODE

    # Parse PROMPT cells: split user prompt from AI response
    output = ""
    if cell_type == CellType.PROMPT:
        separator_prefix = "##### Reply<!-- SOLVEIT_SEPARATOR_"
        if separator_prefix in source:
            parts = source.split(separator_prefix, 1)
            source = parts[0].strip()
            if len(parts) > 1:
                # Find end of separator and extract response
                sep_end = parts[1].find('-->')
                if sep_end != -1:
                    output = parts[1][sep_end + 3:].strip()

    # Parse outputs for code cells
    outputs: List[CellOutput] = []
    if cell_type == CellType.CODE:
        for jout in jcell.get('outputs', []):
            out_type = jout.get('output_type', 'stream')

            if out_type == 'stream':
                text = jout.get('text', '')
                if isinstance(text, list):
                    text = ''.join(text)
                outputs.append(CellOutput(
                    output_type='stream',
                    content=text,
                    stream_name=jout.get('name', 'stdout')
                ))

            elif out_type == 'execute_result':
                data = jout.get('data', {})
                text = data.get('text/plain', '')
                if isinstance(text, list):
                    text = ''.join(text)
                outputs.append(CellOutput(
                    output_type='execute_result',
                    content=text,
                    metadata=jout.get('metadata')
                ))

            elif out_type == 'display_data':
                outputs.append(CellOutput(
                    output_type='display_data',
                    content=jout.get('data', {}),
                    metadata=jout.get('metadata')
                ))

            elif out_type == 'error':
                outputs.append(CellOutput(
                    output_type='error',
                    ename=jout.get('ename', 'Error'),
                    evalue=jout.get('evalue', ''),
                    traceback=jout.get('traceback', [])
                ))

    # Set output for PROMPT cells from parsed response
    if cell_type == CellType.PROMPT and output:
        outputs = [CellOutput(output_type='stream', content=output, stream_name='stdout')]

    # Create cell
    cell = Cell(
        id=metadata.get('id', f"_{uuid.uuid4().hex[:8]}"),
        cell_type=cell_type,
        source=source,
        outputs=outputs,
        execution_count=jcell.get('execution_count'),
        skipped=metadata.get('skipped', False),
        pinned=metadata.get('pinned', False),
        use_thinking=metadata.get('use_thinking', False),
        is_exported=metadata.get('is_exported', False),
        collapsed=metadata.get('collapsed', False),
        input_collapse=CollapseLevel(metadata.get('input_collapse', 0)),
        output_collapse=CollapseLevel(metadata.get('output_collapse', 0)),
    )

    return cell


def load_notebook(path: Path) -> Notebook:
    """
    Load a .ipynb file into a Notebook object.

    Uses execnb.nbio.read_nb for parsing, then converts cells
    to internal format.
    """
    path = Path(path)

    if not path.exists():
        # Create new notebook
        return Notebook(
            id=path.stem,
            title=path.stem,
            cells=[Cell(cell_type=CellType.CODE)],
            path=path
        )

    # Read using execnb
    nb_data = read_nb(path)

    # Convert cells
    cells = []
    for i, jcell in enumerate(nb_data.cells):
        cells.append(_jupyter_to_cell(jcell, i))

    # Extract metadata
    metadata = nb_data.metadata or {}

    return Notebook(
        id=path.stem,
        title=metadata.get('title', path.stem),
        cells=cells,
        path=path,
        dialog_mode=metadata.get('solveit_dialog_mode', 'learning'),
        solveit_ver=metadata.get('solveit_ver', 2)
    )


def save_notebook(notebook: Notebook, path: Optional[Path] = None):
    """
    Save a Notebook to .ipynb format.

    Uses standard Jupyter notebook format with Solveit extensions
    in cell metadata.
    """
    path = Path(path or notebook.path)

    # Convert cells to Jupyter format
    jupyter_cells = [_cell_to_jupyter(c) for c in notebook.cells]

    # Build notebook structure
    nb_dict = {
        'cells': jupyter_cells,
        'metadata': {
            'title': notebook.title,
            'solveit_dialog_mode': notebook.dialog_mode,
            'solveit_ver': notebook.solveit_ver,
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.13'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)

    # Update notebook path and modified flag
    notebook.path = path
    notebook.modified = False


def create_new_notebook(path: Path) -> Notebook:
    """Create a new empty notebook."""
    notebook = Notebook(
        id=path.stem,
        title=path.stem,
        cells=[Cell(cell_type=CellType.CODE)],
        path=path
    )
    save_notebook(notebook, path)
    return notebook
