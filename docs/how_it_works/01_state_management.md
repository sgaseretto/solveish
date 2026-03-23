# State Management - Technical Documentation

This document explains how Dialeng manages notebook and cell state, including the in-memory data structures, persistence to disk, and state synchronization across the application.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Structures](#data-structures)
4. [State Lifecycle](#state-lifecycle)
5. [CRUD Operations](#crud-operations)
6. [Persistence](#persistence)
7. [State Synchronization](#state-synchronization)
8. [Key Files and Functions](#key-files-and-functions)
9. [Extending the State System](#extending-the-state-system)

---

## Overview

Dialeng uses a simple but effective state management approach:

- **In-memory dictionary** holds all active notebooks
- **Lazy loading** - notebooks are loaded from disk on first access
- **Manual save** - changes persist to `.ipynb` files when user clicks Save (Ctrl+S)
- **Direct mutation** - routes modify the in-memory objects directly
- **HTMX re-rendering** - UI updates by re-rendering FastHTML components

```mermaid
flowchart LR
    subgraph Memory["In-Memory State"]
        notebooks["notebooks: Dict[str, Notebook]"]
        nb1["Notebook 'demo'"]
        nb2["Notebook 'project'"]
        notebooks --> nb1
        notebooks --> nb2
    end

    subgraph Disk["File System"]
        dir["notebooks/"]
        f1["demo.ipynb"]
        f2["project.ipynb"]
        dir --> f1
        dir --> f2
    end

    Memory <-->|"load/save"| Disk
```

---

## Architecture

### State Flow

```mermaid
flowchart TB
    subgraph UserAction["User Action"]
        click["Click Run/Add/Delete"]
        type["Type in Cell"]
        save["Click Save"]
    end

    subgraph HTMX["HTMX Layer"]
        post["POST /dialeng/{id}/cell/..."]
    end

    subgraph Route["FastHTML Route"]
        getNb["get_notebook(id)"]
        mutate["Mutate notebook/cell"]
        render["Return HTML component"]
    end

    subgraph State["State Layer"]
        dict["notebooks dict"]
        nb["Notebook object"]
        cells["Cell objects"]
    end

    subgraph Persistence["Persistence Layer"]
        saveNb["save_notebook(id)"]
        ipynb["*.ipynb file"]
    end

    click --> post
    type --> post
    post --> getNb
    getNb --> dict
    dict --> nb
    nb --> cells
    mutate --> cells
    render --> HTMX

    save --> saveNb
    saveNb --> ipynb
```

### Key Principles

1. **Single Source of Truth**: The `notebooks` dictionary is the authoritative state
2. **Lazy Loading**: Notebooks are only loaded when first accessed
3. **Optimistic Updates**: UI updates immediately, save is explicit
4. **No ORM**: Direct JSON serialization to `.ipynb` format

---

## Data Structures

### Global State (`app.py:348-356`)

```python
# In-memory notebook storage
notebooks: Dict[str, Notebook] = {}

# Notebook files directory
NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# WebSocket connections (for collaboration)
ws_connections: Dict[str, List[Any]] = {}

# Track cancelled cell generations
cancelled_cells: set = set()
```

### Notebook Class (`app.py:205-242`)

```python
@dataclass
class Notebook:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "Untitled Notebook"
    cells: List[Cell] = field(default_factory=list)
    dialog_mode: str = "learning"  # Solveit compatibility

    def to_ipynb(self) -> Dict[str, Any]:
        """Convert to Jupyter .ipynb format"""
        return {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", ...},
                "solveit_dialog_mode": self.dialog_mode,
                "solveit_ver": SOLVEIT_VER
            },
            "cells": [cell.to_jupyter_cell() for cell in self.cells]
        }

    @classmethod
    def from_ipynb(cls, data: Dict[str, Any], notebook_id: str) -> "Notebook":
        """Load from Jupyter .ipynb format"""
        metadata = data.get("metadata", {})
        cells = [Cell.from_jupyter_cell(c) for c in data.get("cells", [])]
        return cls(
            id=notebook_id,
            title="Imported Notebook",
            cells=cells,
            dialog_mode=metadata.get("solveit_dialog_mode", "learning")
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_ipynb(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Notebook":
        with open(path) as f:
            data = json.load(f)
        nb_id = Path(path).stem
        nb = cls.from_ipynb(data, nb_id)
        nb.title = Path(path).stem
        return nb
```

### Cell Class (`document/cell.py`)

See [02_cell_types.md](02_cell_types.md) for complete Cell documentation. The unified `Cell` dataclass lives in `document/cell.py` and is imported by both `app.py` and the kernel execution layer.

---

## State Lifecycle

### 1. Application Startup

```mermaid
sequenceDiagram
    participant App as FastHTML App
    participant Dir as notebooks/
    participant Dict as notebooks dict

    App->>Dir: mkdir(exist_ok=True)
    Note over Dict: Empty dictionary {}
    Note over App: Ready to serve requests
```

### 2. First Notebook Access

```mermaid
sequenceDiagram
    participant User
    participant Route as GET /dialeng/{id}
    participant Fn as get_notebook()
    participant Dict as notebooks dict
    participant Disk as notebooks/*.ipynb

    User->>Route: Visit /dialeng/demo
    Route->>Fn: get_notebook("demo")
    Fn->>Dict: Check if "demo" exists
    alt Not in memory
        Fn->>Disk: Check demo.ipynb exists
        alt File exists
            Disk-->>Fn: Load JSON
            Fn->>Dict: Store Notebook
        else File doesn't exist
            Fn->>Dict: Create new Notebook with default cells
        end
    end
    Dict-->>Fn: Return Notebook
    Fn-->>Route: Return Notebook
    Route-->>User: Render HTML
```

### 3. Cell Modification

```mermaid
sequenceDiagram
    participant User
    participant Route as POST /cell/{id}/source
    participant Dict as notebooks dict
    participant WS as WebSocket

    User->>Route: Type in cell (blur or run triggers POST)
    Route->>Dict: get_notebook(nb_id)
    Route->>Dict: Find cell, commit source, bump version
    Route->>WS: Broadcast committed cell state
    Route-->>User: Empty response (hx-swap="none")
    Note over Dict: State updated in memory and marks notebook modified
    Note over User: Initiating tab keeps local editor DOM; other tabs receive canonical committed state
```

### 4. Save to Disk

```mermaid
sequenceDiagram
    participant User
    participant Route as POST /dialeng/{id}/save
    participant Fn as save_notebook()
    participant Nb as Notebook
    participant Disk as demo.ipynb

    User->>Route: Ctrl+S or click Save
    Route->>Fn: save_notebook("demo")
    Fn->>Nb: Get from dict
    Nb->>Nb: to_ipynb()
    Nb->>Disk: Write JSON
    Route-->>User: "✓ Saved" status
```

---

## CRUD Operations

### Create Cell

```python
@rt("/dialeng/{nb_id}/cell/add")
async def post(nb_id: str, pos: int = -1, type: str = "code"):
    nb = get_notebook(nb_id)
    if pos < 0:
        pos = len(nb.cells)

    # Code cells default to scrollable output
    if type == "code":
        nb.cells.insert(pos, Cell(cell_type=type, output_collapse=1))
    else:
        nb.cells.insert(pos, Cell(cell_type=type))

    nb.modified = True

    # Broadcast backend-authoritative structure payload
    await broadcast_json(nb_id, _cell_add_payload(nb, nb_id, new_cell.id))
    await _mark_outline_dirty(nb_id, reason="cell_add")

    return ""
```

**Key points:**
- `pos=-1` means append to end
- Code cells default to `output_collapse=1` (scrollable)
- Returns an empty HTTP response; structure updates are applied from the WebSocket payload
- The payload includes canonical `ordered_cell_ids`, so the browser rebuilds `#cells` from backend order

### Read Cell

Cells are read implicitly when rendering:

```python
def CellView(cell: Cell, notebook_id: str):
    """Render a single cell to HTML"""
    # Reads cell.source, cell.output, cell.cell_type, etc.
    # Returns FastHTML component
```

### Update Cell Source

```python
@rt("/dialeng/{nb_id}/cell/{cid}/source")
async def post(nb_id: str, cid: str, source: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            changed = c.update_source(source)
            if not changed:
                break
            nb.modified = True

            if c.cell_type in {"code", "shell"}:
                await broadcast_json(nb_id, {
                    "type": "cell_source_update",
                    "cell_id": c.id,
                    "source": c.source,
                    "version": c.version,
                })
                await broadcast_to_notebook(nb_id, CellOutputOOB(c))
                await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
            else:
                await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))

            if c.cell_type == "note":
                await _mark_outline_dirty(nb_id, reason="note_source_commit")
            break
    return ""
```

**Key points:**
- Triggered on blur with `hx-trigger="blur changed"` and also when running a cell with fresher in-editor source
- Only committed state is synchronized; keystroke-by-keystroke typing is still local
- Code/shell source commits clear outputs and execution metadata via `CellOutputOOB` + `CellHeaderOOB`
- Note/prompt commits re-render the full cell so rendered markdown stays authoritative across tabs

### Update Cell Output

```python
@rt("/dialeng/{nb_id}/cell/{cid}/output")
async def post(nb_id: str, cid: str, output: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            changed = c.update_output(output)
            if not changed:
                break
            nb.modified = True
            if c.cell_type in {"code", "shell"}:
                await broadcast_to_notebook(nb_id, CellOutputOOB(c))
                await broadcast_to_notebook(nb_id, CellHeaderOOB(c, nb_id))
            else:
                await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
            break
    return ""
```

**Use case:** Editing AI response in prompt cells (double-click to edit).

### Delete Cell

```python
@rt("/dialeng/{nb_id}/cell/{cid}")
async def delete(nb_id: str, cid: str):
    nb = get_notebook(nb_id)
    nb.cells = [c for c in nb.cells if c.id != cid]
    nb.modified = True
    await broadcast_json(nb_id, _cell_delete_payload(nb, cid))
    await _mark_outline_dirty(nb_id, reason="cell_delete")
    return ""
```

### Move Cell

```python
@rt("/dialeng/{nb_id}/cell/{cid}/move/{direction}")
async def post(nb_id: str, cid: str, direction: str):
    nb = get_notebook(nb_id)
    moved = nb.move_cell(cid, {"up": -1, "down": 1}[direction])
    if not moved:
        return ""

    await broadcast_json(nb_id, _cell_move_payload(nb, cid))
    await _mark_outline_dirty(nb_id, reason="cell_move")
    return ""
```

### Change Cell Type

```python
@rt("/dialeng/{nb_id}/cell/{cid}/type")
async def post(nb_id: str, cid: str, cell_type: str):
    nb = get_notebook(nb_id)
    for c in nb.cells:
        if c.id == cid:
            c.cell_type = cell_type
            c.clear_outputs()
            c.version += 1
            c.last_modified = datetime.now()
            nb.modified = True
            await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
            await _mark_outline_dirty(nb_id, reason="cell_type_change")
            return ""
    return ""
```

---

## Persistence

### File Format

Notebooks are stored as standard Jupyter `.ipynb` files in the `notebooks/` directory:

```
notebooks/
├── demo.ipynb
├── project.ipynb
└── tutorial.ipynb
```

### Notebook Identity

Dialeng uses an encoded notebook id as the canonical notebook key instead of the raw
relative path. The shared helpers live in [`dialeng/notebook_id.py`](../../dialeng/notebook_id.py).

- `demo` → `demo`
- `demo_project/colab_test` → `demo_project~colab_test`
- literal `~` characters inside path segments are escaped as `~~`

That encoded id is the single source of truth for:

- `/dialeng/{nb_id}` route parameters
- in-memory notebook and kernel dictionaries
- file explorer active/running state matching for both root and nested notebooks

### Save Operation (`app.py:374-377`)

```python
def save_notebook(notebook_id: str):
    if notebook_id in notebooks:
        path = NOTEBOOKS_DIR / f"{notebook_id}.ipynb"
        notebooks[notebook_id].save(str(path))
```

### Load Operation (`app.py:358-372`)

```python
def get_notebook(notebook_id: str) -> Notebook:
    """Get or create a notebook - ALWAYS requires notebook_id"""
    if notebook_id not in notebooks:
        path = NOTEBOOKS_DIR / f"{notebook_id}.ipynb"
        if path.exists():
            # Load from disk
            notebooks[notebook_id] = Notebook.load(str(path))
        else:
            # Create new with default cells
            nb = Notebook(id=notebook_id, title=notebook_id)
            nb.cells = [
                Cell(cell_type="note", source="# Welcome to Dialeng..."),
                Cell(cell_type="code", source="x = [1, 2, 3]...", output_collapse=1),
                Cell(cell_type="prompt", source="Hello! What can you help me with?"),
            ]
            notebooks[notebook_id] = nb
    return notebooks[notebook_id]
```

### List Notebooks (`app.py:379-380`)

```python
def list_notebooks() -> List[str]:
    return [p.stem for p in NOTEBOOKS_DIR.glob("*.ipynb")]
```

---

## State Synchronization

### What Gets Synchronized

| Action | In-Memory | To Disk | To Collaborators |
|--------|-----------|---------|------------------|
| Type in cell | Local draft immediately; committed on blur or run | Manual save | **Yes** on commit |
| Run cell | Immediate | Manual save | **Yes** (OOB swap) |
| Add cell | Immediate | Manual save | **Yes** (ordered structure payload) |
| Delete cell | Immediate | Manual save | **Yes** (ordered structure payload) |
| Move cell | Immediate | Manual save | **Yes** (ordered structure payload) |
| Change type | Immediate | Manual save | **Yes** (OOB swap) |
| Collapse | Immediate | Manual save | **Yes** (OOB swap) |
| Mode / model / safe mode | Immediate | Manual save | **Yes** (kernel snapshot) |
| Save | N/A | Immediate | **No** |

### Broadcast Pattern

Routes that modify shared notebook state now follow one of three backend-authoritative patterns:

```python
@rt("/dialeng/{nb_id}/cell/{cid}/some-action")
async def post(nb_id: str, cid: str, ...):
    nb = get_notebook(nb_id)

    # 1. Find and modify the cell
    for c in nb.cells:
        if c.id == cid:
            # Modify state
            break

    # 2. Broadcast canonical state from the backend
    await broadcast_json(nb_id, {...})          # in-place cell or structure updates
    await broadcast_to_notebook(nb_id, ...)     # OOB fragments when full section HTML is needed
    await broadcast_kernel_snapshot(nb_id)      # notebook/kernel toolbar state

    # 3. Return an empty HTTP response when the browser should rely on the broadcast
    return ""
```

In practice:

- Structural operations (`add`, `delete`, `move`) broadcast `ordered_cell_ids` and the browser reconciles the whole `#cells` structure from backend order.
- Committed content changes (`source`, `output`) broadcast the committed cell state so other tabs converge on the backend version instead of trusting local draft DOM.
- Notebook-level controls (`mode`, `model`, `safe_mode`, kernel selection/auth/setup state) are synchronized through the kernel snapshot instead of ad hoc client-side mutation.

---

## Key Files and Functions

| Location | Purpose |
|----------|---------|
| `app.py:348` | `notebooks: Dict[str, Notebook]` - global state |
| `app.py:349-350` | `NOTEBOOKS_DIR` - file storage path |
| `app.py:358-372` | `get_notebook()` - lazy load/create |
| `app.py:374-377` | `save_notebook()` - persist to disk |
| `app.py:379-380` | `list_notebooks()` - enumerate saved |
| `app.py:205-242` | `Notebook` class - data model |
| `app.py:71-165` | `Cell` class - data model |
| `app.py:2234-2248` | Add cell route |
| `app.py:2250-2258` | Delete cell route |
| `app.py:2260-2276` | Update source/output routes |
| `app.py:2293-2307` | Move cell route |
| `app.py:2309-2346` | Collapse routes |

---

## Extending the State System

### Adding a New Cell Field

1. **Add to Cell dataclass** (`document/cell.py`):
   ```python
   @dataclass
   class Cell:
       # ... existing fields ...
       new_field: str = ""
   ```

2. **Add serialization** in `to_jupyter_cell()` (`document/cell.py`) and `_cell_to_jupyter()` (`document/serialization.py`):
   ```python
   if self.new_field:
       cell["metadata"]["new_field"] = self.new_field
   ```

3. **Add deserialization** in `from_jupyter_cell()` / `_jupyter_to_cell()` (`document/serialization.py`):
   ```python
   new_field=metadata.get("new_field", "")
   ```

4. **Add route if needed**:
   ```python
   @rt("/dialeng/{nb_id}/cell/{cid}/new-field")
   async def post(nb_id: str, cid: str, value: str):
       nb = get_notebook(nb_id)
       for c in nb.cells:
           if c.id == cid:
               c.new_field = value
               await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
               return CellView(c, nb_id)
   ```

### Adding a New Notebook Field

1. **Add to Notebook dataclass** (`app.py:205`):
   ```python
   @dataclass
   class Notebook:
       # ... existing fields ...
       new_setting: str = "default"
   ```

2. **Add to `to_ipynb()`** (`app.py:211`):
   ```python
   "metadata": {
       # ... existing metadata ...
       "new_setting": self.new_setting
   }
   ```

3. **Add to `from_ipynb()`** (`app.py:224`):
   ```python
   return cls(
       # ... existing fields ...
       new_setting=metadata.get("new_setting", "default")
   )
   ```

### Alternative Storage Backends

To add a different storage backend (e.g., database, cloud):

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    @abstractmethod
    def save(self, notebook_id: str, notebook: Notebook): ...

    @abstractmethod
    def load(self, notebook_id: str) -> Notebook: ...

    @abstractmethod
    def list(self) -> List[str]: ...

    @abstractmethod
    def exists(self, notebook_id: str) -> bool: ...

class FileStorage(StorageBackend):
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, notebook_id: str, notebook: Notebook):
        path = self.directory / f"{notebook_id}.ipynb"
        notebook.save(str(path))

    def load(self, notebook_id: str) -> Notebook:
        path = self.directory / f"{notebook_id}.ipynb"
        return Notebook.load(str(path))

    def list(self) -> List[str]:
        return [p.stem for p in self.directory.glob("*.ipynb")]

    def exists(self, notebook_id: str) -> bool:
        return (self.directory / f"{notebook_id}.ipynb").exists()

# Usage
storage = FileStorage(NOTEBOOKS_DIR)
# Or: storage = DatabaseStorage(connection_string)
# Or: storage = S3Storage(bucket_name)
```

---

## See Also

- [02_cell_types.md](02_cell_types.md) - Cell types and their behavior
- [03_real_time_collaboration.md](03_real_time_collaboration.md) - WebSocket broadcasting
- [../../DEVELOPERS.md](../../DEVELOPERS.md) - General developer guide
