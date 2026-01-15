"""
Dialeng - Global Application State

This module contains the global state containers used throughout the application.
Centralizing state here makes it easier to understand what data is shared
and potentially enables features like state persistence or multi-process sharing.

Usage:
    from state import notebooks, ws_connections, execution_queues

State containers:
    - notebooks: In-memory notebook storage, keyed by notebook_id
    - ws_connections: WebSocket connections per notebook for real-time updates
    - execution_queues: FIFO execution queues per notebook for code cells
    - cancelled_cells: Track cancelled LLM generations
    - data_queues: DialogHelper data queues for bidirectional browser communication

Note: The actual state variables remain in app.py for now due to the tight
coupling with helper functions. This file serves as documentation and can be
used for future refactoring to fully centralize state.
"""

from typing import Dict, List, Any
from pathlib import Path

# Type hints for state containers (actual instances are in app.py)
# These are provided here for documentation and type checking purposes

# notebooks: Dict[str, Notebook] = {}
# """In-memory notebook storage. Key is notebook_id, value is Notebook instance."""

# ws_connections: Dict[str, List[Any]] = {}
# """WebSocket send functions per notebook. Key is notebook_id, value is list of send callables."""

# execution_queues: Dict[str, ExecutionQueue] = {}
# """Execution queues per notebook. Key is notebook_id, value is ExecutionQueue instance."""

# cancelled_cells: set = set()
# """Set of cell IDs that have been cancelled (for LLM generations)."""

# data_queues: Dict[str, Dict[str, asyncio.Queue]] = {}
# """DialogHelper data queues. Structure: {dlg_name: {data_id: asyncio.Queue}}"""

# Constants
NOTEBOOKS_DIR = Path("notebooks")
"""Directory where notebook .ipynb files are stored."""
