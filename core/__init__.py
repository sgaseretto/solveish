"""
Dialeng Core - Extensibility Framework

This package provides the foundation for dialeng's hackable, extensible architecture
using fastcore-inspired patterns:

- **Type Dispatch** (`@typedispatch`): Multi-method dispatch for cell operations
- **2-Way Callbacks**: Callbacks that can observe AND modify execution
- **Extension Registry**: Central registration for cell types, callbacks, services
- **@patch**: Extend classes without inheritance

Usage:
    from core import registry, render_cell, cell_to_llm_messages
    from core.callbacks import Callback, ExecutionContext
    from core.registry import register_cell_type, register_callback
"""

from .registry import (
    registry,
    register_cell_type,
    register_callback,
    register_service,
    CellTypeRegistration,
    ExtensionRegistry,
)

from .dispatch import (
    render_cell,
    cell_to_jupyter,
    jupyter_to_cell,
    cell_to_llm_messages,
)

from .callbacks import (
    Callback,
    CallbackHandler,
    ExecutionContext,
    CancelCellException,
    CancelQueueException,
)

__all__ = [
    # Registry
    'registry',
    'register_cell_type',
    'register_callback',
    'register_service',
    'CellTypeRegistration',
    'ExtensionRegistry',
    # Dispatch functions
    'render_cell',
    'cell_to_jupyter',
    'jupyter_to_cell',
    'cell_to_llm_messages',
    # Callbacks
    'Callback',
    'CallbackHandler',
    'ExecutionContext',
    'CancelCellException',
    'CancelQueueException',
]
