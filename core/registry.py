"""
Dialeng Extension Registry

Central registry for cell types, callbacks, and services.
Extensions register their components here to integrate with dialeng.

Usage:
    from core.registry import register_cell_type, register_callback, registry

    # Register a new cell type
    @register_cell_type(icon="📊", label="Diagram")
    class DiagramCell(Cell):
        cell_type = "diagram"

    # Register a callback
    @register_callback
    class MyCallback(Callback):
        def before_execution(self, ctx):
            ...

    # Access registry directly
    all_cell_types = registry.cell_types
    all_callbacks = registry.get_callback_handler()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Type, Callable, Any, Optional
import logging

if TYPE_CHECKING:
    from document.cell import Cell
    from .callbacks import Callback, CallbackHandler

logger = logging.getLogger(__name__)


@dataclass
class CellTypeRegistration:
    """
    Registration for a custom cell type.

    Contains the cell class and all its handlers (renderer, serializers, etc.).
    Extensions create registrations and add them to the registry.
    """
    cell_class: Type['Cell']
    cell_type: str  # The cell_type string (e.g., "code", "diagram")

    # Display metadata
    icon: str = "📄"
    label: str = ""

    # Optional custom handlers (if not using @typedispatch defaults)
    # These are mainly for documentation - actual dispatch uses @typedispatch
    description: str = ""


@dataclass
class ExtensionRegistry:
    """
    Central registry for dialeng extensions.

    Stores:
    - Cell type registrations
    - Callbacks for execution lifecycle
    - Custom services (LLM providers, storage backends, etc.)

    The registry is a singleton - use the `registry` instance from this module.
    """
    cell_types: Dict[str, CellTypeRegistration] = field(default_factory=dict)
    callbacks: List['Callback'] = field(default_factory=list)
    services: Dict[str, Any] = field(default_factory=dict)

    # Track loaded extensions
    _loaded_extensions: List[str] = field(default_factory=list)

    def register_cell_type(self, registration: CellTypeRegistration) -> None:
        """
        Register a new cell type.

        The cell_type string must be unique. Attempting to register
        a duplicate will log a warning and overwrite.
        """
        cell_type = registration.cell_type
        if cell_type in self.cell_types:
            logger.warning(f"Overwriting existing cell type: {cell_type}")
        self.cell_types[cell_type] = registration
        logger.info(f"Registered cell type: {cell_type} ({registration.label})")

    def register_callback(self, callback: 'Callback') -> None:
        """
        Register a callback for execution lifecycle.

        Callbacks are sorted by order when building the CallbackHandler.
        """
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.order)
        logger.info(f"Registered callback: {callback.name} (order={callback.order})")

    def unregister_callback(self, callback_or_type) -> bool:
        """
        Remove a callback by instance or type.

        Returns True if a callback was removed.
        """
        if isinstance(callback_or_type, type):
            original_len = len(self.callbacks)
            self.callbacks = [c for c in self.callbacks if not isinstance(c, callback_or_type)]
            return len(self.callbacks) < original_len
        else:
            if callback_or_type in self.callbacks:
                self.callbacks.remove(callback_or_type)
                return True
            return False

    def register_service(self, name: str, service: Any) -> None:
        """
        Register a named service.

        Services can be LLM providers, storage backends, custom kernels, etc.
        """
        if name in self.services:
            logger.warning(f"Overwriting existing service: {name}")
        self.services[name] = service
        logger.info(f"Registered service: {name}")

    def get_service(self, name: str, default: Any = None) -> Any:
        """Get a registered service by name."""
        return self.services.get(name, default)

    def get_callback_handler(self) -> 'CallbackHandler':
        """
        Create a CallbackHandler with all registered callbacks.

        Returns a new handler each time (callbacks are copied).
        """
        from .callbacks import CallbackHandler
        return CallbackHandler(self.callbacks.copy())

    def get_cell_type_choices(self) -> List[tuple]:
        """
        Get list of (value, label) tuples for UI dropdowns.

        Includes built-in types (code, note, prompt) plus registered extensions.
        """
        # Built-in types first
        choices = [
            ("code", "Code"),
            ("note", "Note"),
            ("prompt", "Prompt"),
        ]

        # Add registered extensions
        for cell_type, reg in self.cell_types.items():
            if cell_type not in ("code", "note", "prompt"):
                label = reg.label or cell_type.title()
                if reg.icon:
                    label = f"{reg.icon} {label}"
                choices.append((cell_type, label))

        return choices

    def mark_extension_loaded(self, name: str) -> None:
        """Mark an extension as loaded (prevents double-loading)."""
        if name not in self._loaded_extensions:
            self._loaded_extensions.append(name)

    def is_extension_loaded(self, name: str) -> bool:
        """Check if an extension is already loaded."""
        return name in self._loaded_extensions

    def __repr__(self) -> str:
        return (
            f"ExtensionRegistry("
            f"cell_types={list(self.cell_types.keys())}, "
            f"callbacks={len(self.callbacks)}, "
            f"services={list(self.services.keys())})"
        )


# ============================================================================
# Global Registry Instance
# ============================================================================

registry = ExtensionRegistry()


# ============================================================================
# Decorator-style Registration Functions
# ============================================================================

def register_cell_type(
    cell_class: Optional[Type['Cell']] = None,
    *,
    icon: str = "📄",
    label: str = "",
    description: str = ""
) -> Callable:
    """
    Decorator to register a cell type.

    Can be used with or without arguments:

        @register_cell_type
        class DiagramCell(Cell):
            cell_type = "diagram"

        @register_cell_type(icon="📊", label="Diagram")
        class DiagramCell(Cell):
            cell_type = "diagram"
    """
    def decorator(cls: Type['Cell']) -> Type['Cell']:
        # Get cell_type from class attribute
        cell_type = getattr(cls, 'cell_type', None)
        if cell_type is None:
            raise ValueError(f"Cell class {cls.__name__} must have a 'cell_type' attribute")

        # If cell_type is a string, use it directly
        if hasattr(cell_type, 'value'):
            cell_type = cell_type.value

        registration = CellTypeRegistration(
            cell_class=cls,
            cell_type=cell_type,
            icon=icon,
            label=label or cls.__name__.replace('Cell', ''),
            description=description
        )
        registry.register_cell_type(registration)
        return cls

    # Handle @register_cell_type vs @register_cell_type(...)
    if cell_class is not None:
        return decorator(cell_class)
    return decorator


def register_callback(callback_or_class):
    """
    Decorator to register a callback.

    Can be used on a class or an instance:

        @register_callback
        class MyCallback(Callback):
            def before_execution(self, ctx):
                ...

        # Or with an instance
        register_callback(MyCallback())
    """
    from .callbacks import Callback

    if isinstance(callback_or_class, type):
        # It's a class - instantiate it
        callback = callback_or_class()
    else:
        # It's an instance
        callback = callback_or_class

    if not isinstance(callback, Callback):
        raise TypeError(f"Expected Callback instance, got {type(callback)}")

    registry.register_callback(callback)
    return callback_or_class


def register_service(name: str):
    """
    Decorator to register a service.

        @register_service("my_llm")
        class MyLLMService:
            ...
    """
    def decorator(cls):
        registry.register_service(name, cls)
        return cls
    return decorator
