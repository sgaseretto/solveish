"""
Dialeng Extensions

This directory contains extension files that are automatically loaded on startup.

Each .py file (except those starting with _) is imported as a module.
Extensions use decorators from dialeng.core.registry and core.dispatch to register:
- New cell types
- Execution callbacks
- Custom services

Example extension file:
    # extensions/my_extension.py
    from dialeng.core.registry import register_callback
    from dialeng.core.callbacks import Callback

    @register_callback
    class MyCallback(Callback):
        def before_execution(self, ctx):
            print(f"Executing: {ctx.cell.id}")
"""
