"""
Example Callbacks Extension

Demonstrates the 2-way callback system with practical examples.
These callbacks are registered automatically when the extension loads.

Usage:
    Callbacks are active immediately after loading. They hook into
    the cell execution lifecycle and can:
    - Observe execution (logging, timing)
    - Modify code before execution (auto-imports, magic commands)
    - Filter/transform output (truncation, formatting)
"""

import time
import logging
from dialeng.core.registry import register_callback
from dialeng.core.callbacks import Callback, ExecutionContext

logger = logging.getLogger(__name__)


@register_callback
class ExecutionTimingCallback(Callback):
    """
    Track and log execution time for all cells.

    This callback demonstrates:
    - Using before_execution and after_execution hooks
    - Storing metadata in ctx.metadata for cross-hook communication
    - Updating cell properties (time_run)
    """
    order = -100  # Run very early to capture full execution time
    name = "ExecutionTimingCallback"

    def before_execution(self, ctx: ExecutionContext) -> None:
        ctx.metadata['start_time'] = time.time()
        logger.debug(f"Starting execution of cell {ctx.cell.id}")

    def after_execution(self, ctx: ExecutionContext, error=None) -> None:
        start = ctx.metadata.get('start_time')
        if start:
            elapsed = time.time() - start
            ctx.cell.time_run = f"{elapsed:.2f}s"
            logger.debug(f"Cell {ctx.cell.id} completed in {elapsed:.2f}s")


# Note: The AutoImportCallback below is commented out by default.
# Uncomment to enable automatic import injection.

# @register_callback
# class AutoImportCallback(Callback):
#     """
#     Automatically add common imports if they're used but not imported.
#
#     This callback demonstrates:
#     - Modifying ctx.source before execution (2-way modification!)
#     - Pattern matching to detect missing imports
#     """
#     order = 0
#     name = "AutoImportCallback"
#
#     # Map of usage patterns to import statements
#     AUTO_IMPORTS = {
#         'np.': 'import numpy as np',
#         'pd.': 'import pandas as pd',
#         'plt.': 'import matplotlib.pyplot as plt',
#     }
#
#     def before_execution(self, ctx: ExecutionContext) -> None:
#         additions = []
#         for pattern, import_stmt in self.AUTO_IMPORTS.items():
#             if pattern in ctx.source and import_stmt not in ctx.source:
#                 additions.append(import_stmt)
#
#         if additions:
#             ctx.source = '\n'.join(additions) + '\n' + ctx.source
#             logger.info(f"Auto-added imports: {additions}")


# Note: The OutputLimitCallback below is commented out by default.
# Uncomment to enable output truncation.

# @register_callback
# class OutputLimitCallback(Callback):
#     """
#     Limit output size to prevent UI slowdowns.
#
#     This callback demonstrates:
#     - Filtering/transforming outputs via on_output
#     - Returning modified output (or None to filter completely)
#     """
#     order = 50
#     name = "OutputLimitCallback"
#
#     MAX_LINES = 1000
#     MAX_CHARS = 50000
#
#     def on_output(self, ctx: ExecutionContext, output):
#         if output.output_type == 'stream' and isinstance(output.content, str):
#             content = output.content
#
#             # Truncate by lines
#             lines = content.split('\n')
#             if len(lines) > self.MAX_LINES:
#                 content = '\n'.join(lines[:self.MAX_LINES])
#                 content += f"\n... ({len(lines) - self.MAX_LINES} more lines)"
#
#             # Truncate by characters
#             if len(content) > self.MAX_CHARS:
#                 content = content[:self.MAX_CHARS]
#                 content += "\n... (output truncated)"
#
#             output.content = content
#
#         return output


logger.info("Example callbacks extension loaded")
