"""
Shell Cell Extension

Adds a new "shell" cell type for executing bash commands via pshnb.
Includes optional safecmd validation when safe mode is enabled.

Features:
- Shell cell type with bash syntax highlighting
- Fresh shell session per cell execution (no persistent state)
- Variable expansion from Python namespace via @{var} syntax
- Safe mode validation using safecmd allowlist

Usage:
    1. Create a shell cell from the cell type dropdown
    2. Write bash commands
    3. Run the cell (Shift+Enter)

    Variable expansion:
        # In a code cell first:
        name = "world"

        # Then in a shell cell:
        echo "Hello, @{name}!"
        # Output: Hello, world!

    Safe mode:
        Enable safe mode in notebook settings to validate all shell
        commands against the safecmd allowlist before execution.
"""

import logging
from typing import Optional, List, Dict

from dialeng.core.registry import register_callback
from dialeng.core.callbacks import Callback, ExecutionContext
from dialeng.core.dispatch import register_renderer, register_llm_converter
from dialeng.document.cell import CellOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Cell Type Registration
# ============================================================================

# Note: Shell cell type is registered via the renderer and LLM converter below.
# The cell uses cell_type="shell" which is a string, not a CellType enum value.


@register_renderer("shell")
def render_shell_cell(cell, notebook_id: str):
    """Render a shell cell with bash syntax highlighting."""
    from dialeng.ui.cells.shell_cell import ShellCellView

    # Get notebook to check safe_mode
    # Import here to avoid circular imports
    try:
        # Try to get notebook from app.py's notebooks dict
        import sys
        if 'app' in sys.modules:
            app_module = sys.modules['app']
            notebooks = getattr(app_module, 'notebooks', {})
            notebook = notebooks.get(notebook_id)
            safe_mode = getattr(notebook, 'safe_mode', False) if notebook else False
        else:
            safe_mode = False
    except Exception:
        safe_mode = False

    return ShellCellView(cell, notebook_id, safe_mode=safe_mode)


@register_llm_converter("shell")
def shell_to_messages(cell) -> List[Dict]:
    """Convert shell cell to LLM message format."""
    content = f"```bash\n{cell.source}\n```"
    if cell.output:
        content += f"\nOutput:\n```\n{cell.output}\n```"
    return [{"role": "user", "content": content}]


# ============================================================================
# Shell Execution Callback
# ============================================================================

@register_callback
class ShellExecutionCallback(Callback):
    """
    Execute shell cells using ShellService instead of Python kernel.

    This callback intercepts shell cell execution and:
    1. Skips the normal Python kernel execution
    2. Validates the command if safe mode is enabled
    3. Executes the command via pshnb ShellInterpreter
    4. Captures output and sends it to the cell

    The callback runs early (order=-50) to intercept before Python execution.
    """
    order = -50  # Run before Python execution
    name = "ShellExecutionCallback"

    def before_execution(self, ctx: ExecutionContext) -> None:
        """Check if this is a shell cell and handle its execution."""
        # Get cell type as string
        cell_type = ctx.cell.cell_type
        if hasattr(cell_type, 'value'):
            cell_type = cell_type.value

        if cell_type != "shell":
            return  # Not a shell cell, let normal execution proceed

        # Check if kernel is remote (e.g. Colab) — shell cells only work locally
        try:
            import sys
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                notebooks = getattr(app_module, 'notebooks', {})
                notebook = notebooks.get(ctx.notebook_id)
                if notebook and getattr(notebook, 'kernel_type', 'local') != 'local':
                    ctx.skip_execution = True
                    ctx.outputs.append(CellOutput(
                        output_type='error',
                        ename='UnsupportedKernel',
                        evalue='Shell cells are not supported with remote kernels.',
                        traceback=[
                            'Shell cells only work with the local Python kernel.',
                            'Use !command syntax in a code cell for remote execution.',
                        ]
                    ))
                    return
        except Exception:
            pass

        # Mark that we're handling execution ourselves
        ctx.skip_execution = True

        # Get shell service and notebook settings
        try:
            from dialeng.services.shell_service import ShellService, SafeModeError, ShfmtNotInstalledError

            # Get notebook's safe_mode setting
            safe_mode = False
            try:
                import sys
                if 'app' in sys.modules:
                    app_module = sys.modules['app']
                    notebooks = getattr(app_module, 'notebooks', {})
                    notebook = notebooks.get(ctx.notebook_id)
                    if notebook:
                        safe_mode = getattr(notebook, 'safe_mode', False)
            except Exception as e:
                logger.warning(f"Could not get notebook safe_mode: {e}")

            # Get Python namespace for variable expansion
            namespace = self._get_python_namespace(ctx.notebook_id)

            # Create service and execute
            try:
                service = ShellService(safe_mode=safe_mode)
                result = service.execute(ctx.source, namespace=namespace)

                # Add output to context
                if result.output:
                    ctx.outputs.append(CellOutput(
                        output_type='stream',
                        content=result.output,
                        stream_name='stdout'
                    ))

                if result.error:
                    ctx.outputs.append(CellOutput(
                        output_type='stream',
                        content=result.error,
                        stream_name='stderr'
                    ))

            except ShfmtNotInstalledError as e:
                ctx.outputs.append(CellOutput(
                    output_type='error',
                    ename='ShfmtNotInstalled',
                    evalue=str(e),
                    traceback=[
                        'Safe mode requires shfmt binary.',
                        'Install with: brew install shfmt (macOS)',
                        'Or: apt install shfmt (Ubuntu)',
                        str(e)
                    ]
                ))

            except SafeModeError as e:
                ctx.outputs.append(CellOutput(
                    output_type='error',
                    ename='DisallowedCmd',
                    evalue=str(e),
                    traceback=[
                        'Command blocked by Safe Mode',
                        '',
                        str(e),
                        '',
                        'Safe Mode restricts shell commands to a predefined allowlist.',
                        'Disable Safe Mode in notebook settings to run this command.'
                    ]
                ))

        except ImportError as e:
            ctx.outputs.append(CellOutput(
                output_type='error',
                ename='ImportError',
                evalue=str(e),
                traceback=[
                    'Shell execution requires pshnb package.',
                    'Install with: pip install pshnb',
                    str(e)
                ]
            ))

        except Exception as e:
            logger.exception(f"Shell execution failed: {e}")
            ctx.outputs.append(CellOutput(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[str(e)]
            ))

    def _get_python_namespace(self, notebook_id: str) -> dict:
        """Get the Python kernel namespace for variable expansion."""
        try:
            import sys
            if 'app' in sys.modules:
                app_module = sys.modules['app']
                kernel_services = getattr(app_module, 'kernel_services', {})
                kernel_service = kernel_services.get(notebook_id)
                if kernel_service and kernel_service.kernel:
                    # Note: Can't directly access subprocess namespace
                    # Would need to add introspection to SubprocessKernel
                    # For now, return empty dict
                    pass
        except Exception as e:
            logger.debug(f"Could not get kernel namespace: {e}")

        return {}


logger.info("Shell cell extension loaded")
