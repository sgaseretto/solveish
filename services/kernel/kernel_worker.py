"""
Kernel worker that runs in a subprocess.

This module provides streaming output by patching execnb's CaptureShell
to redirect stdout/stderr to a multiprocessing Queue instead of using
IPython's capture_output context manager.
"""
import sys
import traceback
from multiprocessing import Queue
from typing import Optional

from execnb.shell import CaptureShell
from fastcore.basics import patch


class StreamingStdout:
    """
    Custom stdout/stderr that sends output to a queue immediately.

    Each write() call sends a message to the queue, enabling real-time
    streaming of output to the main process.
    """

    def __init__(self, queue: Queue, stream_name: str = 'stdout'):
        self.queue = queue
        self.stream_name = stream_name
        self._original = sys.stdout if stream_name == 'stdout' else sys.stderr

    def write(self, text: str):
        if text:
            self.queue.put({
                'type': 'stream',
                'name': self.stream_name,
                'text': text
            })

    def flush(self):
        pass

    def isatty(self):
        return True  # Report as TTY to enable tqdm progress bars

    def fileno(self):
        return self._original.fileno()


class StreamingDisplayPublisher:
    """
    Capture rich outputs (images, plots, HTML) and send to queue.

    Replaces IPython's display_pub to capture display() calls,
    matplotlib figures, and other rich content.

    Must implement attributes that IPython's internals check:
    - is_publishing: Flag checked by IPython's _tee mechanism
    """

    def __init__(self, queue: Queue, shell: Optional['CaptureShell'] = None):
        self.queue = queue
        self.shell = shell
        self.is_publishing = False  # Required by IPython's _tee

    def publish(self, data: dict, metadata: Optional[dict] = None,
                source: Optional[str] = None, **kwargs):
        """Publish display data to the queue."""
        self.is_publishing = True
        try:
            self.queue.put({
                'type': 'display_data',
                'data': data,
                'metadata': metadata or {}
            })
        finally:
            self.is_publishing = False

    def clear_output(self, wait: bool = False):
        """Clear output - send a clear message."""
        self.queue.put({
            'type': 'clear_output',
            'wait': wait
        })


@patch
def _run_streaming(self: CaptureShell, raw_cell: str, output_queue: Queue,
                   store_history: bool = False, silent: bool = False,
                   shell_futures: bool = True, cell_id: Optional[str] = None):
    """
    Patched method that streams output instead of capturing.

    This replaces CaptureShell's default _run which uses capture_output.
    Instead, we redirect stdout/stderr to StreamingStdout and hook
    the display publisher.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_display_pub = getattr(self, 'display_pub', None)

    try:
        # Replace stdout/stderr with streaming versions
        sys.stdout = StreamingStdout(output_queue, 'stdout')
        sys.stderr = StreamingStdout(output_queue, 'stderr')

        # Replace display publisher for rich outputs
        self.display_pub = StreamingDisplayPublisher(output_queue, self)

        # Run the cell using InteractiveShell's run_cell
        # (bypassing CaptureShell's capture_output wrapper)
        result = super(CaptureShell, self).run_cell(
            raw_cell,
            store_history=store_history,
            silent=silent,
            shell_futures=shell_futures,
            cell_id=cell_id
        )

        # Send final result if there was one
        if result.result is not None:
            output_queue.put({
                'type': 'execute_result',
                'data': {'text/plain': repr(result.result)},
                'metadata': {}
            })

        # Send error if execution failed
        if result.error_in_exec:
            exc = result.error_in_exec
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            output_queue.put({
                'type': 'error',
                'ename': type(exc).__name__,
                'evalue': str(exc),
                'traceback': tb_lines
            })

        return result

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if old_display_pub is not None:
            self.display_pub = old_display_pub


def _get_type_name(annotation) -> str:
    """Convert a type annotation to a string name."""
    import typing
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    elif hasattr(annotation, '__origin__'):
        # Handle generic types like List[int], Dict[str, int], Optional[str]
        origin = annotation.__origin__
        args = getattr(annotation, '__args__', ())
        origin_name = getattr(origin, '__name__', str(origin))
        if args:
            arg_names = ', '.join(_get_type_name(a) for a in args)
            return f"{origin_name}[{arg_names}]"
        return origin_name
    return str(annotation)


def _extract_param_description(docstring: str, param_name: str) -> str:
    """Extract parameter description from Google/numpy style docstring."""
    if not docstring:
        return param_name

    import re
    lines = docstring.split('\n')

    # Look for Args/Parameters section
    in_params_section = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check for section headers
        if stripped.lower() in ('args:', 'arguments:', 'parameters:', 'params:'):
            in_params_section = True
            continue
        elif stripped.lower() in ('returns:', 'return:', 'raises:', 'examples:', 'example:', 'notes:', 'note:'):
            in_params_section = False
            continue

        if in_params_section:
            # Google style: "param_name: description" or "param_name (type): description"
            google_match = re.match(rf'^\s*{re.escape(param_name)}\s*(?:\([^)]*\))?\s*:\s*(.+)', line)
            if google_match:
                desc = google_match.group(1).strip()
                # Check for continuation on next lines
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() and not re.match(r'^\s*\w+\s*(?:\([^)]*\))?\s*:', next_line):
                        if next_line.startswith('        ') or next_line.startswith('\t\t'):
                            desc += ' ' + next_line.strip()
                        else:
                            break
                    else:
                        break
                return desc

            # Numpy style: "param_name : type"
            numpy_match = re.match(rf'^\s*{re.escape(param_name)}\s*:\s*\w+', line)
            if numpy_match:
                # Description is on next line
                if i + 1 < len(lines):
                    return lines[i + 1].strip()

    return param_name


def _format_tool_result(result) -> dict:
    """Format a tool execution result for sending back to the LLM."""
    import io
    import base64

    # Check for matplotlib figure
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        if isinstance(result, Figure):
            buf = io.BytesIO()
            result.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(result)
            return {'type': 'image', 'format': 'png', 'content': img_data}
    except ImportError:
        pass

    # Check for _repr_png_
    if hasattr(result, '_repr_png_'):
        try:
            png_data = result._repr_png_()
            if png_data:
                if isinstance(png_data, bytes):
                    png_data = base64.b64encode(png_data).decode('utf-8')
                return {'type': 'image', 'format': 'png', 'content': png_data}
        except Exception:
            pass

    # Check for _repr_html_
    if hasattr(result, '_repr_html_'):
        try:
            html = result._repr_html_()
            if html:
                # Truncate very long HTML
                if len(html) > 10000:
                    html = html[:10000] + '\n... (truncated)'
                return {'type': 'html', 'content': html}
        except Exception:
            pass

    # Check for pandas DataFrame/Series
    try:
        import pandas as pd
        if isinstance(result, (pd.DataFrame, pd.Series)):
            html = result.to_html(max_rows=50, max_cols=20)
            return {'type': 'html', 'content': html}
    except ImportError:
        pass

    # Default: text representation
    result_str = repr(result)
    if len(result_str) > 5000:
        result_str = result_str[:5000] + '\n... (truncated)'
    return {'type': 'text', 'content': result_str}


def kernel_worker_main(input_queue: Queue, output_queue: Queue):
    """
    Main loop for the kernel subprocess.

    Waits for commands on input_queue and sends results to output_queue.
    The subprocess catches KeyboardInterrupt from SIGINT for hard interrupt.
    """
    import signal

    # Create the shell - this inherits IPython's full environment
    shell = CaptureShell()

    # Setup signal handler to ensure SIGINT raises KeyboardInterrupt
    def sigint_handler(signum, frame):
        raise KeyboardInterrupt("Execution interrupted by user")

    signal.signal(signal.SIGINT, sigint_handler)

    # Setup matplotlib for inline display if available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64
        from IPython.display import display

        # Override plt.show() to capture and display figures
        _original_show = plt.show
        def _inline_show(*args, **kwargs):
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                # Display as image
                display({'image/png': img_data}, raw=True)
                plt.close(fig)
        plt.show = _inline_show
    except ImportError:
        pass

    # Signal ready
    output_queue.put({'type': 'status', 'status': 'ready'})

    # Track which cell defined each variable/function
    # Maps name -> cell_id
    var_cell_map = {}

    while True:
        try:
            msg = input_queue.get()
        except KeyboardInterrupt:
            # SIGINT while waiting - ignore and continue
            continue

        if msg['type'] == 'execute':
            # Send busy status
            output_queue.put({'type': 'status', 'status': 'busy'})

            # Inject dialoghelper magic variables into the namespace
            # These are used by dialoghelper's find_var() to identify context
            notebook_id = msg.get('notebook_id', '')
            cell_id = msg.get('cell_id', '')
            if notebook_id:
                shell.user_ns['__dialog_name'] = notebook_id
            if cell_id:
                shell.user_ns['__msg_id'] = cell_id

            # Capture namespace state before execution
            pre_execution_names = set(shell.user_ns.keys())

            try:
                # Execute with streaming output
                shell._run_streaming(msg['code'], output_queue)

                # Track new variables from this cell (only names that didn't exist before)
                if cell_id:
                    for name in shell.user_ns.keys():
                        # Only track names that are NEW (created in this cell execution)
                        if name not in pre_execution_names:
                            # Only track user-defined names (not private/magic)
                            if not name.startswith('_'):
                                var_cell_map[name] = cell_id

            except KeyboardInterrupt:
                # SIGINT during execution - send interrupt error
                output_queue.put({
                    'type': 'error',
                    'ename': 'KeyboardInterrupt',
                    'evalue': 'Execution interrupted by user',
                    'traceback': ['KeyboardInterrupt: Execution interrupted by user']
                })
            except Exception as e:
                # Unexpected error
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                output_queue.put({
                    'type': 'error',
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': tb_lines
                })

            # Send idle status and done marker
            output_queue.put({'type': 'status', 'status': 'idle'})
            output_queue.put({'type': 'execute_done', 'execution_count': shell.execution_count})

        elif msg['type'] == 'complete':
            # Code completion request
            completions = shell.complete(msg['code'], msg['cursor_pos'])
            output_queue.put({
                'type': 'complete_reply',
                'matches': completions[1] if completions else [],
                'cursor_start': completions[0] if completions else msg['cursor_pos'],
            })

        elif msg['type'] == 'shutdown':
            # Clean shutdown
            output_queue.put({'type': 'status', 'status': 'shutdown'})
            break

        elif msg['type'] == 'restart':
            # Restart by re-creating the shell
            shell = CaptureShell()
            output_queue.put({'type': 'status', 'status': 'restarted'})

        elif msg['type'] == 'introspect_var':
            # Introspect a variable in the kernel namespace
            var_name = msg.get('name', '')
            try:
                if var_name in shell.user_ns:
                    var = shell.user_ns[var_name]
                    var_repr = repr(var)
                    # Truncate repr to 500 chars to avoid huge outputs
                    if len(var_repr) > 500:
                        var_repr = var_repr[:497] + '...'
                    output_queue.put({
                        'type': 'introspect_var_reply',
                        'name': var_name,
                        'exists': True,
                        'var_type': type(var).__name__,
                        'repr': var_repr
                    })
                else:
                    output_queue.put({
                        'type': 'introspect_var_reply',
                        'name': var_name,
                        'exists': False,
                        'error': f"Variable '{var_name}' not found in namespace"
                    })
            except Exception as e:
                output_queue.put({
                    'type': 'introspect_var_reply',
                    'name': var_name,
                    'exists': False,
                    'error': str(e)
                })

        elif msg['type'] == 'introspect_function':
            # Introspect a function in the kernel namespace
            import inspect
            func_name = msg.get('name', '')
            try:
                if func_name in shell.user_ns:
                    func = shell.user_ns[func_name]
                    if callable(func):
                        # Get signature
                        try:
                            sig = str(inspect.signature(func))
                        except (ValueError, TypeError):
                            sig = '(...)'

                        # Get docstring
                        docstring = inspect.getdoc(func) or ''

                        # Get parameter info with types
                        params = {}
                        try:
                            sig_obj = inspect.signature(func)
                            for param_name, param in sig_obj.parameters.items():
                                param_info = {'name': param_name}
                                # Get type annotation
                                if param.annotation != inspect.Parameter.empty:
                                    param_info['type'] = _get_type_name(param.annotation)
                                else:
                                    param_info['type'] = 'any'
                                # Get default value
                                if param.default != inspect.Parameter.empty:
                                    param_info['default'] = repr(param.default)
                                # Get description from docstring (Google/numpy style)
                                param_info['description'] = _extract_param_description(docstring, param_name)
                                params[param_name] = param_info
                        except (ValueError, TypeError):
                            pass

                        # Get return type
                        return_type = None
                        try:
                            sig_obj = inspect.signature(func)
                            if sig_obj.return_annotation != inspect.Parameter.empty:
                                return_type = _get_type_name(sig_obj.return_annotation)
                        except (ValueError, TypeError):
                            pass

                        output_queue.put({
                            'type': 'introspect_function_reply',
                            'name': func_name,
                            'exists': True,
                            'is_callable': True,
                            'signature': sig,
                            'docstring': docstring,
                            'parameters': params,
                            'return_type': return_type
                        })
                    else:
                        output_queue.put({
                            'type': 'introspect_function_reply',
                            'name': func_name,
                            'exists': True,
                            'is_callable': False,
                            'error': f"'{func_name}' is not callable"
                        })
                else:
                    output_queue.put({
                        'type': 'introspect_function_reply',
                        'name': func_name,
                        'exists': False,
                        'error': f"Function '{func_name}' not found in namespace"
                    })
            except Exception as e:
                output_queue.put({
                    'type': 'introspect_function_reply',
                    'name': func_name,
                    'exists': False,
                    'error': str(e)
                })

        elif msg['type'] == 'list_namespace':
            # List all user-defined variables and functions in the namespace
            import inspect
            import types
            import builtins

            # Builtin types and modules to exclude
            builtin_names = set(dir(builtins))
            exclude_names = {
                '__builtins__', '__doc__', '__name__', '__package__', '__loader__',
                '__spec__', '__dialog_name', '__msg_id', 'In', 'Out', 'get_ipython',
                'exit', 'quit', '_', '__', '___', '_i', '_ii', '_iii', '_oh', '_dh',
                '_ih', '_i1', '_i2', '_i3', 'open', 'display', 'set_matplotlib_formats',
                'set_matplotlib_close', 'publish_display_data', 'clear_output'
            }

            variables = []
            functions = []

            for name, obj in shell.user_ns.items():
                # Skip private/magic names and builtins
                if name.startswith('_') or name in exclude_names or name in builtin_names:
                    continue

                # Skip modules
                if isinstance(obj, types.ModuleType):
                    continue

                # Get cell_id from var_cell_map if available
                cell_id_for_var = var_cell_map.get(name)

                if callable(obj) and not isinstance(obj, type):
                    # It's a function or callable
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = '(...)'
                    func_info = {
                        'name': name,
                        'signature': sig,
                        'type': type(obj).__name__
                    }
                    if cell_id_for_var:
                        func_info['cell_id'] = cell_id_for_var
                    functions.append(func_info)
                else:
                    # It's a variable
                    var_type = type(obj).__name__
                    # Get a short preview of the value
                    try:
                        preview = repr(obj)
                        if len(preview) > 50:
                            preview = preview[:47] + '...'
                    except Exception:
                        preview = '<error getting repr>'
                    var_info = {
                        'name': name,
                        'type': var_type,
                        'preview': preview
                    }
                    if cell_id_for_var:
                        var_info['cell_id'] = cell_id_for_var
                    variables.append(var_info)

            # Sort by name
            variables.sort(key=lambda x: x['name'])
            functions.sort(key=lambda x: x['name'])

            output_queue.put({
                'type': 'list_namespace_reply',
                'variables': variables,
                'functions': functions
            })

        elif msg['type'] == 'execute_tool':
            # Execute a function as a tool with given arguments
            import json
            import io
            import base64

            func_name = msg.get('name', '')
            kwargs = msg.get('kwargs', {})

            output_queue.put({'type': 'status', 'status': 'busy'})

            try:
                if func_name not in shell.user_ns:
                    output_queue.put({
                        'type': 'execute_tool_reply',
                        'name': func_name,
                        'status': 'error',
                        'error': f"Function '{func_name}' not found"
                    })
                else:
                    func = shell.user_ns[func_name]
                    if not callable(func):
                        output_queue.put({
                            'type': 'execute_tool_reply',
                            'name': func_name,
                            'status': 'error',
                            'error': f"'{func_name}' is not callable"
                        })
                    else:
                        # Execute the function
                        result = func(**kwargs)

                        # Handle rich output types
                        result_data = _format_tool_result(result)

                        output_queue.put({
                            'type': 'execute_tool_reply',
                            'name': func_name,
                            'status': 'success',
                            'result': result_data
                        })

            except Exception as e:
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                output_queue.put({
                    'type': 'execute_tool_reply',
                    'name': func_name,
                    'status': 'error',
                    'error': str(e),
                    'traceback': tb_lines
                })

            output_queue.put({'type': 'status', 'status': 'idle'})


if __name__ == '__main__':
    # For testing - can be run directly
    from multiprocessing import Queue
    input_q = Queue()
    output_q = Queue()
    kernel_worker_main(input_q, output_q)
