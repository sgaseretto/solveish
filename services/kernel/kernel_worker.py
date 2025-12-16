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

            try:
                # Execute with streaming output
                shell._run_streaming(msg['code'], output_queue)
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


if __name__ == '__main__':
    # For testing - can be run directly
    from multiprocessing import Queue
    input_q = Queue()
    output_q = Queue()
    kernel_worker_main(input_q, output_q)
