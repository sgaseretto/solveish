(function initializeExecutionUi(global) {
    class FrameBatcher {
        constructor({onFlush, maxBytes = 32768, schedule = null, cancel = null}) {
            this.onFlush = onFlush;
            this.maxBytes = maxBytes;
            this.schedule = schedule || (callback => global.requestAnimationFrame(callback));
            this.cancel = cancel || (handle => global.cancelAnimationFrame(handle));
            this.buffers = new Map();
            this.handles = new Map();
        }

        append(key, chunk) {
            const pending = (this.buffers.get(key) || '') + chunk;
            this.buffers.set(key, pending);
            if (pending.length >= this.maxBytes) {
                this.flush(key);
                return;
            }
            if (!this.handles.has(key)) {
                this.handles.set(key, this.schedule(() => this.flush(key)));
            }
        }

        flush(key, options = {}) {
            const chunk = this.buffers.get(key) || '';
            this.buffers.delete(key);
            const handle = this.handles.get(key);
            if (handle !== undefined) this.cancel(handle);
            this.handles.delete(key);
            if (chunk) this.onFlush(key, chunk, options);
            return chunk;
        }

        clear(key) {
            const handle = this.handles.get(key);
            if (handle !== undefined) this.cancel(handle);
            this.handles.delete(key);
            this.buffers.delete(key);
        }
    }

    function setCellRunControlState(cell, running) {
        if (!cell) return;
        cell.classList?.toggle?.('streaming', running);
        const cancelButton = cell.querySelector?.('.btn-cancel');
        const runButton = cell.querySelector?.('.btn-run');
        if (cancelButton) cancelButton.style.display = running ? '' : 'none';
        if (runButton) {
            runButton.style.display = running ? 'none' : '';
            runButton.disabled = running;
        }
    }

    global.DialengFrameBatcher = FrameBatcher;
    global.DialengSetCellRunControlState = setCellRunControlState;
})(typeof window === 'undefined' ? globalThis : window);
