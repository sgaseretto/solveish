(function initializeBrowserDiagnostics(global) {
    const metrics = {
        longTaskCount: 0,
        longTaskTotalMs: 0,
        longTaskMaxMs: 0,
        lastFlushAt: Date.now(),
    };

    if ('PerformanceObserver' in global) {
        try {
            const observer = new PerformanceObserver(list => {
                list.getEntries().forEach(entry => {
                    metrics.longTaskCount += 1;
                    metrics.longTaskTotalMs += entry.duration;
                    metrics.longTaskMaxMs = Math.max(metrics.longTaskMaxMs, entry.duration);
                });
            });
            observer.observe({type: 'longtask', buffered: true});
        } catch (_) {
            // Long Task API is optional.
        }
    }

    function flush() {
        if (!global.NOTEBOOK_ID || document.hidden) return;
        const cells = Array.from(document.querySelectorAll('.cell'));
        const visibleCells = cells.filter(cell => {
            const rect = cell.getBoundingClientRect();
            return rect.bottom >= 0 && rect.top <= global.innerHeight;
        }).length;
        const body = new URLSearchParams({
            notebook_id: global.NOTEBOOK_ID,
            long_task_count: String(metrics.longTaskCount),
            long_task_total_ms: String(Math.round(metrics.longTaskTotalMs)),
            long_task_max_ms: String(Math.round(metrics.longTaskMaxMs)),
            editor_count: String(document.querySelectorAll('.monaco-editor').length),
            visible_cell_count: String(visibleCells),
            dom_cell_count: String(cells.length),
        });
        navigator.sendBeacon('/dialeng/diagnostics/browser', body);
        metrics.longTaskCount = 0;
        metrics.longTaskTotalMs = 0;
        metrics.longTaskMaxMs = 0;
        metrics.lastFlushAt = Date.now();
    }

    global.dialengBrowserDiagnostics = {flush};
    global.setInterval(flush, 30000);
    global.addEventListener('pagehide', flush);
})(window);
