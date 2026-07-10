(function initializeDialengNotifications(global) {
    const history = [];
    let dismissTimeoutId = null;

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function clear() {
        const status = global.document?.getElementById('status');
        if (dismissTimeoutId) {
            global.clearTimeout(dismissTimeoutId);
            dismissTimeoutId = null;
        }
        if (status) status.innerHTML = '';
    }

    function persistMs(statusNode) {
        if (!statusNode) return 0;
        const explicit = Number(statusNode.dataset.persistMs || 0);
        if (Number.isFinite(explicit) && explicit > 0) return explicit;
        if (statusNode.classList.contains('error')) return 8000;
        if (statusNode.classList.contains('warning')) return 6000;
        if (statusNode.classList.contains('success')) return 2800;
        return 4000;
    }

    function schedule(message, duration) {
        if (dismissTimeoutId) global.clearTimeout(dismissTimeoutId);
        dismissTimeoutId = null;
        if (duration > 0) {
            dismissTimeoutId = global.setTimeout(() => {
                const status = global.document?.getElementById('status');
                if (status && status.textContent.trim() === message) clear();
            }, duration);
        }
    }

    function arm(target) {
        const status = target?.id === 'status'
            ? target
            : global.document?.getElementById('status');
        if (!status) return;
        const toast = status.querySelector('.status');
        if (!toast) return;
        const message = toast.textContent.trim();
        if (!message) {
            clear();
            return;
        }
        schedule(message, persistMs(toast));
    }

    function show(message, tone = 'warning', duration = 5000) {
        const status = global.document?.getElementById('status');
        if (!status) return;
        status.innerHTML = `<div class="status ${tone}">${escapeHtml(message)}</div>`;
        history.push({timestamp: new Date().toISOString(), tone, message});
        if (history.length > 50) history.shift();
        schedule(message, duration);
    }

    global.DialengNotifications = {
        arm,
        clear,
        history,
        persistMs,
        schedule,
        show,
    };
})(typeof window === 'undefined' ? globalThis : window);
