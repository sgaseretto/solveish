const extensionId = 'dev.dialeng.diagram-cell';
window.__dialengExtensionMetrics = window.__dialengExtensionMetrics || {};
window.__dialengExtensionMetrics.diagramListeners = (window.__dialengExtensionMetrics.diagramListeners || 0) + 1;
window.__dialengExtensionMetrics.diagramToolRenderers = (window.__dialengExtensionMetrics.diagramToolRenderers || 0) + 1;
const updatePreview = event => {
    const source = event.target.closest?.('.diagram-source');
    if (!source) return;
    const cell = source.closest('[data-cell-id]');
    const preview = cell?.querySelector('.diagram-preview');
    if (preview) preview.textContent = source.value;
};
document.addEventListener('input', updatePreview);
window.dialeng.extensions.subscribe(extensionId, 'cell_source_update', data => {
    const cell = document.querySelector(`[data-cell-id="${CSS.escape(data.cell_id)}"]`);
    if (cell?.dataset.cellType !== 'diagram') return;
    const preview = cell.querySelector('.diagram-preview');
    if (preview) preview.textContent = data.source || '';
});
window.dialeng.extensions.registerToolRenderer(extensionId, 'diagram-preview', {
    renderCall({toolName, input}) {
        const node = document.createElement('div');
        node.className = 'diagram-tool-renderer';
        node.textContent = `${toolName}: ${input.source || ''}`;
        return node;
    },
    renderResult({result, status}) {
        const node = document.createElement('div');
        node.className = 'diagram-tool-result';
        node.textContent = `${status}: ${result.content || ''}`;
        return node;
    },
});
window.dialeng.extensions.registerCleanup(extensionId, () => {
    document.removeEventListener('input', updatePreview);
    window.__dialengExtensionMetrics.diagramListeners -= 1;
    window.__dialengExtensionMetrics.diagramToolRenderers -= 1;
});
