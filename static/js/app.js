/* ==========================================================================
   Dialeng Client-Side JavaScript
   
   This file contains all client-side logic for the Dialeng notebook app:
   - Ace Editor management
   - Cell focus and selection
   - Keyboard shortcuts
   - Markdown rendering
   - Preview/Edit toggle (event delegation for reliability)
   - WebSocket streaming
   - Theme management
   - Cell collapse controls
   - Code cell streaming
   - Queue state management
   - OOB (Out-of-Band) swap handling for collaboration
   ========================================================================== */

// ==================== Global Cell Selection (Event Delegation) ====================
// Use event delegation on document to ensure cell selection works even when clicking buttons
document.addEventListener('mousedown', (e) => {
    const cell = e.target.closest('.cell');
    if (cell) {
        const cellId = cell.id.replace('cell-', '');
        if (cellId && typeof setFocusedCell === 'function') {
            setFocusedCell(cellId);
        }
    }
}, true);  // Use capture phase to get the event before it's stopped

// ==================== Ace Editor Management ====================
const aceEditors = {};

function initAceEditor(cellId) {
    const container = document.getElementById(`ace-${cellId}`);
    if (!container) return null;

    // If editor already exists, destroy it first to ensure fresh state
    if (aceEditors[cellId]) {
        aceEditors[cellId].destroy();
        delete aceEditors[cellId];
    }

    // Also check if Ace left any state on the container
    if (container.env && container.env.editor) {
        try {
            container.env.editor.destroy();
        } catch (e) {}
    }

    // Get initial content from hidden textarea
    const textarea = document.getElementById(`source-${cellId}`);
    const initialContent = textarea ? textarea.value : '';

    // Clear container completely before Ace takes over
    container.innerHTML = '';
    container.className = 'ace-container';  // Reset classes

    const editor = ace.edit(container);
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    editor.setTheme(currentTheme === 'light' ? 'ace/theme/chrome' : 'ace/theme/monokai');
    editor.setOptions({
        fontSize: "14px",
        showPrintMargin: false,
        highlightActiveLine: true,
        wrap: true,
        minLines: 3,
        maxLines: 30,
        tabSize: 4,
        useSoftTabs: true,
    });

    // Set content first, then apply mode (mode triggers re-highlighting)
    editor.setValue(initialContent, -1);

    // Apply mode after setValue - this ensures syntax highlighting works
    // Use a small delay to let Ace settle
    editor.session.setMode("ace/mode/python");

    // Force a complete re-render after a brief delay
    setTimeout(() => {
        editor.session.setMode("ace/mode/python");
        editor.renderer.updateFull();
    }, 50);
    
    // Sync to hidden textarea on change
    if (textarea) {
        editor.session.on('change', () => {
            textarea.value = editor.getValue();
        });
    }
    
    // When Ace editor gets focus, also set cell as focused
    editor.on('focus', () => {
        setFocusedCell(cellId);
    });

    // Shift+Enter to run AND move to next cell (Jupyter style)
    editor.commands.addCommand({
        name: 'runCell',
        bindKey: {win: 'Shift-Enter', mac: 'Shift-Enter'},
        exec: function(editor) {
            const cell = editor.container.closest('.cell');
            if (cell) {
                // Sync content first
                const cellId = cell.id.replace('cell-', '');
                syncAceToTextarea(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
                // Move to next cell immediately (Jupyter behavior)
                moveToNextCell(cell);
            }
        }
    });
    
    // Ctrl/Cmd+Enter also runs
    editor.commands.addCommand({
        name: 'runCellAlt',
        bindKey: {win: 'Ctrl-Enter', mac: 'Cmd-Enter'},
        exec: function(editor) {
            const cell = editor.container.closest('.cell');
            if (cell) {
                const cellId = cell.id.replace('cell-', '');
                syncAceToTextarea(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
            }
        }
    });
    
    // Ctrl/Cmd+S to save
    editor.commands.addCommand({
        name: 'saveNotebook',
        bindKey: {win: 'Ctrl-S', mac: 'Cmd-S'},
        exec: function() {
            document.getElementById('save-btn')?.click();
        }
    });

    // Double-Escape to cancel all (handled by global keydown listener)
    // No need for Ace-specific binding since Escape blurs the editor first

    aceEditors[cellId] = editor;
    return editor;
}

function syncAceToTextarea(cellId) {
    const editor = aceEditors[cellId];
    const textarea = document.getElementById(`source-${cellId}`);
    if (editor && textarea) {
        textarea.value = editor.getValue();
    }
}

function getAceContent(cellId) {
    const editor = aceEditors[cellId];
    return editor ? editor.getValue() : '';
}

function destroyAceEditor(cellId) {
    if (aceEditors[cellId]) {
        aceEditors[cellId].destroy();
        delete aceEditors[cellId];
    }
}

// ==================== Focused Cell Tracking ====================
let focusedCellId = null;
let lastKeyTime = 0;
let lastKey = '';

function setFocusedCell(cellId) {
    document.querySelectorAll('.cell.focused').forEach(c => c.classList.remove('focused'));
    focusedCellId = cellId;
    if (cellId) {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell) cell.classList.add('focused');
    }
}

function focusNextCell(cellId) {
    // Focus a cell and optionally its editor
    setFocusedCell(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    // Scroll cell into view
    cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // If it's a code cell with Ace editor, focus the editor
    if (cell.dataset.type === 'code') {
        const editor = aceEditors[cellId];
        if (editor) {
            editor.focus();
        }
    } else if (cell.dataset.type === 'prompt') {
        // For prompt cells, focus the prompt textarea if visible
        const promptTextarea = cell.querySelector('.prompt-content[name="prompt_source"]');
        if (promptTextarea && promptTextarea.style.display !== 'none') {
            promptTextarea.focus();
        } else {
            // If prompt has been run (has preview), just keep cell selected
            // User can double-click to edit
        }
    } else if (cell.dataset.type === 'note') {
        // For note cells, just ensure the cell is focused/selected
        // Don't auto-open edit mode
    }
}

function getFocusedCellId() {
    const active = document.activeElement;
    if (active) {
        const cell = active.closest('.cell');
        if (cell) return cell.id.replace('cell-', '');
    }
    for (const [cellId, editor] of Object.entries(aceEditors)) {
        if (editor.isFocused()) return cellId;
    }
    return focusedCellId;
}

function moveToNextCell(currentCell) {
    // Find the next cell in DOM order
    // Cells are siblings within #cells, separated by .add-row divs
    let sibling = currentCell.nextElementSibling;
    while (sibling) {
        if (sibling.classList.contains('cell')) {
            const nextCellId = sibling.id.replace('cell-', '');
            focusNextCell(nextCellId);
            return;
        }
        sibling = sibling.nextElementSibling;
    }
    // No next cell found - we're at the last cell
    // Could optionally create a new cell here, but for now just stay on current
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', e => {
    const target = e.target;
    const mod = e.ctrlKey || e.metaKey;
    const inAce = target.closest('.ace_editor');
    const inInput = target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.isContentEditable;
    
    let currentCellId = getFocusedCellId();
    if (!currentCellId && target.closest('.cell')) {
        currentCellId = target.closest('.cell').id.replace('cell-', '');
    }
    
    // ===== D D to delete cell (Jupyter style) =====
    if (e.key === 'd' || e.key === 'D') {
        if (!inInput && !inAce) {
            const now = Date.now();
            if (lastKey === 'd' && (now - lastKeyTime) < 500) {
                // Double D pressed
                if (currentCellId) {
                    e.preventDefault();
                    const cell = document.getElementById(`cell-${currentCellId}`);
                    if (cell) {
                        const deleteBtn = cell.querySelector('button[hx-delete]');
                        if (deleteBtn) deleteBtn.click();
                    }
                }
                lastKey = '';
                lastKeyTime = 0;
                return;
            }
            lastKey = 'd';
            lastKeyTime = now;
        }
    } else {
        // Reset D D sequence on any other key
        lastKey = '';
        lastKeyTime = 0;
    }
    
    // ===== Shift+Enter - Run current cell AND move to next (Jupyter style) =====
    if (e.shiftKey && e.key === 'Enter' && !inAce) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncAceToTextarea(cellId);
                // Also sync prompt textarea
                syncPromptContent(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) {
                    // Click run button first
                    btn.click();
                    // Move focus to next cell IMMEDIATELY (don't wait for server)
                    // This is Jupyter behavior - Shift+Enter runs AND moves
                    moveToNextCell(cell);
                } else {
                    // Note cell - no run button, just move to next cell
                    moveToNextCell(cell);
                }
            }
        }
    }

    // ===== Ctrl/Cmd+Enter - Run current cell =====
    if (mod && e.key === 'Enter' && !inAce) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncAceToTextarea(cellId);
                syncPromptContent(cellId);
                const btn = cell.querySelector('.btn-run');
                if (btn) {
                    btn.click();
                } else {
                    // Note cell - just move to next cell
                    moveToNextCell(cell);
                }
            }
        }
    }
    
    // ===== Ctrl/Cmd+S - Save notebook =====
    if (mod && e.key === 's' && !inAce) {
        e.preventDefault();
        document.getElementById('save-btn')?.click();
    }
    
    // ===== Escape - Exit edit mode OR Cancel All (double Escape) =====
    if (e.key === 'Escape') {
        const now = Date.now();
        // Check for double-Escape (like Jupyter's I I for interrupt)
        if (lastKey === 'Escape' && (now - lastKeyTime) < 500) {
            // Double Escape pressed - cancel all execution
            e.preventDefault();
            cancelAllExecution();
            lastKey = '';
            lastKeyTime = 0;
            return;
        }
        // Single Escape - exit edit mode
        if (document.activeElement) {
            document.activeElement.blur();
        }
        Object.values(aceEditors).forEach(ed => ed.blur());
        lastKey = 'Escape';
        lastKeyTime = now;
    }

    // ===== Z - Collapse shortcuts =====
    // Z: cycle input collapse, Shift+Z: cycle output collapse, Alt+Z: cycle both
    if ((e.key === 'z' || e.key === 'Z') && !inInput && !inAce) {
        if (currentCellId) {
            e.preventDefault();
            if (e.altKey) {
                // Alt+Z: cycle both
                cycleCollapseLevel(currentCellId, 'both');
            } else if (e.shiftKey) {
                // Shift+Z: cycle output
                cycleCollapseLevel(currentCellId, 'output');
            } else {
                // Z: cycle input
                cycleCollapseLevel(currentCellId, 'input');
            }
        }
    }

    // ===== 0-3: Set specific collapse level =====
    // 0-3 for input, Shift+0-3 for output
    if (['0', '1', '2', '3'].includes(e.key) && !inInput && !inAce && !mod) {
        if (currentCellId) {
            const level = parseInt(e.key);
            if (e.shiftKey) {
                e.preventDefault();
                setCollapseLevel(currentCellId, 'output', level);
                // Also save to server
                fetch(`${window.location.pathname}/cell/${currentCellId}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=output&level=${level}`
                });
            } else if (e.altKey) {
                e.preventDefault();
                // Alt+number: set both to same level
                setCollapseLevel(currentCellId, 'input', level);
                setCollapseLevel(currentCellId, 'output', level);
                fetch(`${window.location.pathname}/cell/${currentCellId}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=both&level=${level}`
                });
            }
        }
    }
    
    // ===== Ctrl/Cmd+Shift+D or Ctrl/Cmd+Backspace - Delete cell =====
    if (mod && (e.key === 'Backspace' || (e.shiftKey && e.key === 'D'))) {
        if (currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const deleteBtn = cell.querySelector('button[hx-delete]');
                if (deleteBtn) deleteBtn.click();
            }
        }
    }
    
    // ===== Alt+Up or Ctrl/Cmd+Shift+Up - Move cell up =====
    if ((e.altKey && e.key === 'ArrowUp') || (mod && e.shiftKey && e.key === 'ArrowUp')) {
        if (currentCellId && !inAce) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const moveBtn = cell.querySelector('button[title="Move up"]');
                if (moveBtn) moveBtn.click();
            }
        }
    }
    
    // ===== Alt+Down or Ctrl/Cmd+Shift+Down - Move cell down =====
    if ((e.altKey && e.key === 'ArrowDown') || (mod && e.shiftKey && e.key === 'ArrowDown')) {
        if (currentCellId && !inAce) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const moveBtn = cell.querySelector('button[title="Move down"]');
                if (moveBtn) moveBtn.click();
            }
        }
    }
    
    // ===== Add cell shortcuts (not in input) =====
    if (!inInput && !inAce) {
        if (mod && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=code', {target: '#cells'});
        }
        if (mod && e.shiftKey && e.key === 'N') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=note', {target: '#cells'});
        }
        if (mod && e.shiftKey && e.key === 'P') {
            e.preventDefault();
            htmx.ajax('POST', window.location.pathname + '/cell/add?type=prompt', {target: '#cells'});
        }
    }
});

// Sync prompt content before running
function syncPromptContent(cellId) {
    // Try finding by ID first (newer format), then by name (for compatibility)
    let promptTextarea = document.getElementById(`prompt-${cellId}`);
    if (!promptTextarea) {
        promptTextarea = document.querySelector(`#cell-${cellId} .prompt-content[name="prompt_source"]`);
    }
    const hiddenSource = document.getElementById(`source-${cellId}`);
    if (promptTextarea && hiddenSource) {
        hiddenSource.value = promptTextarea.value;
    }
}

// Also sync when Ace editor content needs to go to hidden field
function syncAllContent(cellId) {
    syncAceToTextarea(cellId);
    syncPromptContent(cellId);
}

// ==================== Markdown Rendering ====================
function renderMarkdown(text) {
    if (!text) return '<p style="color: var(--text-muted);">Click to edit...</p>';
    
    // Process code blocks first and store them
    const codeBlocks = [];
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const idx = codeBlocks.length;
        const escaped = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        codeBlocks.push(`<pre data-lang="${lang || 'text'}"><code class="language-${lang || 'text'}">${escaped}</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></pre>`);
        return `__CODE_BLOCK_${idx}__`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Headers
    text = text.replace(/^### (.*)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.*)$/gm, '<h1>$1</h1>');
    // Lists
    text = text.replace(/^- (.*)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    // Numbered lists
    text = text.replace(/^\d+\. (.*)$/gm, '<li>$1</li>');
    // Paragraphs
    text = text.replace(/\n\n/g, '</p><p>');
    text = text.replace(/\n/g, '<br>');
    
    // Restore code blocks
    codeBlocks.forEach((block, idx) => {
        text = text.replace(`__CODE_BLOCK_${idx}__`, block);
    });
    
    return '<p>' + text + '</p>';
}

// Copy code to clipboard
function copyCode(btn) {
    const pre = btn.closest('pre');
    const code = pre.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });
}

// ==================== Preview/Edit Toggle ====================
// Event delegation for double-click editing - a single listener handles ALL preview elements
// This is more robust than per-element listeners because:
// 1. Works automatically for dynamically added elements (no re-attachment needed after OOB updates)
// 2. No timing issues with DOM updates
// 3. Single listener regardless of notebook size
(function setupPreviewEditingDelegation() {
    document.addEventListener('dblclick', function(e) {
        // Find the preview element (could be the target or an ancestor)
        const preview = e.target.closest('.md-preview, .ai-preview, .prompt-preview');
        if (!preview) return;

        // Don't trigger on copy button clicks
        if (e.target.closest('.copy-btn')) return;

        const cellId = preview.dataset.cellId;
        const field = preview.dataset.field;
        if (cellId && field) {
            switchToEdit(cellId, field);
        }
    });
})();

// Keep setupPreviewEditing as a no-op for backward compatibility
// Existing call sites (htmx:afterSettle, DOMContentLoaded) don't need modification
function setupPreviewEditing() {
    // Event delegation handles this automatically - no individual listeners needed
}

function switchToEdit(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        preview.style.display = 'none';
        textarea.style.display = 'block';
        textarea.focus();
    }
}

function switchToPreview(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        // Update preview content
        preview.innerHTML = renderMarkdown(textarea.value);
        preview.style.display = 'block';
        textarea.style.display = 'none';
    }
}

// Update preview when content changes
function updatePreview(cellId, field) {
    const textarea = document.getElementById(`${field}-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    if (textarea && preview && preview.style.display !== 'none') {
        preview.innerHTML = renderMarkdown(textarea.value);
    }
}

// ==================== Initialization ====================
function initCell(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    
    // Initialize Ace editor for code cells
    if (cell.dataset.type === 'code') {
        initAceEditor(cellId);
    }
    
    // Setup preview for note cells
    const notePreview = document.getElementById(`preview-${cellId}`);
    const noteSource = document.getElementById(`source-${cellId}`);
    if (notePreview && noteSource && cell.dataset.type === 'note') {
        notePreview.innerHTML = renderMarkdown(noteSource.value);
    }
    
    // Setup AI response preview for prompt cells
    const aiPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    const aiTextarea = document.getElementById(`output-${cellId}`);
    if (aiPreview && aiTextarea) {
        const content = aiTextarea.value;
        if (content && content.trim()) {
            aiPreview.innerHTML = renderMarkdown(content);
        } else {
            aiPreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Click ▶ to generate response...</p>';
        }
    }

    // Setup user prompt preview for prompt cells (after they've been run)
    const promptPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="prompt"]`);
    const promptTextarea = document.getElementById(`prompt-${cellId}`);
    if (promptPreview && promptTextarea) {
        const content = promptTextarea.value;
        if (content && content.trim()) {
            promptPreview.innerHTML = renderMarkdown(content);
        } else {
            promptPreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">No prompt...</p>';
        }
    }
    
    // Track focus - both focusin (for editors/inputs) and click (for cell background)
    cell.addEventListener('focusin', () => setFocusedCell(cellId));
    cell.addEventListener('click', (e) => {
        // Only set focus if clicking directly on cell or its non-interactive children
        // This allows clicking anywhere on the cell to select it
        setFocusedCell(cellId);
    });
}

// Cleanup before HTMX swaps
document.addEventListener('htmx:beforeSwap', (e) => {
    // Only destroy Ace editors within the swap target (efficient)
    const target = e.detail.target;
    if (target) {
        // If target itself is a cell with an ace-container
        if (target.classList && target.classList.contains('cell')) {
            const container = target.querySelector('.ace-container');
            if (container) {
                const cellId = container.id.replace('ace-', '');
                destroyAceEditor(cellId);
            }
        } else {
            // If target contains ace-containers
            target.querySelectorAll('.ace-container').forEach(container => {
                const cellId = container.id.replace('ace-', '');
                destroyAceEditor(cellId);
            });
        }
    }
});

// After HTMX settles (fires after all HTMX processing is complete)
document.addEventListener('htmx:afterSettle', (e) => {
    // Small delay to ensure DOM is fully ready and Ace can initialize properly
    setTimeout(() => {
        // Only initialize cells within the swap target (efficient - not ALL cells)
        const target = e.detail.target || e.detail.elt;
        if (target) {
            // If target is a cell, initialize just that cell
            if (target.classList && target.classList.contains('cell')) {
                const cellId = target.id.replace('cell-', '');
                initCell(cellId);
                // Reset streaming state for this cell (HTMX swap means request completed)
                if (streamingCellId === cellId) {
                    finishStreaming(cellId);
                }
            } else {
                // If target contains cells (e.g., OOB swap to #cells), initialize those
                target.querySelectorAll('.cell').forEach(cell => {
                    const cellId = cell.id.replace('cell-', '');
                    initCell(cellId);
                });
            }
        }
        setupPreviewEditing();
    }, 20);
});

// Handle HTMX errors - ensure streaming state is reset for both prompt and code cells
function resetCellOnError(e, errorMsg) {
    // Check if this is a cell-related request
    const target = e.detail?.target;
    if (target && target.id && target.id.startsWith('cell-')) {
        const cellId = target.id.replace('cell-', '');
        const cell = document.getElementById(`cell-${cellId}`);

        if (cell && cell.classList.contains('streaming')) {
            // Determine cell type and reset appropriately
            const isCodeCell = cell.querySelector('.ace-container') !== null;
            const isPromptCell = cell.querySelector('.prompt-source') !== null;

            if (isPromptCell && streamingCellId === cellId) {
                finishStreaming(cellId);
                const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
                if (preview) {
                    preview.innerHTML = `<p style="color: var(--error);">${errorMsg}</p>`;
                }
            } else if (isCodeCell) {
                finishCodeStreaming(cellId, true);
                const outputEl = document.getElementById(`output-${cellId}`);
                if (outputEl) {
                    outputEl.innerHTML = `<pre class="stream-output" style="color: var(--error);">${errorMsg}</pre>`;
                }
            }
        }
    }

    // Fallback: reset prompt streaming if we have a streaming cell
    if (streamingCellId) {
        finishStreaming(streamingCellId);
    }
}

document.addEventListener('htmx:responseError', (e) => {
    console.error('[HTMX] Response error:', e.detail);
    resetCellOnError(e, 'Request failed. Please try again.');
});

document.addEventListener('htmx:sendError', (e) => {
    console.error('[HTMX] Send error:', e.detail);
    resetCellOnError(e, 'Network error. Please check your connection.');
});

document.addEventListener('htmx:timeout', (e) => {
    console.error('[HTMX] Timeout:', e.detail);
    resetCellOnError(e, 'Request timed out. Please try again.');
});

// On page load
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    document.querySelectorAll('.cell').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        initCell(cellId);
    });
    setupPreviewEditing();
});

// Auto-resize textareas
document.addEventListener('input', e => {
    if (e.target.tagName === 'TEXTAREA') {
        e.target.style.height = 'auto';
        e.target.style.height = Math.max(60, e.target.scrollHeight) + 'px';
    }
});

// ==================== Theme Toggle ====================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update Ace editor themes
    const aceTheme = newTheme === 'light' ? 'ace/theme/chrome' : 'ace/theme/monokai';
    Object.values(aceEditors).forEach(editor => {
        editor.setTheme(aceTheme);
    });

    // Update toggle button
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = newTheme === 'light' ? '🌙' : '☀️';
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = savedTheme === 'light' ? '🌙' : '☀️';
}

// ==================== Model Select Toggle ====================
function toggleModelSelect(mode) {
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        // Show model dropdown only for non-mock modes
        modelSelect.style.display = mode === 'mock' ? 'none' : '';
    }
}

// ==================== Cell Collapse ====================
// Collapse levels: 0=expanded, 1=scrollable, 2=summary
const COLLAPSE_LEVELS = ['', 'collapse-scrollable', 'collapse-summary'];
const COLLAPSE_LABELS = ['Expanded', 'Scrollable', 'Summary'];

function toggleCollapse(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.toggle('collapsed');
        // Send update to server
        const isCollapsed = cell.classList.contains('collapsed');
        fetch(`${window.location.pathname}/cell/${cellId}/collapse`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `collapsed=${isCollapsed}`
        });
    }
}

function cycleCollapseLevel(cellId, section) {
    // section can be 'input', 'output', or 'both'
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    if (section === 'both') {
        cycleCollapseLevel(cellId, 'input');
        cycleCollapseLevel(cellId, 'output');
        return;
    }

    // Find the section element
    const sectionEl = cell.querySelector(`[data-collapse-section="${section}"]`);
    const btn = cell.querySelector(`[data-collapse-btn="${section}"]`);
    if (!sectionEl) return;

    // Get current level
    let currentLevel = 0;
    for (let i = COLLAPSE_LEVELS.length - 1; i > 0; i--) {
        if (COLLAPSE_LEVELS[i] && sectionEl.classList.contains(COLLAPSE_LEVELS[i])) {
            currentLevel = i;
            break;
        }
    }

    // Cycle to next level (0 -> 1 -> 2 -> 3 -> 0)
    const nextLevel = (currentLevel + 1) % COLLAPSE_LEVELS.length;

    // Remove all collapse classes
    COLLAPSE_LEVELS.forEach(cls => {
        if (cls) sectionEl.classList.remove(cls);
    });

    // Add new collapse class if not expanded
    if (COLLAPSE_LEVELS[nextLevel]) {
        sectionEl.classList.add(COLLAPSE_LEVELS[nextLevel]);
    }

    // Update button indicator
    if (btn) {
        btn.setAttribute('data-level', nextLevel);
        btn.title = `${section === 'input' ? 'Input' : 'Output'}: ${COLLAPSE_LABELS[nextLevel]} (click to cycle)`;
    }

    // Send update to server
    fetch(`${window.location.pathname}/cell/${cellId}/collapse-section`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `section=${section}&level=${nextLevel}`
    });
}

function setCollapseLevel(cellId, section, level) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    const sectionEl = cell.querySelector(`[data-collapse-section="${section}"]`);
    const btn = cell.querySelector(`[data-collapse-btn="${section}"]`);
    if (!sectionEl) return;

    // Remove all collapse classes
    COLLAPSE_LEVELS.forEach(cls => {
        if (cls) sectionEl.classList.remove(cls);
    });

    // Add new collapse class if not expanded
    if (COLLAPSE_LEVELS[level]) {
        sectionEl.classList.add(COLLAPSE_LEVELS[level]);
    }

    // Update button indicator
    if (btn) {
        btn.setAttribute('data-level', level);
        btn.title = `${section === 'input' ? 'Input' : 'Output'}: ${COLLAPSE_LABELS[level]} (click to cycle)`;
    }
}

// ==================== Cancel Streaming ====================
let cancelledCells = new Set();

function cancelStreaming(cellId) {
    cancelledCells.add(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.remove('streaming');
        // Hide cancel button, show run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }
    // Send cancel message via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: 'cancel', cell_id: cellId}));
    }
    streamingCellId = null;
}

// ==================== WebSocket for Streaming ====================
let ws = null;
let streamingCellId = null;
let currentNotebookId = null;  // Global notebook ID for use in cancelAllExecution, etc.

function connectWebSocket(notebookId) {
    currentNotebookId = notebookId;  // Store globally for other functions to use
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);

    ws.onopen = function() {
        console.log('[WS] Connected to notebook:', notebookId);
        // Send join message to register this connection with the notebook
        ws.send(JSON.stringify({type: 'join', notebook_id: notebookId}));
    };

    ws.onmessage = function(event) {
        const msg = event.data;

        // Debug: Log every message received
        console.log('[WS] RAW message received, length:', msg?.length, 'type:', typeof msg, 'starts:', msg?.substring?.(0, 50));

        // Check if message is HTML (OOB swap from collaborator) or JSON (streaming)
        if (msg && typeof msg === 'string' && msg.startsWith('<')) {
            // HTML with hx-swap-oob - process as OOB swap
            console.log('[WS] Received OOB HTML swap, length:', msg.length);
            processOOBSwap(msg);
            return;
        }

        // JSON message for streaming, thinking indicators, etc.
        let data;
        try {
            data = JSON.parse(msg);
        } catch (e) {
            console.error('[WS] Failed to parse JSON message:', msg?.substring?.(0, 100), e);
            return;
        }
        console.log('[WS] Received message:', data.type, 'cell_id:', data.cell_id || 'none');

        if (data.type === 'stream_chunk') {
            // Skip if cancelled
            if (cancelledCells.has(data.cell_id)) return;
            appendToResponse(data.cell_id, data.chunk, data.thinking);
            // Reset streaming timeout on activity
            resetStreamingTimeout();
        } else if (data.type === 'stream_end') {
            console.log('[WS] stream_end received for cell:', data.cell_id);
            cancelledCells.delete(data.cell_id);
            finishStreaming(data.cell_id);
        } else if (data.type === 'thinking_start') {
            showThinkingIndicator(data.cell_id);
            resetStreamingTimeout();
        } else if (data.type === 'thinking_end') {
            hideThinkingIndicator(data.cell_id);
        } else if (data.type === 'code_stream_start') {
            // Code cell execution started - show streaming indicator
            console.log('[WS] code_stream_start received for cell:', data.cell_id);
            startCodeStreaming(data.cell_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_stream_chunk') {
            // Append output chunk to code cell
            console.log('[WS] code_stream_chunk received for cell:', data.cell_id, 'stream:', data.stream, 'length:', data.chunk?.length || 0);
            appendCodeOutput(data.cell_id, data.chunk, data.stream);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_stream_end') {
            // Code cell execution finished
            console.log('[WS] code_stream_end received for cell:', data.cell_id, 'has_error:', data.has_error);
            finishCodeStreaming(data.cell_id, data.has_error);
        } else if (data.type === 'code_display_data') {
            // Rich output (image, HTML, plot, etc.)
            console.log('[WS] code_display_data received for cell:', data.cell_id);
            appendDisplayData(data.cell_id, data.html);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'queue_update') {
            // Queue state update from server
            console.log('[WS] queue_update received:', data);
            handleQueueUpdate(data);
        } else if (data.type === 'cell_state_change') {
            // Cell state change (queued, running, idle)
            console.log('[WS] cell_state_change received:', data.cell_id, data.state);
            // State changes are now handled via queue_update for consistency
        }
    };

    ws.onclose = function() {
        console.log('[WS] Disconnected, reconnecting in 3s...');
        setTimeout(() => connectWebSocket(notebookId), 3000);
    };

    ws.onerror = function(error) {
        console.error('[WS] Error:', error);
    };
}

function appendToResponse(cellId, chunk, isThinking) {
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        if (textarea.value === 'Generating...' || textarea.value === 'Click ▶ to generate response...' || textarea.value.startsWith('🧠')) {
            textarea.value = '';
        }
        textarea.value += chunk;
        if (preview) {
            preview.innerHTML = renderMarkdown(textarea.value);
        }
    }
}

function showThinkingIndicator(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (preview) {
        preview.innerHTML = '<div class="thinking-indicator"><span>🧠</span> Thinking...</div>';
    }
    if (cell) {
        const header = cell.querySelector('.cell-header');
        if (header && !header.querySelector('.thinking-indicator')) {
            const indicator = document.createElement('span');
            indicator.className = 'thinking-indicator';
            indicator.innerHTML = '🧠 Thinking...';
            indicator.id = `thinking-${cellId}`;
            header.querySelector('.cell-actions')?.prepend(indicator);
        }
    }
}

function hideThinkingIndicator(cellId) {
    const indicator = document.getElementById(`thinking-${cellId}`);
    if (indicator) indicator.remove();
}

function finishStreaming(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.remove('streaming');
        // Hide cancel button, show run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }
    hideThinkingIndicator(cellId);
    streamingCellId = null;
    // Clear safety timeout
    if (streamingTimeoutId) {
        clearTimeout(streamingTimeoutId);
        streamingTimeoutId = null;
    }
}

let streamingTimeoutId = null;
const STREAMING_TIMEOUT_MS = 120000; // 2 minutes safety timeout

function startStreaming(cellId, useThinking) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.classList.add('streaming');
        // Show cancel button, hide run button
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = '';
        if (runBtn) runBtn.style.display = 'none';
    }
    streamingCellId = cellId;
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        textarea.value = useThinking ? '🧠 Thinking...' : 'Generating...';
    }
    if (preview && useThinking) {
        preview.innerHTML = '<div class="thinking-indicator"><span>🧠</span> Thinking...</div>';
    }

    // Set safety timeout to reset streaming state
    if (streamingTimeoutId) clearTimeout(streamingTimeoutId);
    streamingTimeoutId = setTimeout(() => {
        if (streamingCellId === cellId) {
            console.warn('[Streaming] Safety timeout reached, resetting streaming state');
            finishStreaming(cellId);
        }
    }, STREAMING_TIMEOUT_MS);
}

function resetStreamingTimeout() {
    // Call this when we receive streaming activity to reset the timeout
    if (streamingTimeoutId && streamingCellId) {
        clearTimeout(streamingTimeoutId);
        const cellId = streamingCellId;
        streamingTimeoutId = setTimeout(() => {
            if (streamingCellId === cellId) {
                console.warn('[Streaming] Safety timeout reached, resetting streaming state');
                finishStreaming(cellId);
            }
        }, STREAMING_TIMEOUT_MS);
    }
}

// ==================== Code Cell Streaming Functions ====================

// ANSI color code mapping
const ANSI_COLORS = {
    '30': '#000', '31': '#c00', '32': '#0a0', '33': '#a50',
    '34': '#00a', '35': '#a0a', '36': '#0aa', '37': '#aaa',
    '90': '#555', '91': '#f55', '92': '#5f5', '93': '#ff5',
    '94': '#55f', '95': '#f5f', '96': '#5ff', '97': '#fff',
    '40': 'background:#000', '41': 'background:#c00',
    '42': 'background:#0a0', '43': 'background:#a50',
    '44': 'background:#00a', '45': 'background:#a0a',
    '46': 'background:#0aa', '47': 'background:#aaa'
};

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function ansiToHtml(text) {
    let result = '';
    let openSpans = 0;

    const parts = text.split(/(\x1b\[[0-9;]*m)/);
    for (const part of parts) {
        const match = part.match(/\x1b\[([0-9;]*)m/);
        if (match) {
            const codes = match[1].split(';');
            for (const code of codes) {
                if (code === '0' || code === '') {
                    // Reset all styles
                    while (openSpans > 0) {
                        result += '</span>';
                        openSpans--;
                    }
                } else if (code === '1') {
                    result += '<span style="font-weight:bold">';
                    openSpans++;
                } else if (ANSI_COLORS[code]) {
                    const style = ANSI_COLORS[code].includes(':')
                        ? ANSI_COLORS[code]
                        : `color:${ANSI_COLORS[code]}`;
                    result += `<span style="${style}">`;
                    openSpans++;
                }
            }
        } else {
            result += escapeHtml(part);
        }
    }

    // Close any remaining open spans
    while (openSpans > 0) {
        result += '</span>';
        openSpans--;
    }

    return result;
}

// Track raw text content for carriage return handling
const streamTextContent = new Map();

function startCodeStreaming(cellId) {
    console.log('[Code] startCodeStreaming called for cell:', cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (!cell) {
        console.error('[Code] Cell element not found:', `cell-${cellId}`);
        return;
    }

    if (!outputEl) {
        console.error('[Code] Output element not found:', `output-${cellId}`);
    }

    cell.classList.add('streaming');
    const cancelBtn = cell.querySelector('.btn-cancel');
    const runBtn = cell.querySelector('.btn-run');
    if (cancelBtn) cancelBtn.style.display = '';
    if (runBtn) runBtn.style.display = 'none';

    if (outputEl) {
        outputEl.innerHTML = '';  // Clear for fresh output
        outputEl.classList.remove('error');
    }

    // Reset text content tracker for this cell
    streamTextContent.set(cellId, '');

    console.log('[Code] Started streaming for cell:', cellId, 'cell found:', !!cell, 'output found:', !!outputEl);
}

function appendCodeOutput(cellId, chunk, streamName) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Get or create stream output container
    let streamEl = outputEl.querySelector('.stream-output');
    if (!streamEl) {
        streamEl = document.createElement('pre');
        streamEl.className = 'stream-output';
        outputEl.appendChild(streamEl);
    }

    if (streamName === 'stderr') {
        outputEl.classList.add('error');
    }

    // Get current raw text and apply chunk
    let currentText = streamTextContent.get(cellId) || '';

    // Handle carriage return for progress bars (tqdm)
    if (chunk.includes('\r')) {
        const lines = currentText.split('\n');
        const parts = chunk.split('\r');

        for (let i = 0; i < parts.length; i++) {
            if (i === 0) {
                // First part appends to current line
                lines[lines.length - 1] += parts[i];
            } else {
                // After \r, replace current line content
                lines[lines.length - 1] = parts[i];
            }
        }
        currentText = lines.join('\n');
    } else {
        currentText += chunk;
    }

    // Store updated raw text
    streamTextContent.set(cellId, currentText);

    // Render with ANSI color conversion
    streamEl.innerHTML = ansiToHtml(currentText);
    streamEl.scrollTop = streamEl.scrollHeight;
}

function appendDisplayData(cellId, html) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Create display data container
    const displayEl = document.createElement('div');
    displayEl.className = 'display-data';
    displayEl.innerHTML = html;
    outputEl.appendChild(displayEl);

    // Execute any scripts in the HTML (for interactive widgets)
    displayEl.querySelectorAll('script').forEach(script => {
        const newScript = document.createElement('script');
        newScript.textContent = script.textContent;
        script.parentNode.replaceChild(newScript, script);
    });
}

function finishCodeStreaming(cellId, hasError) {
    console.log('[Code] finishCodeStreaming called for cell:', cellId, 'hasError:', hasError);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (cell) {
        cell.classList.remove('streaming');
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (runBtn) runBtn.style.display = '';
    }

    if (outputEl && hasError) {
        outputEl.classList.add('error');
    }

    // Clean up text content tracker
    streamTextContent.delete(cellId);

    // Clear the streaming timeout
    clearCodeStreamingTimeout(cellId);

    console.log('[Code] Finished streaming for cell:', cellId, hasError ? '(with errors)' : '');
}

// ============================================================================
// Queue State Management
// ============================================================================

// Track queue state for cells
const cellQueueState = new Map(); // cellId -> {state: 'queued'|'running'|'idle', position: number}

function handleQueueUpdate(data) {
    // Clear all previous queue states
    cellQueueState.forEach((_, cellId) => {
        clearCellQueueState(cellId);
    });
    cellQueueState.clear();

    // Set running cell
    if (data.running_cell_id) {
        updateCellQueueState(data.running_cell_id, 'running', null);
    }

    // Set queued cells with positions
    if (data.queued_cell_ids) {
        data.queued_cell_ids.forEach((cellId, index) => {
            updateCellQueueState(cellId, 'queued', index + 1);
        });
    }

    // Show/hide Cancel All button based on queue state
    const hasQueuedOrRunning = data.running_cell_id || (data.queued_cell_ids && data.queued_cell_ids.length > 0);
    updateCancelAllButton(hasQueuedOrRunning);
}

function updateCellQueueState(cellId, state, position) {
    cellQueueState.set(cellId, { state, position });
    updateCellVisualState(cellId, state, position);
}

function clearCellQueueState(cellId) {
    cellQueueState.delete(cellId);
    updateCellVisualState(cellId, 'idle', null);
}

function updateCellVisualState(cellId, state, queuePosition) {
    const cell = document.getElementById(`cell-${cellId}`);
    const runBtn = cell?.querySelector('.btn-run');
    const outputEl = document.getElementById(`output-${cellId}`);

    if (!cell) return;

    // Remove queued class only - streaming is managed by startCodeStreaming/stopCodeStreaming
    cell.classList.remove('queued');

    switch(state) {
        case 'queued':
            cell.classList.add('queued');
            if (runBtn) {
                runBtn.style.display = '';
                runBtn.innerHTML = '⏳';
                runBtn.disabled = true;
            }
            if (outputEl) {
                outputEl.innerHTML = `<pre class="stream-output" style="color: var(--accent-yellow);">Queued (position ${queuePosition})...</pre>`;
                outputEl.classList.remove('error');
            }
            break;

        case 'running':
            // Running state is handled by startCodeStreaming
            // Just ensure queued class is removed
            break;

        case 'idle':
        default:
            if (runBtn) {
                runBtn.style.display = '';
                runBtn.innerHTML = '▶';
                runBtn.disabled = false;
            }
            break;
    }
}

function updateCancelAllButton(show) {
    const cancelAllBtn = document.querySelector('.btn-cancel-all');
    if (cancelAllBtn) {
        cancelAllBtn.style.display = show ? '' : 'none';
    }
}

async function cancelAllExecution() {
    if (!currentNotebookId) {
        console.error('[Queue] Cannot cancel: no notebook ID set');
        return;
    }
    try {
        console.log('[Queue] Cancelling all execution for notebook:', currentNotebookId);
        await fetch(`/notebook/${currentNotebookId}/queue/cancel_all`, { method: 'POST' });
    } catch (e) {
        console.error('[Queue] Failed to cancel all:', e);
    }
}

// Code cell streaming timeout mechanism
let codeStreamingTimeouts = new Map();  // Track timeouts per cell
const CODE_STREAMING_TIMEOUT_MS = 30000; // 30 seconds safety timeout (reduced for better UX)

// Called immediately when user clicks run on a code cell
// Provides visual feedback before WebSocket code_stream_start arrives
function prepareCodeRun(cellId) {
    console.log('[Code] prepareCodeRun called for cell:', cellId);

    // Skip if cell is already queued or running
    const queueState = cellQueueState.get(cellId);
    if (queueState && (queueState.state === 'queued' || queueState.state === 'running')) {
        console.log('[Code] Cell already queued/running, skipping prepareCodeRun');
        return;
    }

    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    if (cell) {
        cell.classList.add('streaming');
        const cancelBtn = cell.querySelector('.btn-cancel');
        const runBtn = cell.querySelector('.btn-run');
        if (cancelBtn) cancelBtn.style.display = '';
        if (runBtn) runBtn.style.display = 'none';
    }

    // Clear output and show "Queuing..." indicator (queue_update will update to "Queued (position N)...")
    if (outputEl) {
        outputEl.innerHTML = '<pre class="stream-output" style="color: var(--text-muted);">Queuing...</pre>';
        outputEl.classList.remove('error');
    }

    // Reset text content tracker
    streamTextContent.set(cellId, '');

    // Set safety timeout to reset streaming state if server doesn't respond
    clearCodeStreamingTimeout(cellId);
    const timeoutId = setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.warn('[Code] Safety timeout reached for cell:', cellId);
            finishCodeStreaming(cellId, true);
            const outputEl = document.getElementById(`output-${cellId}`);
            if (outputEl) {
                const currentOutput = outputEl.textContent?.trim();
                if (!currentOutput || currentOutput === 'Running...') {
                    outputEl.innerHTML = '<pre class="stream-output" style="color: var(--error);">Execution timed out. Please try again.</pre>';
                }
            }
        }
    }, CODE_STREAMING_TIMEOUT_MS);
    codeStreamingTimeouts.set(cellId, timeoutId);

    console.log('[Code] Preparing to run cell:', cellId);
}

function clearCodeStreamingTimeout(cellId) {
    const timeoutId = codeStreamingTimeouts.get(cellId);
    if (timeoutId) {
        clearTimeout(timeoutId);
        codeStreamingTimeouts.delete(cellId);
    }
}

function resetCodeStreamingTimeout(cellId) {
    // Call this when we receive streaming activity to reset the timeout
    clearCodeStreamingTimeout(cellId);
    const timeoutId = setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.warn('[Code] Safety timeout reached for cell:', cellId);
            finishCodeStreaming(cellId, true);
        }
    }, CODE_STREAMING_TIMEOUT_MS);
    codeStreamingTimeouts.set(cellId, timeoutId);
}

function interruptCodeCell(notebookId, cellId) {
    console.log('[Code] interruptCodeCell called for cell:', cellId);

    // Use cancelAllExecution to stop running cell AND clear the queue
    // This ensures we don't just interrupt one cell and continue with others
    cancelAllExecution();

    // Clear the streaming timeout for this cell
    clearCodeStreamingTimeout(cellId);

    // Wait a bit for server to finish, then check if cell is still stuck
    setTimeout(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell && cell.classList.contains('streaming')) {
            console.log('[Code] Cell still streaming after interrupt, forcing reset');
            finishCodeStreaming(cellId, true);
            const outputEl = document.getElementById(`output-${cellId}`);
            if (outputEl) {
                const currentOutput = outputEl.textContent?.trim();
                if (!currentOutput || currentOutput === 'Running...' || currentOutput === 'Stopping...') {
                    outputEl.innerHTML = '<pre class="stream-output" style="color: var(--warning);">Execution interrupted</pre>';
                }
            }
        }
    }, 2000); // Wait 2 seconds for server to respond

    // Immediately update UI to show stopping state
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        const outputEl = document.getElementById(`output-${cellId}`);
        if (outputEl) {
            const currentOutput = outputEl.textContent?.trim();
            if (currentOutput === 'Running...' || currentOutput.startsWith('Queued')) {
                outputEl.innerHTML = '<pre class="stream-output" style="color: var(--text-muted);">Stopping...</pre>';
            }
        }
    }
}

// ==================== Collaborative WebSocket OOB Swap Handler ====================

function processOOBSwap(html) {
    // Process HTML with hx-swap-oob attributes from WebSocket
    // This handles both full cells container updates and single cell updates
    console.log('[OOB] processOOBSwap called, HTML length:', html.length);

    // Parse the HTML to extract the element(s)
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    const elements = template.content.children;
    console.log('[OOB] Parsed elements count:', elements.length);

    for (const element of elements) {
        const oobAttr = element.getAttribute('hx-swap-oob');
        console.log('[OOB] Element tag:', element.tagName, 'id:', element.id, 'oobAttr:', oobAttr);

        // Handle swap strategies like "beforeend:#js-script" for script injection
        if (oobAttr && oobAttr.includes(':')) {
            const [swapStrategy, targetSelector] = oobAttr.split(':');
            const target = document.querySelector(targetSelector);
            console.log('[OOB] Swap strategy:', swapStrategy, 'target:', targetSelector, 'found:', !!target);

            if (target) {
                element.removeAttribute('hx-swap-oob');

                // For script injection, we need to manually execute the scripts
                // innerHTML doesn't auto-execute scripts for security reasons
                const scripts = element.querySelectorAll('script');
                if (scripts.length > 0) {
                    console.log('[OOB] Found', scripts.length, 'script(s) to execute');
                    scripts.forEach(script => {
                        const newScript = document.createElement('script');
                        // Copy all attributes
                        Array.from(script.attributes).forEach(attr => {
                            newScript.setAttribute(attr.name, attr.value);
                        });
                        newScript.textContent = script.textContent;

                        if (swapStrategy === 'beforeend') {
                            target.appendChild(newScript);
                        } else if (swapStrategy === 'afterbegin') {
                            target.insertBefore(newScript, target.firstChild);
                        } else {
                            target.appendChild(newScript);
                        }
                        console.log('[OOB] Script executed');
                    });
                } else {
                    // Regular content, use innerHTML based on swap strategy
                    if (swapStrategy === 'beforeend') {
                        target.insertAdjacentHTML('beforeend', element.innerHTML);
                    } else if (swapStrategy === 'afterbegin') {
                        target.insertAdjacentHTML('afterbegin', element.innerHTML);
                    } else if (swapStrategy === 'innerHTML') {
                        target.innerHTML = element.innerHTML;
                    }
                }
            }
            continue;
        }

        if (oobAttr !== 'true') continue;

        // Handle script elements specially - they need to be manually executed
        if (element.tagName === 'SCRIPT') {
            console.log('[OOB] Executing script element with id:', element.id);
            const newScript = document.createElement('script');
            // Copy all attributes except hx-swap-oob
            Array.from(element.attributes).forEach(attr => {
                if (attr.name !== 'hx-swap-oob') {
                    newScript.setAttribute(attr.name, attr.value);
                }
            });
            newScript.textContent = element.textContent;

            // If a script with this ID exists, replace it; otherwise append to body
            const existingScript = element.id ? document.getElementById(element.id) : null;
            if (existingScript) {
                existingScript.replaceWith(newScript);
            } else {
                document.body.appendChild(newScript);
            }
            console.log('[OOB] Script executed successfully');
            continue;
        }

        const targetId = element.id;
        if (!targetId) {
            console.log('[OOB] Skipping - no targetId');
            continue;
        }

        const target = document.getElementById(targetId);
        if (!target) {
            console.log('[OOB] Skipping - target not found for id:', targetId);
            continue;
        }
        console.log('[OOB] Found target element:', targetId);

        // Check if this is a cell update
        if (targetId.startsWith('cell-')) {
            const cellId = targetId.replace('cell-', '');
            const isEditing = target.contains(document.activeElement);
            const isStreaming = target.classList.contains('streaming');

            // Skip update if user is editing this cell or it's streaming
            if (isEditing || isStreaming) {
                console.log('[WS] Skipping OOB swap for cell being edited/streamed:', cellId);
                continue;
            }

            // Replace the cell
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);

            // CRITICAL: Reinitialize HTMX bindings on the new element
            // Without this, hx-post/hx-get attributes won't work!
            const newCell = document.getElementById(targetId);
            if (newCell) {
                htmx.process(newCell);

                // Reinitialize Ace editor if it's a code cell
                if (newCell.dataset.type === 'code') {
                    setTimeout(() => initAceEditor(cellId), 0);
                }
            }

            // Re-render previews for this cell
            renderCellPreviews(cellId);
        }
        else if (targetId === 'cells') {
            // Full cells container update (e.g., from dialoghelper add_msg)
            console.log('[OOB] Processing cells container update');
            // Save currently focused cell ID before update
            const focusedCell = document.activeElement?.closest('.cell');
            const focusedCellId = focusedCell?.id?.replace('cell-', '');

            // Only skip if user is actively typing AND no cell is currently streaming
            // If ANY cell is streaming (executing), we need to allow updates for add_msg() to work
            // The Ace editor's hidden textarea keeps focus during execution, but that's not "real typing"
            const isInInput = document.activeElement?.matches('input, textarea, .ace_text-input');
            const anyCellStreaming = document.querySelector('.cell.streaming') !== null;
            const shouldSkip = isInInput && !anyCellStreaming;
            console.log('[OOB] isInInput:', isInInput, 'anyCellStreaming:', anyCellStreaming, 'shouldSkip:', shouldSkip);

            if (shouldSkip) {
                console.log('[OOB] Skipping cells container update - user is typing and no cell is streaming');
                continue;
            }

            // Replace the cells container
            console.log('[OOB] Replacing cells container');
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            console.log('[OOB] Cells container replaced successfully');

            // CRITICAL: Reinitialize HTMX bindings on the new cells container
            // Without this, hx-post/hx-get attributes won't work!
            const newCells = document.getElementById('cells');
            if (newCells) {
                htmx.process(newCells);
            }

            // Reinitialize Ace editors for all code cells
            reinitializeAceEditors();

            // Re-render all markdown previews
            renderAllPreviews();

            // Restore focus if possible
            if (focusedCellId) {
                const restoredCell = document.getElementById(`cell-${focusedCellId}`);
                if (restoredCell) {
                    setFocusedCell(focusedCellId);
                }
            }
        }
    }
}

function reinitializeAceEditors() {
    // Destroy all existing Ace editors
    for (const cellId of Object.keys(aceEditors)) {
        destroyAceEditor(cellId);
    }

    // Find all code cells and initialize their editors
    document.querySelectorAll('.cell[data-type="code"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => initAceEditor(cellId), 0);
    });
}

function renderAllPreviews() {
    // Re-render all markdown previews after a collaborative update
    document.querySelectorAll('.md-preview, .ai-preview, .prompt-preview').forEach(preview => {
        const cellId = preview.dataset.cellId;
        const field = preview.dataset.field;
        if (cellId && field) {
            renderCellPreviews(cellId);
        }
    });
}

function renderCellPreviews(cellId) {
    // Render markdown preview for a specific cell
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    // Handle note cells
    const notePreview = document.getElementById(`preview-${cellId}`);
    if (notePreview) {
        const textarea = document.getElementById(`source-${cellId}`);
        if (textarea) {
            notePreview.innerHTML = renderMarkdown(textarea.value);
        }
    }

    // Handle prompt cells - render both prompt and output previews
    const promptPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="prompt"]`);
    if (promptPreview) {
        const promptTextarea = document.getElementById(`prompt-${cellId}`);
        if (promptTextarea) {
            promptPreview.innerHTML = renderMarkdown(promptTextarea.value);
        }
    }

    const outputPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (outputPreview) {
        const outputTextarea = document.getElementById(`output-${cellId}`);
        if (outputTextarea && outputTextarea.value) {
            outputPreview.innerHTML = renderMarkdown(outputTextarea.value);
        }
    }
}
