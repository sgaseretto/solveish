/* ==========================================================================
   Dialeng Client-Side JavaScript
   
   This file contains all client-side logic for the Dialeng notebook app:
   - Monaco Editor management
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

// ==================== DialogHelper Bidirectional Data ====================
// Used by screenshot.js and other dialoghelper event handlers to push data
// back to Python via the /push_data_blocking_ endpoint.
async function pushData(idx, data) {
    const params = new URLSearchParams();
    params.append('dlg_name', window.NOTEBOOK_ID);
    params.append('data_id', String(idx));
    params.append('data', JSON.stringify(data));
    await fetch('/push_data_blocking_', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: params.toString()
    });
}

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

// ==================== Monaco Editor Management ====================
const monacoEditors = {};
let monacoReady = false;
const pendingEditorInits = [];

// Monaco AMD loader initialization
require.config({ paths: { 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' }});
require(['vs/editor/editor.main'], function() {
    monacoReady = true;

    // Register kernel-backed completion provider for Python
    let _completionTimer = null;
    monaco.languages.registerCompletionItemProvider('python', {
        triggerCharacters: ['.'],
        provideCompletionItems: function(model, position, context, token) {
            const nbId = window.NOTEBOOK_ID;
            if (!nbId) return { suggestions: [] };

            const code = model.getValue();
            const cursorOffset = model.getOffsetAt(position);

            if (_completionTimer) clearTimeout(_completionTimer);

            return new Promise((resolve) => {
                _completionTimer = setTimeout(async () => {
                    try {
                        const resp = await fetch(`/api/complete/${nbId}`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                            body: new URLSearchParams({ code, cursor_pos: cursorOffset })
                        });
                        if (!resp.ok) { resolve({ suggestions: [] }); return; }
                        const data = await resp.json();

                        const word = model.getWordUntilPosition(position);
                        const range = {
                            startLineNumber: position.lineNumber,
                            startColumn: word.startColumn,
                            endLineNumber: position.lineNumber,
                            endColumn: position.column,
                        };

                        const suggestions = (data.matches || []).map((m, i) => ({
                            label: m,
                            kind: monaco.languages.CompletionItemKind.Variable,
                            insertText: m,
                            range: range,
                            sortText: String(i).padStart(5, '0'),
                        }));
                        resolve({ suggestions });
                    } catch (e) {
                        resolve({ suggestions: [] });
                    }
                }, 150);
            });
        }
    });

    // Initialize any editors that were requested before Monaco loaded
    pendingEditorInits.forEach(args => initMonacoEditor(...args));
    pendingEditorInits.length = 0;
});

// ---------------------------------------------------------------------------
// Monaco Editor Initialization
//
// EXPECTED BEHAVIOR (from the user's perspective):
//   - When a cell is first loaded or created, the editor should appear with
//     full syntax highlighting immediately — no flash of plain white text.
//   - When a cell finishes executing, the output appears below the editor
//     and the editor keeps its highlighted source code intact.
//   - Scrolling inside an editor should propagate to the notebook when the
//     editor content is fully scrolled (no scroll trapping).
//   - Adding/deleting/moving cells should preserve the current scroll
//     position — the notebook should NOT jump to the top or bottom.
//   - Keyboard shortcuts (Shift+Enter, Ctrl+Enter, Ctrl+S) should work
//     even though Monaco has built-in bindings for those keys.
//
// KNOWN ISSUE — Flash of Unstyled Text (FOUST):
//   When cells are re-rendered via HTMX OOB swaps (e.g., after execution),
//   the entire cell DOM is replaced. The new HTML contains an inline Script
//   that calls initMonacoEditor(), which destroys the old editor and creates
//   a fresh one via monaco.editor.create(). Monaco renders text immediately
//   in plain white, then tokenizes asynchronously via web workers. This
//   creates a brief flash where the code appears without syntax colors.
//
//   Current mitigation: we set container opacity to 0 and poll for colored
//   token spans (mtk classes beyond the default mtk1). Once tokenization
//   is detected, we reveal the editor. This works in most cases but the
//   issue can still occasionally appear — particularly when:
//     - The browser tab is under heavy load (tokenization takes longer)
//     - Many cells are re-initialized simultaneously (bulk #cells swap)
//     - The code contains no keywords (only comments or plain identifiers)
//       so all tokens remain mtk1 and we fall through to the 30-frame timeout
//
//   A more robust fix would require one of:
//     (a) Avoid full DOM replacement — use targeted OOB swaps that only
//         update the output section, leaving the editor DOM untouched
//     (b) Cache editor models and re-attach them instead of recreating
//     (c) Use Monaco's built-in tokenization API (not public) to wait for
//         tokenization to complete before rendering
//   See docs/how_it_works/17_editor_cell_transitions.md for full details.
// ---------------------------------------------------------------------------
function initMonacoEditor(cellId, mode = 'python') {
    if (!monacoReady) {
        pendingEditorInits.push([cellId, mode]);
        return null;
    }

    const container = document.getElementById(`monaco-${cellId}`);
    if (!container) return null;

    // If editor already exists AND the container still has the editor DOM (not replaced),
    // skip re-initialization — this prevents FOUST when htmx:afterSettle fires
    // for non-swap responses (e.g., hx_swap="none" on code cell run)
    if (monacoEditors[cellId]) {
        if (container.querySelector('.monaco-editor')) {
            return monacoEditors[cellId];
        }
        // Container was replaced (HTMX re-render) — destroy old editor
        monacoEditors[cellId].dispose();
        delete monacoEditors[cellId];
    }

    // Get initial content from hidden textarea
    const textarea = document.getElementById(`source-${cellId}`);
    const initialContent = textarea ? textarea.value : '';

    // Hide container until tokenization completes to mitigate FOUST (see above).
    // NOTE: This mitigation is imperfect — see "KNOWN ISSUE" comment above.
    container.innerHTML = '';
    container.className = 'monaco-container';
    container.style.opacity = '0';

    const isDark = (document.documentElement.getAttribute('data-theme') || 'dark') !== 'light';
    const langMap = { 'python': 'python', 'sh': 'shell' };

    const editor = monaco.editor.create(container, {
        value: initialContent,
        language: langMap[mode] || 'python',
        theme: isDark ? 'vs-dark' : 'vs',
        fontSize: 14,
        fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'on',
        minimap: { enabled: false },
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        overviewRulerLanes: 0,
        hideCursorInOverviewRuler: true,
        renderLineHighlight: 'line',
        glyphMargin: false,
        folding: false,
        scrollbar: { vertical: 'hidden', horizontal: 'auto', alwaysConsumeMouseWheel: false },
    });

    // Auto-resize to content (replaces Ace minLines/maxLines)
    const minHeight = 60;  // ~3 lines
    const maxHeight = 600; // ~30 lines
    function updateEditorHeight() {
        const contentHeight = Math.max(minHeight, Math.min(maxHeight, editor.getContentHeight()));
        container.style.height = contentHeight + 'px';
        editor.layout();
    }
    editor.onDidContentSizeChange(updateEditorHeight);
    updateEditorHeight();

    // Reveal editor after syntax tokenization (prevents flash of unstyled text).
    // Monaco tokenizes asynchronously after create(). We poll for colored token
    // spans (mtk classes > mtk1) which indicate highlighting has been applied.
    if (initialContent.trim()) {
        let attempts = 0;
        const pollTokens = () => {
            attempts++;
            // mtk1 = default/unstyled text. mtk3+ = keyword/string/etc colored tokens.
            // When we see diverse mtk classes, tokenization is done.
            const tokenSpans = container.querySelectorAll('.view-lines [class*="mtk"]');
            let hasColoredTokens = false;
            for (const span of tokenSpans) {
                if (span.className !== 'mtk1') { hasColoredTokens = true; break; }
            }
            if (hasColoredTokens || attempts >= 30) {
                container.style.opacity = '1';
            } else {
                requestAnimationFrame(pollTokens);
            }
        };
        requestAnimationFrame(pollTokens);
    } else {
        container.style.opacity = '1';
    }

    // Sync to hidden textarea on change
    if (textarea) {
        editor.onDidChangeModelContent(() => {
            textarea.value = editor.getValue();
        });
    }

    // Focus tracking
    editor.onDidFocusEditorText(() => setFocusedCell(cellId));

    // Keyboard shortcuts (use addAction to properly override built-in Monaco keybindings)
    // Shift+Enter: run cell and move to next
    editor.addAction({
        id: 'dialeng-shift-enter-' + cellId,
        label: 'Run Cell and Move Next',
        keybindings: [monaco.KeyMod.Shift | monaco.KeyCode.Enter],
        run: () => {
            syncMonacoToTextarea(cellId);
            const cell = container.closest('.cell');
            if (cell) {
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
                moveToNextCell(cell);
            }
        }
    });

    // Ctrl/Cmd+Enter: run cell
    editor.addAction({
        id: 'dialeng-ctrl-enter-' + cellId,
        label: 'Run Cell',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
        run: () => {
            syncMonacoToTextarea(cellId);
            const cell = container.closest('.cell');
            if (cell) {
                const btn = cell.querySelector('.btn-run');
                if (btn) btn.click();
            }
        }
    });

    // Ctrl/Cmd+S: save notebook
    editor.addAction({
        id: 'dialeng-save-' + cellId,
        label: 'Save Notebook',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
        run: () => {
            document.getElementById('save-btn')?.click();
        }
    });

    monacoEditors[cellId] = editor;
    return editor;
}

function syncMonacoToTextarea(cellId) {
    const editor = monacoEditors[cellId];
    const textarea = document.getElementById(`source-${cellId}`);
    if (editor && textarea) {
        textarea.value = editor.getValue();
    }
}

function getMonacoContent(cellId) {
    const editor = monacoEditors[cellId];
    return editor ? editor.getValue() : '';
}

function destroyMonacoEditor(cellId) {
    if (monacoEditors[cellId]) {
        monacoEditors[cellId].dispose();
        delete monacoEditors[cellId];
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

    // Suppress any pending HTMX scroll restore so it doesn't fight us
    _htmxScrollRestore = null;

    // Scroll cell into view - use 'center' so the focused cell is clearly visible
    cell.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // If it's a code or shell cell with Monaco editor, focus the editor
    if (cell.dataset.type === 'code' || cell.dataset.type === 'shell') {
        const editor = monacoEditors[cellId];
        if (editor) {
            editor.focus();
        }
    } else if (cell.dataset.type === 'prompt') {
        // For prompt cells, focus the prompt textarea if visible
        const promptTextarea = cell.querySelector('.prompt-content[name="prompt_source"]');
        if (promptTextarea && promptTextarea.style.display !== 'none') {
            promptTextarea.focus();
        } else {
            // If prompt has been run (has preview), just select the cell
            // and move DOM focus out of any Monaco editor
            cell.tabIndex = -1;
            cell.focus();
        }
    } else {
        // Note cells and other types: move DOM focus to the cell element
        // so that the previous Monaco editor loses keyboard focus.
        // Without this, Shift+Enter would re-run the previous code cell.
        cell.tabIndex = -1;
        cell.focus();
    }
}

function getFocusedCellId() {
    const active = document.activeElement;
    if (active) {
        const cell = active.closest('.cell');
        if (cell) return cell.id.replace('cell-', '');
    }
    for (const [cellId, editor] of Object.entries(monacoEditors)) {
        if (editor.hasTextFocus()) return cellId;
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
    // Create a new code cell at the end (Jupyter behavior)
    createNewCellAtEnd();
}

function createNewCellAtEnd() {
    // Count current cells to determine position
    const cells = document.querySelectorAll('#cells .cell');
    const position = cells.length;

    // Flag: scroll to the next cell inserted via WS cell_add
    _pendingScrollToNewCell = true;

    // POST to create the cell on the server. We intentionally DON'T use
    // htmx.ajax with outerHTML swap — that would replace the entire #cells
    // container, destroying all Monaco editors and causing race conditions.
    // Instead, the server broadcasts a cell_add WS message, and the WS
    // handler inserts just the new cell (existing editors untouched).
    fetch(`${window.location.pathname}/cell/add?pos=${position}&type=code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    // The WS cell_add handler will insert the cell and scroll to it
    // when it sees _pendingScrollToNewCell is true.
}

function addCellAtRow(btn, nbId, cellType) {
    // Compute position dynamically from the add-row's index in the DOM.
    // This avoids stale hardcoded pos values when cells are added/deleted/moved.
    const addRow = btn.closest('.add-row');
    if (!addRow) return;
    const allAddRows = document.querySelectorAll('#cells .add-row');
    const pos = Array.from(allAddRows).indexOf(addRow);
    if (pos < 0) return;

    fetch(`${window.location.pathname}/cell/add?pos=${pos}&type=${cellType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    // The WS cell_add handler will insert the cell into the DOM.
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', e => {
    const target = e.target;
    const mod = e.ctrlKey || e.metaKey;
    const inMonaco = target.closest('.monaco-editor');
    const inInput = target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.isContentEditable;
    
    let currentCellId = getFocusedCellId();
    if (!currentCellId && target.closest('.cell')) {
        currentCellId = target.closest('.cell').id.replace('cell-', '');
    }

    // ===== Escape to close sidebars =====
    if (e.key === 'Escape') {
        const sidebar = document.getElementById('settings-sidebar');
        if (sidebar && sidebar.classList.contains('open')) {
            toggleSettings();
            return;
        }
        const outlineSidebar = document.getElementById('outline-sidebar');
        if (outlineSidebar && outlineSidebar.classList.contains('outline-open')) {
            toggleOutline();
            return;
        }
    }

    // ===== Ctrl+Shift+O to toggle outline =====
    if (e.ctrlKey && e.shiftKey && e.key === 'O') {
        e.preventDefault();
        toggleOutline();
        return;
    }

    // ===== Ctrl+Shift+E to toggle file explorer =====
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'E') {
        e.preventDefault();
        toggleFileExplorer();
        return;
    }

    // ===== D D to delete cell (Jupyter style) =====
    if (e.key === 'd' || e.key === 'D') {
        if (!inInput && !inMonaco) {
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
    if (e.shiftKey && e.key === 'Enter' && !inMonaco) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncMonacoToTextarea(cellId);
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
    if (mod && e.key === 'Enter' && !inMonaco) {
        // Use currentCellId (from getFocusedCellId) or fall back to target's cell
        const cellId = currentCellId || (target.closest('.cell')?.id.replace('cell-', ''));
        if (cellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${cellId}`);
            if (cell) {
                syncMonacoToTextarea(cellId);
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
    if (mod && e.key === 's' && !inMonaco) {
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
        Object.values(monacoEditors).forEach(ed => ed.trigger('keyboard', 'blur', {}));
        lastKey = 'Escape';
        lastKeyTime = now;
    }

    // ===== Z - Collapse shortcuts =====
    // Z: cycle input collapse, Shift+Z: cycle output collapse, Alt+Z: cycle both
    if ((e.key === 'z' || e.key === 'Z') && !inInput && !inMonaco) {
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
    if (['0', '1', '2', '3'].includes(e.key) && !inInput && !inMonaco && !mod) {
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
        if (currentCellId && !inMonaco) {
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
        if (currentCellId && !inMonaco) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                const moveBtn = cell.querySelector('button[title="Move down"]');
                if (moveBtn) moveBtn.click();
            }
        }
    }
    
    // ===== h/p/e - Toggle cell state shortcuts =====
    if (!inInput && !inMonaco && !mod) {
        if (e.key === 'h' && currentCellId) {
            e.preventDefault();
            toggleCellState(currentCellId, 'skipped');
        }
        if (e.key === 'p' && currentCellId) {
            e.preventDefault();
            toggleCellState(currentCellId, 'pinned');
        }
        if (e.key === 'e' && currentCellId) {
            e.preventDefault();
            toggleCellState(currentCellId, 'is_exported');
        }
    }

    // ===== Add cell shortcuts (not in input) =====
    if (!inInput && !inMonaco) {
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

// Toggle cell state (skipped, pinned, is_exported) via server endpoint
function toggleCellState(cellId, property) {
    const nbPath = window.location.pathname;
    fetch(`${nbPath}/cell/${cellId}/toggle/${property}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'}
    });
}

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

// Also sync when Monaco editor content needs to go to hidden field
function syncAllContent(cellId) {
    syncMonacoToTextarea(cellId);
    syncPromptContent(cellId);
}

// ==================== Markdown Rendering ====================

/**
 * Extract leading HTML blocks from text, properly handling nested tags.
 * Returns [htmlPrefix, remainingText]
 */
function extractLeadingHtmlBlocks(text) {
    if (!text) return ['', text];

    const trimmedText = text.trimStart();
    const blockTags = ['details', 'div', 'table', 'ul', 'ol', 'pre', 'section'];

    // Check if text starts with an HTML block tag
    const openTagMatch = trimmedText.match(/^<(details|div|table|ul|ol|pre|section)(\s[^>]*)?>/i);
    if (!openTagMatch) return ['', text];

    const tagName = openTagMatch[1].toLowerCase();
    let htmlPrefix = '';
    let remaining = trimmedText;

    // Keep extracting complete HTML blocks
    while (remaining) {
        const nextOpenTag = remaining.match(/^<(details|div|table|ul|ol|pre|section)(\s[^>]*)?>/i);
        if (!nextOpenTag) break;

        const currentTagName = nextOpenTag[1].toLowerCase();
        let depth = 0;
        let pos = 0;
        let inString = false;
        let blockEnd = -1;

        // Find the matching closing tag by counting depth
        const openPattern = new RegExp(`<${currentTagName}(\\s[^>]*)?>`, 'gi');
        const closePattern = new RegExp(`</${currentTagName}>`, 'gi');

        // Simple approach: count all opens and closes
        const opens = [...remaining.matchAll(new RegExp(`<${currentTagName}(\\s[^>]*)?>`, 'gi'))];
        const closes = [...remaining.matchAll(new RegExp(`</${currentTagName}>`, 'gi'))];

        if (opens.length === 0 || closes.length === 0) break;

        // Build a list of all tag positions with type
        const tagPositions = [];
        opens.forEach(m => tagPositions.push({ pos: m.index, type: 'open' }));
        closes.forEach(m => tagPositions.push({ pos: m.index, type: 'close', len: m[0].length }));
        tagPositions.sort((a, b) => a.pos - b.pos);

        // Walk through and find where depth returns to 0
        depth = 0;
        for (const tag of tagPositions) {
            if (tag.type === 'open') {
                depth++;
            } else {
                depth--;
                if (depth === 0) {
                    blockEnd = tag.pos + tag.len;
                    break;
                }
            }
        }

        if (blockEnd === -1) break; // Malformed HTML, stop

        htmlPrefix += remaining.slice(0, blockEnd);
        remaining = remaining.slice(blockEnd).trimStart();
    }

    return [htmlPrefix, remaining];
}

function renderMarkdown(text) {
    if (!text) return '<p style="color: var(--text-muted);">Click to edit...</p>';

    // Check if content starts with raw HTML (like tool-steps)
    // These should be rendered as-is, not wrapped in paragraph tags
    let htmlPrefix = '';

    // Extract leading HTML blocks (like <details>) properly handling nesting
    [htmlPrefix, text] = extractLeadingHtmlBlocks(text);

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

    // Return with HTML prefix preserved (not wrapped in p tags)
    if (text.trim()) {
        return htmlPrefix + '<p>' + text + '</p>';
    } else {
        return htmlPrefix;
    }
}

// Server-side markdown rendering via mistlefoot for full-fidelity output
// (subscript, superscript, emoji, highlighting, footnotes, task lists, etc.)
async function renderMarkdownServer(text, targetEl) {
    if (!text || !text.trim()) {
        targetEl.innerHTML = '<p style="color: var(--text-muted);">Click to edit...</p>';
        return;
    }
    try {
        const resp = await fetch('/render-markdown', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'text=' + encodeURIComponent(text)
        });
        const data = await resp.json();
        targetEl.innerHTML = data.html;
        // Add copy buttons and syntax highlighting to code blocks
        targetEl.querySelectorAll('pre').forEach(pre => {
            if (!pre.querySelector('.copy-btn')) {
                const btn = document.createElement('button');
                btn.className = 'copy-btn';
                btn.textContent = 'Copy';
                btn.onclick = function() { copyCode(this); };
                pre.appendChild(btn);
            }
            const code = pre.querySelector('code');
            if (code && typeof hljs !== 'undefined') hljs.highlightElement(code);
        });
    } catch (e) {
        // Fallback to client-side rendering
        targetEl.innerHTML = renderMarkdown(text);
    }
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
        // Show client-side preview immediately, then upgrade with server rendering
        preview.innerHTML = renderMarkdown(textarea.value);
        preview.style.display = 'block';
        textarea.style.display = 'none';
        renderMarkdownServer(textarea.value, preview);
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
    
    // Initialize Monaco editor for code and shell cells
    if (cell.dataset.type === 'code') {
        initMonacoEditor(cellId);
    } else if (cell.dataset.type === 'shell') {
        initMonacoEditor(cellId, 'sh');
    }
    
    // Setup preview for note cells
    const notePreview = document.getElementById(`preview-${cellId}`);
    const noteSource = document.getElementById(`source-${cellId}`);
    if (notePreview && noteSource && cell.dataset.type === 'note') {
        notePreview.innerHTML = renderMarkdown(noteSource.value);
        renderMarkdownServer(noteSource.value, notePreview);
    }

    // Setup AI response preview for prompt cells
    const aiPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    const aiTextarea = document.getElementById(`output-${cellId}`);
    if (aiPreview && aiTextarea) {
        const content = aiTextarea.value;
        if (content && content.trim()) {
            aiPreview.innerHTML = renderMarkdown(content);
            renderMarkdownServer(content, aiPreview);
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
    
    // Abort previous listeners to prevent accumulation on re-init
    if (cell._abortCtrl) cell._abortCtrl.abort();
    cell._abortCtrl = new AbortController();
    const signal = cell._abortCtrl.signal;

    // Track focus - both focusin (for editors/inputs) and click (for cell background)
    cell.addEventListener('focusin', () => setFocusedCell(cellId), { signal });
    cell.addEventListener('click', (e) => {
        // Only set focus if clicking directly on cell or its non-interactive children
        // This allows clicking anywhere on the cell to select it
        setFocusedCell(cellId);
    }, { signal });
}

// ---------------------------------------------------------------------------
// HTMX Swap Lifecycle — Editor Cleanup & Scroll Preservation
//
// FOUST PREVENTION CONTEXT:
// Monaco Editor renders text white first, then asynchronously tokenizes via a
// web worker to apply syntax highlighting. If the editor DOM is destroyed and
// recreated (e.g., by an HTMX swap), users see a flash of white unstyled text
// before highlighting reappears — this is FOUST (Flash of Unstyled Text).
//
// The FOUST prevention strategy has three layers:
//   1. Server sends granular updates (JSON messages, targeted OOB swaps) instead
//      of replacing entire cells/containers (see ws.onmessage handlers below)
//   2. htmx:beforeSwap guard: skip editor disposal when swap style is "none"
//   3. initMonacoEditor skip guard: don't re-create editor if DOM is intact
//
// The HTMX response from the initiating tab (AllCells, CellView, etc.) still
// does a full replacement for that tab's local state. The WS broadcasts use
// granular messages so OTHER tabs (collaborators) don't see FOUST.
//
// Several operations still use HTMX outerHTML as the local response:
//   - Adding a cell (+ Code / + Note / + Prompt buttons, or via dialoghelper)
//   - Deleting a cell
//   - Moving a cell up/down
//   - Cell execution completion (OOB swap from WebSocket)
//
// When #cells is replaced, ALL Monaco editors inside are destroyed and
// recreated. This causes two problems:
//   1. The last editor created may receive focus, scrolling the page to it
//   2. Monaco re-initialization triggers layout, which can also cause scroll
//
// Fix: Save window.scrollY before the swap, restore it after editors init.
// We restore in both htmx:afterSwap (immediate) and htmx:afterSettle
// (after Monaco init) to cover both timing windows.
//
// The hx_swap="outerHTML show:none" on buttons also tells HTMX not to
// scroll any element into view after the swap.
// ---------------------------------------------------------------------------
let _htmxScrollRestore = null;
let _pendingScrollToNewCell = false;

document.addEventListener('htmx:beforeSwap', (e) => {
    const target = e.detail.target;
    if (target) {
        // Save scroll position when swapping the #cells container (multi-cell swap)
        if (target.id === 'cells' || target.querySelectorAll('.cell').length > 1) {
            _htmxScrollRestore = window.scrollY;
        }

        // Skip editor destruction when no actual DOM replacement will happen.
        // hx_swap="none" means the response is discarded — the target DOM stays intact.
        // Without this check, editors get destroyed and recreated for nothing (causing FOUST).
        const swapStyle = e.detail.swapStyle || (e.detail.requestConfig && e.detail.requestConfig.swapStyle);
        if (swapStyle === 'none' || (e.detail.serverResponse === '' && !e.detail.isError)) {
            return;
        }

        // If target itself is a cell with a monaco-container
        if (target.classList && target.classList.contains('cell')) {
            const container = target.querySelector('.monaco-container');
            if (container) {
                const cellId = container.id.replace('monaco-', '');
                destroyMonacoEditor(cellId);
            }
        } else {
            // If target contains monaco-containers
            target.querySelectorAll('.monaco-container').forEach(container => {
                const cellId = container.id.replace('monaco-', '');
                destroyMonacoEditor(cellId);
            });
        }
    }
});

// Restore scroll immediately after DOM swap (before inline scripts run)
document.addEventListener('htmx:afterSwap', (e) => {
    if (_htmxScrollRestore !== null && !_pendingScrollToNewCell) {
        window.scrollTo(0, _htmxScrollRestore);
    }
});

// After HTMX settles (fires after all HTMX processing is complete)
document.addEventListener('htmx:afterSettle', (e) => {
    // Small delay to ensure DOM is fully ready and Monaco can initialize properly
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
                // For outerHTML swaps (e.g., createNewCellAtEnd), the target may be
                // the OLD detached element. Query the live document's #cells instead
                // to ensure newly added cells are also initialized.
                const liveTarget = target.isConnected ? target : (document.getElementById('cells') || target);
                liveTarget.querySelectorAll('.cell').forEach(cell => {
                    const cellId = cell.id.replace('cell-', '');
                    initCell(cellId);
                });
            }
        }
        setupPreviewEditing();
        // Restore scroll position after multi-cell swap (prevents jump-to-bottom)
        _restoreScrollPosition();
    }, 20);
});

function _restoreScrollPosition() {
    if (_pendingScrollToNewCell) {
        // Don't restore old scroll — we're about to scroll to the new cell
        _htmxScrollRestore = null;
        return;
    }
    if (_htmxScrollRestore !== null) {
        const pos = _htmxScrollRestore;
        _htmxScrollRestore = null;
        // Restore immediately and again after Monaco layout settles
        window.scrollTo(0, pos);
        requestAnimationFrame(() => window.scrollTo(0, pos));
    }
}

// Handle HTMX errors - ensure streaming state is reset for both prompt and code cells
function resetCellOnError(e, errorMsg) {
    // Check if this is a cell-related request
    const target = e.detail?.target;
    if (target && target.id && target.id.startsWith('cell-')) {
        const cellId = target.id.replace('cell-', '');
        const cell = document.getElementById(`cell-${cellId}`);

        if (cell && cell.classList.contains('streaming')) {
            // Determine cell type and reset appropriately
            const isCodeCell = cell.querySelector('.monaco-container') !== null;
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

    // Update Monaco editor themes
    if (typeof monaco !== 'undefined') {
        monaco.editor.setTheme(newTheme === 'light' ? 'vs' : 'vs-dark');
    }

    // Update toggle button
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        const use = btn.querySelector('use');
        if (use) use.setAttribute('href', newTheme === 'light' ? '#moon' : '#sun');
    }
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        const use = btn.querySelector('use');
        if (use) use.setAttribute('href', savedTheme === 'light' ? '#moon' : '#sun');
    }
}

// ==================== Model Select Toggle ====================
function toggleModelSelect(mode) {
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        // Show model dropdown only for non-mock modes
        modelSelect.style.display = mode === 'mock' ? 'none' : '';
    }
}

// ==================== Safe Mode Toggle ====================
function toggleSafeMode(nbId) {
    const btn = document.getElementById('safe-mode-toggle');
    if (!btn) return;
    const isActive = btn.classList.contains('active');
    const newState = !isActive;

    // Toggle visual state
    btn.classList.toggle('active');
    const use = btn.querySelector('use');
    if (use) use.setAttribute('href', newState ? '#shield-check' : '#shield-off');

    // Send to server
    fetch(`/notebook/${nbId}/safe_mode`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `safe_mode=${newState}`
    });
}

// ==================== Colab Auth State Listener ====================
function onColabAuthenticated() {
    // Swap button to Disconnect (with after-swap hook for disconnect flow)
    const container = document.getElementById('colab-auth-container');
    if (container) {
        container.innerHTML = '<button class="btn btn-sm" id="colab-disconnect-btn" '
            + 'title="Disconnect Colab account" '
            + 'hx-post="/auth/google/logout" hx-target="#colab-auth-container" '
            + 'hx-on::after-swap="onColabDisconnected()">'
            + 'Disconnect</button>';
        htmx.process(container);
    }
    // Show runtime dropdown now that we're authenticated
    const runtimeSelect = document.getElementById('runtime-select');
    if (runtimeSelect) runtimeSelect.style.display = '';
    // Status dot → green
    const statusDot = document.getElementById('colab-status-dot');
    if (statusDot) statusDot.className = 'colab-status-dot connected';
}

function onColabDisconnected() {
    // Hide runtime dropdown (can't pick runtime without auth)
    const runtimeSelect = document.getElementById('runtime-select');
    if (runtimeSelect) runtimeSelect.style.display = 'none';
    // Status dot → gray
    const statusDot = document.getElementById('colab-status-dot');
    if (statusDot) statusDot.className = 'colab-status-dot disconnected';
}

// Listen for postMessage from popup (works if browser preserves window.opener)
window.addEventListener('message', function(event) {
    if (event.data === 'colab-authenticated') onColabAuthenticated();
});

// Listen for localStorage event from popup (reliable cross-origin fallback)
window.addEventListener('storage', function(event) {
    if (event.key === 'colab-auth-event') onColabAuthenticated();
});

// ==================== Kernel Type Management ====================
function handleKernelTypeChanged(kernelType) {
    // Update kernel selector dropdown
    const kernelSelect = document.getElementById('kernel-select');
    if (kernelSelect) {
        kernelSelect.value = kernelType;
    }

    // Show/hide entire Colab controls group (status dot + runtime + auth button)
    const colabControls = document.getElementById('colab-controls');
    if (colabControls) {
        colabControls.style.display = kernelType === 'colab' ? '' : 'none';
    }

    // Runtime selector: only show if colab AND authenticated
    const runtimeSelect = document.getElementById('runtime-select');
    if (runtimeSelect) {
        if (kernelType === 'colab') {
            fetch('/auth/google/status')
                .then(r => r.json())
                .then(data => {
                    runtimeSelect.style.display = data.authenticated ? '' : 'none';
                    const statusDot = document.getElementById('colab-status-dot');
                    if (statusDot) {
                        statusDot.className = 'colab-status-dot ' + (data.authenticated ? 'connected' : 'disconnected');
                    }
                })
                .catch(() => {
                    runtimeSelect.style.display = 'none';
                });
        } else {
            runtimeSelect.style.display = 'none';
        }
    }

    // Hide shell cell add buttons when using remote kernel
    const shellAddBtns = document.querySelectorAll('.btn-add-shell');
    shellAddBtns.forEach(btn => {
        btn.style.display = kernelType === 'local' ? '' : 'none';
    });

    // Hide shell option in cell type dropdowns
    const cellTypeSelects = document.querySelectorAll('.cell-type-select');
    cellTypeSelects.forEach(select => {
        const shellOption = select.querySelector('option[value="shell"]');
        if (shellOption) {
            shellOption.disabled = kernelType !== 'local';
        }
    });

    // Update safe mode toggle visibility (only relevant for local kernel)
    const safeModeToggle = document.querySelector('.safe-mode-toggle');
    if (safeModeToggle) {
        safeModeToggle.style.display = kernelType === 'local' ? '' : 'none';
    }
}

// ==================== Settings Sidebar Toggle ====================
function toggleSettings() {
    const sidebar = document.getElementById('settings-sidebar');
    const overlay = document.getElementById('settings-overlay');

    if (sidebar) {
        sidebar.classList.toggle('open');
    }
    if (overlay) {
        overlay.classList.toggle('open');
    }
}

// Settings/outline Escape handling is consolidated into the main keydown handler (line ~406)

// ==================== Outline Sidebar Toggle ====================
function toggleOutline() {
    const sidebar = document.getElementById('outline-sidebar');
    if (sidebar) {
        const isOpening = !sidebar.classList.contains('outline-open');
        sidebar.classList.toggle('outline-open');

        // Close file explorer if opening outline
        if (isOpening) {
            const fileExplorer = document.getElementById('file-explorer-sidebar');
            if (fileExplorer && fileExplorer.classList.contains('file-explorer-open')) {
                fileExplorer.classList.remove('file-explorer-open');
            }
            refreshOutline();
        }
    }
}

function refreshOutline() {
    // Trigger HTMX refresh of the outline sidebar
    document.body.dispatchEvent(new CustomEvent('outline-refresh'));
}

function scrollToCell(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) {
        cell.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Add a brief highlight effect
        cell.classList.add('cell-highlight');
        setTimeout(() => cell.classList.remove('cell-highlight'), 1500);
    }
}

// Outline toggle shortcut is consolidated into the main keydown handler (line ~406)

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

let _wsReconnectDelay = 1000;

function connectWebSocket(notebookId) {
    currentNotebookId = notebookId;  // Store globally for other functions to use
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);

    ws.onopen = function() {
        console.log('[WS] Connected to notebook:', notebookId);
        _wsReconnectDelay = 1000;  // Reset backoff on successful connection
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
            // Flash error on kernel dot if cell had an error, then return to connected
            if (data.has_error) {
                updateKernelDot('error');
                setTimeout(() => updateKernelDot('connected'), 3000);
            }
        } else if (data.type === 'code_display_data') {
            // Rich output (image, HTML, plot, etc.)
            console.log('[WS] code_display_data received for cell:', data.cell_id);
            appendDisplayData(data.cell_id, data.html, data.display_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_update_display') {
            // Update existing display data (tqdm progress bars, widgets)
            console.log('[WS] code_update_display received for cell:', data.cell_id, 'display_id:', data.display_id);
            updateDisplayData(data.cell_id, data.html, data.display_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_clear_output') {
            // Clear cell output (used by widgets before updating)
            console.log('[WS] code_clear_output received for cell:', data.cell_id);
            clearCellOutput(data.cell_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'queue_update') {
            // Queue state update from server
            console.log('[WS] queue_update received:', data);
            handleQueueUpdate(data);
        } else if (data.type === 'cell_state_change') {
            // Cell state change (queued, running, idle)
            console.log('[WS] cell_state_change received:', data.cell_id, data.state);
            // State changes are now handled via queue_update for consistency
        } else if (data.type === 'var_substituted') {
            // Variable was substituted in prompt
            console.log('[WS] var_substituted:', data.var_name, '->', data.var_value?.substring?.(0, 50));
            ToolUI.showVarSubstitution(data.cell_id, data.var_name, data.var_value);
        } else if (data.type === 'tool_available') {
            // Tool became available
            console.log('[WS] tool_available:', data.tool_name, 'type:', data.tool_type);
            ToolUI.showToolAvailable(data.cell_id, data.tool_name, data.tool_type);
        } else if (data.type === 'tool_call') {
            // AI is calling a tool
            console.log('[WS] tool_call:', data.tool_name, 'input:', data.tool_input);
            ToolUI.showToolCall(data.cell_id, data.tool_id, data.tool_name, data.tool_input);
        } else if (data.type === 'tool_result') {
            // Tool returned a result
            console.log('[WS] tool_result:', data.tool_name, 'status:', data.status);
            ToolUI.showToolResult(data.cell_id, data.tool_id, data.tool_name, data.result, data.status);
        } else if (data.type === 'tool_confirmation_request') {
            // Server requesting confirmation for file-modifying tool
            console.log('[WS] tool_confirmation_request:', data.tool_name);
            ToolConfirmation.show(data.cell_id, data.tool_name, data.tool_input, data.confirmation_id);
        } else if (data.type === 'kernel_type_changed') {
            // Kernel type changed (local <-> colab)
            console.log('[WS] kernel_type_changed:', data.kernel_type);
            handleKernelTypeChanged(data.kernel_type);
        } else if (data.type === 'kernel_connected') {
            // Kernel connected after first execution — refresh status bar
            document.body.dispatchEvent(new CustomEvent('kernel-connected'));
            updateKernelDot('connected');
        } else if (data.type === 'kernel_error') {
            // Kernel died or disconnected
            updateKernelDot('error');
        } else if (data.type === 'kernel_restarting') {
            // Kernel is restarting — show yellow, will go grey until next execution
            updateKernelDot('busy');
        // ---- FOUST Prevention Handlers ----
        // These handlers receive granular JSON messages from the server and update
        // the DOM surgically, avoiding full cell/container replacement that would
        // destroy Monaco editors and cause FOUST (Flash of Unstyled Text).
        // See docs/how_it_works/17_editor_cell_transitions.md for full architecture.

        } else if (data.type === 'cell_source_update') {
            // In-place source update from dialoghelper — no editor destruction (no FOUST).
            // editor.setValue() preserves the Monaco DOM and re-tokenizes in-place.
            const editor = monacoEditors[data.cell_id];
            if (editor) {
                const currentValue = editor.getValue();
                if (currentValue !== data.source) {
                    const position = editor.getPosition();
                    editor.setValue(data.source);
                    if (position) editor.setPosition(position);
                }
            }
            // Also update hidden textarea
            const textarea = document.getElementById(`source-${data.cell_id}`);
            if (textarea) textarea.value = data.source;
        } else if (data.type === 'cell_class_update') {
            // Update cell wrapper CSS classes (skipped, pinned, etc) without DOM replacement
            const cellEl = document.getElementById(`cell-${data.cell_id}`);
            if (cellEl) cellEl.className = data.cls;
        } else if (data.type === 'cell_collapse_update') {
            // Collapse toggle: update CSS classes in-place via setCollapseLevel().
            // Previously used CellViewOOB (full cell replacement → FOUST).
            if (data.section === 'input' || data.section === 'both')
                setCollapseLevel(data.cell_id, 'input', data.input_collapse);
            if (data.section === 'output' || data.section === 'both')
                setCollapseLevel(data.cell_id, 'output', data.output_collapse);

        } else if (data.type === 'cell_delete') {
            // Granular cell removal: delete one cell + adjacent add-row.
            // Previously used AllCellsOOB (replaced entire #cells → FOUST on ALL cells).
            // DOM structure: .add-row, #cell-A, .add-row, #cell-B, .add-row
            const cellEl = document.getElementById(`cell-${data.cell_id}`);
            if (!cellEl) return; // Already deleted (e.g., HTMX response processed first)

            // Find neighbor to focus BEFORE removing the cell from DOM.
            // Prefer the next cell below; if this was the last cell, use the one above.
            let focusTarget = null;
            if (focusedCellId === data.cell_id) {
                let sibling = cellEl.nextElementSibling;
                while (sibling) {
                    if (sibling.classList.contains('cell')) { focusTarget = sibling.id.replace('cell-', ''); break; }
                    sibling = sibling.nextElementSibling;
                }
                if (!focusTarget) {
                    sibling = cellEl.previousElementSibling;
                    while (sibling) {
                        if (sibling.classList.contains('cell')) { focusTarget = sibling.id.replace('cell-', ''); break; }
                        sibling = sibling.previousElementSibling;
                    }
                }
            }

            const next = cellEl.nextElementSibling;
            const prev = cellEl.previousElementSibling;
            if (next && next.classList.contains('add-row')) next.remove();
            else if (prev && prev.classList.contains('add-row')) prev.remove();
            if (monacoEditors[data.cell_id]) {
                monacoEditors[data.cell_id].dispose();
                delete monacoEditors[data.cell_id];
            }
            cellEl.remove();

            // Focus the neighbor cell in "command mode" (selected but not editing)
            // so the user can keep pressing DD to delete more cells.
            if (focusTarget) {
                setFocusedCell(focusTarget);
                const targetCell = document.getElementById(`cell-${focusTarget}`);
                if (targetCell) {
                    targetCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    targetCell.tabIndex = -1;
                    targetCell.focus();
                }
            }

        } else if (data.type === 'cell_move') {
            // Granular cell reorder: swap two adjacent cells in DOM.
            // Previously used AllCellsOOB (replaced entire #cells → FOUST on ALL cells).
            // Key insight: insertBefore() MOVES DOM nodes (doesn't copy them), so Monaco
            // editors survive the move with their full state — syntax highlighting, cursor
            // position, undo history, etc. are all preserved.
            // DOM structure: .add-row, #cell-A, .add-row, #cell-B, .add-row
            const cellEl = document.getElementById(`cell-${data.cell_id}`);
            if (!cellEl) return;
            const parent = cellEl.parentNode;
            if (data.direction === 'up') {
                const addRow = cellEl.previousElementSibling;
                const prevCell = addRow?.previousElementSibling;
                if (prevCell && prevCell.id?.startsWith('cell-')) {
                    parent.insertBefore(cellEl, prevCell);
                }
            } else {
                const addRow = cellEl.nextElementSibling;
                const nextCell = addRow?.nextElementSibling;
                if (nextCell && nextCell.id?.startsWith('cell-')) {
                    parent.insertBefore(nextCell, cellEl);
                }
            }

        } else if (data.type === 'cell_add') {
            // Granular cell insertion: add one cell + add-row at a position.
            // Previously used AllCellsOOB (replaced entire #cells → FOUST on ALL cells).
            //
            // DUPLICATE GUARD: The initiating tab receives BOTH the HTMX response
            // (which may include OOB content adding this cell) AND this WS message.
            // Without this check, the cell appears twice — the duplicate has no HTMX
            // bindings or Monaco editor, making it unselectable and uneditable.
            if (document.getElementById(`cell-${data.cell_id}`)) return;
            const cells = document.getElementById('cells');
            if (!cells) return;
            const addRows = cells.querySelectorAll('.add-row');
            const target = addRows[data.pos];
            if (target) {
                target.insertAdjacentHTML('afterend', data.html);
                const newCell = document.getElementById(`cell-${data.cell_id}`);
                if (newCell) {
                    htmx.process(newCell); // Enable hx-post/hx-get on new elements
                    initCell(data.cell_id); // Initialize Monaco editor + event listeners
                    const newAddRow = newCell.nextElementSibling;
                    if (newAddRow && newAddRow.classList.contains('add-row')) {
                        htmx.process(newAddRow);
                    }
                }
                renderCellPreviews(data.cell_id); // Render markdown if note cell

                // If Shift+Enter created this cell, scroll to it now that it's
                // in the DOM with its Monaco editor fully initialized.
                if (_pendingScrollToNewCell) {
                    _pendingScrollToNewCell = false;
                    requestAnimationFrame(() => {
                        focusNextCell(data.cell_id);
                        // The new cell is at the bottom of the page — scrollIntoView
                        // with block:'center' can't center it (nothing below to fill
                        // the viewport). Scroll to page bottom to guarantee visibility.
                        window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
                    });
                }
            }
        }
    };

    ws.onclose = function() {
        const delay = _wsReconnectDelay + Math.random() * _wsReconnectDelay * 0.2;
        console.log(`[WS] Disconnected, reconnecting in ${Math.round(delay)}ms...`);
        setTimeout(() => connectWebSocket(notebookId), delay);
        _wsReconnectDelay = Math.min(_wsReconnectDelay * 2, 30000);
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
    // Re-render with server-side mistlefoot for full-fidelity markdown
    const outputPreview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    const outputTextarea = document.getElementById(`output-${cellId}`);
    if (outputPreview && outputTextarea && outputTextarea.value) {
        renderMarkdownServer(outputTextarea.value, outputPreview);
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

// Debounced stream rendering — coalesce multiple chunks per animation frame
const _streamDirty = new Set();
const _streamRafIds = new Map();

function appendCodeOutput(cellId, chunk, streamName) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

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

    // Schedule a single render per animation frame (debounce)
    if (!_streamDirty.has(cellId)) {
        _streamDirty.add(cellId);
        const rafId = requestAnimationFrame(() => _flushStreamRender(cellId));
        _streamRafIds.set(cellId, rafId);
    }
}

function _flushStreamRender(cellId) {
    _streamDirty.delete(cellId);
    _streamRafIds.delete(cellId);

    // Don't render if streaming already finished (prevents overwriting OOB output)
    if (!streamTextContent.has(cellId)) return;

    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    let streamEl = outputEl.querySelector('.stream-output');
    if (!streamEl) {
        streamEl = document.createElement('pre');
        streamEl.className = 'stream-output';
        outputEl.appendChild(streamEl);
    }

    const text = streamTextContent.get(cellId) || '';
    streamEl.innerHTML = ansiToHtml(text);

    // Only auto-scroll if user is near the bottom
    const nearBottom = streamEl.scrollTop + streamEl.clientHeight >= streamEl.scrollHeight - 30;
    if (nearBottom) streamEl.scrollTop = streamEl.scrollHeight;
}

function appendDisplayData(cellId, html, displayId) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Create display data container
    const displayEl = document.createElement('div');
    displayEl.className = 'display-data';
    if (displayId) {
        displayEl.setAttribute('data-display-id', displayId);
    }
    displayEl.innerHTML = html;
    outputEl.appendChild(displayEl);

    // Execute any scripts in the HTML (for interactive widgets)
    displayEl.querySelectorAll('script').forEach(script => {
        const newScript = document.createElement('script');
        newScript.textContent = script.textContent;
        script.parentNode.replaceChild(newScript, script);
    });
}

function updateDisplayData(cellId, html, displayId) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl || !displayId) return;

    // Find existing display element with matching display_id
    const existing = outputEl.querySelector(`[data-display-id="${displayId}"]`);
    if (existing) {
        existing.innerHTML = html;
        // Re-execute scripts
        existing.querySelectorAll('script').forEach(script => {
            const newScript = document.createElement('script');
            newScript.textContent = script.textContent;
            script.parentNode.replaceChild(newScript, script);
        });
    } else {
        // Fallback: append as new display data
        appendDisplayData(cellId, html, displayId);
    }
}

function clearCellOutput(cellId) {
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    outputEl.innerHTML = '';
    outputEl.classList.remove('error');
    streamTextContent.set(cellId, '');
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

    // Cancel any pending debounced stream render to prevent overwriting OOB output
    if (_streamRafIds.has(cellId)) {
        cancelAnimationFrame(_streamRafIds.get(cellId));
        _streamRafIds.delete(cellId);
    }
    _streamDirty.delete(cellId);

    // Clean up text content tracker
    streamTextContent.delete(cellId);

    // Clear the streaming timeout
    clearCodeStreamingTimeout(cellId);

    // Refresh outline sidebar if it's open (to update variables/functions)
    const outlineSidebar = document.getElementById('outline-sidebar');
    if (outlineSidebar && outlineSidebar.classList.contains('outline-open')) {
        refreshOutline();
    }

    console.log('[Code] Finished streaming for cell:', cellId, hasError ? '(with errors)' : '');
}

// ============================================================================
// Queue State Management
// ============================================================================

// Track queue state for cells
const cellQueueState = new Map(); // cellId -> {state: 'queued'|'running'|'idle', position: number}

// Kernel status dot — updates the dot next to notebook title
// States: (no class)=grey/disconnected, connected=green, busy=yellow, error=red
function updateKernelDot(state) {
    const dot = document.getElementById('kernel-dot');
    if (!dot) return;
    dot.classList.remove('connected', 'busy', 'error');
    const titles = {
        connected: 'Kernel: idle',
        busy: 'Kernel: busy',
        error: 'Kernel: error',
        '': 'Kernel: not connected'
    };
    if (state) dot.classList.add(state);
    dot.title = titles[state] || titles[''];
}

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

    // Update kernel dot: busy if running/queued, connected if idle
    updateKernelDot(hasQueuedOrRunning ? 'busy' : 'connected');
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

    // Parse the HTML and find all OOB elements (may be nested in wrapper divs)
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    // Collect OOB elements: check top-level first, then search nested
    const topLevel = Array.from(template.content.children);
    const elements = [];
    for (const el of topLevel) {
        if (el.getAttribute('hx-swap-oob')) {
            elements.push(el);
        } else {
            // Search inside wrapper divs for nested OOB elements
            el.querySelectorAll('[hx-swap-oob]').forEach(nested => elements.push(nested));
        }
    }
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
            const activeEl = document.activeElement;
            const isEditingText = target.contains(activeEl) && activeEl && (
                activeEl.tagName === 'TEXTAREA' || activeEl.tagName === 'INPUT' ||
                activeEl.isContentEditable || activeEl.closest('.monaco-editor')
            );
            const isStreaming = target.classList.contains('streaming');

            // Skip update only if user is editing text in this cell or it's streaming
            if (isEditingText || isStreaming) {
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

                // Reinitialize Monaco editor if it's a code or shell cell
                if (newCell.dataset.type === 'code') {
                    setTimeout(() => initMonacoEditor(cellId), 0);
                } else if (newCell.dataset.type === 'shell') {
                    setTimeout(() => initMonacoEditor(cellId, 'sh'), 0);
                }
            }

            // Re-render previews for this cell
            renderCellPreviews(cellId);
        }
        else if (targetId.startsWith('output-') || targetId.startsWith('header-')) {
            // Targeted OOB swap for output or header sections only
            // This preserves the Monaco editor DOM — no FOUST
            console.log('[OOB] Targeted swap for:', targetId);
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            // Reinitialize HTMX bindings on the new element
            const newEl = document.getElementById(targetId);
            if (newEl) htmx.process(newEl);
        }
        else if (targetId === 'cells') {
            // Full cells container update (e.g., from dialoghelper add_msg)
            console.log('[OOB] Processing cells container update');
            // Save currently focused cell ID before update
            const focusedCell = document.activeElement?.closest('.cell');
            const focusedCellId = focusedCell?.id?.replace('cell-', '');

            // Only skip if user is actively typing AND no cell is currently streaming
            // If ANY cell is streaming (executing), we need to allow updates for add_msg() to work
            // The Monaco editor's hidden textarea keeps focus during execution, but that's not "real typing"
            const isInInput = document.activeElement?.matches('input, textarea, .monaco-editor .inputarea');
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

            // Reinitialize Monaco editors for all code cells
            reinitializeMonacoEditors();

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

function reinitializeMonacoEditors() {
    // Destroy all existing Monaco editors
    for (const cellId of Object.keys(monacoEditors)) {
        destroyMonacoEditor(cellId);
    }

    // Find all code and shell cells and initialize their editors
    document.querySelectorAll('.cell[data-type="code"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => initMonacoEditor(cellId), 0);
    });
    document.querySelectorAll('.cell[data-type="shell"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => initMonacoEditor(cellId, 'sh'), 0);
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
            renderMarkdownServer(textarea.value, notePreview);
        }
    }

    // Handle prompt cells - render both prompt and output previews
    const promptPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="prompt"]`);
    if (promptPreview) {
        const promptTextarea = document.getElementById(`prompt-${cellId}`);
        if (promptTextarea) {
            promptPreview.innerHTML = renderMarkdown(promptTextarea.value);
            renderMarkdownServer(promptTextarea.value, promptPreview);
        }
    }

    const outputPreview = cell.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (outputPreview) {
        const outputTextarea = document.getElementById(`output-${cellId}`);
        if (outputTextarea && outputTextarea.value) {
            outputPreview.innerHTML = renderMarkdown(outputTextarea.value);
            renderMarkdownServer(outputTextarea.value, outputPreview);
        }
    }
}

// ==================== Tool UI Management (IIFE) ====================
// Handles displaying tool calls, results, and variable substitutions in the UI

const ToolUI = (function() {
    // Track tool calls for each cell
    const cellToolCalls = new Map();  // cellId -> array of tool call objects

    // Get or create the tool results area for a cell
    function getToolArea(cellId) {
        const cell = document.getElementById(`cell-${cellId}`);
        if (!cell) return null;

        let toolArea = cell.querySelector('.tool-results-area');
        if (!toolArea) {
            // Create tool results area before the AI response preview
            const outputSection = cell.querySelector('[data-collapse-section="output"]');
            if (outputSection) {
                toolArea = document.createElement('div');
                toolArea.className = 'tool-results-area';
                outputSection.insertBefore(toolArea, outputSection.firstChild);
            }
        }
        return toolArea;
    }

    // Format JSON for display
    function formatJson(obj) {
        try {
            return JSON.stringify(obj, null, 2);
        } catch (e) {
            return String(obj);
        }
    }

    // Truncate long strings for display
    function truncate(str, maxLen = 100) {
        if (!str) return '';
        str = String(str);
        return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
    }

    return {
        // Show variable substitution indicator
        showVarSubstitution: function(cellId, varName, varValue) {
            const toolArea = getToolArea(cellId);
            if (!toolArea) return;

            const varEl = document.createElement('div');
            varEl.className = 'tool-var-substitution';
            varEl.innerHTML = `
                <span class="tool-var-icon">📝</span>
                <span class="tool-var-name">$\`${escapeHtml(varName)}\`</span>
                <span class="tool-var-arrow">→</span>
                <span class="tool-var-value" title="${escapeHtml(varValue)}">${escapeHtml(truncate(varValue, 50))}</span>
            `;
            toolArea.appendChild(varEl);
        },

        // Show tool available indicator
        showToolAvailable: function(cellId, toolName, toolType) {
            const toolArea = getToolArea(cellId);
            if (!toolArea) return;

            const toolEl = document.createElement('div');
            toolEl.className = 'tool-available';
            toolEl.innerHTML = `
                <span class="tool-icon">${toolType === 'builtin' ? '🔧' : '⚡'}</span>
                <span class="tool-name">${escapeHtml(toolName)}</span>
                <span class="tool-badge">${toolType === 'builtin' ? 'built-in' : 'dynamic'}</span>
            `;
            toolArea.appendChild(toolEl);
        },

        // Show tool call in progress
        showToolCall: function(cellId, toolId, toolName, toolInput) {
            const toolArea = getToolArea(cellId);
            if (!toolArea) return;

            // Track this call
            if (!cellToolCalls.has(cellId)) {
                cellToolCalls.set(cellId, []);
            }
            cellToolCalls.get(cellId).push({ id: toolId, name: toolName, input: toolInput });

            const callEl = document.createElement('div');
            callEl.className = 'tool-call';
            callEl.id = `tool-call-${toolId}`;
            callEl.innerHTML = `
                <div class="tool-call-header">
                    <span class="tool-call-icon">🔄</span>
                    <span class="tool-call-name">${escapeHtml(toolName)}</span>
                    <span class="tool-call-status">Running...</span>
                </div>
                <div class="tool-call-input">
                    <button class="tool-toggle-btn" onclick="ToolUI.toggleDetails('${toolId}')">▶ Input</button>
                    <pre class="tool-call-details" id="tool-input-${toolId}" style="display:none;">${escapeHtml(formatJson(toolInput))}</pre>
                </div>
            `;
            toolArea.appendChild(callEl);
        },

        // Show tool result
        showToolResult: function(cellId, toolId, toolName, result, status) {
            const callEl = document.getElementById(`tool-call-${toolId}`);
            if (!callEl) return;

            // Update status
            const statusEl = callEl.querySelector('.tool-call-status');
            if (statusEl) {
                statusEl.textContent = status === 'success' ? 'Done' : 'Error';
                statusEl.className = `tool-call-status ${status === 'success' ? 'success' : 'error'}`;
            }

            // Update icon
            const iconEl = callEl.querySelector('.tool-call-icon');
            if (iconEl) {
                iconEl.textContent = status === 'success' ? '✅' : '❌';
            }

            // Add result section
            const resultContent = result?.content || result?.error || JSON.stringify(result);
            const resultHtml = `
                <div class="tool-call-result ${status === 'success' ? '' : 'error'}">
                    <button class="tool-toggle-btn" onclick="ToolUI.toggleDetails('result-${toolId}')">▶ Result</button>
                    <pre class="tool-call-details" id="tool-result-${toolId}" style="display:none;">${escapeHtml(truncate(resultContent, 500))}</pre>
                </div>
            `;
            callEl.insertAdjacentHTML('beforeend', resultHtml);
        },

        // Toggle visibility of details section
        toggleDetails: function(id) {
            const detailsEl = document.getElementById(`tool-input-${id}`) || document.getElementById(`tool-result-${id}`);
            if (!detailsEl) return;

            const btn = detailsEl.previousElementSibling;
            if (detailsEl.style.display === 'none') {
                detailsEl.style.display = 'block';
                if (btn) btn.textContent = btn.textContent.replace('▶', '▼');
            } else {
                detailsEl.style.display = 'none';
                if (btn) btn.textContent = btn.textContent.replace('▼', '▶');
            }
        },

        // Clear tool area for a cell
        clear: function(cellId) {
            const toolArea = getToolArea(cellId);
            if (toolArea) {
                toolArea.innerHTML = '';
            }
            cellToolCalls.delete(cellId);
        }
    };
})();

// ==================== Tool Confirmation Management (IIFE) ====================
// Handles confirmation dialogs for file-modifying tools

const ToolConfirmation = (function() {
    const pendingConfirmations = new Map();  // confirmationId -> { cellId, toolName, toolInput }

    return {
        // Show confirmation dialog
        show: function(cellId, toolName, toolInput, confirmationId) {
            pendingConfirmations.set(confirmationId, { cellId, toolName, toolInput });

            const toolArea = document.querySelector(`#cell-${cellId} .tool-results-area`);
            if (!toolArea) return;

            const dialogHtml = `
                <div class="tool-confirmation-dialog" id="tool-confirm-${confirmationId}">
                    <div class="tool-confirm-content">
                        <div class="tool-confirm-title">
                            <span class="tool-confirm-icon">⚠️</span>
                            Confirm: ${escapeHtml(toolName)}
                        </div>
                        <pre class="tool-confirm-preview">${escapeHtml(JSON.stringify(toolInput, null, 2))}</pre>
                        <div class="tool-confirm-actions">
                            <button class="btn-secondary" onclick="ToolConfirmation.cancel('${confirmationId}')">Cancel</button>
                            <button class="btn-primary" onclick="ToolConfirmation.confirm('${confirmationId}')">Execute</button>
                        </div>
                    </div>
                </div>
            `;
            toolArea.insertAdjacentHTML('beforeend', dialogHtml);
        },

        // User confirms execution
        confirm: function(confirmationId) {
            const pending = pendingConfirmations.get(confirmationId);
            if (pending && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'tool_confirmation_response',
                    confirmation_id: confirmationId,
                    approved: true
                }));
            }
            this.removeDialog(confirmationId);
            pendingConfirmations.delete(confirmationId);
        },

        // User cancels execution
        cancel: function(confirmationId) {
            const pending = pendingConfirmations.get(confirmationId);
            if (pending && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'tool_confirmation_response',
                    confirmation_id: confirmationId,
                    approved: false
                }));
            }
            this.removeDialog(confirmationId);
            pendingConfirmations.delete(confirmationId);
        },

        // Remove dialog from DOM
        removeDialog: function(confirmationId) {
            const dialog = document.getElementById(`tool-confirm-${confirmationId}`);
            if (dialog) dialog.remove();
        }
    };
})();

// ==================== Kernel Selection Modal ====================
function toggleKernelModal(nbId) {
    const overlay = document.getElementById('kernel-modal-overlay');
    if (!overlay) return;
    overlay.classList.toggle('visible');
}

function selectKernelOption(el, kernelType) {
    // Highlight the selected kernel option (client-side only)
    const body = el.closest('.kernel-modal-body');
    if (!body) return;
    // Remove active from all options
    body.querySelectorAll('.kernel-option').forEach(opt => opt.classList.remove('active'));
    // Hide all runtime pickers
    body.querySelectorAll('.kernel-runtime-options').forEach(rp => rp.style.display = 'none');
    // Activate the clicked option
    el.classList.add('active');
    // Show runtime picker for this option if it has one
    const runtimePicker = el.querySelector('.kernel-runtime-options');
    if (runtimePicker) {
        runtimePicker.style.display = '';
        // Select first runtime if none is active
        if (!runtimePicker.querySelector('.kernel-runtime-btn.active')) {
            const first = runtimePicker.querySelector('.kernel-runtime-btn');
            if (first) {
                first.classList.add('active');
                const rtInput = document.getElementById('kernel-modal-selected-runtime');
                if (rtInput) rtInput.value = first.dataset.runtime;
            }
        }
    }
    // Update hidden input
    const typeInput = document.getElementById('kernel-modal-selected-type');
    if (typeInput) typeInput.value = kernelType;
}

function selectKernelRuntime(btn, runtime) {
    // Highlight the selected runtime button (client-side only)
    const picker = btn.closest('.kernel-runtime-options');
    if (picker) {
        picker.querySelectorAll('.kernel-runtime-btn').forEach(b => b.classList.remove('active'));
    }
    btn.classList.add('active');
    // Update hidden input
    const rtInput = document.getElementById('kernel-modal-selected-runtime');
    if (rtInput) rtInput.value = runtime;
    // Stop click from bubbling to the kernel-option (which would re-trigger selectKernelOption)
    event.stopPropagation();
}

function applyKernelSelection(nbId) {
    const typeInput = document.getElementById('kernel-modal-selected-type');
    const rtInput = document.getElementById('kernel-modal-selected-runtime');
    if (!typeInput) return;
    const kernelType = typeInput.value;
    const runtimeType = rtInput ? rtInput.value : 'cpu';

    // Send kernel type change
    fetch(`/notebook/${nbId}/kernel/type`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `kernel_type=${encodeURIComponent(kernelType)}`
    }).then(response => response.text()).then(html => {
        const statusEl = document.getElementById('status');
        if (statusEl) statusEl.innerHTML = html;

        // If Colab, also set runtime type
        if (runtimeType && kernelType === 'colab') {
            fetch(`/notebook/${nbId}/kernel/runtime`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `runtime_type=${encodeURIComponent(runtimeType)}`
            });
        }

        toggleKernelModal();
        // Refresh kernel toolbar button
        htmx.ajax('GET', `/notebook/${nbId}/kernel/info`, {target: '#kernel-status-bar', swap: 'outerHTML'});
    });
}

// Refresh kernel status bar after first cell execution
document.body.addEventListener('kernel-connected', function(e) {
    const nbId = window.NOTEBOOK_ID;
    if (nbId) {
        htmx.ajax('GET', `/notebook/${nbId}/kernel/info`, {target: '#kernel-status-bar', swap: 'outerHTML'});
    }
});

// Close kernel modal on Escape
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const overlay = document.getElementById('kernel-modal-overlay');
        if (overlay && overlay.classList.contains('visible')) {
            toggleKernelModal();
            return;
        }
    }
});

// ==================== File Explorer ====================
function toggleFileExplorer() {
    const sidebar = document.getElementById('file-explorer-sidebar');
    if (sidebar) {
        const isOpening = !sidebar.classList.contains('file-explorer-open');
        sidebar.classList.toggle('file-explorer-open');

        // Close outline if opening file explorer
        if (isOpening) {
            const outline = document.getElementById('outline-sidebar');
            if (outline && outline.classList.contains('outline-open')) {
                outline.classList.remove('outline-open');
            }
        }
    }
}

function toggleNewItemModal() {
    const modal = document.getElementById('new-item-modal');
    if (modal) {
        modal.classList.toggle('visible');
        // Focus the name input when opening
        if (modal.classList.contains('visible')) {
            const input = document.getElementById('new-item-name');
            if (input) { input.value = ''; input.focus(); }
            // Reset type to dialog
            selectNewItemType('dialog');
        }
    }
}

function selectNewItemType(type) {
    const dialogBtn = document.getElementById('new-item-type-dialog');
    const folderBtn = document.getElementById('new-item-type-folder');
    const typeInput = document.getElementById('new-item-type');
    if (typeInput) typeInput.value = type;
    if (dialogBtn) dialogBtn.classList.toggle('active', type === 'dialog');
    if (folderBtn) folderBtn.classList.toggle('active', type === 'folder');
}

function createNewItem(currentPath) {
    const name = document.getElementById('new-item-name')?.value?.trim();
    const type = document.getElementById('new-item-type')?.value || 'dialog';
    if (!name) return;

    if (type === 'folder') {
        // Create folder via HTMX-style fetch
        fetch('/files/new-folder', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `path=${encodeURIComponent(currentPath)}&name=${encodeURIComponent(name)}`
        }).then(r => r.text()).then(html => {
            const container = document.getElementById('file-list-content');
            if (container) {
                container.outerHTML = html;
                const newContainer = document.getElementById('file-list-content');
                if (newContainer && typeof htmx !== 'undefined') htmx.process(newContainer);
            }
        });
        toggleNewItemModal();
    } else {
        // Navigate to create new notebook with name
        toggleNewItemModal();
        window.location.href = `/notebook/new?dir=${encodeURIComponent(currentPath)}&name=${encodeURIComponent(name)}`;
    }
}

// Delete confirmation modal
function showDeleteConfirm(filePath, displayName) {
    const modal = document.getElementById('delete-confirm-modal');
    const pathInput = document.getElementById('delete-file-path');
    const displaySpan = document.getElementById('delete-file-display');
    if (modal && pathInput && displaySpan) {
        pathInput.value = filePath;
        displaySpan.textContent = displayName;
        modal.classList.add('visible');
    }
}

function hideDeleteConfirm() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('visible');
}

function confirmDeleteFile() {
    const filePath = document.getElementById('delete-file-path')?.value;
    if (!filePath) return;
    fetch('/files/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `path=${encodeURIComponent(filePath)}`
    }).then(r => r.text()).then(html => {
        const container = document.getElementById('file-list-content');
        if (container) {
            container.outerHTML = html;
            const newContainer = document.getElementById('file-list-content');
            if (newContainer && typeof htmx !== 'undefined') htmx.process(newContainer);
        }
    });
    hideDeleteConfirm();
}

// File explorer toggle shortcut is consolidated into the main keydown handler (line ~406)
