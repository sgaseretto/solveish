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

// ==================== Notebook API Path ====================
// Returns the base API path for the current notebook, e.g. "/dialeng/myid"
// Uses window.NOTEBOOK_ID so it works regardless of whether the URL is
// /dialeng/myid or /dialeng/?name=path/to/notebook
function nbApiPath() {
    return `/dialeng/${window.NOTEBOOK_ID}`;
}

function isFilePage() {
    return window.DIALENG_PAGE_KIND === 'file';
}

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
        const clickedHeader = e.target.closest('.cell-header');
        const clickedHeaderControl = e.target.closest(
            '.cell-header button, .cell-header select, .cell-header input, .cell-header a',
        );

        // A blank area of the cell header is the notebook's command-mode
        // affordance. Explicitly release Monaco/textarea focus so subsequent
        // keys are interpreted as cell shortcuts instead of source text.
        if (clickedHeader && !clickedHeaderControl) {
            const activeElement = document.activeElement;
            const activeCell = activeElement?.closest?.('.cell');
            const activeCellId = activeCell?.id?.replace('cell-', '');
            if (!finishTextCellEdit(activeElement)) activeElement?.blur?.();
            if (activeCellId && monacoEditors[activeCellId]) {
                monacoEditors[activeCellId].trigger('keyboard', 'blur', {});
            }
        }
        if (cellId && typeof selectCell === 'function') {
            if (e.shiftKey) {
                selectCell(cellId, {extend: true});
            } else if (e.metaKey || e.ctrlKey) {
                selectCell(cellId, {toggle: true});
            } else {
                selectCell(cellId);
            }
        }
    }
}, true);  // Use capture phase to get the event before it's stopped

// ==================== Monaco Editor Management ====================
const monacoEditors = {};
const monacoModels = {};
const monacoViewStates = new Map();
const standaloneMonacoEditors = {};
const runIntentTracker = new window.DialengRunIntentTracker();
const setCellRunControlState = window.DialengSetCellRunControlState;
let monacoReady = false;
let monacoLoadPromise = null;

function configureMonaco() {
    // Register kernel-backed completion provider for Python once.
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
}

function loadMonaco() {
    if (monacoReady) return Promise.resolve(window.monaco);
    if (monacoLoadPromise) return monacoLoadPromise;
    monacoLoadPromise = new Promise((resolve, reject) => {
        const loadEditor = () => {
            window.require.config({
                paths: {vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs'}
            });
            window.require(['vs/editor/editor.main'], () => {
                monacoReady = true;
                configureMonaco();
                resolve(window.monaco);
            }, reject);
        };
        if (window.require?.config) {
            loadEditor();
            return;
        }
        const loader = document.createElement('script');
        loader.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/loader.min.js';
        loader.async = true;
        loader.onload = loadEditor;
        loader.onerror = () => reject(new Error('Failed to load Monaco editor'));
        document.head.appendChild(loader);
    }).catch(error => {
        monacoLoadPromise = null;
        console.error('[Editor] Monaco failed to load:', error);
        throw error;
    });
    return monacoLoadPromise;
}

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
        loadMonaco().then(() => initMonacoEditor(cellId, mode));
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
        monacoViewStates.set(cellId, monacoEditors[cellId].saveViewState());
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
    const language = langMap[mode] || 'python';
    let model = monacoModels[cellId];
    if (!model || model.isDisposed()) {
        const notebookId = encodeURIComponent(window.NOTEBOOK_ID || 'notebook');
        const modelUri = monaco.Uri.parse(`inmemory://dialeng/${notebookId}/${cellId}.${language === 'shell' ? 'sh' : 'py'}`);
        model = monaco.editor.createModel(initialContent, language, modelUri);
        monacoModels[cellId] = model;
    }

    const editor = monaco.editor.create(container, {
        model,
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

    // Auto-resize to content (replaces Ace minLines/maxLines).
    // Uses requestAnimationFrame to batch height changes and avoid a feedback loop
    // where resize → automaticLayout → word-wrap recalc → content size change → resize.
    const minHeight = 60;  // ~3 lines
    const maxHeight = 600; // ~30 lines
    let _heightRaf = null;
    function updateEditorHeight() {
        if (_heightRaf) return;  // already scheduled
        _heightRaf = requestAnimationFrame(() => {
            _heightRaf = null;
            const contentHeight = Math.max(minHeight, Math.min(maxHeight, editor.getContentHeight()));
            const currentHeight = container.style.height;
            const newHeight = contentHeight + 'px';
            if (currentHeight !== newHeight) {
                container.style.height = newHeight;
                editor.layout();
            }
        });
    }
    editor.onDidContentSizeChange(updateEditorHeight);
    // Initial sizing (synchronous to avoid flash)
    const initHeight = Math.max(minHeight, Math.min(maxHeight, editor.getContentHeight()));
    container.style.height = initHeight + 'px';
    editor.layout();

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
            setNotebookSaveState('modified');
        });
    }

    // Focus tracking
    editor.onDidFocusEditorText(() => setFocusedCell(cellId));
    const savedViewState = monacoViewStates.get(cellId);
    if (savedViewState) editor.restoreViewState(savedViewState);

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
                clickCellRunButton(cell, {advance: true});
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
                clickCellRunButton(cell, {advance: false});
            }
        }
    });

    // Alt/Opt+Enter: run cell and create new cell below
    editor.addAction({
        id: 'dialeng-alt-enter-' + cellId,
        label: 'Run Cell and Add New',
        keybindings: [monaco.KeyMod.Alt | monaco.KeyCode.Enter],
        run: () => {
            syncMonacoToTextarea(cellId);
            const cell = container.closest('.cell');
            if (cell) {
                clickCellRunButton(cell, {advance: false});
                const cells = getOrderedCells();
                const idx = cells.indexOf(cell);
                const pos = idx >= 0 ? idx + 1 : cells.length;
                const nextCellId = cells[idx + 1]?.id?.replace('cell-', '') || null;
                _pendingScrollToNewCell = true;
                requestCellAdd('code', pos, {
                    positionResolver: () => resolveAddPositionBefore(nextCellId),
                });
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
    const model = monacoModels[cellId];
    const textarea = document.getElementById(`source-${cellId}`);
    if (textarea && (editor || model)) {
        textarea.value = (editor || model).getValue();
    }
}

function getCellSourceForRequest(cellId) {
    const editor = monacoEditors[cellId];
    const model = monacoModels[cellId];
    if (editor || model) return (editor || model).getValue();

    const prompt = document.getElementById(`prompt-${cellId}`);
    if (prompt?.tagName === 'TEXTAREA' && prompt.style.display !== 'none') {
        return prompt.value;
    }
    return document.getElementById(`source-${cellId}`)?.value || '';
}

const pendingCellTypeChanges = new Set();

async function requestCellTypeChange(cellId, cellType) {
    const requestKey = `${cellId}:${cellType}`;
    if (pendingCellTypeChanges.has(requestKey)) return;
    const cell = document.getElementById(`cell-${cellId}`);
    const select = cell?.querySelector('select[name="cell_type"]');
    if (!cell || !select || select.value === cellType) return;
    pendingCellTypeChanges.add(requestKey);
    syncAllContent(cellId);
    const body = new URLSearchParams({
        cell_type: cellType,
        source: getCellSourceForRequest(cellId),
        expected_version: document.getElementById(`version-${cellId}`)?.value || '-1',
        expected_cell_type: select.value,
        expected_revision: String(getCurrentNotebookRevision()),
    });
    try {
        const response = await fetch(`${nbApiPath()}/cell/${cellId}/type`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body,
        });
        if (!response.ok) {
            showKernelStatusMessage('The cell changed before its type could be updated. Resynchronizing.', 'warning', 5000);
            await recoverNotebookState(`cell type change failed with HTTP ${response.status}`);
        }
    } finally {
        setTimeout(() => pendingCellTypeChanges.delete(requestKey), 500);
    }
}

function getMonacoContent(cellId) {
    const editor = monacoEditors[cellId];
    const model = monacoModels[cellId];
    return editor ? editor.getValue() : model ? model.getValue() : '';
}

function destroyMonacoEditor(cellId) {
    if (monacoEditors[cellId]) {
        monacoEditors[cellId].dispose();
        delete monacoEditors[cellId];
    }
    if (monacoModels[cellId]) {
        monacoModels[cellId].dispose();
        delete monacoModels[cellId];
    }
    monacoViewStates.delete(cellId);
}

function initStandaloneMonacoEditor(containerId, textareaId, mode = 'plaintext') {
    if (!monacoReady) {
        loadMonaco().then(() => initStandaloneMonacoEditor(containerId, textareaId, mode));
        return null;
    }

    const container = document.getElementById(containerId);
    const textarea = document.getElementById(textareaId);
    if (!container || !textarea) return null;

    if (standaloneMonacoEditors[containerId]) {
        if (container.querySelector('.monaco-editor')) {
            return standaloneMonacoEditors[containerId];
        }
        standaloneMonacoEditors[containerId].dispose();
        delete standaloneMonacoEditors[containerId];
    }

    container.innerHTML = '';
    container.className = 'monaco-container file-monaco-editor';
    container.style.opacity = '0';

    const isDark = (document.documentElement.getAttribute('data-theme') || 'dark') !== 'light';
    const editor = monaco.editor.create(container, {
        value: textarea.value || '',
        language: mode || 'plaintext',
        theme: isDark ? 'vs-dark' : 'vs',
        fontSize: 14,
        fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'off',
        minimap: { enabled: false },
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        overviewRulerLanes: 0,
        hideCursorInOverviewRuler: true,
        renderLineHighlight: 'line',
        glyphMargin: false,
        folding: true,
        scrollbar: { vertical: 'auto', horizontal: 'auto', alwaysConsumeMouseWheel: false },
    });

    textarea.value = editor.getValue();
    editor.onDidChangeModelContent(() => {
        textarea.value = editor.getValue();
    });

    const minHeight = 320;
    const maxHeight = Math.max(window.innerHeight - 220, minHeight);
    let heightRaf = null;
    function updateEditorHeight() {
        if (heightRaf) return;
        heightRaf = requestAnimationFrame(() => {
            heightRaf = null;
            const nextHeight = Math.max(minHeight, Math.min(maxHeight, editor.getContentHeight()));
            const newHeight = `${nextHeight}px`;
            if (container.style.height !== newHeight) {
                container.style.height = newHeight;
                editor.layout();
            }
        });
    }
    editor.onDidContentSizeChange(updateEditorHeight);
    container.style.height = `${Math.max(minHeight, Math.min(maxHeight, editor.getContentHeight()))}px`;
    editor.layout();

    if (textarea.value.trim()) {
        let attempts = 0;
        const pollTokens = () => {
            attempts++;
            const tokenSpans = container.querySelectorAll('.view-lines [class*="mtk"]');
            let hasColoredTokens = false;
            for (const span of tokenSpans) {
                if (span.className !== 'mtk1') {
                    hasColoredTokens = true;
                    break;
                }
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

    editor.addAction({
        id: `dialeng-file-save-${containerId}`,
        label: 'Save File',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
        run: () => saveOpenFile(),
    });

    standaloneMonacoEditors[containerId] = editor;
    return editor;
}

function syncStandaloneMonacoToTextarea(containerId, textareaId) {
    const editor = standaloneMonacoEditors[containerId];
    const textarea = document.getElementById(textareaId);
    if (editor && textarea) {
        textarea.value = editor.getValue();
    }
}

function destroyStandaloneMonacoEditor(containerId) {
    if (standaloneMonacoEditors[containerId]) {
        standaloneMonacoEditors[containerId].dispose();
        delete standaloneMonacoEditors[containerId];
    }
}

// ==================== Cell Selection & Focus Tracking ====================
let focusedCellId = null;
let lastKeyTime = 0;
let lastKey = '';

// Multi-selection state
let selectedCellIds = new Set();
let anchorCellId = null;

// Cell clipboard
let cellClipboard = [];
let clipboardMode = null; // 'cut' or 'copy'

/** Get all .cell elements in DOM order */
function getOrderedCells() {
    return Array.from(document.querySelectorAll('#cells .cell'));
}

/** Get selected cell IDs in DOM order */
function getSelectedCellIds() {
    const ordered = getOrderedCells();
    return ordered
        .filter(c => selectedCellIds.has(c.id.replace('cell-', '')))
        .map(c => c.id.replace('cell-', ''));
}

/** Update DOM classes to reflect selection state */
function _updateSelectionClasses() {
    document.querySelectorAll('.cell.focused').forEach(c => c.classList.remove('focused'));
    document.querySelectorAll('.cell.selected').forEach(c => c.classList.remove('selected'));
    if (anchorCellId) {
        const anchor = document.getElementById(`cell-${anchorCellId}`);
        if (anchor) anchor.classList.add('focused');
    }
    for (const id of selectedCellIds) {
        const el = document.getElementById(`cell-${id}`);
        if (el) el.classList.add('selected');
    }
}

/**
 * Select a cell. If extend=true, select range from anchor to cellId.
 * If toggle=true, toggle cellId in/out of selection without changing anchor.
 */
function selectCell(cellId, {extend = false, toggle = false} = {}) {
    if (!cellId) return;
    if (toggle) {
        if (selectedCellIds.has(cellId)) {
            selectedCellIds.delete(cellId);
            if (anchorCellId === cellId) {
                // Pick another selected cell as anchor, or clear
                anchorCellId = selectedCellIds.size > 0 ? [...selectedCellIds][0] : null;
            }
        } else {
            selectedCellIds.add(cellId);
            anchorCellId = cellId;
        }
    } else if (extend && anchorCellId) {
        // Range select from anchor to cellId
        const cells = getOrderedCells();
        const anchorIdx = cells.findIndex(c => c.id === `cell-${anchorCellId}`);
        const targetIdx = cells.findIndex(c => c.id === `cell-${cellId}`);
        if (anchorIdx >= 0 && targetIdx >= 0) {
            const start = Math.min(anchorIdx, targetIdx);
            const end = Math.max(anchorIdx, targetIdx);
            selectedCellIds.clear();
            for (let i = start; i <= end; i++) {
                selectedCellIds.add(cells[i].id.replace('cell-', ''));
            }
        }
        // Keep anchorCellId unchanged for range selection
    } else {
        // Single select
        selectedCellIds.clear();
        selectedCellIds.add(cellId);
        anchorCellId = cellId;
    }
    focusedCellId = anchorCellId;
    _updateSelectionClasses();
}

/** Select all cells */
function selectAllCells() {
    const cells = getOrderedCells();
    selectedCellIds.clear();
    cells.forEach(c => selectedCellIds.add(c.id.replace('cell-', '')));
    // Keep current anchor, or set to first cell
    if (!anchorCellId && cells.length > 0) {
        anchorCellId = cells[0].id.replace('cell-', '');
    }
    focusedCellId = anchorCellId;
    _updateSelectionClasses();
}

function setFocusedCell(cellId) {
    // Backward-compatible alias: single-select the cell
    selectCell(cellId);
}

async function focusCellWithoutScroll(cellId, preserveY = window.scrollY) {
    setFocusedCell(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;

    // Suppress any pending HTMX scroll restore so it doesn't fight us
    _htmxScrollRestore = null;

    // If it's a code or shell cell with Monaco editor, focus the editor
    if (cell.dataset.type === 'code' || cell.dataset.type === 'shell') {
        const editor = monacoEditors[cellId] || await ensureMonacoEditor(
            cellId,
            cell.dataset.type === 'shell' ? 'sh' : 'python',
        );
        editor?.focus();
    } else if (cell.dataset.type === 'prompt') {
        // For prompt cells, focus the prompt textarea if visible
        const promptTextarea = cell.querySelector('.prompt-content[name="prompt_source"]');
        if (promptTextarea && promptTextarea.style.display !== 'none') {
            promptTextarea.focus({preventScroll: true});
        } else {
            // If prompt has been run (has preview), just select the cell
            // and move DOM focus out of any Monaco editor
            cell.tabIndex = -1;
            cell.focus({preventScroll: true});
        }
    } else {
        // Note cells and other types: move DOM focus to the cell element
        // so that the previous Monaco editor loses keyboard focus.
        // Without this, Shift+Enter would re-run the previous code cell.
        cell.tabIndex = -1;
        cell.focus({ preventScroll: true });
    }

    window.scrollTo(0, preserveY);
    await new Promise(resolve => requestAnimationFrame(resolve));
    window.scrollTo(0, preserveY);
}

async function focusNextCell(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    cell.scrollIntoView({behavior: 'auto', block: 'nearest'});
    const revealedY = window.scrollY;
    await focusCellWithoutScroll(cellId, revealedY);
    document.getElementById(`cell-${cellId}`)?.scrollIntoView({
        behavior: 'auto',
        block: 'nearest',
    });
}

function getFocusedCellId() {
    const active = document.activeElement;
    if (active) {
        const cell = active.closest('.cell');
        if (cell) return cell.id.replace('cell-', '');
    }
    // In command mode the explicit selection is authoritative. Monaco's
    // hasTextFocus() can remain true briefly after a header click and used to
    // route destructive shortcuts to the previously edited cell.
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

function clickCellRunButton(cell, {advance = true} = {}) {
    const button = cell?.querySelector('.btn-run');
    if (!button) return false;
    runIntentTracker.mark(button, {advance});
    button.click();
    return true;
}

// Prompt execution is an HTTP request whose response arrives only after the
// stream has been persisted. Defer navigation until the matching terminal
// WebSocket event so a trailing-cell add cannot race the active prompt.
const pendingPromptAdvances = new Set();

function completePromptAdvance(cellId, status) {
    if (!pendingPromptAdvances.delete(cellId) || status === 'cancelled') return;
    requestAnimationFrame(() => {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell) moveToNextCell(cell);
    });
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
    requestCellAdd('code', position, {
        positionResolver: () => getOrderedCells().length,
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

    const cells = getOrderedCells();
    const anchorCell = cells[Math.min(Math.max(pos - 1, 0), Math.max(cells.length - 1, 0))];
    const anchorCellId = anchorCell ? anchorCell.id.replace('cell-', '') : null;
    const nextCell = addRow.nextElementSibling?.classList?.contains('cell')
        ? addRow.nextElementSibling
        : null;
    const nextCellId = nextCell?.id?.replace('cell-', '') || null;
    requestCellAdd(cellType, pos, {
        anchorCellId,
        positionResolver: () => resolveAddPositionBefore(nextCellId),
    });
    // The WS cell_add handler will insert the cell into the DOM.
}

function _firstVisibleCellId() {
    const cells = getOrderedCells();
    const visible = cells.find(cell => {
        const rect = cell.getBoundingClientRect();
        return rect.bottom > 0 && rect.top < window.innerHeight;
    });
    return visible ? visible.id.replace('cell-', '') : null;
}

let _pendingStructureViewport = null;

function rememberStructureViewport(anchorCellId = null) {
    _pendingStructureViewport = {
        cellId: anchorCellId || getFocusedCellId() || _firstVisibleCellId(),
        scrollY: window.scrollY,
    };
}

function resolveAddPositionBefore(nextCellId = null) {
    const cells = getOrderedCells();
    if (!nextCellId) return cells.length;
    const index = cells.findIndex(cell => cell.id === `cell-${nextCellId}`);
    return index >= 0 ? index : cells.length;
}

function waitForStructureRevision(minimumRevision, timeoutMs = 5000) {
    if (!Number.isFinite(minimumRevision)) return Promise.resolve();
    const deadline = Date.now() + timeoutMs;
    return new Promise(resolve => {
        const check = () => {
            if (getCurrentStructureRevision() >= minimumRevision || Date.now() >= deadline) {
                resolve();
                return;
            }
            setTimeout(check, 20);
        };
        check();
    });
}

let _cellAddRequestChain = Promise.resolve();

function consumeStructureViewport() {
    const viewport = _pendingStructureViewport;
    _pendingStructureViewport = null;
    return viewport;
}

function requestCellAdd(cellType, pos = null, options = {}) {
    const { anchorCellId = null, positionResolver = null } = options;
    const execute = async () => {
        rememberStructureViewport(anchorCellId);
        const resolvedPos = typeof positionResolver === 'function'
            ? positionResolver()
            : pos;
        const params = new URLSearchParams({ type: cellType });
        if (Number.isInteger(resolvedPos)) {
            params.set('pos', String(resolvedPos));
        }
        const mutation = new URLSearchParams({
            operation_id: crypto.randomUUID
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        });
        const structureRevision = getCurrentStructureRevision();
        if (Number.isFinite(structureRevision)) {
            mutation.set('expected_structure_revision', String(structureRevision));
        }
        const response = await fetch(`${nbApiPath()}/cell/add?${params.toString()}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: mutation.toString(),
        });
        if (response.status === 409) {
            recoverNotebookState('cell add revision conflict');
            showKernelStatusMessage(
                'The notebook changed before the cell was added. Resynchronizing…',
                'warning',
                5000,
            );
        } else if (response.ok && Number.isFinite(structureRevision)) {
            await waitForStructureRevision(structureRevision + 1);
        }
        return response;
    };
    const pending = _cellAddRequestChain.then(execute, execute);
    _cellAddRequestChain = pending.catch(() => {});
    return pending;
}

function snapshotCellStructure(cellsRoot) {
    if (!cellsRoot) return null;
    const children = Array.from(cellsRoot.children);
    const leadingAddRow = children[0];
    if (!leadingAddRow || !leadingAddRow.classList.contains('add-row')) {
        return null;
    }

    const cellUnits = new Map();
    for (let i = 1; i < children.length; i += 1) {
        const child = children[i];
        if (!child.classList.contains('cell')) continue;
        const addRowAfter = children[i + 1];
        cellUnits.set(child.id.replace('cell-', ''), {
            cellEl: child,
            addRowEl: addRowAfter && addRowAfter.classList.contains('add-row') ? addRowAfter : null,
        });
    }

    return { leadingAddRow, cellUnits };
}

function buildCellUnitFromHtml(cellId, html) {
    if (!html) return null;
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    const children = Array.from(template.content.children);
    const cellEl = template.content.querySelector(`#cell-${cellId}`) || children.find(node => node.classList?.contains('cell'));
    const addRowEl = children.find(node => node.classList?.contains('add-row')) || template.content.querySelector('.add-row');
    if (!cellEl || !addRowEl) {
        return null;
    }
    return { cellEl, addRowEl };
}

function reconcileCellStructure({
    orderedIds,
    insertedCells = [],
    preserveViewportCellId = null,
    preserveScrollY = null,
} = {}) {
    if (!Array.isArray(orderedIds)) {
        return false;
    }

    const cellsRoot = document.getElementById('cells');
    const structure = snapshotCellStructure(cellsRoot);
    if (!cellsRoot || !structure) {
        return false;
    }

    const viewportCell = preserveViewportCellId ? document.getElementById(`cell-${preserveViewportCellId}`) : null;
    const viewportTop = viewportCell ? viewportCell.getBoundingClientRect().top : null;
    const { leadingAddRow, cellUnits } = structure;

    // Structural reconciles are WebSocket-authored, not HTMX swaps. Clear any
    // stale HTMX scroll restore state so htmx.process(newCell/newAddRow) cannot
    // fight the explicit viewport preservation below.
    _htmxScrollRestore = null;
    _suppressHtmxScrollRestoreUntil = Date.now() + 250;

    insertedCells.forEach(({ cellId, html }) => {
        if (!cellUnits.has(cellId)) {
            const unit = buildCellUnitFromHtml(cellId, html);
            if (unit) cellUnits.set(cellId, unit);
        }
    });

    const plan = window.dialengStructure.plan(orderedIds, cellId => cellUnits.get(cellId));
    if (!plan.valid) {
        const availableIds = Array.from(cellUnits.keys());
        const missingIds = orderedIds.filter(cellId => !cellUnits.has(cellId));
        console.warn(
            `[cells] Unable to reconcile notebook structure; refreshing notebook view ` +
            `(reason=${plan.reason}, missing=${missingIds.join(',') || 'none'}, ` +
            `ordered=${orderedIds.join(',')}, available=${availableIds.join(',')})`,
            {
            reason: plan.reason,
            orderedIds,
            availableIds,
            missingIds,
            },
        );
        window.location.reload();
        return false;
    }

    const fragment = document.createDocumentFragment();
    fragment.appendChild(leadingAddRow);
    plan.units.forEach(unit => {
        fragment.appendChild(unit.cellEl);
        fragment.appendChild(unit.addRowEl);
    });
    cellsRoot.replaceChildren(fragment);

    insertedCells.forEach(({ cellId }) => {
        const newCell = document.getElementById(`cell-${cellId}`);
        if (!newCell) return;
        htmx.process(newCell);
        initCell(cellId);
        const newAddRow = newCell.nextElementSibling;
        if (newAddRow && newAddRow.classList.contains('add-row')) {
            htmx.process(newAddRow);
        }
        renderCellPreviews(cellId);
    });

    const restoreViewportAnchor = () => {
        const liveCell = document.getElementById(`cell-${preserveViewportCellId}`);
        if (!liveCell) return;
        window.scrollBy(0, liveCell.getBoundingClientRect().top - viewportTop);
    };

    if (preserveViewportCellId && viewportTop !== null) {
        restoreViewportAnchor();
        requestAnimationFrame(restoreViewportAnchor);
    } else if (preserveScrollY !== null) {
        window.scrollTo(0, preserveScrollY);
        requestAnimationFrame(() => window.scrollTo(0, preserveScrollY));
    }

    return true;
}

function applyCellTypeChange(data) {
    const cellId = data?.cell_id;
    const target = cellId ? document.getElementById(`cell-${cellId}`) : null;
    if (!target || !data.html) {
        recoverNotebookState('missing cell during type change');
        return;
    }

    const template = document.createElement('template');
    template.innerHTML = data.html.trim();
    const replacement = template.content.querySelector(`#cell-${cellId}`);
    if (!replacement) {
        recoverNotebookState('invalid cell type replacement');
        return;
    }

    const viewportTop = target.getBoundingClientRect().top;
    const wasFocused = focusedCellId === cellId;
    destroyMonacoEditor(cellId);
    target.replaceWith(replacement);
    htmx.process(replacement);
    initCell(cellId);
    renderCellPreviews(cellId);
    updateCellVersion(cellId, data.version);

    if (replacement.dataset.type === 'code') {
        setTimeout(() => observeMonacoCell(cellId), 0);
    } else if (replacement.dataset.type === 'shell') {
        setTimeout(() => observeMonacoCell(cellId, 'sh'), 0);
    }
    if (wasFocused) setFocusedCell(cellId);

    // Replacing a tall editor with a compact note/prompt (or vice versa)
    // should not make the user's current cell jump away from the viewport.
    const restoreViewport = () => {
        const liveCell = document.getElementById(`cell-${cellId}`);
        if (liveCell) window.scrollBy(0, liveCell.getBoundingClientRect().top - viewportTop);
    };
    restoreViewport();
    requestAnimationFrame(restoreViewport);
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

    // ===== Escape to close sidebars/modals =====
    if (e.key === 'Escape') {
        const diagnosticsOverlay = document.getElementById('kernel-diagnostics-overlay');
        const contextOverlay = document.getElementById('context-inspector-overlay');
        if (diagnosticsOverlay?.classList.contains('visible')) {
            diagnosticsOverlay.classList.remove('visible');
            return;
        }
        if (contextOverlay?.classList.contains('visible')) {
            contextOverlay.classList.remove('visible');
            return;
        }
        const shortcutsModal = document.getElementById('shortcuts-modal-overlay');
        if (shortcutsModal && shortcutsModal.classList.contains('visible')) {
            shortcutsModal.classList.remove('visible');
            return;
        }
        const sidebar = document.getElementById('settings-sidebar');
        if (sidebar && sidebar.classList.contains('open')) {
            closeSettings();
            return;
        }
        const outlineSidebar = document.getElementById('outline-sidebar');
        if (outlineSidebar && outlineSidebar.classList.contains('outline-open')) {
            toggleOutline();
            return;
        }
    }

    // ===== ? - Toggle keyboard shortcuts modal =====
    if (e.key === '?' && !inInput && !inMonaco) {
        e.preventDefault();
        toggleKeyboardShortcuts();
        return;
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

    if (isFilePage()) {
        if (mod && e.key === 's') {
            e.preventDefault();
            saveOpenFile();
            return;
        }
        if (e.key === 'Escape' && document.activeElement) {
            document.activeElement.blur();
        }
        return;
    }

    // ===== Arrow/j/k Navigation (nav mode only) =====
    if (!inInput && !inMonaco && !mod) {
        const navKeys = {'ArrowUp': -1, 'ArrowDown': 1, 'k': -1, 'j': 1};
        if (navKeys[e.key] !== undefined && !e.altKey) {
            e.preventDefault();
            const cells = getOrderedCells();
            if (cells.length === 0) return;
            const currentIdx = anchorCellId
                ? cells.findIndex(c => c.id === `cell-${anchorCellId}`)
                : -1;
            const dir = navKeys[e.key];
            let nextIdx = currentIdx + dir;
            if (nextIdx < 0) nextIdx = 0;
            if (nextIdx >= cells.length) nextIdx = cells.length - 1;
            const nextId = cells[nextIdx].id.replace('cell-', '');
            if (e.shiftKey) {
                selectCell(nextId, {extend: true});
            } else {
                selectCell(nextId);
            }
            cells[nextIdx].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            return;
        }
    }

    // ===== Cmd/Ctrl+A - Select all cells =====
    if (mod && e.key === 'a' && !inInput && !inMonaco) {
        e.preventDefault();
        selectAllCells();
        return;
    }

    // ===== Enter - Enter edit mode =====
    if (e.key === 'Enter' && !inInput && !inMonaco && !mod && !e.shiftKey && !e.altKey) {
        if (currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (!cell) return;
            const type = cell.dataset.type;
            if (type === 'code' || type === 'shell') {
                const editor = monacoEditors[currentCellId];
                if (editor) editor.focus();
                else ensureMonacoEditor(currentCellId, type === 'shell' ? 'sh' : 'python').then(
                    nextEditor => nextEditor?.focus()
                );
            } else if (type === 'prompt') {
                const textarea = cell.querySelector('.prompt-content[name="prompt_source"]');
                if (textarea) {
                    if (textarea.style.display === 'none') {
                        // Switch from preview to edit
                        switchToEdit(currentCellId, 'prompt');
                    }
                    textarea.focus();
                }
            } else if (type === 'note') {
                // Switch from preview to edit mode
                switchToEdit(currentCellId, 'source');
                const textarea = cell.querySelector('textarea[name="source"]');
                if (textarea) textarea.focus();
            }
            return;
        }
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
                if (!clickCellRunButton(cell, {advance: true})) {
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
                if (!clickCellRunButton(cell, {advance: false})) {
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
        if (!finishTextCellEdit(document.activeElement)) document.activeElement?.blur?.();
        Object.values(monacoEditors).forEach(ed => ed.trigger('keyboard', 'blur', {}));
        lastKey = 'Escape';
        lastKeyTime = now;
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
                fetch(`${nbApiPath()}/cell/${currentCellId}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=output&level=${level}`
                });
            } else if (e.altKey) {
                e.preventDefault();
                // Alt+number: set both to same level
                setCollapseLevel(currentCellId, 'input', level);
                setCollapseLevel(currentCellId, 'output', level);
                fetch(`${nbApiPath()}/cell/${currentCellId}/collapse-section`, {
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
    
    // ===== h/p/e - Toggle cell state shortcuts (multi-select aware) =====
    if (!inInput && !inMonaco && !mod) {
        if (e.key === 'h' && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => toggleCellState(id, 'skipped'));
        }
        if (e.key === 'p' && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => toggleCellState(id, 'pinned'));
        }
        if (e.key === 'e' && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => toggleCellState(id, 'is_exported'));
        }
    }

    // ===== Add cell shortcuts (not in input) =====
    if (!inInput && !inMonaco) {
        if (mod && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            requestCellAdd('code');
        }
        if (mod && e.shiftKey && e.key === 'N') {
            e.preventDefault();
            requestCellAdd('note');
        }
        if (mod && e.shiftKey && e.key === 'P') {
            e.preventDefault();
            requestCellAdd('prompt');
        }

        const typeShortcuts = {J: 'code', K: 'note', L: 'prompt'};
        const shortcutType = typeShortcuts[e.key.toUpperCase()];
        if (mod && e.shiftKey && shortcutType && currentCellId) {
            e.preventDefault();
            syncAllContent(currentCellId);
            const select = document.querySelector(
                `#cell-${CSS.escape(currentCellId)} select[name="cell_type"]`,
            );
            if (select && select.value !== shortcutType) {
                requestCellTypeChange(currentCellId, shortcutType);
            }
        }
    }

    // ===== Cell clipboard shortcuts (nav mode) =====
    if (!inInput && !inMonaco && !mod) {
        // x - Cut selected cells
        if (e.key === 'x' && currentCellId) {
            e.preventDefault();
            _clipboardSerialize('cut');
        }
        // c - Copy selected cells (also to system clipboard as markdown)
        if (e.key === 'c' && !e.shiftKey && currentCellId) {
            e.preventDefault();
            _clipboardSerialize('copy');
        }
        // v - Paste after anchor cell
        if (e.key === 'v' && currentCellId) {
            e.preventDefault();
            _clipboardPaste();
        }
        // , - Copy cell input to system clipboard
        if (e.key === ',' && currentCellId) {
            e.preventDefault();
            _copyCellContent(currentCellId, 'input');
        }
        // . - Copy cell output to system clipboard
        if (e.key === '.' && currentCellId) {
            e.preventDefault();
            _copyCellContent(currentCellId, 'output');
        }
        // q - Duplicate selected cells
        if (e.key === 'q' && currentCellId) {
            e.preventDefault();
            rememberStructureViewport(currentCellId);
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => {
                fetch(`${nbApiPath()}/cell/${id}/duplicate`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
                });
            });
        }
        // Backspace - Clear outputs from selected cells
        if (e.key === 'Backspace' && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => {
                fetch(`${nbApiPath()}/cell/${id}/clear-output`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
                });
            });
        }
        // m - Copy code blocks from AI response to clipboard
        if (e.key === 'm' && currentCellId) {
            e.preventDefault();
            _copyCodeBlocks(currentCellId);
        }
        // n - Edit AI response (output) of prompt cell
        if (e.key === 'n' && currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell && cell.dataset.type === 'prompt') {
                switchToEdit(currentCellId, 'output');
                const textarea = cell.querySelector('.prompt-content[name="output"]');
                if (textarea) textarea.focus();
            }
        }

        // ===== Phase 3: Cell operations =====
        // a - Add cell above current
        if (e.key === 'a' && !e.shiftKey && currentCellId) {
            e.preventDefault();
            const cells = getOrderedCells();
            const idx = cells.findIndex(c => c.id === `cell-${currentCellId}`);
            const pos = idx >= 0 ? idx : 0;
            requestCellAdd('code', pos);
        }
        // b - Add cell below current
        if (e.key === 'b' && !e.shiftKey && currentCellId) {
            e.preventDefault();
            const cells = getOrderedCells();
            const idx = cells.findIndex(c => c.id === `cell-${currentCellId}`);
            const pos = idx >= 0 ? idx + 1 : cells.length;
            requestCellAdd('code', pos);
        }
        // w - Extract fenced code blocks from prompt/note output
        if (e.key === 'w' && currentCellId) {
            e.preventDefault();
            rememberStructureViewport(currentCellId);
            fetch(`${nbApiPath()}/cell/${currentCellId}/extract-code-blocks`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'}
            });
        }

        // ===== Phase 4: Execution shortcuts =====
        // r - Re-run all code cells
        if (e.key === 'r' && !e.shiftKey) {
            e.preventDefault();
            getOrderedCells().forEach(c => {
                if (c.dataset.type === 'code') {
                    clickCellRunButton(c, {advance: false});
                }
            });
        }

        // ===== Phase 5: Display shortcuts =====
        // i - Toggle input collapse (multi-select aware)
        if (e.key === 'i' && !e.shiftKey && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => {
                cycleCollapseLevel(id, 'input');
                fetch(`${nbApiPath()}/cell/${id}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=input&level=${_getNextCollapseLevel(id, 'input')}`
                });
            });
        }
        // o - Toggle output collapse (multi-select aware)
        if (e.key === 'o' && !e.shiftKey && currentCellId) {
            e.preventDefault();
            const ids = getSelectedCellIds();
            (ids.length > 0 ? ids : [currentCellId]).forEach(id => {
                cycleCollapseLevel(id, 'output');
                fetch(`${nbApiPath()}/cell/${id}/collapse-section`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: `section=output&level=${_getNextCollapseLevel(id, 'output')}`
                });
            });
        }
        // s - Save (S S double press)
        if (e.key === 's' || e.key === 'S') {
            const now = Date.now();
            if (lastKey === 's' && now - lastKeyTime < 500) {
                e.preventDefault();
                const saveBtn = document.getElementById('save-btn');
                if (saveBtn) saveBtn.click();
                lastKey = '';
                return;
            }
            lastKey = 's';
            lastKeyTime = now;
        }
        // ArrowLeft - Collapse full cell
        if (e.key === 'ArrowLeft' && currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell && !cell.classList.contains('collapsed')) {
                toggleCollapse(currentCellId);
            }
        }
        // ArrowRight - Expand full cell
        if (e.key === 'ArrowRight' && currentCellId) {
            e.preventDefault();
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell && cell.classList.contains('collapsed')) {
                toggleCollapse(currentCellId);
            }
        }
    }

    // ===== Phase 3: Cell type switching (works with mod, any mode) =====
    if (mod && e.shiftKey && !inInput && !inMonaco && currentCellId) {
        const typeMap = {'J': 'code', 'K': 'note', 'L': 'prompt', ';': 'raw'};
        if (typeMap[e.key]) {
            e.preventDefault();
            fetch(`${nbApiPath()}/cell/${currentCellId}/type`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `cell_type=${typeMap[e.key]}`
            });
        }
    }

    // ===== Phase 3: Shift+M - Merge with cell below =====
    if (e.shiftKey && e.key === 'M' && !inInput && !inMonaco && !mod && currentCellId) {
        e.preventDefault();
        fetch(`${nbApiPath()}/cell/${currentCellId}/merge-below`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'}
        });
    }

    // ===== Phase 4: Shift execution shortcuts =====
    if (e.shiftKey && !inInput && !inMonaco && !mod) {
        // Shift+R - Restart kernel
        if (e.key === 'R') {
            e.preventDefault();
            htmx.ajax('POST', nbApiPath() + '/kernel/restart', {target: '#status'});
        }
        // Shift+S - Stop all execution
        if (e.key === 'S') {
            e.preventDefault();
            if (typeof cancelAllExecution === 'function') cancelAllExecution();
        }
        // Shift+A - Run all code cells above
        if (e.key === 'A' && currentCellId) {
            e.preventDefault();
            const cells = getOrderedCells();
            const idx = cells.findIndex(c => c.id === `cell-${currentCellId}`);
            cells.slice(0, idx).forEach(c => {
                if (c.dataset.type === 'code') {
                    clickCellRunButton(c, {advance: false});
                }
            });
        }
        // Shift+B - Run all code cells below
        if (e.key === 'B' && currentCellId) {
            e.preventDefault();
            const cells = getOrderedCells();
            const idx = cells.findIndex(c => c.id === `cell-${currentCellId}`);
            cells.slice(idx + 1).forEach(c => {
                if (c.dataset.type === 'code') {
                    clickCellRunButton(c, {advance: false});
                }
            });
        }
        // Shift+O - Clamp output height (set to scrollable)
        if (e.key === 'O' && currentCellId) {
            e.preventDefault();
            setCollapseLevel(currentCellId, 'output', 1);
            fetch(`${nbApiPath()}/cell/${currentCellId}/collapse-section`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'section=output&level=1'
            });
        }
    }

    // ===== Phase 4: Alt/Opt+Enter - Run and create new cell =====
    if (e.altKey && e.key === 'Enter' && !mod) {
        if (currentCellId) {
            e.preventDefault();
            // Run current cell
            const cell = document.getElementById(`cell-${currentCellId}`);
            if (cell) {
                clickCellRunButton(cell, {advance: false});
            }
            // Create new cell below
            const cells = getOrderedCells();
            const idx = cells.findIndex(c => c.id === `cell-${currentCellId}`);
            const pos = idx >= 0 ? idx + 1 : cells.length;
            const nextCellId = cells[idx + 1]?.id?.replace('cell-', '') || null;
            _pendingScrollToNewCell = true;
            requestCellAdd('code', pos, {
                positionResolver: () => resolveAddPositionBefore(nextCellId),
            });
        }
    }
});

// Toggle cell state (skipped, pinned, is_exported) via server endpoint
function toggleCellState(cellId, property) {
    fetch(`${nbApiPath()}/cell/${cellId}/toggle/${property}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'}
    });
}

// ==================== Cell Clipboard Helpers ====================

/** Serialize selected cells into clipboard */
function _clipboardSerialize(mode) {
    // Clear previous cut styling
    document.querySelectorAll('.cell.cell-cut').forEach(c => c.classList.remove('cell-cut'));

    const ids = getSelectedCellIds();
    const cellIds = ids.length > 0 ? ids : (anchorCellId ? [anchorCellId] : []);
    cellClipboard = cellIds.map(id => {
        const el = document.getElementById(`cell-${id}`);
        if (!el) return null;
        // Read source from hidden textarea or Monaco
        let source = '';
        const textarea = el.querySelector(`#source-${id}`);
        if (textarea) source = textarea.value;
        else if (monacoEditors[id]) source = monacoEditors[id].getValue();
        // Read output text
        const outputEl = el.querySelector('.cell-output, .ai-response');
        const output = outputEl ? outputEl.textContent : '';
        return { id, type: el.dataset.type, source, output };
    }).filter(Boolean);

    clipboardMode = mode;
    if (mode === 'cut') {
        cellIds.forEach(id => {
            const el = document.getElementById(`cell-${id}`);
            if (el) el.classList.add('cell-cut');
        });
    }
    // Also copy to system clipboard as markdown
    if (cellClipboard.length > 0) {
        const md = cellClipboard.map(c => {
            if (c.type === 'code') return '```python\n' + c.source + '\n```';
            return c.source;
        }).join('\n\n');
        navigator.clipboard.writeText(md).catch(() => {});
    }
}

/** Paste clipboard cells after anchor */
async function _clipboardPaste() {
    if (cellClipboard.length === 0) return;
    const cells = getOrderedCells();
    const anchorIdx = anchorCellId
        ? cells.findIndex(c => c.id === `cell-${anchorCellId}`)
        : cells.length - 1;
    const basePos = anchorIdx + 1;

    for (let i = 0; i < cellClipboard.length; i++) {
        const item = cellClipboard[i];
        const pos = basePos + i;
        // Create cell
        await requestCellAdd(item.type, pos);
        // Wait a bit for the WS cell_add to insert the cell
        await new Promise(r => setTimeout(r, 100));
        // Find the newly added cell and set its source
        const newCells = getOrderedCells();
        if (newCells[pos]) {
            const newId = newCells[pos].id.replace('cell-', '');
            await fetch(`${nbApiPath()}/cell/${newId}/source`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `source=${encodeURIComponent(item.source)}`
            });
            // Update Monaco or textarea
            if (monacoEditors[newId]) {
                monacoEditors[newId].setValue(item.source);
            } else {
                const ta = document.querySelector(`#source-${newId}`);
                if (ta) ta.value = item.source;
                // Also update preview if it's a note/prompt
                const preview = document.querySelector(`#preview-${newId}`);
                if (preview) preview.innerHTML = renderMarkdown(item.source);
            }
        }
    }
    // If cut, delete originals
    if (clipboardMode === 'cut') {
        for (const item of cellClipboard) {
            const el = document.getElementById(`cell-${item.id}`);
            if (el) {
                const deleteBtn = el.querySelector('button[hx-delete]');
                if (deleteBtn) deleteBtn.click();
            }
        }
        document.querySelectorAll('.cell.cell-cut').forEach(c => c.classList.remove('cell-cut'));
    }
    cellClipboard = [];
    clipboardMode = null;
}

/** Copy cell input or output to system clipboard */
function _copyCellContent(cellId, section) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    let text = '';
    if (section === 'input') {
        const textarea = cell.querySelector(`#source-${cellId}`);
        if (textarea) text = textarea.value;
        else if (monacoEditors[cellId]) text = monacoEditors[cellId].getValue();
    } else {
        // Output: get text content from output area
        const outputEl = cell.querySelector('.cell-output, .ai-response, .output-content');
        if (outputEl) text = outputEl.textContent || '';
    }
    if (text) navigator.clipboard.writeText(text).catch(() => {});
}

/** Copy code blocks from AI response to clipboard */
function _copyCodeBlocks(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    const codeEls = cell.querySelectorAll('.ai-response pre code, .cell-output pre code, .output-content pre code');
    const blocks = Array.from(codeEls).map(el => el.textContent.trim()).filter(Boolean);
    if (blocks.length > 0) {
        navigator.clipboard.writeText(blocks.join('\n\n')).catch(() => {});
    }
}

// ==================== Keyboard Shortcuts Modal ====================

function toggleKeyboardShortcuts() {
    let overlay = document.getElementById('shortcuts-modal-overlay');
    if (!overlay) {
        overlay = _createShortcutsModal();
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('visible');
}

function _createShortcutsModal() {
    const overlay = document.createElement('div');
    overlay.id = 'shortcuts-modal-overlay';
    overlay.className = 'shortcuts-modal-overlay';
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.classList.remove('visible');
    });

    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const mod = isMac ? '⌘' : 'Ctrl';
    const alt = isMac ? '⌥' : 'Alt';

    const sections = [
        {
            title: 'Navigation',
            shortcuts: [
                ['↑ / k', 'Move to previous cell'],
                ['↓ / j', 'Move to next cell'],
                ['Shift+↑ / Shift+k', 'Extend selection up'],
                ['Shift+↓ / Shift+j', 'Extend selection down'],
                [`${mod}+A`, 'Select all cells'],
                ['Enter', 'Enter edit mode'],
                ['Escape', 'Exit edit mode'],
                ['← / →', 'Collapse / expand cell'],
            ]
        },
        {
            title: 'Cell Operations',
            shortcuts: [
                ['a', 'Add cell above'],
                ['b', 'Add cell below'],
                ['D D', 'Delete cell'],
                ['q', 'Duplicate cell'],
                ['Shift+M', 'Merge with cell below'],
                [`${mod}+Shift+D`, 'Delete cell (alt)'],
                [`${alt}+↑`, 'Move cell up'],
                [`${alt}+↓`, 'Move cell down'],
            ]
        },
        {
            title: 'Clipboard',
            shortcuts: [
                ['x', 'Cut selected cells'],
                ['c', 'Copy selected cells'],
                ['v', 'Paste cells'],
                [',', 'Copy cell input to clipboard'],
                ['.', 'Copy cell output to clipboard'],
                ['m', 'Copy code blocks from output'],
            ]
        },
        {
            title: 'Execution',
            shortcuts: [
                [`${mod}+Enter`, 'Run cell'],
                ['Shift+Enter', 'Run cell and move to next'],
                [`${alt}+Enter`, 'Run cell and create new below'],
                ['r', 'Re-run all code cells'],
                ['Shift+A', 'Run all code above'],
                ['Shift+B', 'Run all code below'],
                ['Shift+R', 'Restart kernel'],
                ['Shift+S', 'Stop all execution'],
                ['Esc Esc', 'Cancel all execution'],
            ]
        },
        {
            title: 'Cell State',
            shortcuts: [
                ['h', 'Toggle hidden from AI'],
                ['p', 'Toggle pinned'],
                ['e', 'Toggle export'],
            ]
        },
        {
            title: 'Display',
            shortcuts: [
                ['i', 'Toggle input collapse'],
                ['o', 'Toggle output collapse'],
                ['Shift+O', 'Clamp output height'],
                ['S S', 'Save notebook'],
                [`${mod}+S`, 'Save notebook'],
            ]
        },
        {
            title: 'Cell Type',
            shortcuts: [
                [`${mod}+Shift+J`, 'Switch to Code'],
                [`${mod}+Shift+K`, 'Switch to Note'],
                [`${mod}+Shift+L`, 'Switch to Prompt'],
                [`${mod}+Shift+;`, 'Switch to Raw'],
            ]
        },
        {
            title: 'Content',
            shortcuts: [
                ['n', 'Edit AI response'],
                ['w', 'Extract code blocks to cells'],
            ]
        },
        {
            title: 'Sidebar',
            shortcuts: [
                ['Ctrl+Shift+O', 'Toggle outline'],
                ['Ctrl+Shift+E', 'Toggle file explorer'],
                ['?', 'This shortcuts modal'],
            ]
        },
    ];

    let html = `<div class="shortcuts-modal">
        <div class="shortcuts-modal-header">
            <h3>Keyboard Shortcuts</h3>
            <button class="btn btn-sm shortcuts-modal-close" onclick="toggleKeyboardShortcuts()">&times;</button>
        </div>
        <div class="shortcuts-modal-body">`;

    for (const section of sections) {
        html += `<div class="shortcuts-category">
            <h4>${section.title}</h4>
            <table class="shortcuts-table">`;
        for (const [keys, desc] of section.shortcuts) {
            const kbds = keys.split(/(?<=\s)\+\s*|\s+(?=\+)/).join('').split('+').map(k =>
                `<kbd>${k.trim()}</kbd>`
            ).join(' + ');
            // Better: split by visual groups
            const keyBadges = keys.split(/\s*\+\s*/).map(k => `<kbd>${k}</kbd>`).join('');
            html += `<tr><td class="shortcut-keys">${keyBadges}</td><td>${desc}</td></tr>`;
        }
        html += `</table></div>`;
    }

    html += `</div></div>`;
    overlay.innerHTML = html;
    return overlay;
}

/** Get the current collapse level for a cell section (for persistence after cycling) */
function _getNextCollapseLevel(cellId, section) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return 0;
    const selector = section === 'input' ? '.cell-input' : '.cell-output, .output-section';
    const el = cell.querySelector(selector);
    if (!el) return 0;
    if (el.classList.contains('collapse-summary')) return 0; // was 2, cycled to 0
    if (el.classList.contains('collapse-scrollable')) return 2; // was 1, cycled to 2
    return 1; // was 0, cycled to 1
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
    const renderVersion = String(Number(targetEl.dataset.renderVersion || '0') + 1);
    targetEl.dataset.renderVersion = renderVersion;
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
        if (!targetEl.isConnected || targetEl.dataset.renderVersion !== renderVersion) return;
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
        if (!targetEl.isConnected || targetEl.dataset.renderVersion !== renderVersion) return;
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
const promptResponseEditor = window.DialengPromptResponseEditor
    ? new window.DialengPromptResponseEditor()
    : null;

function registerPromptResponseEditor(cellId) {
    const textarea = document.getElementById(`output-${cellId}`);
    if (textarea?.tagName === 'TEXTAREA') {
        promptResponseEditor?.register(cellId, textarea.value);
    }
}

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

document.addEventListener('input', function trackPromptResponseDraft(event) {
    const textarea = event.target;
    if (textarea?.tagName !== 'TEXTAREA' || !textarea.id.startsWith('output-')) return;
    promptResponseEditor?.input(textarea.id.slice('output-'.length), textarea.value);
});

document.addEventListener('focusout', function finishBlurredPromptResponse(event) {
    const textarea = event.target;
    if (textarea?.tagName !== 'TEXTAREA' || !textarea.id.startsWith('output-')) return;
    finishPromptResponseEdit(textarea.id.slice('output-'.length), textarea);
}, true);

// Keep setupPreviewEditing as a no-op for backward compatibility
// Existing call sites (htmx:afterSettle, DOMContentLoaded) don't need modification
function setupPreviewEditing() {
    // Event delegation handles this automatically - no individual listeners needed
}

function switchToEdit(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        if (field === 'output') {
            if (textarea.dataset.responseSaveStarted === 'true') {
                showKernelStatusMessage('Saving the previous response edit…', 'info', 2000);
                return;
            }
            const draft = promptResponseEditor?.begin(cellId, textarea.value);
            if (typeof draft === 'string' && textarea.value !== draft) textarea.value = draft;
            textarea.dataset.responseEditing = 'true';
        }
        preview.style.display = 'none';
        textarea.style.display = 'block';
        textarea.focus();
    }
}

function switchToPreview(cellId, field) {
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="${field}"]`);
    const textarea = document.getElementById(`${field}-${cellId}`);
    if (preview && textarea) {
        if (field === 'output') {
            promptResponseEditor?.finish(cellId, textarea.value);
            delete textarea.dataset.responseEditing;
        }
        // Show client-side preview immediately, then upgrade with server rendering
        preview.innerHTML = renderMarkdown(textarea.value);
        preview.style.display = 'block';
        textarea.style.display = 'none';
        renderMarkdownServer(textarea.value, preview);
    }
}

async function finishPromptResponseEdit(cellId, textarea) {
    if (textarea.dataset.responseSaveStarted === 'true') {
        switchToPreview(cellId, 'output');
        return;
    }
    switchToPreview(cellId, 'output');
    const draft = promptResponseEditor
        ? promptResponseEditor.startSave(cellId)
        : textarea.value;
    if (draft === null) return;
    textarea.dataset.responseSaveStarted = 'true';

    const expectedVersion = document.getElementById(`version-${cellId}`)?.value || '-1';
    const operationId = crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const body = new URLSearchParams({
        output: draft,
        expected_version: expectedVersion,
        operation_id: operationId,
    });

    try {
        const response = await fetch(`${nbApiPath()}/cell/${cellId}/output`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: body.toString(),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const responseText = await response.text();
        let result = {};
        if (responseText.trim()) {
            try { result = JSON.parse(responseText); } catch (_) {}
        }
        if (result.status === 'conflict') {
            applyPromptOutputUpdate(cellId, result.output, {force: true});
            updateCellVersion(cellId, result.version);
            switchToPreview(cellId, 'output');
            showKernelStatusMessage(
                'That response changed in another tab. Dialeng kept the newer backend version.',
                'warning',
                7000,
            );
            return;
        }
        promptResponseEditor?.commit(cellId, draft);
    } catch (error) {
        promptResponseEditor?.saveFailed(cellId);
        console.error('[Prompt] Failed to save edited response:', error);
        showKernelStatusMessage(
            'The response could not be saved. Your draft is still available; double-click it to retry.',
            'error',
            8000,
        );
    } finally {
        delete textarea.dataset.responseSaveStarted;
    }
}

function finishTextCellEdit(element = document.activeElement) {
    if (!element || element.tagName !== 'TEXTAREA' || !element.closest('.cell')) return false;
    if (getComputedStyle(element).display === 'none') return false;

    const match = element.id.match(/^(source|prompt|output)-(.+)$/);
    if (!match) return false;
    const [, field, cellId] = match;

    // Trigger HTMX before releasing focus. The `changed` modifier on the
    // textarea suppresses the subsequent native blur request, while this
    // explicit event makes command-mode transitions deterministic.
    if (typeof htmx !== 'undefined' && element.hasAttribute('hx-post')) {
        htmx.trigger(element, 'blur');
    }
    if (field === 'output') {
        finishPromptResponseEdit(cellId, element);
        element.blur();
        return true;
    }
    switchToPreview(cellId, field);
    element.blur();
    return true;
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
const editorDisposeTimers = new Map();
let monacoIntersectionObserver = null;

function ensureMonacoEditor(cellId, mode = 'python') {
    const existing = monacoEditors[cellId];
    if (existing) return Promise.resolve(existing);
    return loadMonaco().then(() => initMonacoEditor(cellId, mode));
}

function disposeMonacoView(cellId) {
    const editor = monacoEditors[cellId];
    if (!editor) return;
    syncMonacoToTextarea(cellId);
    monacoViewStates.set(cellId, editor.saveViewState());
    editor.dispose();
    delete monacoEditors[cellId];
    const container = document.getElementById(`monaco-${cellId}`);
    const textarea = document.getElementById(`source-${cellId}`);
    if (container && textarea) {
        container.replaceChildren(Object.assign(document.createElement('pre'), {
            className: 'monaco-placeholder',
            textContent: textarea.value,
        }));
        container.style.opacity = '1';
    }
}

function getMonacoIntersectionObserver() {
    if (monacoIntersectionObserver || !('IntersectionObserver' in window)) {
        return monacoIntersectionObserver;
    }
    monacoIntersectionObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            const cellId = entry.target.id.replace('monaco-', '');
            const cell = entry.target.closest('.cell');
            const mode = cell?.dataset.type === 'shell' ? 'sh' : 'python';
            if (entry.isIntersecting) {
                const timer = editorDisposeTimers.get(cellId);
                if (timer) clearTimeout(timer);
                editorDisposeTimers.delete(cellId);
                ensureMonacoEditor(cellId, mode);
            } else if (monacoEditors[cellId] && !editorDisposeTimers.has(cellId)) {
                editorDisposeTimers.set(cellId, setTimeout(() => {
                    editorDisposeTimers.delete(cellId);
                    if (focusedCellId !== cellId) disposeMonacoView(cellId);
                }, 15000));
            }
        });
    }, {rootMargin: '900px 0px'});
    return monacoIntersectionObserver;
}

function observeMonacoCell(cellId, mode = 'python') {
    const container = document.getElementById(`monaco-${cellId}`);
    if (!container) return;
    const observer = getMonacoIntersectionObserver();
    if (observer) observer.observe(container);
    else ensureMonacoEditor(cellId, mode);
}

function initCell(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    if (!cell) return;
    
    // Initialize Monaco editor for code and shell cells
    if (cell.dataset.type === 'code') {
        observeMonacoCell(cellId);
    } else if (cell.dataset.type === 'shell') {
        observeMonacoCell(cellId, 'sh');
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
        registerPromptResponseEditor(cellId);
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
    setupLargeOutput(cellId);
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
// Structural operations (add/delete/move) now reconcile entirely from the
// backend-authored WebSocket payloads. The remaining HTMX-driven DOM swaps are
// cell-local operations such as type changes or execution-completion OOB swaps.
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
let _suppressHtmxScrollRestoreUntil = 0;

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
    if (Date.now() < _suppressHtmxScrollRestoreUntil) {
        _htmxScrollRestore = null;
        return;
    }
    if (_htmxScrollRestore !== null && !_pendingScrollToNewCell) {
        window.scrollTo(0, _htmxScrollRestore);
    }
    if (e.detail?.target?.id === 'status') {
        armStatusToastDismiss(e.detail.target);
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
    if (Date.now() < _suppressHtmxScrollRestoreUntil) {
        _htmxScrollRestore = null;
        return;
    }
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

function setupLargeOutput(cellId) {
    const output = document.getElementById(`output-${cellId}`);
    if (!output || output.dataset.largeOutputReady === 'true' || !output.children.length) return;
    const textLength = output.textContent?.length || 0;
    const lineCount = (output.textContent?.match(/\n/g) || []).length + 1;
    if (textLength < 50000 && lineCount < 300 && output.scrollHeight < 700) return;

    output.dataset.largeOutputReady = 'true';
    const content = document.createElement('div');
    content.className = 'large-output-content collapsed';
    while (output.firstChild) content.appendChild(output.firstChild);
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'btn btn-sm large-output-toggle';
    toggle.textContent = 'Show full output';
    toggle.onclick = () => {
        const expanded = content.classList.toggle('expanded');
        content.classList.toggle('collapsed', !expanded);
        toggle.textContent = expanded ? 'Collapse output' : 'Show full output';
    };
    output.append(content, toggle);
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
        clearPendingRunRequest(cellId);
    }

    // Fallback: reset prompt streaming if we have a streaming cell
    if (streamingCellId) {
        finishStreaming(streamingCellId);
    }
}

document.addEventListener('htmx:responseError', (e) => {
    console.error('[HTMX] Response error:', e.detail);
    if (e.detail?.xhr?.status === 409) {
        let conflict = {};
        try { conflict = JSON.parse(e.detail.xhr.responseText || '{}'); } catch (_) {}
        if (conflict.type === 'external_file_changed') {
            setNotebookSaveState('failed');
            showExternalSaveConflict(conflict);
        } else if (conflict.type === 'prompt_operation_conflict') {
            if (conflict.operation === 'save') setNotebookSaveState('modified');
            showKernelStatusMessage(
                conflict.message || 'A prompt is still running. Wait or cancel it before changing the notebook.',
                'warning',
                7000,
            );
        } else {
            recoverNotebookState('backend revision conflict');
            showKernelStatusMessage('The notebook changed in another tab. Resynchronizing…', 'warning', 5000);
        }
        return;
    }
    resetCellOnError(e, 'Request failed. Please try again.');
});

document.addEventListener('htmx:configRequest', (event) => {
    const path = event.detail?.path || event.detail?.pathInfo?.requestPath || '';
    const isCellMutation = /\/cell\/(?:add|[^/?]+(?:\/(?:type|move\/(?:up|down)|duplicate|clear-output|merge-below|extract-code-blocks|collapse(?:-section)?))?)(?:$|\?)/.test(path);
    if (!isCellMutation) return;

    const isStructureMutation = /\/cell\/(?:add|[^/?]+(?:\/(?:move\/(?:up|down)|duplicate|merge-below|extract-code-blocks))?)(?:$|\?)/.test(path);

    // Type changes use cell version + expected old type because a textarea
    // blur save can legitimately advance the notebook revision just before
    // the select change reaches the server.
    if (isStructureMutation) {
        event.detail.parameters.expected_structure_revision = getCurrentStructureRevision();
        delete event.detail.parameters.expected_revision;
    } else if (!/\/cell\/[^/?]+\/type(?:$|\?)/.test(path)) {
        event.detail.parameters.expected_revision = getCurrentNotebookRevision();
    }
    event.detail.parameters.operation_id = crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
});

document.addEventListener('htmx:beforeRequest', (event) => {
    const path = event.detail?.pathInfo?.requestPath || event.detail?.requestConfig?.path || '';
    if (/\/dialeng\/[^/]+\/save(?:\?|$)/.test(path)) setNotebookSaveState('saving');
});

document.addEventListener('htmx:afterRequest', (event) => {
    const path = event.detail?.pathInfo?.requestPath || event.detail?.requestConfig?.path || '';
    if (!/\/dialeng\/[^/]+\/save(?:\?|$)/.test(path)) return;
    setNotebookSaveState(event.detail?.successful ? 'saved' : 'failed');
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
    if (window.AUTOSAVE_ENABLED) checkNotebookRecovery();
    if (document.getElementById('extension-manager-btn')) refreshExtensionManager();
});

document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.NOTEBOOK_ID) {
        if (!isWebSocketHealthy()) fetchKernelSnapshot(window.NOTEBOOK_ID, 'visibility');
        else sendWebSocketHeartbeat();
    }
});

window.addEventListener('online', () => {
    if (window.NOTEBOOK_ID) {
        if (!isWebSocketHealthy()) fetchKernelSnapshot(window.NOTEBOOK_ID, 'online');
        else sendWebSocketHeartbeat();
    }
});

// Auto-resize textareas
document.addEventListener('input', e => {
    if (e.target.tagName === 'TEXTAREA') {
        e.target.style.height = 'auto';
        e.target.style.height = Math.max(60, e.target.scrollHeight) + 'px';
        if (!isFilePage()) setNotebookSaveState('modified');
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
    const providerSelect = document.getElementById('provider-select');
    if (modelSelect) {
        // Show model dropdown only for non-mock modes
        modelSelect.style.display = mode === 'mock' ? 'none' : '';
    }
    if (providerSelect) providerSelect.style.display = mode === 'mock' ? 'none' : '';
}

function syncLlmSelectors(llm, notebook) {
    if (!llm || !notebook) return;
    const providerSelect = document.getElementById('provider-select');
    if (providerSelect) {
        const selected = notebook.llm_provider || llm.selected_provider || '';
        const signature = JSON.stringify((llm.providers || []).map(
            p => [p.id, p.label, p.authenticated, p.validation_state]
        ));
        if (providerSelect.dataset.inventorySignature !== signature) {
            providerSelect.replaceChildren(...(llm.providers || []).map(provider => {
                const option = document.createElement('option');
                option.value = provider.id;
                option.textContent = provider.validation_state === 'validating'
                    ? `${provider.label} (validating...)`
                    : provider.authenticated ? provider.label : `${provider.label} (unavailable)`;
                option.disabled = !provider.authenticated;
                option.selected = provider.id === selected;
                return option;
            }));
            providerSelect.dataset.inventorySignature = signature;
        }
        if (providerSelect.value !== selected) providerSelect.value = selected;
    }

    const selectedProvider = (llm.providers || []).find(p => p.id === notebook.llm_provider);
    const modelSelect = document.getElementById('model-select');
    if (modelSelect && selectedProvider) {
        const models = selectedProvider.models || [];
        const signature = JSON.stringify(models.map(model => [model.id, model.name]));
        if (modelSelect.dataset.modelSignature !== signature) {
            modelSelect.replaceChildren(...models.map(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                option.selected = model.id === notebook.model;
                return option;
            }));
            modelSelect.dataset.modelSignature = signature;
        }
        if (modelSelect.value !== notebook.model) modelSelect.value = notebook.model || '';
    }
}

let openAICodexPollId = null;

function updateOpenAICodexAuthUi(status) {
    const state = document.getElementById('openai-codex-auth-state');
    if (state) {
        state.textContent = status.authenticated
            ? `Connected (${String(status.account_id || '').slice(0, 12)}...)`
            : (status.error || (status.pending ? 'Waiting for OpenAI authorization...' : 'Not connected'));
    }
    const logout = document.getElementById('openai-codex-logout');
    if (logout) logout.style.display = status.authenticated ? '' : 'none';
    const login = document.getElementById('openai-codex-login');
    if (login) login.style.display = status.authenticated ? 'none' : '';
}

async function pollOpenAICodexStatus() {
    const response = await fetch('/auth/openai/status', { cache: 'no-store' });
    const status = await response.json();
    updateOpenAICodexAuthUi(status);
    if (status.authenticated || (!status.pending && status.error)) {
        if (openAICodexPollId) clearInterval(openAICodexPollId);
        openAICodexPollId = null;
        if (status.authenticated) showKernelStatusMessage('OpenAI Codex connected.', 'success', 3500);
    }
    return status;
}

async function completeOpenAICodexLogin(code) {
    if (!code) return;
    const response = await fetch('/auth/openai/complete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({code})
    });
    const status = await response.json();
    updateOpenAICodexAuthUi(status);
    if (!response.ok) throw new Error(status.error || 'OpenAI authorization failed');
    showKernelStatusMessage('OpenAI Codex connected.', 'success', 3500);
}

async function startOpenAICodexLogin() {
    try {
        const response = await fetch('/auth/openai/start', { method: 'POST' });
        const status = await response.json();
        if (!response.ok) throw new Error(status.error || 'Could not start OpenAI login');
        updateOpenAICodexAuthUi(status);
        if (status.authorization_url) window.open(status.authorization_url, '_blank', 'noopener');
        if (openAICodexPollId) clearInterval(openAICodexPollId);
        openAICodexPollId = setInterval(pollOpenAICodexStatus, 1500);
        if (!status.callback_available) {
            const value = window.prompt('After authorizing OpenAI, paste the final redirect URL or code#state here:');
            if (value) await completeOpenAICodexLogin(value);
        }
    } catch (error) {
        showKernelStatusMessage(error.message, 'error', 7000);
    }
}

async function logoutOpenAICodex() {
    const response = await fetch('/auth/openai/logout', { method: 'POST' });
    const status = await response.json();
    updateOpenAICodexAuthUi(status);
    const message = status.authenticated
        ? 'Stored OpenAI credentials were removed; the environment token remains active.'
        : 'OpenAI Codex disconnected.';
    showKernelStatusMessage(message, status.authenticated ? 'warning' : 'success', 4000);
}

let kernelModalDismissed = false;

function shouldKeepKernelModalOpen() {
    if (kernelModalDismissed) return false;
    const kernelState = getKernelState();
    const kernel = kernelState?.kernel;
    return !kernel?.exists || !!kernel?.auth_required;
}

function refreshKernelChrome({ refreshModal = true } = {}) {
    if (!window.NOTEBOOK_ID || typeof htmx === 'undefined') return;
    htmx.ajax('GET', `${nbApiPath()}/kernel/info`, { target: '#kernel-status-bar', swap: 'outerHTML' });
    if (refreshModal) {
        const modal = document.getElementById('kernel-modal-overlay');
        if (modal) {
            const keepOpen = modal.classList.contains('visible') || shouldKeepKernelModalOpen();
            if (keepOpen) {
                const restoreVisible = function(event) {
                    const target = event.detail?.target;
                    if (target?.id === 'kernel-modal-overlay') {
                        target.classList.add('visible');
                        document.body.removeEventListener('htmx:afterSwap', restoreVisible);
                    }
                };
                document.body.addEventListener('htmx:afterSwap', restoreVisible);
            }
            htmx.ajax('GET', `${nbApiPath()}/kernel/modal`, { target: '#kernel-modal-overlay', swap: 'outerHTML' });
        }
    }
}

function applyNotebookSnapshot(notebook, previousNotebook = null, llm = null) {
    if (!notebook) return;
    documentRevisionTracker.recordNotebook(notebook);

    applyCellVersions(notebook.cell_versions || {});

    const modeSelect = document.getElementById('mode-select');
    if (modeSelect && modeSelect.value !== notebook.dialog_mode) {
        modeSelect.value = notebook.dialog_mode;
    }
    toggleModelSelect(notebook.dialog_mode);
    syncLlmSelectors(llm, notebook);

    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        const nextModel = notebook.model || '';
        if (modelSelect.value !== nextModel) {
            modelSelect.value = nextModel;
        }
    }

    const safeModeBtn = document.getElementById('safe-mode-toggle');
    if (safeModeBtn) {
        safeModeBtn.classList.toggle('active', !!notebook.safe_mode);
        const use = safeModeBtn.querySelector('use');
        if (use) use.setAttribute('href', notebook.safe_mode ? '#shield-check' : '#shield-off');
    }

    if (!previousNotebook || previousNotebook.kernel_type !== notebook.kernel_type) {
        handleKernelTypeChanged(notebook.kernel_type);
    }
}

function updateCellVersion(cellId, version) {
    if (!cellId || version === undefined || version === null) return;
    const versionInput = document.getElementById(`version-${cellId}`);
    if (versionInput) versionInput.value = String(version);
}

function applyCellVersions(cellVersions) {
    Object.entries(cellVersions || {}).forEach(([cellId, version]) => {
        updateCellVersion(cellId, version);
    });
}

function syncCurrentExplorerKernelState(snapshot) {
    if (!window.NOTEBOOK_ID) return;
    const item = document.querySelector(`.file-explorer-item[data-notebook-id="${window.NOTEBOOK_ID}"]`);
    if (!item) return;
    item.classList.toggle('has-kernel', !!snapshot?.kernel?.is_alive);
}

// ==================== Safe Mode Toggle ====================
function toggleSafeMode(nbId) {
    const btn = document.getElementById('safe-mode-toggle');
    if (!btn) return;
    const isActive = btn.classList.contains('active');
    const newState = !isActive;
    btn.disabled = true;

    // Send to server
    fetch(`/dialeng/${nbId}/safe_mode`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `safe_mode=${newState}`
    }).finally(() => {
        btn.disabled = false;
    });
}

// ==================== Colab Auth State Listener ====================
function onColabAuthenticated() {
    if (window.NOTEBOOK_ID) {
        fetchKernelSnapshot(window.NOTEBOOK_ID, 'auth');
        refreshKernelChrome({ refreshModal: true });
    }
}

function onColabDisconnected() {
    if (window.NOTEBOOK_ID) {
        fetchKernelSnapshot(window.NOTEBOOK_ID, 'auth');
        refreshKernelChrome({ refreshModal: true });
    }
}

// Listen for postMessage from popup (works if browser preserves window.opener)
window.addEventListener('message', function(event) {
    if (event.data === 'colab-authenticated') onColabAuthenticated();
});

// Listen for localStorage event from popup (reliable cross-origin fallback)
window.addEventListener('storage', function(event) {
    if (event.key === 'colab-auth-event') onColabAuthenticated();
    if (event.key === 'dialeng-settings-changed') {
        refreshSettings({preserveDirty: true});
    }
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
    const safeModeToggle = document.getElementById('safe-mode-toggle');
    if (safeModeToggle) {
        safeModeToggle.style.display = kernelType === 'local' ? '' : 'none';
    }
}

// ==================== Notebook-to-skill creator ====================
const skillCreatorState = {
    buildId: '',
    buildHash: '',
    files: {},
    loading: false,
};

function setSkillCreatorStatus(message = '', tone = '') {
    const status = document.getElementById('skill-creator-status');
    if (!status) return;
    status.textContent = message;
    status.className = `skill-creator-status${tone ? ` ${tone}` : ''}`;
}

async function openSkillCreator() {
    const overlay = document.getElementById('skill-creator-overlay');
    if (!overlay) return;
    overlay.classList.add('visible');
    document.body.classList.add('modal-open');
    setSkillCreatorStatus('Loading project sources...', 'pending');
    try {
        const response = await fetch('/dialeng/skills', {headers: {'Accept': 'application/json'}});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Skill request failed (${response.status})`);
        const sources = document.getElementById('skill-creator-sources');
        if (sources) {
            sources.replaceChildren();
            const projectSources = [
                ...(payload.notebooks || []).map(path => ({path, kind: 'Notebook'})),
                ...(payload.source_files || []).map(path => ({path, kind: 'File'})),
            ];
            if (!projectSources.length) {
                const empty = document.createElement('p');
                empty.textContent = 'No supported project sources were found.';
                sources.appendChild(empty);
            } else {
                projectSources.forEach(({path, kind}) => {
                    const label = document.createElement('label');
                    label.className = 'skill-source-option';
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.value = path;
                    checkbox.name = 'skill-source';
                    const text = document.createElement('span');
                    text.textContent = path;
                    text.title = kind;
                    label.append(checkbox, text);
                    sources.appendChild(label);
                });
            }
        }
        setSkillCreatorStatus('Select project sources, add optional web pages, and generate a reviewable draft.');
        requestAnimationFrame(() => document.getElementById('skill-creator-name')?.focus());
    } catch (error) {
        setSkillCreatorStatus(error.message || 'Could not load project skills.', 'error');
    }
}

function closeSkillCreator() {
    document.getElementById('skill-creator-overlay')?.classList.remove('visible');
    document.body.classList.remove('modal-open');
}

function showSkillPreviewFile(path) {
    const preview = document.getElementById('skill-creator-preview');
    if (preview) preview.textContent = skillCreatorState.files[path] || '';
    document.querySelectorAll('.skill-preview-file').forEach(button => {
        button.classList.toggle('active', button.dataset.path === path);
    });
}

function renderSkillPreviewFiles(files = {}) {
    skillCreatorState.files = files;
    const container = document.getElementById('skill-creator-files');
    if (!container) return;
    container.replaceChildren();
    Object.keys(files).sort((a, b) => (a === 'SKILL.md' ? -1 : b === 'SKILL.md' ? 1 : a.localeCompare(b))).forEach(path => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'skill-preview-file';
        button.dataset.path = path;
        button.textContent = path;
        button.addEventListener('click', () => showSkillPreviewFile(path));
        container.appendChild(button);
    });
    showSkillPreviewFile(files['SKILL.md'] !== undefined ? 'SKILL.md' : Object.keys(files)[0]);
}

async function generateSkillDraft() {
    if (skillCreatorState.loading) return;
    const name = document.getElementById('skill-creator-name')?.value.trim() || '';
    const description = document.getElementById('skill-creator-description')?.value.trim() || '';
    const instructions = document.getElementById('skill-creator-instructions')?.value.trim() || '';
    const sources = Array.from(document.querySelectorAll('input[name="skill-source"]:checked')).map(input => input.value);
    const urls = (document.getElementById('skill-creator-urls')?.value || '')
        .split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    sources.push(...urls);
    if (!name || !description || !sources.length) {
        setSkillCreatorStatus('Provide a name, description, and at least one project file or web page.', 'error');
        return;
    }
    skillCreatorState.loading = true;
    document.getElementById('skill-generate-btn')?.setAttribute('disabled', '');
    document.getElementById('skill-install-btn')?.setAttribute('disabled', '');
    setSkillCreatorStatus('Extracting sources with provenance and generating the staged draft...', 'pending');
    try {
        const response = await fetch('/dialeng/skills/build', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: JSON.stringify({
                name,
                description,
                instructions,
                sources,
                include_outputs: Boolean(document.getElementById('skill-creator-outputs')?.checked),
                use_llm: Boolean(document.getElementById('skill-creator-use-llm')?.checked),
                provider: document.getElementById('provider-select')?.value || '',
                model: document.getElementById('model-select')?.value || '',
            }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Skill generation failed (${response.status})`);
        skillCreatorState.buildId = payload.build_id;
        skillCreatorState.buildHash = payload.build_hash;
        renderSkillPreviewFiles(payload.files || {});
        const label = document.getElementById('skill-creator-build-label');
        if (label) label.textContent = `${payload.target} · ${payload.sources.length} source(s)`;
        const warnings = [...(payload.warnings || [])];
        if (payload.refinement_error) warnings.push(`LLM refinement skipped: ${payload.refinement_error}`);
        if (payload.validation_errors?.length) {
            setSkillCreatorStatus(payload.validation_errors.join(' '), 'error');
        } else {
            setSkillCreatorStatus(warnings.join(' ') || 'Draft validated. Review it before installing.', warnings.length ? 'warning' : 'success');
            document.getElementById('skill-install-btn')?.removeAttribute('disabled');
        }
    } catch (error) {
        setSkillCreatorStatus(error.message || 'Could not generate the skill.', 'error');
    } finally {
        skillCreatorState.loading = false;
        document.getElementById('skill-generate-btn')?.removeAttribute('disabled');
    }
}

async function installSkillDraft() {
    if (!skillCreatorState.buildId || !skillCreatorState.buildHash || skillCreatorState.loading) return;
    if (!window.confirm('Install this reviewed skill into .dialeng/skills?')) return;
    skillCreatorState.loading = true;
    document.getElementById('skill-install-btn')?.setAttribute('disabled', '');
    setSkillCreatorStatus('Installing skill atomically...', 'pending');
    try {
        const response = await fetch(`/dialeng/skills/build/${encodeURIComponent(skillCreatorState.buildId)}/commit`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: JSON.stringify({expected_hash: skillCreatorState.buildHash}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Skill installation failed (${response.status})`);
        setSkillCreatorStatus(`Installed ${payload.name}. It is available on the next Codex prompt.`, 'success');
        window.DialengNotifications?.show(`Installed skill ${payload.name}`, 'success', 3500);
        skillCreatorState.buildId = '';
        skillCreatorState.buildHash = '';
    } catch (error) {
        setSkillCreatorStatus(error.message || 'Could not install the skill.', 'error');
        document.getElementById('skill-install-btn')?.removeAttribute('disabled');
    } finally {
        skillCreatorState.loading = false;
    }
}

// ==================== Backend-authoritative Settings ====================
const settingsState = {
    revision: '',
    baseline: new Map(),
    dirty: new Set(),
    lastSnapshot: null,
    initialized: false,
    saving: false,
};

function getSettingControl(path) {
    const row = document.querySelector(`.setting-row[data-setting-path="${CSS.escape(path)}"]`);
    return row?.querySelector('input[type="checkbox"], select, input[type="number"], input[type="range"], input[type="text"], input[type="password"]') || null;
}

function readSettingControl(control) {
    if (!control) return undefined;
    if (control.type === 'checkbox') return Boolean(control.checked);
    if (control.type === 'number' || control.type === 'range') return Number(control.value);
    return control.value;
}

function writeSettingControl(control, value) {
    if (!control) return;
    if (control.type === 'checkbox') control.checked = Boolean(value);
    else control.value = value ?? '';
    if (control.type === 'range') {
        const valueEl = document.getElementById(`${control.id}-value`);
        if (valueEl) valueEl.textContent = control.value;
    }
}

function setSettingsStatus(message = '', tone = '') {
    const status = document.getElementById('settings-status');
    if (!status) return;
    status.textContent = message;
    status.className = `settings-status${tone ? ` ${tone}` : ''}`;
}

function clearSettingErrors() {
    document.querySelectorAll('[data-setting-error]').forEach(el => {
        el.textContent = '';
        el.closest('.setting-row')?.classList.remove('invalid');
    });
}

function showSettingErrors(errors = {}) {
    clearSettingErrors();
    Object.entries(errors).forEach(([path, message]) => {
        const error = document.querySelector(`[data-setting-error="${CSS.escape(path)}"]`);
        if (error) {
            error.textContent = message;
            error.closest('.setting-row')?.classList.add('invalid');
            error.closest('.settings-group')?.setAttribute('open', '');
        }
    });
}

function updateSettingsDirtyState() {
    settingsState.dirty.clear();
    document.querySelectorAll('.setting-row[data-setting-path]').forEach(row => {
        const path = row.dataset.settingPath;
        const control = getSettingControl(path);
        if (!control || control.disabled) return;
        const dirty = JSON.stringify(readSettingControl(control)) !== JSON.stringify(settingsState.baseline.get(path));
        row.classList.toggle('dirty', dirty);
        if (dirty) settingsState.dirty.add(path);
    });
    const count = settingsState.dirty.size;
    const summary = document.getElementById('settings-change-summary');
    if (summary) summary.textContent = count ? `${count} unsaved change${count === 1 ? '' : 's'}` : 'No unsaved changes';
    const save = document.getElementById('settings-save-btn');
    const revert = document.getElementById('settings-revert-btn');
    if (save) save.disabled = count === 0 || settingsState.saving;
    if (revert) revert.disabled = count === 0 || settingsState.saving;
}

function applyDisplaySettings(display = {}) {
    const root = document.documentElement.style;
    const size = display.button_size || 'normal';
    const padding = {compact: '3px 6px', normal: '5px 10px', large: '8px 14px'}[size] || '5px 10px';
    const font = {compact: '0.75rem', normal: '0.8rem', large: '0.9rem'}[size] || '0.8rem';
    const smallPadding = {compact: '2px 5px', normal: '4px 8px', large: '6px 12px'}[size] || '4px 8px';
    if (display.notebook_width) root.setProperty('--notebook-width', `${display.notebook_width}px`);
    if (display.font_size) root.setProperty('--base-font-size', `${display.font_size}px`);
    root.setProperty('--btn-padding', padding);
    root.setProperty('--btn-font-size', font);
    root.setProperty('--btn-sm-padding', smallPadding);
}

function applySettingsSnapshot(snapshot, {preserveDirty = false} = {}) {
    if (!snapshot?.fields) return;
    const previousDirty = new Set(settingsState.dirty);
    settingsState.revision = snapshot.revision || settingsState.revision;
    settingsState.lastSnapshot = snapshot;
    const revision = document.getElementById('settings-revision');
    if (revision) revision.value = settingsState.revision;

    Object.entries(snapshot.fields).forEach(([path, field]) => {
        const control = getSettingControl(path);
        if (!control) return;
        const baselineValue = field.secret ? '' : field.value;
        if (!(preserveDirty && previousDirty.has(path))) writeSettingControl(control, baselineValue);
        settingsState.baseline.set(path, baselineValue);
        control.disabled = Boolean(field.managed);
        const source = document.querySelector(`[data-setting-source="${CSS.escape(path)}"]`);
        if (source) {
            source.textContent = field.source || '';
            source.className = `setting-source-badge ${field.source || ''}`;
            source.title = field.diagnostic
                ? `${field.env_var}: ${field.diagnostic}`
                : field.managed && field.env_var
                    ? `Managed by ${field.env_var}`
                    : `Value source: ${field.source || 'project'}`;
        }
    });
    applyDisplaySettings(snapshot.display);
    const autosave = snapshot.fields['persistence.autosave.enabled'];
    if (autosave) window.AUTOSAVE_ENABLED = Boolean(autosave.effective_value);
    updateSettingsDirtyState();

    const activation = snapshot.activation || {};
    if (activation.status === 'validating' || activation.status === 'pending') {
        setSettingsStatus(activation.message || 'Settings saved. Validating runtime...', 'pending');
    } else if (activation.status === 'failed') {
        setSettingsStatus(`${activation.message || 'Activation failed'} ${activation.error || ''}`.trim(), 'error');
    } else if (!settingsState.dirty.size) {
        setSettingsStatus(activation.message || 'Settings are current.', activation.message ? 'success' : '');
    } else {
        setSettingsStatus('Settings loaded. Review the unsaved changes.', 'pending');
    }
}

function requestSettingsSnapshot() {
    return new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open('GET', '/settings', true);
        request.responseType = 'json';
        request.timeout = 10000;
        request.setRequestHeader('Accept', 'application/json');
        request.onload = () => {
            if (request.status >= 200 && request.status < 300) resolve(request.response);
            else reject(new Error(`Settings request failed (${request.status})`));
        };
        request.onerror = () => reject(new Error('Could not load settings.'));
        request.ontimeout = () => reject(new Error('Loading settings timed out.'));
        request.send();
    });
}

async function refreshSettings({preserveDirty = false} = {}) {
    setSettingsStatus('Loading current settings...', 'pending');
    try {
        const snapshot = await requestSettingsSnapshot();
        applySettingsSnapshot(snapshot, {preserveDirty});
    } catch (error) {
        setSettingsStatus(error.message || 'Could not load settings.', 'error');
    }
}

function initializeSettingsForm() {
    if (settingsState.initialized) return;
    const form = document.getElementById('settings-form');
    if (!form) return;
    settingsState.initialized = true;
    form.addEventListener('input', event => {
        event.target.closest('.setting-row')?.classList.remove('invalid');
        updateSettingsDirtyState();
    });
    form.addEventListener('change', updateSettingsDirtyState);
}

function toggleSettings() {
    const sidebar = document.getElementById('settings-sidebar');
    const overlay = document.getElementById('settings-overlay');
    if (!sidebar) return;
    if (sidebar.classList.contains('open')) {
        closeSettings();
        return;
    }
    initializeSettingsForm();
    sidebar.classList.add('open');
    overlay?.classList.add('open');
    refreshSettings({preserveDirty: false});
    requestAnimationFrame(() => document.getElementById('settings-search')?.focus());
}

function closeSettings(force = false) {
    if (!force && settingsState.dirty.size && !window.confirm('Discard unsaved settings changes?')) return false;
    if (settingsState.dirty.size) revertSettings();
    document.getElementById('settings-sidebar')?.classList.remove('open');
    document.getElementById('settings-overlay')?.classList.remove('open');
    return true;
}

function revertSettings() {
    settingsState.baseline.forEach((value, path) => writeSettingControl(getSettingControl(path), value));
    clearSettingErrors();
    updateSettingsDirtyState();
    setSettingsStatus('Unsaved changes reverted.', '');
}

function filterSettings(query) {
    const normalized = String(query || '').trim().toLowerCase();
    document.querySelectorAll('.settings-group').forEach(group => {
        const matches = !normalized || group.textContent.toLowerCase().includes(normalized);
        group.hidden = !matches;
        if (normalized && matches) group.open = true;
    });
}

async function saveSettings(event) {
    event?.preventDefault();
    if (!settingsState.dirty.size || settingsState.saving) return;
    settingsState.saving = true;
    updateSettingsDirtyState();
    clearSettingErrors();
    setSettingsStatus('Saving...', 'pending');
    const changes = {};
    settingsState.dirty.forEach(path => {
        changes[path] = readSettingControl(getSettingControl(path));
    });
    try {
        const response = await fetch('/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: JSON.stringify({revision: settingsState.revision, changes}),
        });
        const payload = await response.json();
        if (response.status === 409) {
            applySettingsSnapshot(payload.snapshot, {preserveDirty: true});
            setSettingsStatus('Settings changed elsewhere. Your edits were kept; review and save again.', 'error');
            return;
        }
        if (response.status === 422) {
            showSettingErrors(payload.errors);
            setSettingsStatus('Fix the highlighted settings before saving.', 'error');
            return;
        }
        if (!response.ok) throw new Error(payload.message || `Settings save failed (${response.status})`);
        applySettingsSnapshot(payload);
        const modes = payload.apply_modes || [];
        const suffix = modes.includes('server_restart')
            ? ' Restart Dialeng to apply the marked settings.'
            : modes.includes('new_kernel') ? ' New-kernel defaults apply on the next attachment.' : '';
        setSettingsStatus(`Settings saved.${suffix}`, modes.includes('server_restart') ? 'pending' : 'success');
        try { localStorage.setItem('dialeng-settings-changed', `${Date.now()}:${payload.revision}`); } catch (_) {}
    } catch (error) {
        setSettingsStatus(error.message || 'Could not save settings.', 'error');
    } finally {
        settingsState.saving = false;
        updateSettingsDirtyState();
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
        fetch(`${nbApiPath()}/cell/${cellId}/collapse`, {
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
    fetch(`${nbApiPath()}/cell/${cellId}/collapse-section`, {
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
    flushPromptChunks(cellId);
    cancelledCells.add(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    setCellRunControlState(cell, false);
    // Send cancel message via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        sendWebSocketControl({type: 'cancel', cell_id: cellId});
    }
    showKernelStatusMessage(
        'LLM generation cancelled. The Python kernel was not interrupted.',
        'warning',
        4500,
    );
    streamingCellId = null;
}

// ==================== WebSocket for Streaming ====================
let ws = null;
let streamingCellId = null;
let currentNotebookId = null;  // Global notebook ID for use in cancelAllExecution, etc.
let kernelStateSyncIntervalId = null;
let wsHeartbeatIntervalId = null;
let wsHeartbeatDeadlineId = null;
let wsReconnectTimerId = null;
const dialengStatusHistory = window.DialengNotifications.history;
let wsRecoveryPromise = null;
let wsLastHeartbeatAck = 0;
const wsRevisionTracker = new window.DialengRevisionTracker([
    'notebook', 'kernel', 'outline', 'llm', 'output', 'persistence',
]);
const wsRevisions = wsRevisionTracker.revisions;
const documentRevisionTracker = new window.DialengDocumentRevisionTracker();
const documentRevisions = documentRevisionTracker.revisions;

let _wsReconnectDelay = 1000;

function debugWebSocket(...args) {
    if (window.DIALENG_DEBUG_WEBSOCKET === true) console.debug(...args);
}

function getKernelState() {
    return window.KERNEL_STATE || null;
}

function getCurrentNotebookRevision() {
    const documentRevision = Number(getKernelState()?.notebook?.revision);
    if (Number.isFinite(documentRevision)) return documentRevision;
    return documentRevisions.notebook;
}

function getCurrentStructureRevision() {
    const documentRevision = Number(getKernelState()?.notebook?.structure_revision);
    if (Number.isFinite(documentRevision)) return documentRevision;
    return documentRevisions.structure;
}

function recordNotebookMutationRevisions(data) {
    if (!data) return;
    documentRevisionTracker.recordEvent(data);
    const notebook = getKernelState()?.notebook;
    if (notebook) {
        notebook.revision = documentRevisions.notebook;
        notebook.structure_revision = documentRevisions.structure;
    }
}

function isWebSocketHealthy() {
    return !!ws && ws.readyState === WebSocket.OPEN;
}

function recordSnapshotRevisions(snapshot, streams = null) {
    wsRevisionTracker.recordSnapshot(snapshot, streams);
}

function acceptRevisionedEvent(data) {
    const result = wsRevisionTracker.accept(data);
    if (result.gap) {
        recoverNotebookState(
            `revision gap in ${result.gap.stream}: ${result.gap.from} -> ${result.gap.to}`,
        );
        return false;
    }
    if (result.accepted) recordNotebookMutationRevisions(data);
    if (result.accepted && data?.stream === 'notebook') setNotebookSaveState('modified');
    return result.accepted;
}

async function recoverNotebookState(reason = 'reconnect') {
    if (!window.NOTEBOOK_ID || wsRecoveryPromise) return wsRecoveryPromise;
    wsRecoveryPromise = (async () => {
        console.warn('[WS] Recovering authoritative notebook state:', reason);
        const focusedBefore = focusedCellId;
        const scrollBefore = window.scrollY;
        const recoveryUrl = new URL(`/dialeng/${window.NOTEBOOK_ID}/recovery`, window.location.origin);
        recoveryUrl.searchParams.set('reason', reason);
        const response = await fetch(recoveryUrl, {cache: 'no-store'});
        if (response.status === 410) {
            handleNotebookDeleted(await response.json());
            return;
        }
        if (!response.ok) throw new Error(`Recovery failed with HTTP ${response.status}`);
        const recovery = await response.json();
        const currentCells = document.getElementById('cells');
        if (currentCells && recovery.cells_html) {
            Object.keys(monacoModels).forEach(cellId => destroyMonacoEditor(cellId));
            const template = document.createElement('template');
            template.innerHTML = recovery.cells_html.trim();
            const nextCells = template.content.firstElementChild;
            if (nextCells) {
                currentCells.replaceWith(nextCells);
                nextCells.querySelectorAll('.cell').forEach(cell => {
                    initCell(cell.id.replace('cell-', ''));
                });
            }
        }
        applyKernelSnapshot(recovery.snapshot, 'recovery');
        requestAnimationFrame(() => {
            if (focusedBefore && document.getElementById(`cell-${focusedBefore}`)) {
                setFocusedCell(focusedBefore);
            }
            window.scrollTo({top: scrollBefore, behavior: 'auto'});
        });
    })().catch(error => {
        console.error('[WS] Notebook recovery failed:', error);
        showKernelStatusMessage('Could not fully resynchronize this notebook. Reconnecting…', 'error', 7000);
        if (ws && ws.readyState === WebSocket.OPEN) ws.close(1012, 'Recovery failed');
    }).finally(() => {
        wsRecoveryPromise = null;
    });
    return wsRecoveryPromise;
}

function stopWebSocketHeartbeat() {
    if (wsHeartbeatIntervalId) clearInterval(wsHeartbeatIntervalId);
    if (wsHeartbeatDeadlineId) clearTimeout(wsHeartbeatDeadlineId);
    wsHeartbeatIntervalId = null;
    wsHeartbeatDeadlineId = null;
}

function sendWebSocketHeartbeat() {
    if (!isWebSocketHealthy()) return;
    sendWebSocketControl({type: 'heartbeat', timestamp: Date.now()});
    if (wsHeartbeatDeadlineId) clearTimeout(wsHeartbeatDeadlineId);
    wsHeartbeatDeadlineId = setTimeout(() => {
        if (Date.now() - wsLastHeartbeatAck > 45000 && isWebSocketHealthy()) {
            ws.close(1012, 'Heartbeat timeout');
        }
    }, 45000);
}

function sendWebSocketControl(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    // FastHTML binds handler arguments from the outer JSON object. Keep the
    // Dialeng protocol payload intact inside the handler's `msg` field.
    ws.send(JSON.stringify({msg: JSON.stringify(payload)}));
    return true;
}

function startWebSocketHeartbeat() {
    stopWebSocketHeartbeat();
    wsLastHeartbeatAck = Date.now();
    wsHeartbeatIntervalId = setInterval(sendWebSocketHeartbeat, 20000);
}

function canRunCurrentKernel() {
    const state = getKernelState();
    if (!state) return !!window.KERNEL_ALIVE;
    return !!state.kernel?.can_run;
}

function clearKernelStatusMessage() {
    window.DialengNotifications.clear();
}

function getStatusPersistMs(statusNode) {
    return window.DialengNotifications.persistMs(statusNode);
}

function scheduleKernelStatusDismiss(message, persistMs) {
    window.DialengNotifications.schedule(message, persistMs);
}

function armStatusToastDismiss(target) {
    window.DialengNotifications.arm(target);
}

function showKernelStatusMessage(message, tone = 'warning', persistMs = 5000) {
    window.DialengNotifications.show(message, tone, persistMs);
}

// ==================== Extension Manager ====================
const extensionClientState = {
    snapshot: null,
    loadedAssets: new Map(),
    studioOpen: false,
    cleanupHooks: new Map(),
    toolRenderers: new Map(),
    settingsSignature: '',
    surfaceGeneration: 0,
};

function setExtensionManagerStatus(message = '', tone = '') {
    const status = document.getElementById('extension-manager-status');
    if (!status) return;
    status.textContent = message;
    status.className = `extension-manager-status${tone ? ` ${tone}` : ''}`;
}

function extensionButton(label, action, extensionId, {primary = false} = {}) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `btn btn-sm${primary ? ' btn-primary' : ''}`;
    button.textContent = label;
    button.addEventListener('click', () => mutateExtension(extensionId, action));
    return button;
}

function extensionTrustButton(label, extensionId, mode, {primary = false} = {}) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `btn btn-sm${primary ? ' btn-primary' : ''}`;
    button.textContent = label;
    button.addEventListener('click', () => mutateExtension(extensionId, 'trust', mode));
    return button;
}

function extensionStudioButton(label, action, extensionId) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-sm';
    button.textContent = label;
    button.addEventListener('click', () => runExtensionStudioAction(extensionId, action));
    return button;
}

function renderExtensionManager(snapshot) {
    const previousExtensions = new Map(
        (extensionClientState.snapshot?.extensions || [])
            .filter(item => item.state === 'active')
            .map(item => [item.id, item]),
    );
    const nextExtensions = new Map(
        (snapshot.extensions || [])
            .filter(item => item.state === 'active')
            .map(item => [item.id, item]),
    );
    previousExtensions.forEach((previous, extensionId) => {
        const next = nextExtensions.get(extensionId);
        if (!next || next.source_hash !== previous.source_hash) {
            cleanupExtensionClient(extensionId);
        }
    });
    extensionClientState.snapshot = snapshot;
    const list = document.getElementById('extension-manager-list');
    const diagnostics = document.getElementById('extension-manager-diagnostics');
    if (!list) return;
    list.replaceChildren();
    const extensions = snapshot.extensions || [];
    if (!extensions.length) {
        const empty = document.createElement('p');
        empty.textContent = 'No project or user extensions were discovered.';
        list.appendChild(empty);
    }
    extensions.forEach(extension => {
        const card = document.createElement('section');
        card.className = `extension-manager-card ${extension.state || 'disabled'}`;
        card.dataset.extensionId = extension.id;

        const header = document.createElement('div');
        header.className = 'extension-manager-card-header';
        const heading = document.createElement('div');
        const title = document.createElement('h3');
        title.textContent = extension.name || extension.id;
        const meta = document.createElement('div');
        meta.className = 'extension-manager-meta';
        meta.textContent = `${extension.id} · ${extension.version} · ${extension.origin} · ${extension.state}`;
        heading.append(title, meta);
        const actions = document.createElement('div');
        actions.className = 'extension-manager-actions';
        if (extension.managed === false) {
            const managed = document.createElement('span');
            managed.className = 'extension-manager-meta';
            managed.textContent = extension.legacy ? 'Migration available' : 'Managed by Dialeng';
            actions.append(managed);
        } else if (extension.state === 'blocked') {
            actions.append(extensionTrustButton('Review & trust', extension.id, 'version', {primary: true}));
            actions.append(extensionTrustButton('Once', extension.id, 'once'));
            actions.append(extensionTrustButton('Trust project', extension.id, 'project'));
        } else if (extension.state === 'active') {
            if (extension.update_available && !extension.candidate_trusted) {
                actions.append(extensionTrustButton('Review update', extension.id, 'version', {primary: true}));
            } else {
                actions.append(extensionButton(extension.update_available ? 'Apply update' : 'Reload', 'reload', extension.id));
            }
            actions.append(extensionButton('Disable', 'disable', extension.id));
        } else {
            actions.append(extensionButton('Enable', 'enable', extension.id, {primary: true}));
        }
        if (extension.origin === 'project') {
            actions.append(extensionStudioButton('Build', 'build', extension.id));
            actions.append(extensionStudioButton('Validate', 'prepare', extension.id));
            const open = document.createElement('button');
            open.type = 'button';
            open.className = 'btn btn-sm';
            open.textContent = 'Open source';
            open.addEventListener('click', () => {
                if (extension.project_path) {
                    if (extension.has_extension_notebook) {
                        const path = `${extension.project_path}/extension`;
                        window.open(`/dialeng/?name=${encodeURIComponent(path)}`, '_blank', 'noopener');
                    } else {
                        const path = `${extension.project_path}/${extension.entrypoint_file || 'extension.py'}`;
                        window.open(`/dialeng/file?path=${encodeURIComponent(path)}`, '_blank', 'noopener');
                    }
                }
            });
            actions.appendChild(open);
        }
        (snapshot.resources?.commands || [])
            .filter(command => command.name.startsWith(`${extension.id}:`))
            .forEach(command => {
                const run = document.createElement('button');
                run.type = 'button';
                run.className = 'btn btn-sm';
                run.textContent = `Run: ${command.title}`;
                run.title = command.description || command.title;
                run.addEventListener('click', () => invokeExtensionCommand(command.name).then(result => {
                    const message = result.message || result.result;
                    if (message) showKernelStatusMessage(String(message), 'success', 4000);
                }).catch(error => setExtensionManagerStatus(error.message, 'error')));
                actions.appendChild(run);
            });
        header.append(heading, actions);

        const description = document.createElement('p');
        description.textContent = extension.update_available
            ? `${extension.description || ''} Source changed; the active version remains in use until the candidate is reviewed.`
            : extension.description || '';
        const badges = document.createElement('div');
        badges.className = 'extension-manager-badges';
        const capabilities = Object.entries(extension.contributes || {})
            .filter(([, enabled]) => enabled).map(([name]) => name);
        const contributions = (extension.contributions || []).map(item => item.kind.split('.').pop());
        [...new Set([...capabilities, ...contributions])].sort().forEach(name => {
            const badge = document.createElement('span');
            badge.className = 'extension-manager-badge';
            badge.textContent = name;
            badges.appendChild(badge);
        });
        card.append(header, description, badges);
        Object.keys(extension.bundled_resources || {}).sort().forEach(kind => {
            const enabled = (extension.enabled_resources || []).includes(kind);
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-sm extension-resource-button';
            button.textContent = `${enabled ? 'Disable' : 'Enable'} ${kind}`;
            button.addEventListener('click', async () => {
                try {
                    const response = await fetch(
                        `/dialeng/extensions/${encodeURIComponent(extension.id)}/resources/${encodeURIComponent(kind)}/${enabled ? 'disable' : 'enable'}`,
                        {method: 'POST', headers: {'Accept': 'application/json'}},
                    );
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.error || 'Resource update failed');
                    renderExtensionManager(payload);
                } catch (error) {
                    setExtensionManagerStatus(error.message, 'error');
                }
            });
            actions.appendChild(button);
        });
        if (extension.error) {
            const error = document.createElement('div');
            error.className = 'extension-manager-error';
            error.textContent = extension.error;
            card.appendChild(error);
        }
        list.appendChild(card);
    });
    if (diagnostics) {
        diagnostics.replaceChildren();
        (snapshot.diagnostics || []).forEach(item => {
            const error = document.createElement('div');
            error.className = 'extension-manager-error';
            error.textContent = `${item.path}: ${item.error}`;
            diagnostics.appendChild(error);
        });
    }
    setExtensionManagerStatus(
        `${extensions.filter(item => item.state === 'active').length} active · ${extensions.length} discovered`
    );
    applyExtensionAssets(extensions);
    refreshExtensionCellTypeChoices(snapshot);
    renderExtensionSurfaces(snapshot);
    refreshExtensionSettingsSurface(snapshot);
}

async function refreshExtensionSettingsSurface(snapshot) {
    const signature = JSON.stringify(
        (snapshot.extensions || []).flatMap(extension =>
            (extension.contributions || [])
                .filter(item => item.kind === 'settings.definition' || item.kind === 'registry.settings_section')
                .map(item => `${extension.id}:${item.kind}:${item.name}`)
        ).sort()
    );
    if (signature === extensionClientState.settingsSignature) return;
    extensionClientState.settingsSignature = signature;
    const current = document.getElementById('settings-sidebar');
    if (!current) return;
    if (settingsState.dirty.size) {
        setSettingsStatus('Extensions changed settings while local edits are unsaved. Save or revert before refreshing.', 'warning');
        return;
    }
    const wasOpen = current.classList.contains('open');
    try {
        const response = await fetch('/settings/view', {headers: {'Accept': 'text/html'}});
        if (!response.ok) throw new Error(`Settings view failed (${response.status})`);
        const documentFragment = new DOMParser().parseFromString(await response.text(), 'text/html');
        const replacement = documentFragment.getElementById('settings-sidebar');
        if (!replacement) throw new Error('Settings view did not include the sidebar');
        current.replaceWith(replacement);
        htmx.process(replacement);
        settingsState.initialized = false;
        settingsState.baseline.clear();
        settingsState.dirty.clear();
        initializeSettingsForm();
        if (wasOpen) {
            replacement.classList.add('open');
            await refreshSettings({preserveDirty: false});
        }
    } catch (error) {
        console.warn('[Extensions] Could not refresh settings surface', error);
    }
}

function toggleExtensionStudio(force) {
    const form = document.getElementById('extension-studio-form');
    if (!form) return;
    extensionClientState.studioOpen = force ?? !extensionClientState.studioOpen;
    form.hidden = !extensionClientState.studioOpen;
    if (extensionClientState.studioOpen) {
        requestAnimationFrame(() => document.getElementById('extension-studio-name')?.focus());
    }
}

function setExtensionStudioStatus(message = '', tone = '') {
    const status = document.getElementById('extension-studio-status');
    if (!status) return;
    status.textContent = message;
    status.className = `extension-manager-status${tone ? ` ${tone}` : ''}`;
}

function showExtensionStudioOutput(payload) {
    const output = document.getElementById('extension-studio-output');
    if (output) output.textContent = JSON.stringify(payload, null, 2);
}

async function createExtensionFromStudio() {
    const name = document.getElementById('extension-studio-name')?.value.trim() || '';
    const extensionId = document.getElementById('extension-studio-id')?.value.trim() || '';
    const template = document.getElementById('extension-studio-template')?.value || 'basic';
    if (!name) {
        setExtensionStudioStatus('Provide a lowercase extension name.', 'error');
        return;
    }
    setExtensionStudioStatus(`Creating ${name} without activating it...`, 'pending');
    try {
        const response = await fetch('/dialeng/extensions/studio/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: JSON.stringify({name, extension_id: extensionId, template}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Extension creation failed (${response.status})`);
        renderExtensionManager(payload);
        showExtensionStudioOutput(payload.build);
        setExtensionStudioStatus('Created and statically validated. Review the notebook, then build and validate.', 'success');
        const notebookName = payload.build?.notebook_name;
        if (notebookName && window.confirm('Open the generated extension notebook in a new tab?')) {
            window.open(`/dialeng/?name=${encodeURIComponent(notebookName)}`, '_blank', 'noopener');
        }
    } catch (error) {
        setExtensionStudioStatus(error.message || 'Could not create the extension.', 'error');
    }
}

async function runExtensionStudioAction(extensionId, action) {
    setExtensionStudioStatus(`${action === 'prepare' ? 'Validating' : 'Building'} ${extensionId}...`, 'pending');
    toggleExtensionStudio(true);
    try {
        const response = await fetch(
            `/dialeng/extensions/studio/${encodeURIComponent(extensionId)}/${action}`,
            {method: 'POST', headers: {'Accept': 'application/json'}},
        );
        const payload = await response.json();
        if (!response.ok || payload.ok === false) {
            throw new Error(payload.error || `Extension ${action} failed (${response.status})`);
        }
        renderExtensionManager(payload);
        showExtensionStudioOutput(payload.result || payload.build || payload);
        setExtensionStudioStatus(
            action === 'prepare'
                ? 'Isolated validation and extension tests passed. Trust and activation still require confirmation.'
                : 'Notebook exported and static validation passed.',
            'success',
        );
    } catch (error) {
        setExtensionStudioStatus(error.message || `Could not ${action} extension.`, 'error');
    }
}

async function refreshExtensionManager() {
    setExtensionManagerStatus('Refreshing extension inventory...');
    try {
        const response = await fetch('/dialeng/extensions', {headers: {'Accept': 'application/json'}});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Extension request failed (${response.status})`);
        renderExtensionManager(payload);
    } catch (error) {
        setExtensionManagerStatus(error.message || 'Could not load extensions.', 'error');
    }
}

async function copyExtensionDiagnostics() {
    const snapshot = extensionClientState.snapshot || {};
    try {
        await navigator.clipboard.writeText(JSON.stringify(snapshot, null, 2));
        setExtensionManagerStatus('Redacted extension diagnostics copied.', 'success');
    } catch (error) {
        setExtensionManagerStatus(`Could not copy diagnostics: ${error.message}`, 'error');
    }
}

async function mutateExtension(extensionId, action, trustMode = 'version') {
    const previousExtension = (extensionClientState.snapshot?.extensions || []).find(item => item.id === extensionId);
    if (action === 'trust') {
        const extension = (extensionClientState.snapshot?.extensions || []).find(item => item.id === extensionId);
        const permissions = JSON.stringify(extension?.permissions || {}, null, 2);
        if (!window.confirm(
            `Trust ${extension?.name || extensionId} (${trustMode})?\n\nExtensions run Python with the Dialeng server's full process permissions.\n\nDeclared permissions:\n${permissions}`
        )) return;
    }
    setExtensionManagerStatus(`${action[0].toUpperCase()}${action.slice(1)} ${extensionId}...`);
    try {
        const response = await fetch(`/dialeng/extensions/${encodeURIComponent(extensionId)}/${action}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: action === 'trust' ? JSON.stringify({mode: trustMode}) : '{}',
        });
        const payload = await response.json();
        if (!response.ok || payload.ok === false) {
            throw new Error(payload.error || `Extension ${action} failed (${response.status})`);
        }
        renderExtensionManager(payload);
        if (action === 'trust') {
            await mutateExtension(extensionId, 'enable');
            return;
        }
        const changed = (payload.extensions || []).find(item => item.id === extensionId);
        const contributionSet = [
            ...(previousExtension?.contributions || []),
            ...(changed?.contributions || []),
        ];
        const customCellTypes = contributionSet
            .filter(item => item.kind === 'registry.cell_type')
            .map(item => item.name);
        const customCellIsRendered = customCellTypes.some(cellType =>
            document.querySelector(`.cell[data-type="${CSS.escape(cellType)}"], .cell[data-cell-type="${CSS.escape(cellType)}"]`)
        );
        const requiresChromeReload = contributionSet.some(item => [
            'registry.kernel', 'registry.provider',
        ].includes(item.kind));
        const updatesActiveCustomCell = customCellIsRendered
            && (action !== 'enable' || previousExtension?.state === 'active');
        if (requiresChromeReload || updatesActiveCustomCell) {
            showKernelStatusMessage('Extension changed server-rendered UI. Reloading the page...', 'info', 1500);
            setTimeout(() => window.location.reload(), 250);
        }
    } catch (error) {
        setExtensionManagerStatus(error.message || `Could not ${action} extension.`, 'error');
    }
}

function refreshExtensionCellTypeChoices(snapshot) {
    const customTypes = new Set(
        (snapshot.extensions || [])
            .filter(extension => extension.state === 'active')
            .flatMap(extension => extension.contributions || [])
            .filter(item => item.kind === 'registry.cell_type')
            .map(item => item.name)
    );
    document.querySelectorAll('.type-select').forEach(select => {
        Array.from(select.options).forEach(option => {
            if (!['code', 'note', 'prompt', 'shell'].includes(option.value)) option.remove();
        });
        customTypes.forEach(cellType => {
            if (Array.from(select.options).some(option => option.value === cellType)) return;
            const option = document.createElement('option');
            option.value = cellType;
            option.textContent = cellType.replaceAll('-', ' ');
            option.dataset.extensionCellType = 'true';
            select.appendChild(option);
        });
    });
}

function openExtensionManager() {
    document.getElementById('extension-manager-overlay')?.classList.add('visible');
    document.body.classList.add('modal-open');
    refreshExtensionManager();
}

function closeExtensionManager() {
    document.getElementById('extension-manager-overlay')?.classList.remove('visible');
    document.body.classList.remove('modal-open');
}

function cleanupExtensionClient(extensionId) {
    const hooks = extensionClientState.cleanupHooks.get(extensionId) || [];
    hooks.splice(0).forEach(cleanup => {
        try { cleanup(); } catch (error) { console.warn('[Extension cleanup]', extensionId, error); }
    });
    extensionClientState.cleanupHooks.delete(extensionId);
    for (const [toolName, registration] of extensionClientState.toolRenderers) {
        if (registration.extensionId === extensionId) {
            extensionClientState.toolRenderers.delete(toolName);
        }
    }
}

function applyExtensionAssets(extensions = []) {
    const desired = new Map();
    extensions.filter(item => item.state === 'active').forEach(extension => {
        Object.entries(extension.assets || {}).forEach(([kind, paths]) => {
            (paths || []).forEach(path => {
                const key = `${extension.id}:${kind}:${path}:${extension.source_hash}`;
                desired.set(key, {extension, kind, path});
            });
        });
    });
    extensionClientState.loadedAssets.forEach((element, key) => {
        if (!desired.has(key)) {
            element.remove();
            extensionClientState.loadedAssets.delete(key);
        }
    });
    desired.forEach(({extension, kind, path}, key) => {
        if (extensionClientState.loadedAssets.has(key)) return;
        const url = `/dialeng/extensions/${encodeURIComponent(extension.id)}/asset?path=${encodeURIComponent(path)}&v=${encodeURIComponent(extension.source_hash || '')}`;
        const element = kind === 'css' ? document.createElement('link') : document.createElement('script');
        if (kind === 'css') {
            element.rel = 'stylesheet';
            element.href = url;
        } else {
            element.type = 'module';
            element.src = url;
        }
        element.dataset.dialengExtensionAsset = key;
        document.head.appendChild(element);
        extensionClientState.loadedAssets.set(key, element);
    });
}

function extensionNameParts(qualifiedName) {
    const separator = String(qualifiedName || '').indexOf(':');
    if (separator < 1) return ['', ''];
    return [qualifiedName.slice(0, separator), qualifiedName.slice(separator + 1)];
}

async function loadExtensionSurface(kind, registration) {
    const [extensionId, localName] = extensionNameParts(registration.name);
    if (!extensionId || !localName) return null;
    const notebookId = window.NOTEBOOK_ID || '_workspace';
    const response = await fetch(
        `/dialeng/${encodeURIComponent(notebookId)}/extensions/surfaces/${kind}/${encodeURIComponent(extensionId)}/${encodeURIComponent(localName)}`,
        {headers: {'Accept': 'text/html'}},
    );
    if (!response.ok) return null;
    const wrapper = document.createElement('section');
    wrapper.className = `extension-surface extension-${kind} ${registration.location || ''}`;
    wrapper.dataset.extensionSurface = registration.name;
    wrapper.innerHTML = await response.text();
    htmx.process(wrapper);
    return wrapper;
}

async function renderExtensionSurfaces(snapshot) {
    const generation = ++extensionClientState.surfaceGeneration;
    const resources = snapshot?.resources || {};
    const toolbar = document.getElementById('extension-toolbar-items');
    if (toolbar) {
        toolbar.replaceChildren();
        (resources.toolbar_items || []).forEach(item => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-sm extension-toolbar-button';
            button.textContent = item.title;
            button.title = item.title;
            button.dataset.extensionToolbar = item.name;
            button.addEventListener('click', () => invokeExtensionCommand(item.command).then(result => {
                const message = result.message || result.result;
                if (message) showKernelStatusMessage(String(message), 'success', 4000);
            }).catch(error => showKernelStatusMessage(error.message, 'error', 7000)));
            toolbar.appendChild(button);
        });
    }

    const statusHost = document.getElementById('extension-status-items');
    if (statusHost) {
        statusHost.replaceChildren();
        const statusNodes = await Promise.all((resources.status_items || []).map(item => loadExtensionSurface('status', item)));
        if (generation === extensionClientState.surfaceGeneration) {
            statusNodes.filter(Boolean).forEach(node => statusHost.appendChild(node));
        }
    }

    const panelHost = document.getElementById('extension-panel-host');
    if (panelHost) {
        panelHost.replaceChildren();
        const panelNodes = await Promise.all((resources.panels || []).map(item => loadExtensionSurface('panel', item)));
        if (generation === extensionClientState.surfaceGeneration) {
            panelNodes.filter(Boolean).forEach(node => panelHost.appendChild(node));
        }
    }
}

function invokeExtensionCommand(qualifiedName, args = {}) {
    const separator = qualifiedName.indexOf(':');
    if (separator < 1) throw new Error(`Invalid extension command: ${qualifiedName}`);
    const extensionId = qualifiedName.slice(0, separator);
    const commandName = qualifiedName.slice(separator + 1);
    const notebookId = window.NOTEBOOK_ID || '_workspace';
    const operationId = `ext-command-${crypto.randomUUID().replaceAll('-', '').slice(0, 16)}`;
    const invocation = fetch(
        `/dialeng/${encodeURIComponent(notebookId)}/extensions/commands/${encodeURIComponent(extensionId)}/${encodeURIComponent(commandName)}`,
        {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-Dialeng-Operation-Id': operationId},
            body: JSON.stringify(args),
        },
    ).then(async response => {
        const payload = await response.json();
        if (!response.ok || payload.ok === false) {
            throw new Error(payload.error || `Command failed (${response.status})`);
        }
        return payload;
    });
    invocation.operationId = operationId;
    invocation.cancel = () => cancelExtensionCommand(operationId);
    return invocation;
}

async function cancelExtensionCommand(operationId) {
    const response = await fetch(
        `/dialeng/extensions/commands/${encodeURIComponent(operationId)}/cancel`,
        {method: 'POST', headers: {'Accept': 'application/json'}},
    );
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || `Command cancellation failed (${response.status})`);
    }
    return payload;
}

async function invokeExtensionAction(qualifiedName, args = {}) {
    const [extensionId, actionName] = extensionNameParts(qualifiedName);
    if (!extensionId || !actionName) throw new Error(`Invalid extension action: ${qualifiedName}`);
    const notebookId = window.NOTEBOOK_ID || '_workspace';
    const response = await fetch(
        `/dialeng/${encodeURIComponent(notebookId)}/extensions/${encodeURIComponent(extensionId)}/actions/${encodeURIComponent(actionName)}`,
        {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(args)},
    );
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || `Extension action failed (${response.status})`);
    }
    return payload;
}

window.DialengExtensions = {
    openManager: openExtensionManager,
    refresh: refreshExtensionManager,
    invokeCommand: invokeExtensionCommand,
    cancelCommand: cancelExtensionCommand,
    invokeAction: invokeExtensionAction,
    snapshot: () => extensionClientState.snapshot,
    create: createExtensionFromStudio,
    prepare: extensionId => runExtensionStudioAction(extensionId, 'prepare'),
    registerCleanup(extensionId, cleanup) {
        if (typeof cleanup !== 'function') throw new TypeError('Extension cleanup must be a function');
        const hooks = extensionClientState.cleanupHooks.get(extensionId) || [];
        hooks.push(cleanup);
        extensionClientState.cleanupHooks.set(extensionId, hooks);
    },
    subscribe(extensionId, eventType, handler) {
        if (typeof handler !== 'function') throw new TypeError('Extension event handler must be a function');
        const eventName = `dialeng:${eventType}`;
        const listener = event => handler(event.detail);
        window.addEventListener(eventName, listener);
        this.registerCleanup(extensionId, () => window.removeEventListener(eventName, listener));
        return () => window.removeEventListener(eventName, listener);
    },
    notify(message, tone = 'info', duration = 4000) {
        showKernelStatusMessage(String(message), tone, duration);
    },
    progress(message, {tone = 'info', duration = 2500} = {}) {
        showKernelStatusMessage(String(message), tone, duration);
    },
    async openSurface(qualifiedName, kind = 'panel') {
        const resources = extensionClientState.snapshot?.resources || {};
        const collection = kind === 'status' ? resources.status_items : resources.panels;
        const registration = (collection || []).find(item => item.name === qualifiedName);
        if (!registration) throw new Error(`Unknown extension ${kind}: ${qualifiedName}`);
        const surface = await loadExtensionSurface(kind, registration);
        const host = kind === 'status'
            ? document.getElementById('extension-status-items')
            : document.getElementById('extension-panel-host');
        if (surface && host) host.replaceChildren(surface);
        return surface;
    },
    registerToolRenderer(extensionId, toolName, renderer) {
        if (!extensionId || !toolName || !renderer || typeof renderer !== 'object') {
            throw new TypeError('Tool renderer requires extensionId, toolName, and renderer');
        }
        if (typeof renderer.renderCall !== 'function' && typeof renderer.renderResult !== 'function') {
            throw new TypeError('Tool renderer must define renderCall or renderResult');
        }
        const registration = {extensionId, ...renderer};
        extensionClientState.toolRenderers.set(toolName, registration);
        this.registerCleanup(extensionId, () => {
            if (extensionClientState.toolRenderers.get(toolName) === registration) {
                extensionClientState.toolRenderers.delete(toolName);
            }
        });
        return () => extensionClientState.toolRenderers.delete(toolName);
    },
    publicState() {
        return {
            notebookId: window.NOTEBOOK_ID || '',
            kernel: window.KERNEL_STATE?.kernel || {},
            extensionsRevision: extensionClientState.snapshot?.revision || 0,
        };
    },
};
window.dialeng = window.dialeng || {};
window.dialeng.extensions = window.DialengExtensions;

function extensionShortcutMatches(event, shortcut) {
    const parts = String(shortcut || '').toLowerCase().split('+').map(part => part.trim());
    const key = parts.pop();
    return Boolean(key)
        && event.key.toLowerCase() === key.toLowerCase()
        && event.ctrlKey === parts.includes('ctrl')
        && event.metaKey === parts.includes('meta')
        && event.altKey === parts.includes('alt')
        && event.shiftKey === parts.includes('shift');
}

document.addEventListener('keydown', event => {
    const shortcuts = extensionClientState.snapshot?.resources?.shortcuts || [];
    const match = shortcuts.find(item => extensionShortcutMatches(event, item.shortcut));
    if (!match) return;
    event.preventDefault();
    invokeExtensionCommand(match.command).catch(error => {
        showKernelStatusMessage(error.message || 'Extension command failed.', 'error', 7000);
    });
});

function showLlmFailureAction(data) {
    showKernelStatusMessage(data.message || 'The prompt failed.', data.tone || 'error', data.persist_ms || 10000);
    if (!data.action || !data.action_label) return;
    const toast = document.querySelector('#status .status');
    if (!toast) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'status-action';
    button.textContent = data.action_label;
    button.addEventListener('click', () => {
        if (data.action === 'connect') {
            if (!document.getElementById('settings-sidebar')?.classList.contains('open')) toggleSettings();
            startOpenAICodexLogin();
        } else if (data.action === 'choose_model') {
            document.getElementById('model-select')?.focus();
        } else if (data.action === 'retry' && data.cell_id) {
            const cell = document.getElementById(`cell-${data.cell_id}`);
            if (cell) clickCellRunButton(cell, {advance: false});
        }
    });
    toast.appendChild(button);
    window.DialengNotifications.schedule(toast.textContent.trim(), data.persist_ms || 10000);
}

function renderPromptExecution(cellId, execution = {}, notebookUsage = null) {
    let panel = document.getElementById(`prompt-diagnostics-${cellId}`);
    if (panel) {
        if (panel.tagName !== 'DETAILS') {
            const details = document.createElement('details');
            details.id = panel.id;
            details.className = panel.className;
            panel.replaceWith(details);
            panel = details;
        }
        const usage = execution.usage || {};
        const total = Number(usage.total_tokens || 0);
        const duration = Number(execution.duration_ms);
        const summary = [
            execution.model,
            Number.isFinite(duration) ? `${(duration / 1000).toFixed(1)}s` : '',
            total ? `${total.toLocaleString()} tokens` : '',
            execution.status,
        ].filter(Boolean).join(' · ') || 'Execution details';
        const rows = [
            ['Operation', execution.operation_id],
            ['Provider', execution.provider],
            ['Transport', execution.transport],
            ['First token', execution.ttft_ms == null ? null : `${Number(execution.ttft_ms).toFixed(1)} ms`],
            ['Retries', execution.retry_count],
            ['Input / cached / output', `${usage.input_tokens || 0} / ${usage.cached_tokens || 0} / ${usage.output_tokens || 0}`],
            ['Tool steps', execution.tool_steps],
        ].filter(([, value]) => value !== null && value !== undefined);
        const previousOperationId = panel.dataset.operationId || '';
        const operationId = String(execution.operation_id || '');
        panel.hidden = false;
        panel.innerHTML = `<summary>${escapeHtml(summary)}</summary><div class="prompt-diagnostic-grid">${rows.map(([label, value]) => `<div class="prompt-diagnostic-row"><span>${escapeHtml(label)}</span><code>${escapeHtml(value)}</code></div>`).join('')}</div>`;
        panel.dataset.operationId = operationId;
        if (!previousOperationId || previousOperationId !== operationId) panel.open = false;
    }
    if (notebookUsage) {
        const total = Number(notebookUsage.total_tokens || 0);
        const label = document.getElementById('notebook-llm-usage');
        if (label) label.textContent = `${total.toLocaleString()} tok`;
    }
}

function setNotebookSaveState(state) {
    const stateEl = document.getElementById('notebook-save-state');
    if (!stateEl) return;
    const labels = {
        modified: 'Modified',
        saving: 'Saving…',
        saved: 'Saved',
        failed: 'Save failed',
    };
    stateEl.textContent = labels[state] || state;
    stateEl.className = `notebook-save-state ${state}`;
}

function showExternalSaveConflict(conflict) {
    const statusEl = document.getElementById('status');
    if (!statusEl) return;
    const notice = document.createElement('div');
    notice.className = 'status error recovery-notice';
    const message = document.createElement('span');
    message.textContent = conflict.message || 'This notebook changed on disk.';
    const reload = document.createElement('button');
    reload.className = 'btn btn-sm';
    reload.textContent = 'Reload disk version';
    reload.onclick = () => window.location.reload();
    const overwrite = document.createElement('button');
    overwrite.className = 'btn btn-sm btn-save';
    overwrite.textContent = 'Overwrite disk file';
    overwrite.onclick = () => htmx.ajax('POST', conflict.overwrite_url, {
        target: '#status',
        swap: 'innerHTML',
    });
    notice.append(message, reload, overwrite);
    statusEl.replaceChildren(notice);
}

async function checkNotebookRecovery() {
    if (!window.NOTEBOOK_ID) return;
    try {
        const response = await fetch(`/dialeng/${window.NOTEBOOK_ID}/recovery/status`, {cache: 'no-store'});
        if (!response.ok) return;
        const recovery = await response.json();
        if (!recovery.available) return;
        const statusEl = document.getElementById('status');
        if (!statusEl) return;
        const notice = document.createElement('div');
        notice.className = 'status warning recovery-notice';
        const message = document.createElement('span');
        message.textContent = 'A newer autosave recovery snapshot is available.';
        const inspect = document.createElement('button');
        inspect.className = 'btn btn-sm';
        inspect.textContent = 'Inspect';
        inspect.onclick = () => window.open(recovery.inspect_url, '_blank', 'noopener');
        const restore = document.createElement('button');
        restore.className = 'btn btn-sm btn-save';
        restore.textContent = 'Restore';
        restore.onclick = () => applyNotebookRecovery('restore');
        const discard = document.createElement('button');
        discard.className = 'btn btn-sm';
        discard.textContent = 'Discard';
        discard.onclick = () => applyNotebookRecovery('discard');
        notice.append(message, inspect, restore, discard);
        statusEl.replaceChildren(notice);
    } catch (error) {
        console.warn('[Persistence] Recovery check failed:', error);
    }
}

async function applyNotebookRecovery(action) {
    const body = new URLSearchParams({action});
    const response = await fetch(`/dialeng/${window.NOTEBOOK_ID}/recovery`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body,
    });
    if (!response.ok) {
        showKernelStatusMessage('Recovery action failed. The canonical notebook was not changed.', 'error', 7000);
        return;
    }
    if (action === 'restore') window.location.reload();
    else showKernelStatusMessage('Recovery snapshot discarded.', 'success', 3000);
}

async function showContextInspector(cellId) {
    let overlay = document.getElementById('context-inspector-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'context-inspector-overlay';
        overlay.className = 'context-inspector-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-label', 'Prompt context inspector');
        overlay.addEventListener('click', event => {
            if (event.target === overlay) overlay.classList.remove('visible');
        });
        document.body.appendChild(overlay);
    }
    overlay.classList.add('visible');
    overlay.innerHTML = '<section class="context-inspector"><p>Preparing context inspection…</p></section>';
    try {
        const response = await fetch(`${nbApiPath()}/cell/${cellId}/context`, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const report = await response.json();
        const panel = document.createElement('section');
        panel.className = 'context-inspector';
        const close = document.createElement('button');
        close.className = 'btn btn-sm context-inspector-close';
        close.textContent = 'Close';
        close.onclick = () => overlay.classList.remove('visible');
        const title = document.createElement('h2');
        title.textContent = 'Prompt context';
        const summary = document.createElement('p');
        summary.textContent = `${report.provider || 'Provider'} · ${report.model || 'default model'} · approximately ${report.estimated_input_tokens || 0} input tokens`;
        panel.append(close, title, summary);

        const addSection = (heading, items, formatter) => {
            const section = document.createElement('section');
            const h3 = document.createElement('h3');
            h3.textContent = `${heading} (${items.length})`;
            const list = document.createElement('ul');
            items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = formatter(item);
                list.appendChild(li);
            });
            section.append(h3, list);
            panel.appendChild(section);
        };
        addSection('Included cells', report.included_cells || [], item =>
            `#${item.index + 1} ${item.cell_type}${item.pinned ? ' · pinned' : ''} · ${item.message_count} message(s)`
        );
        addSection('Skipped cells', report.skipped_cells || [], item => `#${item.index + 1} · ${item.reason}`);
        addSection('Truncated cells', report.truncated_cells || [], item => `#${item.index + 1} · ${item.reason}`);
        addSection('Dynamic expressions', report.dynamic_expressions || [], item => `$\`${item}\` (evaluated fresh)`);
        addSection('Declared tools', report.declared_tools || [], item => item);
        const media = document.createElement('p');
        media.textContent = `Images: ${report.image_count || 0} (${report.encoded_image_chars || 0} encoded characters). CRAFT messages: ${report.craft_message_count || 0}. Built-in tools: ${report.built_in_tools_enabled ? 'enabled' : 'disabled'}.`;
        panel.appendChild(media);
        overlay.replaceChildren(panel);
    } catch (error) {
        overlay.innerHTML = `<section class="context-inspector"><h2>Context inspection failed</h2><p>${escapeHtml(error.message)}</p></section>`;
    }
}

async function showKernelDiagnostics() {
    let overlay = document.getElementById('kernel-diagnostics-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'kernel-diagnostics-overlay';
        overlay.className = 'context-inspector-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-label', 'Kernel diagnostics');
        overlay.addEventListener('click', event => {
            if (event.target === overlay) overlay.classList.remove('visible');
        });
        document.body.appendChild(overlay);
    }
    overlay.classList.add('visible');
    overlay.innerHTML = '<section class="context-inspector"><p>Loading diagnostics…</p></section>';
    try {
        const response = await fetch(`${nbApiPath()}/diagnostics`, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const diagnostics = await response.json();
        const panel = document.createElement('section');
        panel.className = 'context-inspector diagnostics-panel';
        const close = document.createElement('button');
        close.className = 'btn btn-sm context-inspector-close';
        close.textContent = 'Close';
        close.onclick = () => overlay.classList.remove('visible');
        const title = document.createElement('h2');
        title.textContent = 'Kernel diagnostics';
        const grid = document.createElement('dl');
        grid.className = 'diagnostics-grid';
        const values = {
            'Display state': diagnostics.kernel.display_state,
            'Connection': diagnostics.kernel.connection_state,
            'Kernel type': diagnostics.kernel.selected_type,
            'Runtime ID': diagnostics.kernel.runtime_id || 'n/a',
            'Kernel ID': diagnostics.kernel.kernel_id || 'n/a',
            'Last activity': diagnostics.kernel.last_activity_at || 'n/a',
            'Queue': diagnostics.queue.running_cell_id
                ? `running ${diagnostics.queue.running_cell_id}`
                : `${diagnostics.queue.queued_count || 0} queued`,
            'Setup': diagnostics.setup.is_active
                ? `${diagnostics.setup.phase || 'active'} · ${diagnostics.setup.detail || ''}`
                : 'idle',
            'Browser tabs': diagnostics.runtime.websocket_clients,
            'Notebook revision': diagnostics.revisions.notebook,
            'Kernel revision': diagnostics.revisions.kernel,
            'Autosave': diagnostics.runtime.autosave_pending ? 'pending' : 'idle',
        };
        Object.entries(values).forEach(([label, value]) => {
            const dt = document.createElement('dt');
            dt.textContent = label;
            const dd = document.createElement('dd');
            dd.textContent = String(value);
            grid.append(dt, dd);
        });
        const actions = document.createElement('div');
        actions.className = 'diagnostics-actions';
        const interrupt = document.createElement('button');
        interrupt.className = 'btn btn-sm';
        interrupt.textContent = 'Interrupt execution';
        interrupt.onclick = () => htmx.ajax('POST', `${nbApiPath()}/kernel/interrupt`, '#status');
        const restart = document.createElement('button');
        restart.className = 'btn btn-sm';
        restart.textContent = 'Restart kernel';
        restart.onclick = () => htmx.ajax('POST', `${nbApiPath()}/kernel/restart`, '#status');
        actions.append(interrupt, restart);
        const historyTitle = document.createElement('h3');
        historyTitle.textContent = 'Recent status history';
        const history = document.createElement('ol');
        history.className = 'diagnostics-history';
        dialengStatusHistory.slice(-10).reverse().forEach(item => {
            const li = document.createElement('li');
            li.textContent = `${new Date(item.timestamp).toLocaleTimeString()} · ${item.message}`;
            history.appendChild(li);
        });
        panel.append(close, title, grid, actions, historyTitle, history);
        overlay.replaceChildren(panel);
        close.focus();
    } catch (error) {
        overlay.innerHTML = `<section class="context-inspector"><h2>Diagnostics unavailable</h2><p>${escapeHtml(error.message)}</p></section>`;
    }
}

async function fetchKernelSnapshot(notebookId, source = 'poll') {
    try {
        const resp = await fetch(`/dialeng/${notebookId}/kernel/snapshot`, { cache: 'no-store' });
        if (resp.status === 410) {
            const deletedState = await resp.json();
            handleNotebookDeleted(deletedState);
            return;
        }
        if (!resp.ok) return;
        const snapshot = await resp.json();
        applyKernelSnapshot(snapshot, source);
    } catch (err) {
        console.warn('[Kernel] Failed to fetch snapshot:', err);
    }
}

function handleNotebookDeleted(data = {}) {
    if (window.NOTEBOOK_TERMINAL_STATE) return;
    window.NOTEBOOK_TERMINAL_STATE = true;
    stopWebSocketHeartbeat();

    if (kernelStateSyncIntervalId) {
        clearInterval(kernelStateSyncIntervalId);
        kernelStateSyncIntervalId = null;
    }
    if (wsReconnectTimerId) {
        clearTimeout(wsReconnectTimerId);
        wsReconnectTimerId = null;
    }
    if (ws) {
        const socket = ws;
        ws = null;
        socket.onclose = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close(1000, 'Notebook deleted');
        }
    }

    showKernelStatusMessage(
        data.message || 'Notebook deleted. Its kernel and pending work were stopped.',
        'success',
        1800,
    );
    const workspaceUrl = data.workspace_url || '/dialeng/workspace?deleted=1';
    setTimeout(() => window.location.replace(workspaceUrl), 450);
}

function startKernelStateSync(notebookId) {
    if (kernelStateSyncIntervalId) clearInterval(kernelStateSyncIntervalId);
    if (!getKernelState()) fetchKernelSnapshot(notebookId, 'startup');
    kernelStateSyncIntervalId = null;
}

function applyKernelSnapshot(snapshot, source = 'ws') {
    if (!snapshot || snapshot.type !== 'kernel_snapshot') return;
    recordSnapshotRevisions(
        snapshot,
        source === 'recovery' ? null : [snapshot.stream || 'kernel'],
    );

    const previous = getKernelState();
    window.KERNEL_STATE = snapshot;
    window.KERNEL_ALIVE = !!snapshot.kernel?.can_run;
    applyNotebookSnapshot(snapshot.notebook, previous?.notebook || null, snapshot.llm || null);
    syncCurrentExplorerKernelState(snapshot);

    handleQueueUpdate({
        running_cell_id: snapshot.queue?.running_cell_id || null,
        queued_cell_ids: snapshot.queue?.queued_cell_ids || []
    });

    const dotStateMap = {
        connected: 'connected',
        busy: 'busy',
        connecting: 'busy',
        initializing: 'busy',
        restarting: 'busy',
        degraded: 'error',
        disconnected: '',
        auth_required: '',
        error: 'error'
    };
    updateKernelDot(dotStateMap[snapshot.kernel.display_state] || '');

    const dot = document.getElementById('kernel-dot');
    if (dot) {
        const detail = snapshot.setup?.detail || snapshot.kernel?.connection_detail || '';
        const labelMap = {
            connected: 'Kernel: idle',
            busy: 'Kernel: busy',
            connecting: 'Kernel: connecting',
            initializing: 'Kernel: initializing',
            restarting: 'Kernel: restarting',
            degraded: 'Kernel: degraded',
            disconnected: 'Kernel: disconnected',
            auth_required: 'Kernel: authentication required'
        };
        dot.title = detail ? `${labelMap[snapshot.kernel.display_state] || 'Kernel'} (${detail})`
            : (labelMap[snapshot.kernel.display_state] || 'Kernel');
    }

    const prevRuntimeId = previous?.kernel?.runtime_id;
    const nextRuntimeId = snapshot.kernel?.runtime_id;
    if (prevRuntimeId && nextRuntimeId && prevRuntimeId !== nextRuntimeId) {
        showKernelStatusMessage('Attached to a new Colab runtime. Remote kernel state was reset.', 'warning', 9000);
    }

    const prevDisplayState = previous?.kernel?.display_state;
    const nextDisplayState = snapshot.kernel?.display_state;
    if (prevDisplayState !== nextDisplayState) {
        if (nextDisplayState === 'degraded') {
            showKernelStatusMessage('Colab connection is degraded. Dialeng will keep trying to recover.', 'warning', 7000);
        } else if (nextDisplayState === 'disconnected' && snapshot.kernel?.exists) {
            showKernelStatusMessage('Kernel connection was lost. Reattach or restart if it does not recover.', 'error', 9000);
        } else if (nextDisplayState === 'auth_required') {
            showKernelStatusMessage('Sign in with Google before using a Colab kernel.', 'warning', 7000);
        }
    }

    const prevOutlineRevision = previous?.outline?.revision ?? null;
    const nextOutlineRevision = snapshot.outline?.revision ?? null;
    if (
        prevOutlineRevision !== null &&
        nextOutlineRevision !== null &&
        prevOutlineRevision !== nextOutlineRevision
    ) {
        const outlineSidebar = document.getElementById('outline-sidebar');
        if (outlineSidebar && outlineSidebar.classList.contains('outline-open')) {
            refreshOutline();
        }
    }

    const prevSetupDetail = previous?.setup?.detail;
    const nextSetupDetail = snapshot.setup?.detail;
    if (snapshot.setup?.is_active && nextSetupDetail && nextSetupDetail !== prevSetupDetail) {
        showKernelStatusMessage(nextSetupDetail, 'warning', 4500);
    }

    const prevCanRun = !!previous?.kernel?.can_run;
    const nextCanRun = !!snapshot.kernel?.can_run;
    const prevAuthEnabled = !!previous?.auth?.enabled;
    const nextAuthEnabled = !!snapshot.auth?.enabled;
    const prevAuthenticated = !!previous?.auth?.authenticated;
    const nextAuthenticated = !!snapshot.auth?.authenticated;
    const prevSelectedType = previous?.kernel?.selected_type;
    const nextSelectedType = snapshot.kernel?.selected_type;
    const prevRuntime = previous?.notebook?.colab_runtime_type;
    const nextRuntime = snapshot.notebook?.colab_runtime_type;

    if (
        previous &&
        (
            prevCanRun !== nextCanRun ||
            prevAuthEnabled !== nextAuthEnabled ||
            prevAuthenticated !== nextAuthenticated ||
            prevSelectedType !== nextSelectedType ||
            prevRuntime !== nextRuntime
        )
    ) {
        refreshKernelChrome({ refreshModal: false });
    }

    if (!prevCanRun && nextCanRun) {
        if (previous?.setup?.is_active && previous?.setup?.source && ['kernel_select', 'kernel_restart'].includes(previous.setup.source)) {
            showKernelStatusMessage('Kernel is ready.', 'success', 2600);
        }
        document.body.dispatchEvent(new CustomEvent('kernel-connected'));
        _runPendingCell();
    }

    if (source === 'ws') {
        console.log('[Kernel] Applied snapshot from WebSocket:', snapshot);
    }
}

function connectWebSocket(notebookId) {
    if (window.NOTEBOOK_TERMINAL_STATE) return;
    recordSnapshotRevisions({revisions: window.INITIAL_REVISIONS || {}});
    currentNotebookId = notebookId;  // Store globally for other functions to use
    startKernelStateSync(notebookId);
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);

    ws.onopen = function() {
        console.log('[WS] Connected to notebook:', notebookId);
        _wsReconnectDelay = 1000;  // Reset backoff on successful connection
        // Send join message to register this connection with the notebook
        sendWebSocketControl({type: 'join', notebook_id: notebookId, revisions: wsRevisions});
        startWebSocketHeartbeat();
    };

    ws.onmessage = function(event) {
        const msg = event.data;

        debugWebSocket(
            '[WS] RAW message received, length:', msg?.length,
            'type:', typeof msg,
            'starts:', msg?.substring?.(0, 50),
        );

        // Check if message is HTML (OOB swap from collaborator) or JSON (streaming)
        if (msg && typeof msg === 'string' && msg.startsWith('<')) {
            // HTML with hx-swap-oob - process as OOB swap
            debugWebSocket('[WS] Received OOB HTML swap, length:', msg.length);
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
        if (!acceptRevisionedEvent(data)) return;
        window.dispatchEvent(new CustomEvent(`dialeng:${data.type}`, {detail: data}));
        debugWebSocket('[WS] Received message:', data.type, 'cell_id:', data.cell_id || 'none');
        if (window.dialengEvents?.dispatch(data)) return;

        if (data.type === 'e2e_force_disconnect') {
            ws.close(4001, 'E2E forced disconnect');
            return;
        } else if (data.type === 'heartbeat_ack') {
            wsLastHeartbeatAck = Date.now();
            if (wsHeartbeatDeadlineId) clearTimeout(wsHeartbeatDeadlineId);
            wsHeartbeatDeadlineId = null;
        } else if (data.type === 'notebook_deleted') {
            handleNotebookDeleted(data);
        } else if (data.type === 'kernel_snapshot') {
            applyKernelSnapshot(data, 'ws');
        } else if (data.type === 'llm_status') {
            const notebook = getKernelState()?.notebook || {
                llm_provider: data.selected_provider,
                model: data.selected_model,
            };
            syncLlmSelectors(data, notebook);
        } else if (data.type === 'notebook_saved') {
            setNotebookSaveState('saved');
        } else if (data.type === 'notebook_reloaded') {
            window.location.reload();
        } else if (data.type === 'file_explorer_changed') {
            const sidebar = document.getElementById('file-explorer-sidebar');
            const currentPath = document.getElementById('current-explorer-path')?.value || '';
            if (
                sidebar?.classList.contains('file-explorer-open')
                && (data.paths || []).includes(currentPath)
            ) {
                refreshFileExplorer();
            }
        } else if (data.type === 'status_message') {
            if (data.action) showLlmFailureAction(data);
            else showKernelStatusMessage(data.message || 'Dialeng status changed.', data.tone || 'warning', data.persist_ms || 7000);
        } else if (data.type === 'stream_chunk') {
            // Skip if cancelled
            if (cancelledCells.has(data.cell_id)) return;
            appendToResponse(data.cell_id, data.chunk, data.thinking);
            // Reset streaming timeout on activity
            resetStreamingTimeout();
        } else if (data.type === 'prompt_run_terminal') {
            console.log('[WS] prompt_run_terminal received for cell:', data.cell_id, data.status);
            cancelledCells.delete(data.cell_id);
            finishStreaming(data.cell_id);
            if (data.version != null) updateCellVersion(data.cell_id, data.version);
            completePromptAdvance(data.cell_id, data.status);
            document.body.dispatchEvent(new CustomEvent('prompt-run-terminal', {detail: data}));
        } else if (data.type === 'prompt_execution_update') {
            renderPromptExecution(data.cell_id, data.execution, data.notebook_usage);
        } else if (data.type === 'stream_end') {
            console.log('[WS] stream_end received for cell:', data.cell_id);
            cancelledCells.delete(data.cell_id);
            finishStreaming(data.cell_id);
        } else if (data.type === 'prompt_output_update') {
            console.log('[WS] prompt_output_update received for cell:', data.cell_id, 'version:', data.version);
            // The backend payload is authoritative. Discard any chunk queued
            // for a later animation frame or finishStreaming() can append the
            // final delta a second time after this replacement.
            promptChunkBatcher.clear(data.cell_id);
            const result = applyPromptOutputUpdate(data.cell_id, data.output);
            if (result?.safeVersion !== false) updateCellVersion(data.cell_id, data.version);
        } else if (data.type === 'prompt_stream_start') {
            // Prompt cell LLM generation started (e.g., from _add_msg_unsafe with run_mode='run').
            // Shows "Generating..." indicator so the user knows the LLM is working.
            console.log('[WS] prompt_stream_start received for cell:', data.cell_id);
            startStreaming(data.cell_id, false);
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
            debugWebSocket('[WS] code_stream_chunk received for cell:', data.cell_id, 'stream:', data.stream, 'length:', data.chunk?.length || 0);
            appendCodeOutput(data.cell_id, data.chunk, data.output_stream || data.stream);
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
            appendDisplayData(data.cell_id, data.html, data.display_id, data.output_kind);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_update_display') {
            // Update existing display data (tqdm progress bars, widgets)
            console.log('[WS] code_update_display received for cell:', data.cell_id, 'display_id:', data.display_id);
            updateDisplayData(data.cell_id, data.html, data.display_id);
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'code_clear_output') {
            // Clear cell output (used by widgets before updating)
            console.log('[WS] code_clear_output received for cell:', data.cell_id);
            clearCellOutput(data.cell_id, Boolean(data.wait));
            resetCodeStreamingTimeout(data.cell_id);
        } else if (data.type === 'queue_update') {
            // Queue state update from server
            console.log('[WS] queue_update received:', data);
            handleQueueUpdate(data);
        } else if (data.type === 'cell_state_change') {
            // Terminal state is authoritative even when an extension guard
            // blocks execution before code_stream_start/code_stream_end.
            console.log('[WS] cell_state_change received:', data.cell_id, data.state);
            if (['idle', 'success', 'error'].includes(data.state)) {
                clearCellQueueState(data.cell_id);
            } else if (['queued', 'running'].includes(data.state)) {
                updateCellQueueState(data.cell_id, data.state, null);
            }
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
        } else if (data.type === 'llm_usage') {
            console.log('[WS] llm_usage:', data.provider, data.usage);
        } else if (data.type === 'tool_confirmation_request') {
            // Server requesting confirmation for file-modifying tool
            console.log('[WS] tool_confirmation_request:', data.tool_name);
            ToolConfirmation.show(data.cell_id, data.tool_name, data.tool_input, data.confirmation_id);
        } else if (data.type === 'kernel_type_changed') {
            // Kernel type changed (local <-> colab)
            console.log('[WS] kernel_type_changed:', data.kernel_type);
            handleKernelTypeChanged(data.kernel_type);
        } else if (data.type === 'kernel_connected') {
            // Kernel setup complete (CRAFT init, restart, or first execution)
            document.body.dispatchEvent(new CustomEvent('kernel-connected'));
            // Run any cell that was pending while waiting for kernel init
            _runPendingCell();
        } else if (data.type === 'kernel_error') {
            // Kernel died or disconnected
            showKernelStatusMessage('Kernel setup failed. Check the server logs for details.', 'error', 8000);
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
            const model = monacoModels[data.cell_id];
            if (editor) {
                const currentValue = editor.getValue();
                if (currentValue !== data.source) {
                    const position = editor.getPosition();
                    editor.setValue(data.source);
                    if (position) editor.setPosition(position);
                }
            } else if (model && model.getValue() !== data.source) {
                model.setValue(data.source);
            }
            // Also update hidden textarea
            const textarea = document.getElementById(`source-${data.cell_id}`);
            if (textarea) textarea.value = data.source;
            const promptTextarea = document.getElementById(`prompt-${data.cell_id}`);
            if (promptTextarea && promptTextarea.value !== data.source) {
                promptTextarea.value = data.source;
            }
            let outputVersionSafe = true;
            if (data.cell_type === 'prompt' && typeof data.output === 'string') {
                const result = applyPromptOutputUpdate(data.cell_id, data.output);
                outputVersionSafe = result?.safeVersion !== false;
            }
            if (data.execution_count == null && data.time_run == null) {
                document.querySelector(`#header-${data.cell_id} .cell-meta`)?.remove();
            }
            if (outputVersionSafe) updateCellVersion(data.cell_id, data.version);
            renderCellPreviews(data.cell_id);
        } else if (data.type === 'cell_conflict') {
            console.warn('[WS] cell_conflict received:', data);
            if (data.conflict_field === 'output') {
                applyPromptOutputUpdate(data.cell_id, data.output, {force: true});
                switchToPreview(data.cell_id, 'output');
            } else {
                const textarea = document.getElementById(`source-${data.cell_id}`);
                if (textarea && typeof data.source === 'string') textarea.value = data.source;
                const promptTextarea = document.getElementById(`prompt-${data.cell_id}`);
                if (promptTextarea && typeof data.source === 'string') promptTextarea.value = data.source;
            }
            updateCellVersion(data.cell_id, data.actual_version ?? data.version);
            renderCellPreviews(data.cell_id);
            const conflictLabel = data.conflict_field === 'output' ? 'response' : 'cell';
            showKernelStatusMessage(`That ${conflictLabel} changed in another tab. Dialeng kept the newer backend version.`, 'warning', 7000);
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
            const cellEl = document.getElementById(`cell-${data.cell_id}`);

            // Find neighbor to focus BEFORE removing the cell from DOM.
            // Prefer the next cell below; if this was the last cell, use the one above.
            let focusTarget = null;
            if (cellEl && focusedCellId === data.cell_id) {
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

            destroyMonacoEditor(data.cell_id);
            promptResponseEditor?.remove(data.cell_id);
            if (!reconcileCellStructure({ orderedIds: data.ordered_cell_ids || [] })) {
                return;
            }
            applyCellVersions(data.cell_versions || {});

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
            // Reorder cells from the backend-authoritative notebook order.
            if (!reconcileCellStructure({
                orderedIds: data.ordered_cell_ids || [],
                preserveViewportCellId: data.cell_id,
            })) {
                return;
            }
            applyCellVersions(data.cell_versions || {});
            if (data.cell_id) {
                setFocusedCell(data.cell_id);
            }

        } else if (data.type === 'cell_add') {
            // Granular cell insertion: add one cell + add-row at a position.
            const pendingViewport = consumeStructureViewport();
            if (!reconcileCellStructure({
                orderedIds: data.ordered_cell_ids || [],
                insertedCells: [{ cellId: data.cell_id, html: data.html }],
                preserveViewportCellId: pendingViewport?.cellId ?? null,
                preserveScrollY: pendingViewport?.scrollY ?? null,
            })) {
                return;
            }
            applyCellVersions(data.cell_versions || {});

            // If Shift+Enter created this cell at the end, keep the current
            // viewport stable through the structural reconcile, then reveal the
            // new cell with the smallest possible adjustment.
            if (_pendingScrollToNewCell) {
                _pendingScrollToNewCell = false;
                requestAnimationFrame(() => {
                    const newCell = document.getElementById(`cell-${data.cell_id}`);
                    if (newCell) {
                        newCell.scrollIntoView({ behavior: 'auto', block: 'nearest' });
                    }
                    focusCellWithoutScroll(data.cell_id, window.scrollY);
                });
            }

        // Extension system: page refresh after hot-reload
        } else if (data.type === 'extensions_reloaded') {
            console.log('[WS] Extensions reloaded, refreshing page...');
            location.reload();
        }
    };

    ws.onclose = function() {
        if (window.NOTEBOOK_TERMINAL_STATE) return;
        stopWebSocketHeartbeat();
        const delay = _wsReconnectDelay + Math.random() * _wsReconnectDelay * 0.2;
        console.log(`[WS] Disconnected, reconnecting in ${Math.round(delay)}ms...`);
        showKernelStatusMessage('Browser connection dropped. Reconnecting…', 'warning', Math.round(delay) + 1000);
        wsReconnectTimerId = setTimeout(() => {
            wsReconnectTimerId = null;
            connectWebSocket(notebookId);
        }, delay);
        _wsReconnectDelay = Math.min(_wsReconnectDelay * 2, 30000);
    };

    ws.onerror = function(error) {
        console.error('[WS] Error:', error);
    };
}

if (window.dialengEvents) {
    window.dialengEvents.register('heartbeat_ack', () => {
        wsLastHeartbeatAck = Date.now();
        if (wsHeartbeatDeadlineId) clearTimeout(wsHeartbeatDeadlineId);
        wsHeartbeatDeadlineId = null;
    });
    window.dialengEvents.register('notebook_deleted', handleNotebookDeleted);
    window.dialengEvents.register('resync_required', data => {
        recoverNotebookState(
            `server revision mismatch: ${(data.streams || []).join(', ') || 'unknown'}`,
        );
    });
    window.dialengEvents.register('kernel_snapshot', data => applyKernelSnapshot(data, 'ws'));
    window.dialengEvents.register('llm_status', data => {
        const notebook = getKernelState()?.notebook || {
            llm_provider: data.selected_provider,
            model: data.selected_model,
        };
        syncLlmSelectors(data, notebook);
    });
    window.dialengEvents.register('settings_changed', data => {
        applySettingsSnapshot(data, {preserveDirty: true});
    });
    window.dialengEvents.register('extension_snapshot', data => {
        renderExtensionManager(data);
    });
    window.dialengEvents.register('extension_command_progress', data => {
        if (data.message === 'Started') return;
        showKernelStatusMessage(
            `${data.command || 'Extension command'}: ${data.message || 'updated'}`,
            data.message === 'Cancelled' ? 'warning' : 'info',
            3000,
        );
    });
    window.dialengEvents.register('status_message', data => {
        if (data.action) showLlmFailureAction(data);
        else showKernelStatusMessage(
            data.message || 'Dialeng status changed.', data.tone || 'warning', data.persist_ms || 7000,
        );
    });
    window.dialengEvents.register('notebook_saved', () => setNotebookSaveState('saved'));
    window.dialengEvents.register('notebook_reloaded', () => window.location.reload());
    window.dialengEvents.register('file_explorer_changed', data => {
        const sidebar = document.getElementById('file-explorer-sidebar');
        const currentPath = document.getElementById('current-explorer-path')?.value || '';
        if (
            sidebar?.classList.contains('file-explorer-open')
            && (data.paths || []).includes(currentPath)
        ) {
            refreshFileExplorer();
        }
    });
    window.dialengEvents.register('cell_type_change', applyCellTypeChange);
    window.dialengEvents.register('cell_output_commit', data => {
        updateCellVersion(data.cell_id, data.version);
    });
}

function renderPromptChunk(cellId, chunk, {render = true} = {}) {
    if (!chunk) return;

    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        if (textarea.value === 'Generating...' || textarea.value === 'Click ▶ to generate response...' || textarea.value.startsWith('🧠')) {
            textarea.value = '';
        }
        textarea.value += chunk;
        if (preview && render && !document.hidden) {
            preview.innerHTML = renderMarkdown(textarea.value);
        }
    }
}

const promptChunkBatcher = new window.DialengFrameBatcher({onFlush: renderPromptChunk});

function flushPromptChunks(cellId, render = true) {
    promptChunkBatcher.flush(cellId, {render});
}

function appendToResponse(cellId, chunk, isThinking) {
    promptChunkBatcher.append(cellId, chunk);
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
    flushPromptChunks(cellId, false);
    const cell = document.getElementById(`cell-${cellId}`);
    setCellRunControlState(cell, false);
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

function applyPromptOutputUpdate(cellId, output, {force = false} = {}) {
    const normalizedOutput = String(output ?? '');
    const result = force
        ? {
            apply: true,
            safeVersion: true,
            value: promptResponseEditor?.resolveConflict(cellId, normalizedOutput) ?? normalizedOutput,
        }
        : (promptResponseEditor?.receive(cellId, normalizedOutput) ?? {
            apply: true,
            safeVersion: true,
            value: normalizedOutput,
        });
    if (!result.apply) return result;

    const textarea = document.getElementById(`output-${cellId}`);
    if (textarea) {
        textarea.value = result.value;
    }

    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (preview) {
        if (result.value && result.value.trim()) {
            preview.innerHTML = renderMarkdown(result.value);
            renderMarkdownServer(result.value, preview);
        } else {
            preview.innerHTML = '<p style="color: var(--text-muted);">Click to edit...</p>';
        }
    }

    ToolUI.clear(cellId);
    return result;
}

function startStreaming(cellId, useThinking) {
    promptChunkBatcher.clear(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    setCellRunControlState(cell, true);
    streamingCellId = cellId;
    const textarea = document.getElementById(`output-${cellId}`);
    const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
    if (textarea) {
        textarea.value = useThinking ? '🧠 Thinking...' : 'Generating...';
    }
    if (preview && useThinking) {
        preview.innerHTML = '<div class="thinking-indicator"><span>🧠</span> Thinking...</div>';
    }
    ToolUI.clear(cellId);

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
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function ansiToHtml(text) {
    // Strip non-SGR ANSI sequences (cursor control, erase codes like \x1b[2K,
    // \x1b[?25h, etc.) that we don't render — tqdm uses these extensively
    text = text.replace(/\x1b\[[0-9;]*[A-HJKSTfn]/g, '');  // cursor/erase
    text = text.replace(/\x1b\[\?[0-9;]*[hl]/g, '');       // private modes

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

// Per-cell live output state. Each stream segment retains its own stdout/stderr
// identity and position relative to rich displays, matching Jupyter ordering.
const streamRenderState = new Map();
const pendingClearOutputs = new Set();

function resetOutputRenderState(cellId) {
    streamRenderState.set(cellId, {pending: [], active: null});
    pendingClearOutputs.delete(cellId);
}

function startCodeStreaming(cellId) {
    console.log('[Code] startCodeStreaming called for cell:', cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    clearPendingRunRequest(cellId);

    if (!cell) {
        console.error('[Code] Cell element not found:', `cell-${cellId}`);
        return;
    }

    if (!outputEl) {
        console.error('[Code] Output element not found:', `output-${cellId}`);
    }

    setCellRunControlState(cell, true);

    if (outputEl) {
        outputEl.innerHTML = '';  // Clear for fresh output
        outputEl.classList.remove('error');
    }

    resetOutputRenderState(cellId);

    console.log('[Code] Started streaming for cell:', cellId, 'cell found:', !!cell, 'output found:', !!outputEl);
}

// Debounced stream rendering — coalesce multiple chunks per animation frame
const _streamDirty = new Set();
const _streamRafIds = new Map();

function appendCodeOutput(cellId, chunk, streamName) {
    applyPendingOutputClear(cellId);
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    // Note: stderr does NOT mean error — tools like tqdm write progress to stderr.
    // The 'error' class is only applied by finishCodeStreaming when has_error is true.

    const state = streamRenderState.get(cellId) || {pending: [], active: null};
    state.pending.push({chunk: String(chunk ?? ''), streamName: streamName === 'stderr' ? 'stderr' : 'stdout'});
    streamRenderState.set(cellId, state);

    // Schedule a single render per animation frame (debounce)
    if (!_streamDirty.has(cellId)) {
        _streamDirty.add(cellId);
        const rafId = requestAnimationFrame(() => _flushStreamRender(cellId));
        _streamRafIds.set(cellId, rafId);
    }
}

function applyCarriageReturn(currentText, chunk) {
    // tqdm, pip, and terminal-style libraries use carriage returns to replace
    // only the current logical line while preserving completed lines.
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
            // Promote embedded \n into separate array entries so
            // future \r only overwrites the actual last line
            if (parts[i].includes('\n')) {
                const tail = lines.pop();
                const sub = tail.split('\n');
                lines.push(...sub);
            }
        }
        return lines.join('\n');
    }
    return currentText + chunk;
}

function _flushStreamRender(cellId) {
    _streamDirty.delete(cellId);
    _streamRafIds.delete(cellId);

    const state = streamRenderState.get(cellId);
    if (!state) return;

    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    for (const event of state.pending) {
        let active = state.active;
        if (
            !active || active.streamName !== event.streamName ||
            !active.element?.isConnected || !outputEl.contains(active.element)
        ) {
            const element = document.createElement('pre');
            element.className = `output-item stream-output ${event.streamName}`;
            element.dataset.outputType = 'stream';
            outputEl.appendChild(element);
            active = {streamName: event.streamName, text: '', element};
            state.active = active;
        }

        const nearBottom = (
            active.element.scrollTop + active.element.clientHeight >=
            active.element.scrollHeight - 30
        );
        active.text = applyCarriageReturn(active.text, event.chunk);
        active.element.innerHTML = ansiToHtml(active.text);
        if (nearBottom) active.element.scrollTop = active.element.scrollHeight;
    }
    state.pending = [];
}

function flushPendingCodeOutput(cellId) {
    if (!_streamRafIds.has(cellId)) return;
    cancelAnimationFrame(_streamRafIds.get(cellId));
    _streamRafIds.delete(cellId);
    _flushStreamRender(cellId);
}

function activateOutputScripts(container) {
    container.querySelectorAll('script').forEach(script => {
        const replacement = document.createElement('script');
        Array.from(script.attributes).forEach(attr => replacement.setAttribute(attr.name, attr.value));
        replacement.textContent = script.textContent;
        script.parentNode.replaceChild(replacement, script);
    });
}

function applyPendingOutputClear(cellId) {
    if (!pendingClearOutputs.has(cellId)) return;
    pendingClearOutputs.delete(cellId);
    clearCellOutput(cellId, false);
}

function appendDisplayData(cellId, html, displayId, outputKind = 'display_data') {
    applyPendingOutputClear(cellId);
    flushPendingCodeOutput(cellId);
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    const state = streamRenderState.get(cellId);
    if (state) state.active = null;

    // Create display data container
    const displayEl = document.createElement('div');
    const kindClass = outputKind === 'execute_result' ? 'execute-result' : 'display-data';
    displayEl.className = `output-item display-data ${kindClass}`;
    displayEl.dataset.outputType = outputKind;
    if (displayId) {
        displayEl.setAttribute('data-display-id', displayId);
    }
    displayEl.innerHTML = html;
    outputEl.appendChild(displayEl);

    activateOutputScripts(displayEl);
}

function updateDisplayData(cellId, html, displayId) {
    applyPendingOutputClear(cellId);
    flushPendingCodeOutput(cellId);
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl || !displayId) return;

    // Jupyter permits multiple outputs to share a display_id. An update must
    // replace all matching views, not only the first one.
    const existing = Array.from(outputEl.querySelectorAll('[data-display-id]'))
        .filter(element => element.dataset.displayId === String(displayId));
    if (existing.length) {
        existing.forEach(element => {
            element.innerHTML = html;
            activateOutputScripts(element);
        });
    } else {
        // Fallback: append as new display data
        appendDisplayData(cellId, html, displayId);
    }
}

function clearCellOutput(cellId, wait = false) {
    if (wait) {
        pendingClearOutputs.add(cellId);
        return;
    }
    const outputEl = document.getElementById(`output-${cellId}`);
    if (!outputEl) return;

    if (_streamRafIds.has(cellId)) cancelAnimationFrame(_streamRafIds.get(cellId));
    _streamRafIds.delete(cellId);
    _streamDirty.delete(cellId);
    outputEl.innerHTML = '';
    outputEl.classList.remove('error');
    resetOutputRenderState(cellId);
}

function finishCodeStreaming(cellId, hasError) {
    console.log('[Code] finishCodeStreaming called for cell:', cellId, 'hasError:', hasError);
    const cell = document.getElementById(`cell-${cellId}`);
    const outputEl = document.getElementById(`output-${cellId}`);

    clearPendingRunRequest(cellId);

    setCellRunControlState(cell, false);

    if (outputEl && hasError) {
        outputEl.classList.add('error');
    }

    // Flush the last frame before removing the accumulator. Otherwise a fast
    // stream_end can arrive before requestAnimationFrame and lose final text.
    if (_streamRafIds.has(cellId)) {
        flushPendingCodeOutput(cellId);
    }
    _streamDirty.delete(cellId);

    streamRenderState.delete(cellId);
    pendingClearOutputs.delete(cellId);
    setupLargeOutput(cellId);

    // Clear the streaming timeout
    clearCodeStreamingTimeout(cellId);

    console.log('[Code] Finished streaming for cell:', cellId, hasError ? '(with errors)' : '');
}

// ============================================================================
// Queue State Management
// ============================================================================

// Track queue state for cells
const cellQueueState = new Map(); // cellId -> {state: 'queued'|'running'|'idle', position: number}
const pendingRunRequests = new Set();

function markPendingRunRequest(cellId) {
    pendingRunRequests.add(cellId);
}

function clearPendingRunRequest(cellId) {
    pendingRunRequests.delete(cellId);
    const cell = document.getElementById(`cell-${cellId}`);
    const runBtn = cell?.querySelector('.btn-run');
    if (runBtn) runBtn.disabled = false;
}

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
    if (state !== 'idle') clearPendingRunRequest(cellId);

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
            clearPendingRunRequest(cellId);
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
        await fetch(`/dialeng/${currentNotebookId}/queue/cancel_all`, { method: 'POST' });
    } catch (e) {
        console.error('[Queue] Failed to cancel all:', e);
    }
}

// Code cell streaming timeout mechanism
let codeStreamingTimeouts = new Map();  // Track timeouts per cell

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

    markPendingRunRequest(cellId);

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
    // Quiet long-running code cells are valid. The backend snapshot is now
    // authoritative for completion/liveness, so this client-side watchdog is
    // intentionally a no-op.
}

function interruptCodeCell(notebookId, cellId) {
    console.log('[Code] interruptCodeCell called for cell:', cellId);

    // Use cancelAllExecution to stop running cell AND clear the queue
    // This ensures we don't just interrupt one cell and continue with others
    cancelAllExecution();

    // Clear the streaming timeout for this cell
    clearCodeStreamingTimeout(cellId);
    clearPendingRunRequest(cellId);

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
            if (target.dataset.type !== element.dataset.type) destroyMonacoEditor(cellId);

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
                    setTimeout(() => observeMonacoCell(cellId), 0);
                } else if (newCell.dataset.type === 'shell') {
                    setTimeout(() => observeMonacoCell(cellId, 'sh'), 0);
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
            if (targetId.startsWith('header-')) {
                // Update execution metadata in place and leave action controls
                // connected. Replacing or detaching those controls can lose
                // either HTMX listeners or compiled inline source-sync handlers.
                const currentSummary = target.firstElementChild;
                const replacementSummary = element.firstElementChild;
                if (currentSummary && replacementSummary) {
                    currentSummary.replaceWith(replacementSummary);
                    continue;
                }
            }
            target.replaceWith(element);
            // Reinitialize HTMX bindings on the new element
            const newEl = document.getElementById(targetId);
            if (newEl) {
                if (targetId.startsWith('output-')) {
                    htmx.process(newEl);
                }
                // Re-execute scripts in output OOB swaps (e.g., YouTube embeds, interactive widgets).
                // innerHTML/replaceWith does NOT execute <script> tags — we must clone them
                // into new <script> elements so the browser runs them. This mirrors the
                // same pattern used in appendDisplayData() during streaming.
                if (targetId.startsWith('output-')) {
                    newEl.querySelectorAll('script').forEach(script => {
                        const newScript = document.createElement('script');
                        if (script.src) newScript.src = script.src;
                        newScript.textContent = script.textContent;
                        script.parentNode.replaceChild(newScript, script);
                    });
                    setupLargeOutput(targetId.replace('output-', ''));
                }
            }
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
    // Full-container recovery invalidates every prior view/model.
    for (const cellId of Object.keys(monacoModels)) {
        destroyMonacoEditor(cellId);
    }

    // Observe all code and shell cells; only near-visible views initialize.
    document.querySelectorAll('.cell[data-type="code"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => observeMonacoCell(cellId), 0);
    });
    document.querySelectorAll('.cell[data-type="shell"]').forEach(cell => {
        const cellId = cell.id.replace('cell-', '');
        setTimeout(() => observeMonacoCell(cellId, 'sh'), 0);
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

            const customRenderer = extensionClientState.toolRenderers.get(toolName);
            if (customRenderer?.renderCall) {
                const rendered = customRenderer.renderCall({
                    cellId, toolId, toolName, input: toolInput, container: toolArea,
                });
                if (rendered instanceof Node) {
                    rendered.id ||= `tool-call-${toolId}`;
                    rendered.classList.add('tool-call', 'extension-tool-call');
                    toolArea.appendChild(rendered);
                    return;
                }
            }

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

            const customRenderer = extensionClientState.toolRenderers.get(toolName);
            if (customRenderer?.renderResult) {
                const rendered = customRenderer.renderResult({
                    cellId, toolId, toolName, result, status, callElement: callEl,
                });
                if (rendered instanceof Node) {
                    callEl.appendChild(rendered);
                    return;
                }
            }

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
                sendWebSocketControl({
                    type: 'tool_confirmation_response',
                    confirmation_id: confirmationId,
                    approved: true
                });
            }
            this.removeDialog(confirmationId);
            pendingConfirmations.delete(confirmationId);
        },

        // User cancels execution
        cancel: function(confirmationId) {
            const pending = pendingConfirmations.get(confirmationId);
            if (pending && ws && ws.readyState === WebSocket.OPEN) {
                sendWebSocketControl({
                    type: 'tool_confirmation_response',
                    confirmation_id: confirmationId,
                    approved: false
                });
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

// Track pending cell execution when kernel is not yet selected
let pendingCellRunId = null;
let pendingCellRunAdvance = true;

function toggleKernelModal(nbId) {
    const overlay = document.getElementById('kernel-modal-overlay');
    if (!overlay) return;
    const wasVisible = overlay.classList.contains('visible');
    overlay.classList.toggle('visible');
    if (wasVisible) {
        kernelModalDismissed = true;
        pendingCellRunId = null;
        pendingCellRunAdvance = true;
    } else {
        kernelModalDismissed = false;
    }
}

function closeKernelModalForApply() {
    const overlay = document.getElementById('kernel-modal-overlay');
    if (!overlay) return;
    overlay.classList.remove('visible');
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

// Helper: run a pending cell after kernel + CRAFT init
function _runPendingCell() {
    if (!pendingCellRunId) return;
    const cellId = pendingCellRunId;
    const advance = pendingCellRunAdvance;
    pendingCellRunId = null;
    pendingCellRunAdvance = true;
    const cell = document.getElementById(`cell-${cellId}`);
    if (cell) clickCellRunButton(cell, {advance});
}

function applyKernelSelection(nbId) {
    const typeInput = document.getElementById('kernel-modal-selected-type');
    const rtInput = document.getElementById('kernel-modal-selected-runtime');
    if (!typeInput) return;
    const kernelType = typeInput.value;
    const runtimeType = rtInput ? rtInput.value : 'cpu';

    console.log('[Kernel] applyKernelSelection:', kernelType, runtimeType);

    // Close modal immediately and show initializing state (yellow dot)
    closeKernelModalForApply();
    window.KERNEL_ALIVE = false;
    window.KERNEL_STATE = window.KERNEL_STATE || {};
    window.KERNEL_STATE.kernel = Object.assign({}, window.KERNEL_STATE.kernel, {
        selected_type: kernelType,
        can_run: false,
        display_state: 'initializing'
    });
    updateKernelDot('busy');
    showKernelStatusMessage('Attaching kernel and running notebook setup…', 'warning', 4000);

    // Send kernel type change in background
    fetch(`/dialeng/${nbId}/kernel/type`, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `kernel_type=${encodeURIComponent(kernelType)}`
    }).then(response => {
        console.log('[Kernel] kernel/type response:', response.status);
        if (!response.ok) throw new Error(`kernel/type failed: ${response.status}`);
        return response.text();
    }).then(async () => {
        // If Colab, set runtime type and WAIT before craft-init
        // (set_runtime_type shuts down the old kernel and creates a new one)
        if (runtimeType && kernelType === 'colab') {
            const rtResp = await fetch(`/dialeng/${nbId}/kernel/runtime`, {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `runtime_type=${encodeURIComponent(runtimeType)}`
            });
            console.log('[Kernel] runtime type response:', rtResp.status);
        }

        // Refresh kernel toolbar button
        htmx.ajax('GET', `/dialeng/${nbId}/kernel/info`, {target: '#kernel-status-bar', swap: 'outerHTML'});

        // Always trigger craft-init on kernel selection/switch.
        // The backend handles: lib sys.path injection, Colab uploads, CRAFT cell
        // execution. It's safe to call even with no CRAFT files (just does lib init).
        // The dot stays yellow (busy) — the backend broadcasts kernel_connected
        // via WebSocket when the background init task completes.
        console.log('[Kernel] Triggering craft-init (dot will update via WebSocket)');
        fetch(`/dialeng/${nbId}/kernel/craft-init`, { method: 'POST' })
            .then(resp => {
                if (!resp.ok) throw new Error(`craft-init failed: ${resp.status}`);
                window.HAS_CRAFT_CODE = false;
                // Pending cell runs when kernel_connected arrives via WebSocket
            })
            .catch(err => {
                console.error('[Kernel] CRAFT init error:', err);
                updateKernelDot('error');
                showKernelStatusMessage('Kernel setup failed. Check the server logs for details.', 'error', 8000);
            });
    }).catch(err => {
        console.error('[Kernel] applyKernelSelection error:', err);
        updateKernelDot('error');
        showKernelStatusMessage('Kernel selection failed. Check the server logs for details.', 'error', 8000);
    });
}

// Refresh kernel status bar after first cell execution.
// Use document instead of document.body because this shared bundle is loaded
// from <head>, before body is guaranteed to exist on both notebook and file pages.
document.addEventListener('kernel-connected', function(e) {
    const nbId = window.NOTEBOOK_ID;
    if (nbId) {
        htmx.ajax('GET', `/dialeng/${nbId}/kernel/info`, {target: '#kernel-status-bar', swap: 'outerHTML'});
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

// Intercept cell run requests when no kernel is selected
document.addEventListener('htmx:beforeRequest', function(e) {
    // Get request path from HTMX element or requestConfig
    const elt = e.detail.elt;
    const path = elt?.getAttribute('hx-post') || e.detail.requestConfig?.path || e.detail.pathInfo?.requestPath || '';
    if (!path.match(/\/cell\/[^/]+\/run$/)) return;
    const match = path.match(/\/cell\/([^/]+)\/run$/);
    const cellId = match ? match[1] : null;
    const advance = runIntentTracker.consume(elt, {defaultAdvance: true});
    if (canRunCurrentKernel()) {
        // Disable only after HTMX has accepted the request. Disabling from the
        // inline click handler can make HTMX discard reruns on a replaced header.
        if (elt?.classList?.contains('btn-run')) elt.disabled = true;
        const cell = cellId ? document.getElementById(`cell-${cellId}`) : null;
        const isPrompt = cell?.dataset.type === 'prompt';
        if (advance && cellId && isPrompt) {
            pendingPromptAdvances.add(cellId);
        } else if (advance && cellId) {
            // Let the originating click finish bubbling first. Otherwise the
            // cell click listener can re-select the run cell after focus has
            // already moved to the next editor.
            requestAnimationFrame(() => {
                const currentCell = document.getElementById(`cell-${cellId}`);
                if (currentCell) moveToNextCell(currentCell);
            });
        }
        return;
    }

    // Cancel the HTMX request
    e.preventDefault();
    if (cellId) pendingPromptAdvances.delete(cellId);

    // Revert UI state that was set by prepareCodeRun() or startStreaming()
    if (cellId) {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell) {
            cell.classList.remove('streaming');
            const cancelBtn = cell.querySelector('.btn-cancel');
            const runBtn = cell.querySelector('.btn-run');
            if (cancelBtn) cancelBtn.style.display = 'none';
            if (runBtn) runBtn.style.display = '';
        }
        clearPendingRunRequest(cellId);
        clearCodeStreamingTimeout(cellId);
    }

    // Store pending cell and show kernel modal
    pendingCellRunId = cellId;
    pendingCellRunAdvance = advance;
    const kernelState = getKernelState();
    const displayState = kernelState?.kernel?.display_state;
    if (kernelState?.kernel?.exists && ['connecting', 'initializing', 'restarting', 'busy'].includes(displayState)) {
        showKernelStatusMessage(kernelState.setup?.detail || 'Kernel is still initializing. The cell will run once it is ready.', 'warning', 5000);
        return;
    }
    if (kernelState?.kernel?.auth_required) {
        showKernelStatusMessage('Sign in with Google before using a Colab kernel.', 'warning', 6000);
    }
    const overlay = document.getElementById('kernel-modal-overlay');
    if (overlay && !overlay.classList.contains('visible')) {
        kernelModalDismissed = false;
        overlay.classList.add('visible');
    }
});

// HTMX evaluates inline values while constructing the request, which can race
// Monaco's final input event during fast edits. Overwrite run parameters at
// the last configurable phase so the backend always receives the live model.
document.addEventListener('htmx:configRequest', function(e) {
    const path = e.detail.path || e.detail.pathInfo?.requestPath || '';
    const match = path.match(/\/cell\/([^/]+)\/run$/);
    if (!match) return;
    const cellId = match[1];
    syncAllContent(cellId);
    e.detail.parameters.source = getCellSourceForRequest(cellId);
    e.detail.parameters.expected_version =
        document.getElementById(`version-${cellId}`)?.value || '-1';
});

// Server-side safety net: kernel-required HX-Trigger
document.addEventListener('kernel-required', function(e) {
    const detail = e.detail || {};
    const cellId = detail.cellId || null;

    // Revert UI state
    if (cellId) {
        const cell = document.getElementById(`cell-${cellId}`);
        if (cell) {
            cell.classList.remove('streaming');
            const cancelBtn = cell.querySelector('.btn-cancel');
            const runBtn = cell.querySelector('.btn-run');
            if (cancelBtn) cancelBtn.style.display = 'none';
            if (runBtn) runBtn.style.display = '';
        }
        clearPendingRunRequest(cellId);
        clearCodeStreamingTimeout(cellId);
    }

    pendingCellRunId = cellId;
    pendingCellRunAdvance = true;
    const overlay = document.getElementById('kernel-modal-overlay');
    if (overlay && !overlay.classList.contains('visible')) {
        kernelModalDismissed = false;
        overlay.classList.add('visible');
    }
});

// ==================== Standalone File Editor ====================
let fileEditorHeartbeatId = null;
let fileEditorOwnsLock = false;

function getDialengClientId() {
    let clientId = sessionStorage.getItem('dialeng-client-id');
    if (!clientId) {
        clientId = (window.crypto?.randomUUID?.() || `dialeng-${Date.now()}-${Math.random().toString(16).slice(2)}`);
        sessionStorage.setItem('dialeng-client-id', clientId);
    }
    return clientId;
}

function stopFileEditorHeartbeat() {
    if (fileEditorHeartbeatId) {
        clearInterval(fileEditorHeartbeatId);
        fileEditorHeartbeatId = null;
    }
}

function startFileEditorHeartbeat() {
    stopFileEditorHeartbeat();
    const path = document.getElementById('file-editor-path')?.value || window.DIALENG_FILE_PATH;
    if (!path || !fileEditorOwnsLock) return;
    const clientId = getDialengClientId();
    fileEditorHeartbeatId = setInterval(async () => {
        try {
            const resp = await fetch('/dialeng/file/heartbeat', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: `path=${encodeURIComponent(path)}&client_id=${encodeURIComponent(clientId)}`
            });
            const data = await resp.json();
            if (!data.ok) {
                fileEditorOwnsLock = false;
                stopFileEditorHeartbeat();
                showKernelStatusMessage('This file is no longer locked by this tab. Reloading editor state…', 'warning', 5000);
                reloadFileEditorView();
            }
        } catch (err) {
            console.warn('[FileEditor] Heartbeat failed:', err);
        }
    }, 10000);
}

function applyFileEditorFragmentState() {
    destroyStandaloneMonacoEditor('file-monaco');
    stopFileEditorHeartbeat();

    const saveBtn = document.getElementById('save-btn');
    const state = document.getElementById('file-editor-state')?.value || 'missing';
    fileEditorOwnsLock = state === 'editable';

    if (saveBtn) saveBtn.disabled = !fileEditorOwnsLock;

    if (fileEditorOwnsLock) {
        const language = document.getElementById('file-editor-language')?.value || 'plaintext';
        initStandaloneMonacoEditor('file-monaco', 'file-source', language);
        startFileEditorHeartbeat();
    }
}

async function reloadFileEditorView() {
    if (!window.DIALENG_FILE_PATH) return;

    const container = document.getElementById('file-editor-container');
    if (!container) return;

    container.innerHTML = '<div class="file-editor-loading">Loading file…</div>';
    const clientId = getDialengClientId();
    try {
        const resp = await fetch(`/dialeng/file/view?path=${encodeURIComponent(window.DIALENG_FILE_PATH)}&client_id=${encodeURIComponent(clientId)}`, {
            headers: {'HX-Request': 'true'},
            cache: 'no-store'
        });
        if (!resp.ok) throw new Error(`view failed: ${resp.status}`);
        const fragmentHtml = await resp.text();
        container.innerHTML = fragmentHtml;
        applyFileEditorFragmentState();
    } catch (err) {
        console.error('[FileEditor] Failed to load file view:', err);
        container.innerHTML = '<div class="file-editor-message error"><h3 class="file-editor-message-title">File editor error</h3><p class="file-editor-message-body">Dialeng could not load this file.</p></div>';
        fileEditorOwnsLock = false;
        const saveBtn = document.getElementById('save-btn');
        if (saveBtn) saveBtn.disabled = true;
    }
}

function initializeFileEditorPage(path) {
    window.DIALENG_PAGE_KIND = 'file';
    window.DIALENG_FILE_PATH = path;
    reloadFileEditorView();
}

async function saveOpenFile() {
    if (!fileEditorOwnsLock) {
        showKernelStatusMessage('This file is not editable in the current tab.', 'warning', 3500);
        return;
    }
    const path = document.getElementById('file-editor-path')?.value || window.DIALENG_FILE_PATH;
    const textarea = document.getElementById('file-source');
    if (!path || !textarea) return;

    syncStandaloneMonacoToTextarea('file-monaco', 'file-source');
    const clientId = getDialengClientId();
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) saveBtn.disabled = true;

    try {
        const params = new URLSearchParams({
            path,
            client_id: clientId,
            content: textarea.value,
            expected_mtime_ns: document.getElementById('file-editor-mtime-ns')?.value || '-1',
            expected_size: document.getElementById('file-editor-size')?.value || '-1',
        });
        const resp = await fetch('/dialeng/file/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: params.toString()
        });
        const data = await resp.json();
        showKernelStatusMessage(data.message || (data.ok ? 'File saved.' : 'Save failed.'), data.ok ? 'success' : 'error', 3000);
        if (data.ok) {
            const mtimeInput = document.getElementById('file-editor-mtime-ns');
            const sizeInput = document.getElementById('file-editor-size');
            if (mtimeInput && data.mtime_ns !== undefined) mtimeInput.value = String(data.mtime_ns);
            if (sizeInput && data.size !== undefined) sizeInput.value = String(data.size);
        }
        if (!data.ok && data.status === 'locked') {
            fileEditorOwnsLock = false;
            reloadFileEditorView();
        } else if (!data.ok && data.status === 'conflict') {
            reloadFileEditorView();
        }
    } catch (err) {
        console.error('[FileEditor] Save failed:', err);
        showKernelStatusMessage('File save failed.', 'error', 4000);
    } finally {
        if (saveBtn) saveBtn.disabled = !fileEditorOwnsLock;
    }
}

function releaseOpenFileLock() {
    if (!fileEditorOwnsLock || !window.DIALENG_FILE_PATH) return;
    const params = new URLSearchParams({
        path: window.DIALENG_FILE_PATH,
        client_id: getDialengClientId(),
    });
    navigator.sendBeacon('/dialeng/file/release', new Blob([params.toString()], {
        type: 'application/x-www-form-urlencoded;charset=UTF-8'
    }));
}

window.addEventListener('beforeunload', () => {
    releaseOpenFileLock();
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

function refreshFileExplorer() {
    const currentPath = document.getElementById('current-explorer-path')?.value || '';
    const activeNotebookId = document.getElementById('current-explorer-active-notebook')?.value || window.NOTEBOOK_ID || '';
    const activeFileRelpath = document.getElementById('current-explorer-active-file')?.value || window.DIALENG_FILE_PATH || '';
    htmx.ajax('GET', `/files?path=${encodeURIComponent(currentPath)}&active_notebook_id=${encodeURIComponent(activeNotebookId)}&active_file_relpath=${encodeURIComponent(activeFileRelpath)}`, '#file-list-content');
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

function createNewItem() {
    const name = document.getElementById('new-item-name')?.value?.trim();
    const type = document.getElementById('new-item-type')?.value || 'dialog';
    if (!name) return;

    // Read current path from hidden input updated by HTMX folder navigation
    const currentPath = document.getElementById('current-explorer-path')?.value || '';
    const activeNotebookId = document.getElementById('current-explorer-active-notebook')?.value || window.NOTEBOOK_ID || '';
    const activeFileRelpath = document.getElementById('current-explorer-active-file')?.value || window.DIALENG_FILE_PATH || '';

    if (type === 'folder') {
        // Create folder via HTMX-style fetch
        fetch('/files/new-folder', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `path=${encodeURIComponent(currentPath)}&name=${encodeURIComponent(name)}&active_notebook_id=${encodeURIComponent(activeNotebookId)}&active_file_relpath=${encodeURIComponent(activeFileRelpath)}`
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
        window.location.href = `/dialeng/new?dir=${encodeURIComponent(currentPath)}&name=${encodeURIComponent(name)}`;
    }
}

// Delete confirmation modal
function showDeleteConfirm(filePath, displayName) {
    const modal = document.getElementById('delete-confirm-modal');
    const pathInput = document.getElementById('delete-file-path');
    const displaySpan = document.getElementById('delete-file-display');
    const notebookWarning = document.getElementById('delete-notebook-warning');
    if (modal && pathInput && displaySpan) {
        pathInput.value = filePath;
        displaySpan.textContent = displayName;
        if (notebookWarning) {
            notebookWarning.style.display = filePath.toLowerCase().endsWith('.ipynb') ? 'block' : 'none';
        }
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
    const activeNotebookId = document.getElementById('current-explorer-active-notebook')?.value || window.NOTEBOOK_ID || '';
    const activeFileRelpath = document.getElementById('current-explorer-active-file')?.value || window.DIALENG_FILE_PATH || '';
    fetch('/files/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `path=${encodeURIComponent(filePath)}&active_notebook_id=${encodeURIComponent(activeNotebookId)}&active_file_relpath=${encodeURIComponent(activeFileRelpath)}`
    }).then(async r => {
        const result = await r.json();
        if (!r.ok || !result.ok) throw new Error(result.message || `Delete failed: ${r.status}`);
        return result;
    }).then(result => {
        const container = document.getElementById('file-list-content');
        if (container && result.html) {
            container.outerHTML = result.html;
            const newContainer = document.getElementById('file-list-content');
            if (newContainer && typeof htmx !== 'undefined') htmx.process(newContainer);
        }
        if (result.deleted_notebook_id && result.deleted_notebook_id === window.NOTEBOOK_ID) {
            handleNotebookDeleted({
                notebook_id: result.deleted_notebook_id,
                workspace_url: result.workspace_url,
            });
        }
    }).catch(error => {
        console.error('[Files] Delete failed:', error);
        showKernelStatusMessage(error.message || 'The file could not be deleted.', 'error', 7000);
    });
    hideDeleteConfirm();
}

// File explorer toggle shortcut is consolidated into the main keydown handler (line ~406)
