/* Browser-owned workspace view state. The server still renders all panel data. */
(function () {
    'use strict';

    const byId = id => document.getElementById(id);
    const panelIds = {files: 'file-explorer-sidebar', outline: 'outline-sidebar'};
    const panelClasses = {files: 'file-explorer-open', outline: 'outline-open'};
    const sectionNames = ['headings', 'variables', 'functions'];
    const scrollValue = value => Number.isFinite(value) ? Math.max(0, value) : 0;

    class WorkspaceUI {
        constructor() {
            this.files = byId(panelIds.files);
            this.outline = byId(panelIds.outline);
            this.notebookId = this.outline?.dataset.notebookId || '';
            this.storageKey = `dialeng.workspace-ui.v1:${this.files.dataset.workspaceId}`;
            this.narrow = window.matchMedia('(max-width: 768px)');
            this.state = this.readState();
            this.requestedFilesPath = this.state.files.path;
            this.settingsOpen = false;
            this.jobs = {};
            this.stale = {files: true, outline: true};
            this.saveTimer = null;
            this.returnFocus = {};
            this.backdrop = document.createElement('div');
            this.backdrop.className = 'workspace-sidebar-backdrop';
            this.backdrop.hidden = true;
            this.backdrop.addEventListener('click', () => this.selectPanel(null));
            document.body.append(this.backdrop);
            this.bind();
            this.render();
            this.refreshVisible();
        }

        readState() {
            const initialPath = byId('current-explorer-path')?.value || '';
            const state = {
                version: 1, leftPanel: this.narrow.matches ? null : 'files',
                files: {path: initialPath, scrollTop: 0}, outlines: {},
            };
            try {
                const saved = JSON.parse(sessionStorage.getItem(this.storageKey));
                if (saved?.version !== 1) return state;
                if ([null, 'files', 'outline'].includes(saved.leftPanel)) state.leftPanel = saved.leftPanel;
                if (typeof saved.files?.path === 'string' && saved.files.path.length < 4096) {
                    state.files.path = saved.files.path;
                    state.files.scrollTop = scrollValue(saved.files.scrollTop);
                }
                for (const [id, view] of Object.entries(saved.outlines || {}).slice(-30)) {
                    Object.defineProperty(state.outlines, id, {enumerable: true, configurable: true, writable: true,
                        value: {scrollTop: scrollValue(view?.scrollTop),
                            collapsed: sectionNames.filter(name => view?.collapsed?.includes?.(name))}});
                }
            } catch (_) { /* Storage may be unavailable or contain an older/broken value. */ }
            return state;
        }

        persist() {
            clearTimeout(this.saveTimer);
            try { sessionStorage.setItem(this.storageKey, JSON.stringify(this.state)); }
            catch (_) { /* View state remains usable without browser storage. */ }
        }

        outlineView() {
            if (!Object.hasOwn(this.state.outlines, this.notebookId)) {
                // Bound session storage even when many notebooks are visited in one tab.
                const keys = Object.keys(this.state.outlines);
                if (keys.length >= 30) delete this.state.outlines[keys[0]];
                Object.defineProperty(this.state.outlines, this.notebookId, {
                    enumerable: true, configurable: true, writable: true,
                    value: {collapsed: [], scrollTop: 0},
                });
            }
            return this.state.outlines[this.notebookId];
        }

        get leftPanel() {
            // File/workspace pages have no Outline; keep the preference for the next notebook.
            return this.state.leftPanel === 'outline' && !this.outline ? 'files' : this.state.leftPanel;
        }

        get filesPath() { return byId('current-explorer-path')?.value || ''; }

        scroller(kind) {
            return kind === 'files' ? this.files.querySelector('.file-explorer-content') : byId('outline-content');
        }

        rememberView(kind) {
            if (kind === 'outline' && !this.outline?.dataset.loaded) return;
            if (kind === 'files' && this.filesPath !== this.state.files.path) return;
            const scroller = this.scroller(kind);
            if (!scroller) return;
            const view = kind === 'files' ? this.state.files : this.outlineView();
            view.scrollTop = scroller.scrollTop;
            if (kind === 'outline') {
                view.collapsed = [...scroller.querySelectorAll('details[data-outline-section]')]
                    .filter(section => !section.open).map(section => section.dataset.outlineSection);
            }
        }

        restoreScroll(kind) {
            if (kind === 'outline' && !this.outline?.dataset.loaded) return;
            if (kind === 'files' && this.filesPath !== this.state.files.path) return;
            const view = kind === 'files' ? this.state.files : this.outlineView();
            const scroller = this.scroller(kind);
            if (scroller) scroller.scrollTop = view.scrollTop;
        }

        togglePanel(kind) { this.selectPanel(this.leftPanel === kind ? null : kind); }

        revealContent() {
            if (this.narrow.matches && this.leftPanel) this.selectPanel(null);
        }

        selectPanel(kind) {
            if (kind && !byId(panelIds[kind])) return;
            const previous = this.leftPanel;
            if (previous) {
                this.rememberView(previous);
                this.cancel(previous);
            }
            if (kind) this.returnFocus[kind] = document.activeElement;
            this.state.leftPanel = kind;
            this.render();
            this.persist();
            if (kind) {
                this.restoreScroll(kind);
                this.refresh(kind);
                if (this.narrow.matches) byId(panelIds[kind]).querySelector('button')?.focus();
            } else if (previous) {
                const trigger = document.querySelector(`[data-workspace-toggle="${previous}"]`);
                (this.returnFocus[previous]?.isConnected ? this.returnFocus[previous] : trigger)?.focus();
                // Closing from a panel button must not leave focus inside its inert subtree.
                if (byId(panelIds[previous]).contains(document.activeElement)) trigger?.focus();
            }
        }

        render() {
            for (const [kind, id] of Object.entries(panelIds)) {
                const panel = byId(id);
                if (!panel) continue;
                const open = this.leftPanel === kind;
                panel.classList.toggle(panelClasses[kind], open);
                panel.inert = !open || this.settingsOpen;
                panel.setAttribute('aria-hidden', String(!open || this.settingsOpen));
                document.querySelectorAll(`[data-workspace-toggle="${kind}"]`).forEach(button => {
                    button.setAttribute('aria-expanded', String(open));
                    button.setAttribute('aria-controls', id);
                    button.classList.toggle('active', open);
                });
            }
            const drawerOpen = this.narrow.matches && !!this.leftPanel;
            this.backdrop.hidden = !drawerOpen || this.settingsOpen;
            const main = document.querySelector('.main-layout > .container');
            if (main) main.inert = drawerOpen || this.settingsOpen;
            const settings = byId('settings-sidebar');
            if (settings) {
                settings.classList.toggle('open', this.settingsOpen);
                settings.inert = !this.settingsOpen;
                settings.setAttribute('aria-hidden', String(!this.settingsOpen));
            }
            byId('settings-overlay')?.classList.toggle('open', this.settingsOpen);
            byId('settings-btn')?.setAttribute('aria-expanded', String(this.settingsOpen));
            byId('settings-btn')?.setAttribute('aria-controls', 'settings-sidebar');
        }

        setSettingsOpen(open) {
            if (open) this.returnFocus.settings = document.activeElement;
            this.settingsOpen = open;
            this.render();
            if (!open) {
                const previous = this.returnFocus.settings;
                if (previous?.isConnected && !previous.closest('[inert]')) previous.focus();
                else if (this.narrow.matches && this.leftPanel) byId(panelIds[this.leftPanel]).querySelector('button')?.focus();
                else byId('settings-btn')?.focus();
            }
        }

        status(kind, message = '', failed = false) {
            const status = byId(`${kind}-panel-status`);
            if (!status) return;
            status.hidden = !message;
            status.querySelector('span').textContent = message;
            status.querySelector('button').hidden = !failed;
        }

        cancel(kind) {
            this.jobs[kind]?.abort.abort();
            delete this.jobs[kind];
            byId(panelIds[kind])?.setAttribute('aria-busy', 'false');
            this.status(kind);
        }

        async refresh(kind, path = this.requestedFilesPath) {
            this.stale[kind] = true;
            if (this.leftPanel !== kind || !byId(panelIds[kind])) return;
            if (kind === 'files') this.requestedFilesPath = path;
            this.cancel(kind);
            const job = {abort: new AbortController()};
            this.jobs[kind] = job;
            byId(panelIds[kind]).setAttribute('aria-busy', 'true');
            this.status(kind, 'Updating…');
            const query = new URLSearchParams({path,
                active_notebook_id: this.files.dataset.activeNotebook || '',
                active_file_relpath: this.files.dataset.activeFile || ''});
            const url = kind === 'files' ? `/files?${query}` : this.outline.dataset.contentUrl;
            try {
                const response = await fetch(url, {signal: job.abort.signal, headers: {'HX-Request': 'true'}});
                if (!response.ok) throw new Error(`Request failed (${response.status})`);
                const html = await response.text();
                // Abort is best effort: object identity also rejects responses already in flight.
                if (this.jobs[kind] !== job || this.leftPanel !== kind) return;
                const contentId = kind === 'files' ? 'file-list-content' : 'outline-content';
                const template = document.createElement('template');
                template.innerHTML = html;
                const next = template.content.querySelector(`#${contentId}`);
                if (!next || template.content.querySelectorAll(`#${contentId}`).length !== 1) {
                    throw new Error('Invalid panel response');
                }
                const current = byId(contentId);
                const focused = current.contains(document.activeElement);
                const focusId = document.activeElement?.id;
                const focusSection = document.activeElement?.closest('[data-outline-section]')?.dataset.outlineSection;
                const focusPath = document.activeElement?.dataset.explorerPath;
                const focusHref = document.activeElement?.getAttribute('href');
                if (kind === 'outline') {
                    // Capture the user's latest choice, including changes made during the request.
                    if (this.outline.dataset.loaded) this.rememberView(kind);
                    const view = this.outlineView();
                    next.querySelectorAll('details[data-outline-section]').forEach(section => {
                        section.open = !view.collapsed.includes(section.dataset.outlineSection);
                    });
                    this.outline.dataset.loaded = 'true';
                } else {
                    const canonicalPath = next.querySelector('#current-explorer-path')?.value || '';
                    if (canonicalPath !== this.state.files.path) this.state.files.scrollTop = 0;
                    this.state.files.path = canonicalPath;
                    this.requestedFilesPath = canonicalPath;
                }
                current.replaceWith(next);
                window.htmx?.process(next);
                this.restoreScroll(kind);
                if (focused) {
                    // A refresh should not silently drop keyboard focus onto the document body.
                    const matching = focusId ? byId(focusId)
                        : focusSection ? next.querySelector(`[data-outline-section="${CSS.escape(focusSection)}"] summary`)
                        : focusPath !== undefined ? next.querySelector(`[data-explorer-path="${CSS.escape(focusPath)}"]`)
                        : focusHref ? next.querySelector(`a[href="${CSS.escape(focusHref)}"]`) : null;
                    (matching || next.querySelector('button, summary, a'))?.focus({preventScroll: true});
                }
                this.stale[kind] = false;
                this.persist();
                this.status(kind);
            } catch (error) {
                if (this.jobs[kind] === job && error.name !== 'AbortError') {
                    this.status(kind, 'Could not update. Your previous view is still shown.', true);
                }
            } finally {
                if (this.jobs[kind] === job) {
                    delete this.jobs[kind];
                    byId(panelIds[kind]).setAttribute('aria-busy', 'false');
                }
            }
        }

        invalidateFiles(paths = null) {
            if (paths && !paths.includes(this.requestedFilesPath)) return;
            this.stale.files = true;
            if (this.leftPanel === 'files') this.refresh('files');
        }

        invalidateOutline() {
            this.stale.outline = true;
            if (this.leftPanel === 'outline') this.refresh('outline');
        }

        refreshVisible() {
            this.stale.files = this.stale.outline = true;
            if (this.leftPanel) this.refresh(this.leftPanel);
        }

        bind() {
            document.addEventListener('click', event => {
                const folder = event.target.closest('[data-explorer-path]');
                if (folder) this.refresh('files', folder.dataset.explorerPath);
                const retry = event.target.closest('[data-panel-retry]');
                if (retry) this.refresh(retry.dataset.panelRetry);
                if (event.target.closest('#file-list-content a[href]')
                    && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
                    this.revealContent();
                }
            });
            document.addEventListener('toggle', event => {
                if (event.target.matches('details[data-outline-section]') && this.outline?.dataset.loaded) {
                    this.rememberView('outline');
                    this.persist();
                }
            }, true);
            document.addEventListener('scroll', event => {
                const kind = event.target.matches?.('.file-explorer-content') ? 'files'
                    : event.target.id === 'outline-content' ? 'outline' : null;
                if (!kind || this.leftPanel !== kind) return;
                this.rememberView(kind);
                clearTimeout(this.saveTimer);
                this.saveTimer = setTimeout(() => this.persist(), 100);
            }, true);
            document.addEventListener('outline-refresh', () => this.invalidateOutline());
            document.addEventListener('visibilitychange', () => {
                if (!document.hidden) this.refreshVisible();
            });
            window.addEventListener('online', () => this.refreshVisible());
            window.addEventListener('focus', () => this.refreshVisible());
            window.addEventListener('pageshow', event => { if (event.persisted) this.refreshVisible(); });
            window.addEventListener('pagehide', () => {
                if (this.leftPanel) this.rememberView(this.leftPanel);
                this.persist();
                this.cancel('files');
                this.cancel('outline');
            });
            this.narrow.addEventListener('change', () => {
                this.render();
                if (this.narrow.matches && this.leftPanel && document.activeElement?.closest('[inert]')) {
                    byId(panelIds[this.leftPanel]).querySelector('button')?.focus();
                }
            });
            document.addEventListener('keydown', event => {
                const panel = this.settingsOpen ? byId('settings-sidebar')
                    : this.narrow.matches && this.leftPanel ? byId(panelIds[this.leftPanel]) : null;
                // Do not trap focus belonging to another dialog above this panel.
                if (event.key !== 'Tab' || !panel?.contains(document.activeElement)) return;
                const targets = [...panel.querySelectorAll('button, a[href], input, select, textarea, summary, [tabindex="0"]')]
                    .filter(element => !element.disabled && !element.closest('[inert]') && element.getClientRects().length);
                const first = targets[0], last = targets[targets.length - 1];
                if (event.shiftKey && document.activeElement === first) {
                    event.preventDefault(); last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                    event.preventDefault(); first?.focus();
                }
            });
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        if (byId('file-explorer-sidebar')) window.dialengWorkspaceUI = new WorkspaceUI();
    });
})();
