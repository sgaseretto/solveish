(function initializePromptResponseEditor(global) {
    class PromptResponseEditor {
        constructor() {
            this.sessions = new Map();
        }

        register(cellId, value = '') {
            const normalized = String(value ?? '');
            const existing = this.sessions.get(cellId);
            if (!existing) {
                const session = {
                    authoritative: normalized,
                    draft: normalized,
                    editing: false,
                    dirty: false,
                    saving: false,
                    pendingRemote: null,
                };
                this.sessions.set(cellId, session);
                return session;
            }
            if (!existing.editing && !existing.dirty) {
                existing.authoritative = normalized;
                existing.draft = normalized;
                existing.pendingRemote = null;
            }
            return existing;
        }

        session(cellId, initialValue = '') {
            return this.sessions.get(cellId) || this.register(cellId, initialValue);
        }

        begin(cellId, currentValue = '') {
            const current = String(currentValue ?? '');
            const session = this.session(cellId, current);
            session.editing = true;
            session.pendingRemote = null;

            // A response textarea may be stale after a delayed DOM update. The
            // last backend value is safer than presenting an unexplained blank.
            session.draft = !current && session.authoritative
                ? session.authoritative
                : current;
            return session.draft;
        }

        input(cellId, value) {
            const normalized = String(value ?? '');
            const session = this.session(cellId, normalized);
            session.draft = normalized;
            session.dirty = true;
            return session;
        }

        finish(cellId, value) {
            const normalized = String(value ?? '');
            const session = this.session(cellId, normalized);
            session.draft = normalized;
            session.editing = false;
            return session;
        }

        startSave(cellId) {
            const session = this.sessions.get(cellId);
            if (!session || !session.dirty || session.saving) return null;
            session.editing = false;
            session.saving = true;
            return session.draft;
        }

        saveFailed(cellId) {
            const session = this.sessions.get(cellId);
            if (session) session.saving = false;
            return session;
        }

        receive(cellId, value) {
            const normalized = String(value ?? '');
            const session = this.session(cellId, normalized);
            if (session.editing || session.saving) {
                if (session.saving && normalized === session.draft) {
                    this.commit(cellId, normalized);
                    return {apply: true, safeVersion: true, value: normalized};
                }
                if (normalized === session.authoritative) {
                    return {apply: false, safeVersion: true, value: session.draft};
                }
                session.pendingRemote = normalized;
                return {apply: false, safeVersion: false, value: session.draft};
            }
            session.authoritative = normalized;
            session.draft = normalized;
            session.dirty = false;
            session.pendingRemote = null;
            return {apply: true, safeVersion: true, value: normalized};
        }

        commit(cellId, value) {
            const normalized = String(value ?? '');
            const session = this.session(cellId, normalized);
            session.authoritative = normalized;
            session.draft = normalized;
            session.editing = false;
            session.dirty = false;
            session.saving = false;
            session.pendingRemote = null;
            return normalized;
        }

        resolveConflict(cellId, value) {
            return this.commit(cellId, value);
        }

        remove(cellId) {
            this.sessions.delete(cellId);
        }
    }

    global.DialengPromptResponseEditor = PromptResponseEditor;
})(typeof window === 'undefined' ? globalThis : window);
