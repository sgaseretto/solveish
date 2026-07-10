(function initializeDialengNotebookClient(global) {
    class RevisionTracker {
        constructor(streams = []) {
            this.revisions = Object.fromEntries(streams.map(stream => [stream, 0]));
            this.seenEventIds = new Set();
            this.seenEventOrder = [];
        }

        remember(eventId) {
            if (!eventId) return;
            this.seenEventIds.add(eventId);
            this.seenEventOrder.push(eventId);
            if (this.seenEventOrder.length > 1000) {
                this.seenEventIds.delete(this.seenEventOrder.shift());
            }
        }

        recordSnapshot(snapshot, streams = null) {
            const revisions = snapshot?.revisions || {};
            const entries = streams
                ? streams.map(stream => [stream, revisions[stream]])
                : Object.entries(revisions);
            entries.forEach(([stream, revision]) => {
                const numericRevision = Number(revision);
                if (Number.isFinite(numericRevision)) {
                    this.revisions[stream] = Math.max(
                        this.revisions[stream] || 0,
                        numericRevision,
                    );
                }
            });
            if (snapshot?.stream && Number.isFinite(Number(snapshot.revision))) {
                this.revisions[snapshot.stream] = Math.max(
                    this.revisions[snapshot.stream] || 0,
                    Number(snapshot.revision),
                );
            }
        }

        accept(event) {
            if (!event || event.type === 'heartbeat_ack' || event.type === 'notebook_deleted') {
                return {accepted: true};
            }
            if (event.event_id && this.seenEventIds.has(event.event_id)) {
                return {accepted: false, duplicate: true};
            }
            // Kernel snapshots contain diagnostics for other revisions but are
            // authoritative only for their own stream.
            if (event.type === 'kernel_snapshot') {
                this.remember(event.event_id);
                return {accepted: true};
            }
            const stream = event.stream;
            const revision = Number(event.revision);
            if (!stream || !Number.isFinite(revision)) return {accepted: true};
            const lastRevision = this.revisions[stream] || 0;
            if (revision <= lastRevision) {
                return {accepted: false, duplicate: true};
            }
            const revisionStart = Number(event.revision_start ?? revision);
            if (!event.authoritative && revisionStart > lastRevision + 1) {
                return {
                    accepted: false,
                    gap: {stream, from: lastRevision, to: revision},
                };
            }
            this.revisions[stream] = revision;
            this.remember(event.event_id);
            return {accepted: true};
        }
    }

    class DocumentRevisionTracker {
        constructor() {
            this.revisions = {notebook: 0, structure: 0};
        }

        recordNotebook(notebook) {
            this._record('notebook', notebook?.revision);
            this._record('structure', notebook?.structure_revision);
        }

        recordEvent(event) {
            this._record('notebook', event?.notebook_revision);
            this._record('structure', event?.structure_revision);
        }

        _record(name, revision) {
            const numericRevision = Number(revision);
            if (Number.isFinite(numericRevision)) {
                this.revisions[name] = Math.max(
                    this.revisions[name] || 0,
                    numericRevision,
                );
            }
        }
    }

    class RunIntentTracker {
        mark(button, {advance = true} = {}) {
            if (!button?.dataset) return;
            button.dataset.runAdvance = advance ? 'true' : 'false';
        }

        consume(button, {defaultAdvance = true} = {}) {
            const encoded = button?.dataset?.runAdvance;
            if (button?.dataset) delete button.dataset.runAdvance;
            if (encoded === 'true') return true;
            if (encoded === 'false') return false;
            return defaultAdvance;
        }
    }

    global.DialengRevisionTracker = RevisionTracker;
    global.DialengDocumentRevisionTracker = DocumentRevisionTracker;
    global.DialengRunIntentTracker = RunIntentTracker;
})(typeof window === 'undefined' ? globalThis : window);
