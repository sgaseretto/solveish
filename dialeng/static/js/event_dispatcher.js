(function initializeDialengEventDispatcher(global) {
    class EventDispatcher {
        constructor() {
            this.handlers = new Map();
        }

        register(eventType, handler) {
            this.handlers.set(eventType, handler);
            return () => this.handlers.delete(eventType);
        }

        dispatch(event) {
            const handler = this.handlers.get(event?.type);
            if (!handler) return false;
            try {
                const result = handler(event);
                if (result?.catch) result.catch(error => {
                    console.error(`[WS] Async handler failed (${event.type}):`, error);
                });
            } catch (error) {
                console.error(`[WS] Handler failed (${event.type}):`, error);
            }
            return true;
        }
    }

    global.DialengEventDispatcher = EventDispatcher;
    global.dialengEvents = new EventDispatcher();
})(window);
