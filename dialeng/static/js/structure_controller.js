(function initializeStructureController(global) {
    class StructureController {
        plan(orderedIds, resolveUnit) {
            if (!Array.isArray(orderedIds)) {
                return {valid: false, reason: 'invalid_order', units: []};
            }
            if (new Set(orderedIds).size !== orderedIds.length) {
                return {valid: false, reason: 'duplicate_id', units: []};
            }
            const units = orderedIds.map(cellId => resolveUnit(cellId));
            if (units.some(unit => !unit?.cellEl || !unit?.addRowEl)) {
                return {valid: false, reason: 'missing_unit', units: []};
            }
            return {valid: true, reason: null, units};
        }
    }

    global.DialengStructureController = StructureController;
    global.dialengStructure = new StructureController();
})(typeof window === 'undefined' ? globalThis : window);
