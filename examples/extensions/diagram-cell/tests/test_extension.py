from extension import DiagramCell, deserialize_diagram, serialize_diagram


def test_diagram_round_trip():
    cell = DiagramCell(id="diagram-1", source="A --> B")
    restored = deserialize_diagram(serialize_diagram(cell))
    assert restored.id == cell.id
    assert restored.source == cell.source
