# Dialeng Documentation

Welcome to the Dialeng documentation. This folder contains detailed technical documentation for developers.

## Documentation Structure

```
docs/
├── README.md                       # This file - documentation overview
└── how_it_works/                   # Deep dives into feature implementations
    ├── README.md                   # Index with recommended reading order
    ├── 01_state_management.md      # Notebook/cell state, persistence
    ├── 02_cell_types.md            # Code, Note, Prompt cells
    └── 03_real_time_collaboration.md  # WebSocket collaboration
```

## Quick Links

### For Users
- [Main README](../README.md) - Quick start, features, keyboard shortcuts

### For Developers
- [DEVELOPERS.md](../DEVELOPERS.md) - Extension points, patterns, architecture
- [ROADMAP.md](../ROADMAP.md) - Planned features and contribution areas
- [CHANGELOG.md](../CHANGELOG.md) - Version history and changes

### Technical Deep Dives (Recommended Reading Order)

| # | Document | Description |
|---|----------|-------------|
| 1 | [State Management](how_it_works/01_state_management.md) | How notebooks/cells are stored, loaded, saved |
| 2 | [Cell Types](how_it_works/02_cell_types.md) | Code, Note, Prompt - data model, rendering, execution |
| 3 | [Real-time Collaboration](how_it_works/03_real_time_collaboration.md) | WebSocket architecture, OOB swaps, conflict handling |

## Contributing Documentation

When writing documentation:

1. **Use clear headings** - Makes documents scannable
2. **Include code examples** - Show, don't just tell
3. **Add line references** - Point to specific code locations (e.g., `app.py:386`)
4. **Document both what and why** - Implementation details AND design decisions
5. **Keep it updated** - Update docs when you change code
