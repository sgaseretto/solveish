# Dialeng Documentation

Welcome to the Dialeng documentation. This folder contains detailed technical documentation for developers.

## Documentation Structure

```
docs/
├── README.md                       # This file - documentation overview
├── guides/
│   ├── gui_overview.md             # Visual guide to the notebook interface
│   ├── notebook_regression_qa.md   # Repeatable notebook interaction QA suite
│   ├── colab_oauth_setup.md        # Google Colab OAuth configuration
│   ├── autorun_extensions.md      # Writing AUTORUN extensions (tutorial)
│   └── notebook_to_package.md    # Explore → Reuse → Package progression
└── how_it_works/                   # Deep dives into feature implementations
    ├── README.md                   # Index with recommended reading order
    ├── 01_state_management.md      # Notebook/cell state, persistence
    ├── 02_cell_types.md            # Code, Note, Prompt cells
    └── 03_real_time_collaboration.md  # WebSocket collaboration
```

## Quick Links

### For Users
- [Main README](../README.md) - Quick start, features, keyboard shortcuts
- [GUI Overview](guides/gui_overview.md) - Visual guide to every section of the interface
- [Notebook Regression QA](guides/notebook_regression_qa.md) - Manual interaction, collaboration, execution, and output regression suite
- [Writing AUTORUN Extensions](guides/autorun_extensions.md) - Tutorial for creating extensions from notebooks
- [Notebook to Package](guides/notebook_to_package.md) - Three-phase progression from exploration to published package

### For Developers
- [DEVELOPERS.md](../DEVELOPERS.md) - Extension points, patterns, architecture
- [ROADMAP.md](../ROADMAP.md) - Planned features and contribution areas
- [Performance, Reliability, and UX Roadmap](superpowers/plans/2026-07-09-performance-reliability-ux-roadmap.md) - Implemented roadmap and acceptance targets for startup, synchronization, runtime state, rendering, persistence, and diagnostics
- [CHANGELOG.md](../CHANGELOG.md) - Version history and changes

### Technical Deep Dives (Recommended Reading Order)

| # | Document | Description |
|---|----------|-------------|
| 1 | [State Management](how_it_works/01_state_management.md) | How notebooks/cells are stored, loaded, saved |
| 2 | [Cell Types](how_it_works/02_cell_types.md) | Code, Note, Prompt - data model, rendering, execution |
| 3 | [Real-time Collaboration](how_it_works/03_real_time_collaboration.md) | WebSocket architecture, OOB swaps, conflict handling |
| 20 | [Multi-provider LLM and Codex](how_it_works/20_multi_provider_llm_and_codex.md) | Provider inventory, Codex OAuth/transport, and project skills |
| 21 | [Performance and Reliability](how_it_works/21_performance_reliability.md) | Runtime lifecycle, revision recovery, persistence, lazy rendering, caches, and diagnostics |

## Contributing Documentation

When writing documentation:

1. **Use clear headings** - Makes documents scannable
2. **Include code examples** - Show, don't just tell
3. **Add line references** - Point to specific code locations (e.g., `app.py:386`)
4. **Document both what and why** - Implementation details AND design decisions
5. **Keep it updated** - Update docs when you change code
