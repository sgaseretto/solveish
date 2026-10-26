# Dialeng Documentation

Welcome to the Dialeng documentation. Start with [How to Use a Dialog](guides/how_to_use_a_dialog.md)
for the user workflow and [How a Dialog Works](how_it_works/00_dialog_overview.md)
for an overview of the internals. The guides cover using Dialeng; the technical
deep dives explain its implementation.

## Documentation Structure

```
docs/
├── README.md                       # This file - documentation overview
├── guides/
│   ├── how_to_use_a_dialog.md      # First dialog: notes, code, AI, checks, and saving
│   ├── gui_overview.md             # Visual guide to the notebook interface
│   ├── notebook_regression_qa.md   # Repeatable notebook interaction QA suite
│   ├── colab_oauth_setup.md        # Google Colab OAuth configuration
│   ├── autorun_extensions.md      # Writing AUTORUN extensions (tutorial)
│   ├── extensions.md              # Manifest extensions, Studio, API, trust, packaging
│   ├── migrating_autorun_extensions.md # Legacy migration workflow
│   └── notebook_to_package.md    # Explore → Reuse → Package progression
└── how_it_works/                   # Deep dives into feature implementations
    ├── README.md                   # Index with recommended reading order
    ├── 00_dialog_overview.md       # End-to-end document, kernel, and AI lifecycle
    ├── 01_state_management.md      # Notebook/cell state, persistence
    ├── 02_cell_types.md            # Code, Note, Prompt cells
    └── 03_real_time_collaboration.md  # WebSocket collaboration
```

## Quick Links

### For Users

- [How to Use a Dialog](guides/how_to_use_a_dialog.md) - What a dialog is, which cell to use, and a complete worked example
- [Main README](../README.md) - Quick start, features, keyboard shortcuts
- [GUI Overview](guides/gui_overview.md) - Visual guide to every section of the interface
- [Notebook Regression QA](guides/notebook_regression_qa.md) - Manual interaction, collaboration, execution, and output regression suite
- [Writing AUTORUN Extensions](guides/autorun_extensions.md) - Tutorial for creating extensions from notebooks
- [Building Extensions](guides/extensions.md) - Public API, Studio, trust, resources, and packaging
- [Migrating AUTORUN Extensions](guides/migrating_autorun_extensions.md) - Non-destructive migration to manifests
- [Notebook to Package](guides/notebook_to_package.md) - Three-phase progression from exploration to published package
- [Codex Skills](guides/codex_skills.md) - Project skills and reviewable notebook-to-skill creation

### For Developers

- [How a Dialog Works](how_it_works/00_dialog_overview.md) - How document state, kernel execution, AI context, collaboration, and persistence fit together
- [DEVELOPERS.md](../DEVELOPERS.md) - Extension points, patterns, architecture
- [ROADMAP.md](../ROADMAP.md) - Planned features and contribution areas
- [Performance, Reliability, and UX Roadmap](superpowers/plans/2026-07-09-performance-reliability-ux-roadmap.md) - Implemented roadmap and acceptance targets for startup, synchronization, runtime state, rendering, persistence, and diagnostics
- [CHANGELOG.md](../CHANGELOG.md) - Version history and changes

### Technical Deep Dives (Recommended Reading Order)

| # | Document | Description |
|---|----------|-------------|
| 0 | [How a Dialog Works](how_it_works/00_dialog_overview.md) | Start here for the overall lifecycle and implementation map |
| 1 | [State Management](how_it_works/01_state_management.md) | How notebooks/cells are stored, loaded, saved |
| 2 | [Cell Types](how_it_works/02_cell_types.md) | Code, Note, Prompt - data model, rendering, execution |
| 3 | [Real-time Collaboration](how_it_works/03_real_time_collaboration.md) | WebSocket architecture, OOB swaps, conflict handling |
| 20 | [Multi-provider LLM and Codex](how_it_works/20_multi_provider_llm_and_codex.md) | Provider inventory, Codex OAuth/transport, and project skills |
| 21 | [Performance and Reliability](how_it_works/21_performance_reliability.md) | Runtime lifecycle, revision recovery, persistence, lazy rendering, caches, and diagnostics |
| 24 | [Notebook-to-Skill Creation](how_it_works/24_notebook_skill_creation.md) | Built-in creator, deterministic extraction, staged refinement, and atomic installation |
| 25 | [Extension Platform](how_it_works/25_extension_platform.md) | Discovery, ownership, trust, transactional activation, events, UI, and packaging |
| 26 | [Extension Verification](how_it_works/26_extension_verification.md) | Reproducible unit, stress, E2E, package, security, and manual acceptance baseline |
| 27 | [Extension Runtime And Environments](how_it_works/27_extension_runtime_environments.md) | Isolated hosts, extension lockfiles, command palette, and notebook-native development |

## Contributing Documentation

When writing documentation:

1. **Use clear headings** - Makes documents scannable
2. **Include code examples** - Show, don't just tell
3. **Add line references** - Point to specific code locations (e.g., `app.py:386`)
4. **Document both what and why** - Implementation details AND design decisions
5. **Keep it updated** - Update docs when you change code
