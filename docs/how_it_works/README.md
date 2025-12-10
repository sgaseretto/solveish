# How It Works - Technical Documentation

This folder contains detailed technical documentation explaining how various features of LLM Notebook are implemented.

## Recommended Reading Order

Read the documents in numerical order for the best understanding of the system:

| # | Document | Description |
|---|----------|-------------|
| 1 | [01_state_management.md](01_state_management.md) | **Start here** - How notebooks and cells are stored, loaded, saved, and synchronized |
| 2 | [02_cell_types.md](02_cell_types.md) | The three cell types (Code, Note, Prompt) - data model, rendering, execution, serialization |
| 3 | [03_real_time_collaboration.md](03_real_time_collaboration.md) | WebSocket-based real-time collaboration - OOB swaps, broadcasting, conflict avoidance |
| 4 | [04_kernel_execution.md](04_kernel_execution.md) | Subprocess kernel with streaming output, hard interrupt (SIGINT), and execution queue |

## Purpose

These documents are intended for developers who want to:
- Understand how features are implemented
- Debug issues in specific subsystems
- Extend or improve existing functionality
- Contribute to the project

## Quick Reference

### If you want to...

| Goal | Start with |
|------|------------|
| Understand the data model | [01_state_management.md](01_state_management.md) |
| Add a new cell type | [02_cell_types.md](02_cell_types.md) |
| Fix collaboration bugs | [03_real_time_collaboration.md](03_real_time_collaboration.md) |
| Add a new cell field | [01_state_management.md](01_state_management.md) → [02_cell_types.md](02_cell_types.md) |
| Understand persistence | [01_state_management.md](01_state_management.md) |
| Add new storage backend | [01_state_management.md](01_state_management.md) |
| Debug code execution | [04_kernel_execution.md](04_kernel_execution.md) |
| Add streaming outputs | [04_kernel_execution.md](04_kernel_execution.md) |
| Implement cell queue | [04_kernel_execution.md](04_kernel_execution.md) |

## Related Documentation

- [../../README.md](../../README.md) - Project overview and quick start
- [../../DEVELOPERS.md](../../DEVELOPERS.md) - General developer guide
- [../../ROADMAP.md](../../ROADMAP.md) - Planned features
- [../../CHANGELOG.md](../../CHANGELOG.md) - Version history

## Adding New Documentation

When adding new "how it works" documentation:

1. Create a new `.md` file with a number prefix (e.g., `04_new_feature.md`)
2. Follow the structure of existing documents:
   - Table of Contents
   - Architecture Overview
   - Key Concepts
   - Implementation Details
   - How to Extend/Improve
3. Update this README to include the new document
4. Cross-reference from other docs where relevant
5. Update the CHANGELOG
