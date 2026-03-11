# How It Works - Technical Documentation

This folder contains detailed technical documentation explaining how various features of Dialeng are implemented.

## Recommended Reading Order

Read the documents in numerical order for the best understanding of the system:

| # | Document | Description |
|---|----------|-------------|
| 1 | [01_state_management.md](01_state_management.md) | **Start here** - How notebooks and cells are stored, loaded, saved, and synchronized |
| 2 | [02_cell_types.md](02_cell_types.md) | The three cell types (Code, Note, Prompt) - data model, rendering, execution, serialization |
| 3 | [03_real_time_collaboration.md](03_real_time_collaboration.md) | WebSocket-based real-time collaboration - OOB swaps, broadcasting, conflict avoidance |
| 4 | [04_kernel_execution.md](04_kernel_execution.md) | Subprocess kernel with streaming output, hard interrupt (SIGINT), and execution queue |
| 5 | [05_dialoghelper_integration.md](05_dialoghelper_integration.md) | DialogHelper library integration - remote notebook API, headless interaction |
| 6 | [06_llm_integration.md](06_llm_integration.md) | LLM service integration - Claude, context building, streaming responses |
| 7 | [07_code_organization.md](07_code_organization.md) | **Architecture guide** - Directory structure, `core/` vs `services/` distinction, how to extend |
| 8 | [08_extension_system.md](08_extension_system.md) | Extension system - type dispatch, 2-way callbacks, creating extensions |
| 10 | [10_tool_calling.md](10_tool_calling.md) | Tool calling - LLM tool usage and integration |
| 11 | [11_shell_integration.md](11_shell_integration.md) | Shell commands - pshnb magic, shell cells, Safe Mode with safecmd |
| 12 | [12_colab_kernel.md](12_colab_kernel.md) | Google Colab kernel - remote execution via Jupyter wire protocol over WebSocket |
| 13 | [13_colab_dialoghelper_proxy.md](13_colab_dialoghelper_proxy.md) | DialogHelper proxy for Colab - auto-install, stdin proxy, monkey-patching |
| 14 | [14_package_structure.md](14_package_structure.md) | uv project setup - pyproject.toml, entry points, configurable paths |
| 15 | [15_extension_registries.md](15_extension_registries.md) | Kernel, provider, toolbar, and settings registries |
| 16 | [16_craft_template_autorun.md](16_craft_template_autorun.md) | CRAFT, TEMPLATE, and AUTORUN folder support |
| 17 | [17_editor_cell_transitions.md](17_editor_cell_transitions.md) | Monaco editor lifecycle, cell transitions, scroll preservation, FOUST known issue |

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
| Learn the interface / UI | [GUI Overview](../guides/gui_overview.md) |
| Understand the data model | [01_state_management.md](01_state_management.md) |
| Add a new cell type | [08_extension_system.md](08_extension_system.md) → [07_code_organization.md](07_code_organization.md) |
| Create a custom callback | [08_extension_system.md](08_extension_system.md) |
| Understand `core/` vs `services/` | [07_code_organization.md](07_code_organization.md) (Section 8) |
| Fix collaboration bugs | [03_real_time_collaboration.md](03_real_time_collaboration.md) |
| Add a new cell field | [01_state_management.md](01_state_management.md) → [02_cell_types.md](02_cell_types.md) |
| Understand persistence | [01_state_management.md](01_state_management.md) |
| Add new storage backend | [01_state_management.md](01_state_management.md) |
| Debug code execution | [04_kernel_execution.md](04_kernel_execution.md) |
| Add streaming outputs | [04_kernel_execution.md](04_kernel_execution.md) |
| Implement cell queue | [04_kernel_execution.md](04_kernel_execution.md) |
| Colab kernel internals | [12_colab_kernel.md](12_colab_kernel.md) |
| Add remote kernel backend | [12_colab_kernel.md](12_colab_kernel.md) → [15_extension_registries.md](15_extension_registries.md) |
| Colab OAuth & auth | [12_colab_kernel.md](12_colab_kernel.md) |
| DialogHelper on Colab | [13_colab_dialoghelper_proxy.md](13_colab_dialoghelper_proxy.md) |
| Colab stdin proxy | [13_colab_dialoghelper_proxy.md](13_colab_dialoghelper_proxy.md) |
| Set up own Colab OAuth | [Colab OAuth Setup Guide](../guides/colab_oauth_setup.md) |
| Remote notebook API | [05_dialoghelper_integration.md](05_dialoghelper_integration.md) |
| LLM integration | [06_llm_integration.md](06_llm_integration.md) |
| Headless interaction | [05_dialoghelper_integration.md](05_dialoghelper_integration.md) |
| Run shell commands | [11_shell_integration.md](11_shell_integration.md) |
| Add shell cell type | [11_shell_integration.md](11_shell_integration.md) |
| Safe Mode validation | [11_shell_integration.md](11_shell_integration.md) |
| Install as uv project | [14_package_structure.md](14_package_structure.md) |
| Add custom kernel type | [15_extension_registries.md](15_extension_registries.md) |
| Add LLM provider | [15_extension_registries.md](15_extension_registries.md) |
| Add toolbar/settings extension | [15_extension_registries.md](15_extension_registries.md) |
| CRAFT context injection | [16_craft_template_autorun.md](16_craft_template_autorun.md) |
| TEMPLATE notebooks | [16_craft_template_autorun.md](16_craft_template_autorun.md) |
| AUTORUN extensions | [16_craft_template_autorun.md](16_craft_template_autorun.md) |
| Editor flash / transition bugs | [17_editor_cell_transitions.md](17_editor_cell_transitions.md) |
| Scroll jumping on cell operations | [17_editor_cell_transitions.md](17_editor_cell_transitions.md) |
| Monaco editor lifecycle | [17_editor_cell_transitions.md](17_editor_cell_transitions.md) |

## Related Documentation

- [../../README.md](../../README.md) - Project overview and quick start
- [../guides/gui_overview.md](../guides/gui_overview.md) - Visual guide to every section of the interface
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
