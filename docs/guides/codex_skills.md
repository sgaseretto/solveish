# Project Skills For OpenAI Codex

Dialeng discovers skills only from the current project's `.dialeng/skills/`
directory. Skills are available to OpenAI Codex prompt cells and are not enabled
for Claude providers in this release.

Create one directory per skill:

```text
.dialeng/skills/
└── release-notes/
    ├── SKILL.md
    └── references.md
```

`SKILL.md` must start with YAML frontmatter:

```markdown
---
name: release-notes
description: Prepare release notes from a set of repository changes.
---

Read `references.md`, group user-facing changes, and call out migrations.
```

Dialeng places only skill names and descriptions in the Codex system prompt.
When a description matches the task, Codex can call `read_skill(name, path)` to
load `SKILL.md` or another resource inside that skill directory. Full skill text
is never injected into notebook-authored context.

Set `disable-model-invocation: true` in frontmatter to keep a skill registered
but omit it from automatic model discovery. Skill files are rescanned before
prompt execution, so edits do not require a server restart.

Paths are resolved before reading. Symlink escapes, traversal, duplicate names,
oversized files, and malformed frontmatter are ignored or rejected and logged as
non-fatal diagnostics.
