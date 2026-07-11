# Project Skills For OpenAI Codex

Dialeng loads packaged built-in skills and project-owned skills from the current
project's `.dialeng/skills/` directory. A project skill with the same name as a
built-in overrides it. Skills are available to OpenAI Codex prompt cells and are
not enabled for Claude providers in this release.

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

## Creating A Skill From Project Sources

Dialeng ships a built-in `skill-creator` that uses a staged, review-first
workflow. Open the sparkles button in the notebook toolbar, then:

1. Enter a lowercase skill name and a description that states when it applies.
2. Select one or more project notebooks or supported text/code files, and add
   optional HTTP(S) URLs one per line. URLs are converted through `web2md`.
3. Leave outputs excluded unless their text is essential to the skill.
4. Generate the draft, inspect `SKILL.md`, every `references/*.md` file, and any
   redaction or validation warnings.
5. Install only after the preview is correct.

The same workflow is available from the CLI:

```bash
uv run dialeng skill create data-analysis \
  --from lessons/01_data.ipynb lessons/02_workflow.ipynb \
  --description "Use when analyzing the project's tabular datasets." \
  --instructions "Prefer the validation workflow demonstrated in lesson 2."
```

By default, Dialeng asks the selected provider to refine the deterministic
draft. Use `--no-llm` to keep the deterministic version and `--yes` for
non-interactive installation. `dialeng skill update` uses the same pipeline but
explicitly replaces an existing skill. `dialeng skill list` and
`dialeng skill validate` inspect the project.

Notebook extraction preserves Note content, Code source, Prompt user input, and
the currently edited assistant response. Skipped cells are excluded. Text
outputs are excluded by default, rich/binary outputs are never copied, likely
credentials are redacted, and total source size is bounded by
`skills.max_source_bytes`.

Generated skills contain:

```text
.dialeng/skills/data-analysis/
├── SKILL.md
├── .dialeng-sources.json
└── references/
    ├── lessons-01-data-<hash>.md
    └── lessons-02-workflow-<hash>.md
```

`.dialeng-sources.json` records source kind, path or URL, content hash, and web
fetch timestamp for provenance. Temporary previews live under
`.dialeng/skill-builds/`, may contain private source context, and are ignored by
project initialization. `refresh_skill_build_sources` refreshes a staged build's
references while preserving reviewed instructions; inspect its new hash before
committing.

The model-facing creator tools are `build_skill_from_notebooks`,
`inspect_skill_build`, `update_skill_build`, `refresh_skill_build_sources`, and
`commit_skill_build`. The final
commit requires the hash from the latest preview, preventing a stale review from
installing a changed build.
