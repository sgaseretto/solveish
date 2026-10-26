# Project Skills For OpenAI Codex

Dialeng loads packaged built-in skills, enabled extension skills, and
project-owned skills from `.agents/skills/` and `.dialeng/skills/`. Skills are
available to OpenAI Codex prompt cells and are not enabled for Claude providers
in this release.

For matching frontmatter names, precedence is:

1. `.dialeng/skills/`: Dialeng-specific project overrides.
2. `.agents/skills/`: shared project skills used by other agents too.
3. Packaged built-ins and enabled extension skills, using their existing discovery
   order (built-ins first; later duplicate names are diagnosed and skipped).

Duplicates within either project directory are also diagnosed and skipped. An
invalid override is skipped, leaving a valid lower-precedence skill available.
Only the selected project root is scanned, not parent projects or `~/.agents/`.
An existing `.agents/skills/<name>/SKILL.md` works without copying or configuration
when project skills are enabled. `dialeng skill list` includes shared skills with
source `agents`; Dialeng-local skills retain source `project`.

Create one directory per skill under either root:

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
but omit it from automatic model discovery and block `read_skill` access. A
disabled override still masks lower-precedence skills with the same name.
Skill files in both project roots are rescanned before
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
4. Choose the original skill location and whether to link it in the other folder.
5. Generate the draft, inspect its destination, `SKILL.md`, every `references/*.md` file, and any
   redaction or validation warnings.
6. Install only after the preview is correct.

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

## Original Location And Symlinks

By default, creation and update install the original into `.dialeng/skills/`
without a link. To share newly created skills with other agents, configure the
project's `dialeng_config.json`:

```json
{
  "skills": {
    "install_root": "agents",
    "link_other_root": true
  }
}
```

This creates one original and a relative directory symlink:

```text
.agents/skills/data-analysis/          # original SKILL.md and references
.dialeng/skills/data-analysis         # -> ../../.agents/skills/data-analysis
```

`install_root` accepts `agents` or `dialeng`; the latter reverses the arrangement.
`link_other_root: false` installs just the original. Dialeng already discovers
both roots, so the link is optional; it makes the same files available to tools
that only inspect the other folder. Linked skills are discovered from their
original directory, without a duplicate catalog entry. Relative links continue
working when the whole project is moved.

The settings panel exposes **Created skill location** and **Link skills in the
other folder**. The creator modal initializes from those settings and lets you
override them for a new draft. CLI equivalents are:

```bash
uv run dialeng skill create data-analysis \
  --from lessons/01_data.ipynb \
  --description "Use when analyzing project datasets." \
  --install-root agents --link-other-root
```

Use `--no-link-other-root` to override a project setting that enables linking.
The model-facing `build_skill_from_notebooks` tool accepts `install_root` and
`link_other_root` too; omitted values use the project settings. The build records
these choices, and its preview/hash covers both the files and installation plan.
Settings or form changes after staging apply to future drafts, not the reviewed
one. Generate a new draft to change its destination.

Updates must select the original's root. Updating the original also updates what
the symlink exposes. Existing independent skills or unrelated links in the other
folder are never replaced, even by `skill update`; installation reports a
conflict. If link creation fails, installation rolls back and leaves the draft
available for retry. Existing files are not moved when changing settings, and
turning linking off does not remove existing links. To keep a separate
Dialeng-specific override, install into `.dialeng/skills/` with linking disabled.
