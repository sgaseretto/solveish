# Migrating AUTORUN Extensions

`AUTORUN` remains supported for compatibility and for background notebooks. New
server-process extensions should use `.dialeng/extensions/`, a manifest, and the
public `ExtensionAPI`.

Legacy `AUTORUN/*.py` and exported notebook modules are loaded transactionally
under synthetic owners such as `legacy.autorun.notebook.example`. They appear in
Extension Manager diagnostics, and removing a legacy file removes its owned
registrations on reload. Background AUTORUN notebook execution remains separate.

Create a non-destructive migration copy:

```bash
uv run dialeng extension migrate notebooks/AUTORUN/example.ipynb --path notebooks
```

The command creates a project extension without changing the original notebook.
Then:

1. Review the generated `extension.toml` and `extension.ipynb`.
2. Replace direct registry decorators with `setup(api, ctx)` registrations.
3. Run `dialeng extension build PATH`.
4. Run `dialeng extension prepare PATH`.
5. Start Dialeng, review trust, and enable the project extension.
6. Confirm all UI, execution, and cleanup behavior.
7. Remove or archive the original AUTORUN server-extension cells only after the
   manifest extension is accepted.

Do not migrate AUTORUN cells whose purpose is background kernel execution; that
is a different lifecycle and remains in `AUTORUN`.
