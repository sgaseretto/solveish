# Install And Initialize A Dialeng Project

## Install From GitHub

Install Dialeng as a global `uv` tool:

```bash
uv tool install "dialeng @ git+https://github.com/sgaseretto/solveish.git"
dialeng init ./my-project --name my_pkg
dialeng ./my-project
```

For a project-local dependency:

```bash
uv add "dialeng @ git+https://github.com/sgaseretto/solveish.git"
uv run dialeng init . --name my_pkg
uv run dialeng .
```

## What `dialeng init` Creates

The command initializes and exits. It creates or merges:

- `dialeng_config.json`
- `CRAFT.ipynb`
- the configured Python package directory and `__init__.py`
- `.dialeng/skills/README.md`
- `.gitignore` entries for private recovery snapshots and staged notebook-to-skill builds
- `[tool.dialeng] lib_name` in `pyproject.toml`

The command is idempotent. Existing configuration and user-authored CRAFT cells
are preserved; only generated CRAFT cells are refreshed. The older
`dialeng --init [NAME]` command remains available and starts the server after
initialization.
