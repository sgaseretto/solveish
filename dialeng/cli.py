"""CLI entry point for dialeng.

Usage:
    dialeng                    # Start server rooted at current directory
    dialeng ./my-project       # Start server rooted at ./my-project
    dialeng /abs/path          # Start server rooted at absolute path
    dialeng --port 9000        # Custom port
    dialeng --init             # Start server + create CRAFT.ipynb (auto-detect name)
    dialeng --init my_pkg      # Start server + create CRAFT.ipynb with package name
    dialeng new_folder --init  # Create folder, init CRAFT, start server
    dialeng init [path]        # Initialize a project and exit
    dialeng package init       # Scaffold nbdev-compatible package
    dialeng skill create NAME  # Create a project skill from notebooks
"""
import argparse
import os
import sys
from pathlib import Path

# Subcommand names that should not be treated as path arguments
_SUBCOMMANDS = {"init", "package", "skill", "extension", "doctor"}


def parse_args(argv=None):
    """Parse CLI arguments.

    Handles the ambiguity between positional `path` and subcommands by
    checking whether the first positional argument is a known subcommand
    name before parsing.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Determine if we're invoking a subcommand or the default serve mode.
    # We need to check this up front because argparse can't cleanly handle
    # both an optional positional arg and subparsers at the same level.
    has_subcommand = len(argv) > 0 and argv[0] in _SUBCOMMANDS

    if has_subcommand:
        return _parse_subcommand(argv)
    else:
        return _parse_serve(argv)


def _parse_serve(argv):
    """Parse args for the default serve mode: dialeng [path] [--port PORT] [--init [NAME]]."""
    epilog = """\
examples:
  dialeng                       Start server in current directory
  dialeng ./my-project          Start server rooted at ./my-project
  dialeng new_folder            Create new_folder/ and start server there
  dialeng --port 9000           Start server on port 9000
  dialeng --init                Start server + init reuse workflow (auto-detect name)
  dialeng --init my_pkg         Start server + init reuse workflow as "my_pkg"
  dialeng ./project --init      Init reuse workflow in existing project
  dialeng new_folder --init     Create folder, init reuse workflow, start server

  dialeng package init          Scaffold nbdev-compatible package (Phase 3)
  dialeng package init --name x Scaffold with explicit package name

reuse workflow (--init):
  Creates pyproject.toml, CRAFT.ipynb, and a package directory so notebooks
  can share code via #| export directives. The package name is resolved as:
    1. Explicit name from --init NAME
    2. Existing [tool.dialeng] or [tool.nbdev] lib_name in pyproject.toml
    3. Directory name (sanitized to a valid Python identifier)
  Safe to re-run: updates generated cells, preserves your custom cells.

phases:
  Phase 1 (Explore)  Just run dialeng and create notebooks freely.
  Phase 2 (Reuse)    Use --init or the toolbar button to share code between
                     notebooks via #| export. No pip install needed.
  Phase 3 (Package)  Run 'dialeng package init' to add [tool.nbdev] and
                     get nbdev_export, nbdev_test, nbdev_docs, nbdev_pypi.
"""
    parser = argparse.ArgumentParser(
        prog="dialeng",
        description="Notebook-based dialog engine for exploration and development",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=".",
                        help="Root directory for notebooks (default: current directory). "
                             "Created automatically if it doesn't exist.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--init", nargs="?", const=True, default=False, metavar="PKG_NAME",
                        help="Initialize reuse CRAFT.ipynb on startup. "
                             "Optionally provide a package name (default: auto-detect or directory name)")
    parser.add_argument(
        "--extension",
        dest="extension_paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Add a development extension path for this server session (repeatable)",
    )

    args = parser.parse_args(argv)
    args.subcommand = None
    return args


def _parse_subcommand(argv):
    """Parse args when a subcommand is detected."""
    parser = argparse.ArgumentParser(
        prog="dialeng",
        description="Notebook-based dialog engine for exploration and development",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # --- `dialeng init [path]` ---
    project_init = subparsers.add_parser("init", help="Initialize a Dialeng project and exit")
    project_init.add_argument("path", nargs="?", default=".",
                              help="Project directory (default: current directory)")
    project_init.add_argument("--name", dest="project_name", default=None,
                              help="Package/reuse name (default: detect or derive from directory)")

    doctor = subparsers.add_parser("doctor", help="Report project and runtime readiness")
    doctor.add_argument("--path", default=".", help="Dialeng project directory")

    # --- `dialeng package init` ---
    pkg_parser = subparsers.add_parser("package", help="Package management commands")
    pkg_sub = pkg_parser.add_subparsers(dest="package_action")
    init_parser = pkg_sub.add_parser("init", help="Scaffold nbdev-compatible package from notebooks")
    init_parser.add_argument("--name", dest="package_name", default=None,
                             help="Package name (default: directory name)")

    # --- `dialeng skill ...` ---
    skill_parser = subparsers.add_parser("skill", help="Create and manage project skills")
    skill_sub = skill_parser.add_subparsers(dest="skill_action")

    def add_skill_build_arguments(command):
        command.add_argument("name", help="Lowercase skill name")
        command.add_argument("--from", dest="sources", nargs="+", required=True,
                             help="Notebook files, directories, or glob patterns")
        command.add_argument("--description", required=True,
                             help="Specific description of when the skill should be used")
        command.add_argument("--instructions", default="",
                             help="Additional workflow requirements")
        command.add_argument("--include-outputs", action="store_true",
                             help="Include bounded text outputs (off by default)")
        command.add_argument("--install-root", choices=("dialeng", "agents"), default=None,
                             help="Original skill location; defaults to skills.install_root")
        command.add_argument("--link-other-root", action=argparse.BooleanOptionalAction, default=None,
                             help="Create a relative symlink in the other skills folder; defaults to project settings")
        command.add_argument("--provider", default=None,
                             help="LLM provider used to refine the deterministic draft")
        command.add_argument("--model", default=None,
                             help="Model used to refine the deterministic draft")
        command.add_argument("--no-llm", action="store_true",
                             help="Keep the deterministic draft without LLM refinement")
        command.add_argument("--yes", action="store_true",
                             help="Install without an interactive confirmation prompt")
        command.add_argument("--path", default=".", help="Dialeng project directory")

    create_skill = skill_sub.add_parser("create", help="Create a skill from notebooks")
    add_skill_build_arguments(create_skill)
    update_skill = skill_sub.add_parser("update", help="Replace an existing notebook-derived skill")
    add_skill_build_arguments(update_skill)
    list_skills = skill_sub.add_parser("list", help="List discovered skills")
    list_skills.add_argument("--path", default=".", help="Dialeng project directory")
    validate_skill = skill_sub.add_parser("validate", help="Validate discovered skills")
    validate_skill.add_argument("--path", default=".", help="Dialeng project directory")

    # --- `dialeng extension ...` ---
    extension_parser = subparsers.add_parser("extension", help="Create and manage Dialeng extensions")
    extension_sub = extension_parser.add_subparsers(dest="extension_action")
    extension_create = extension_sub.add_parser("create", help="Create a notebook-authored extension")
    extension_create.add_argument("name")
    extension_create.add_argument("--template", choices=("basic", "tool", "cell", "lifecycle", "provider", "package"), default="basic")
    extension_create.add_argument("--id", dest="extension_id", default=None)
    extension_create.add_argument("--path", default=".")
    for action in ("list", "doctor"):
        command = extension_sub.add_parser(action, help=f"{action.title()} project extensions")
        command.add_argument("--path", default=".")
    extension_schema = extension_sub.add_parser("schema", help="Print Extension API JSON schemas")
    extension_schema.add_argument("--output", default="", help="Write the schema catalog to a file")
    extension_schema.add_argument("--path", default=".")
    for action in ("inspect", "trust", "enable", "disable", "reload", "remove"):
        command = extension_sub.add_parser(action, help=f"{action.title()} one extension")
        command.add_argument("extension_id")
        command.add_argument("--path", default=".")
    extension_check = extension_sub.add_parser("check", help="Check a source or locked project for drift")
    extension_check.add_argument("extension_path", nargs="?", default="")
    extension_check.add_argument("--path", default=".")
    for action in ("build", "validate", "test", "prepare", "package", "install", "migrate"):
        command = extension_sub.add_parser(action, help=f"{action.title()} an extension source")
        command.add_argument("extension_path")
        command.add_argument("--path", default=".")
        if action == "test":
            command.add_argument("--browser", action="store_true")
    for action in ("lock", "sync"):
        command = extension_sub.add_parser(action, help=f"{action.title()} the project extension environment")
        command.add_argument("--path", default=".")
        if action == "sync":
            command.add_argument(
                "--offline", action="store_true",
                help="Restore only from project artifacts and the populated extension/uv caches",
            )
    extension_update = extension_sub.add_parser("update", help="Update locked extension coordinates deliberately")
    extension_update.add_argument("extension_id", nargs="?", default="")
    extension_update.add_argument("--path", default=".")

    return parser.parse_args(argv)


def resolve_root_dir(path_str: str, respect_env: bool = True) -> Path:
    """Resolve the root directory from CLI argument or environment.

    Priority: DIALENG_NOTEBOOKS_DIR env var (if respect_env) > CLI path argument.
    Creates the directory if it doesn't exist.
    """
    if respect_env:
        env_dir = os.environ.get("DIALENG_NOTEBOOKS_DIR")
        if env_dir:
            resolved = Path(env_dir).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved

    resolved = Path(path_str).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def cli(argv=None):
    """Main CLI entry point."""
    args = parse_args(argv)

    if args.subcommand == "init":
        from dialeng.services.project_init_service import initialize_project
        root = resolve_root_dir(args.path, respect_env=False)
        print(f"Initializing Dialeng project in {root}...")
        result = initialize_project(root, name=args.project_name)
        if result.get("error"):
            raise SystemExit(result["error"])
        print(f"  Config: {result['config_path']}")
        print(f"  Skills: {result['skills_dir']}")
        print("Dialeng project is ready.")
        return

    if args.subcommand == "package":
        if args.package_action == "init":
            from dialeng.services.package_scaffold_service import scaffold_package
            root = resolve_root_dir(".", respect_env=False)
            scaffold_package(root, package_name=args.package_name)
        else:
            print("Usage: dialeng package init [--name NAME]")
        return

    if args.subcommand == "skill":
        return _run_skill_command(args)

    if args.subcommand == "extension":
        return _run_extension_command(args)

    if args.subcommand == "doctor":
        return _run_doctor(args)

    # Default: start the server
    root_dir = resolve_root_dir(args.path)
    if args.extension_paths:
        os.environ["DIALENG_EXTENSION_PATHS"] = os.pathsep.join(
            str(Path(path).expanduser().resolve()) for path in args.extension_paths
        )

    # --init: create CRAFT.ipynb before starting the server
    if args.init is not False:
        from dialeng.services.craft_init_service import resolve_and_init
        pkg_name = args.init if isinstance(args.init, str) else None
        print(f"Initializing reuse workflow in {root_dir}...")
        resolve_and_init(root_dir, pkg_name=pkg_name)
        print()

    from dialeng.app import main
    main(root_dir=root_dir, port=args.port)


def _run_skill_command(args):
    """Execute project skill commands without starting the Dialeng server."""
    import asyncio
    import json

    from dialeng.services.dialeng_config import load_config
    from dialeng.services.skill_creation_service import get_skill_creation_service
    from dialeng.services.skill_service import set_skill_root

    if not args.skill_action:
        print("Usage: dialeng skill {create,update,list,validate}")
        return
    root = resolve_root_dir(args.path, respect_env=False)
    config = load_config(
        config_path=root / "dialeng_config.json",
        force_reload=True,
        create_if_missing=False,
    )
    service = set_skill_root(root)

    if args.skill_action == "list":
        for skill in service.skills:
            print(f"{skill.name}\t{skill.source}\t{skill.description}")
        for diagnostic in service.diagnostics:
            print(f"INVALID\t{diagnostic.path}\t{diagnostic.message}")
        return

    if args.skill_action == "validate":
        if service.diagnostics:
            for diagnostic in service.diagnostics:
                print(f"INVALID {diagnostic.path}: {diagnostic.message}")
            raise SystemExit(1)
        print(f"Validated {len(service.skills)} skill(s).")
        return

    if not config.skills_creator_enabled:
        raise SystemExit("Skill creation is disabled by skills.creator_enabled")
    creation = get_skill_creation_service()
    preview = creation.create_build(
        args.name,
        args.description,
        args.sources,
        instructions=args.instructions,
        include_outputs=args.include_outputs,
        install_root=args.install_root,
        link_other_root=args.link_other_root,
    )
    if not args.no_llm:
        try:
            preview = asyncio.run(creation.refine_build(
                preview["build_id"],
                provider=args.provider,
                model=args.model,
                instructions=args.instructions,
            ))
        except Exception as exc:
            print(f"LLM refinement unavailable; keeping deterministic draft: {exc}")

    print(json.dumps({
        "build_id": preview["build_id"],
        "target": preview["target"],
        "link_target": preview["link_target"],
        "build_hash": preview["build_hash"],
        "warnings": preview["warnings"],
        "validation_errors": preview["validation_errors"],
    }, indent=2))
    print("\n--- SKILL.md preview ---\n")
    print(preview["files"]["SKILL.md"])
    if preview["validation_errors"]:
        raise SystemExit("The staged skill is invalid and was not installed")

    confirmed = bool(args.yes)
    if not confirmed and sys.stdin.isatty():
        confirmed = input(f"Install {preview['target']}? [y/N] ").strip().lower() in {"y", "yes"}
    if not confirmed:
        print(f"Build staged as {preview['build_id']}; rerun with --yes to install automatically.")
        return
    result = creation.commit_build(
        preview["build_id"],
        preview["build_hash"],
        overwrite=args.skill_action == "update",
    )
    print(f"Installed {result['name']} at {result['path']}")
    if result["link_path"]:
        print(f"Linked {result['link_path']} to {result['path']}")


def _run_extension_command(args):
    """Execute extension authoring and static management commands."""
    import json

    from dialeng.extensions.discovery import discover_extensions
    from dialeng.extensions.state import ExtensionStateStore
    from dialeng.extensions.trust import ExtensionTrustStore
    from dialeng.services.extension_project_service import ExtensionProjectService

    if not args.extension_action:
        print("Usage: dialeng extension {create,list,inspect,trust,enable,disable,reload,build,check,validate,test,prepare,package,install,remove,migrate,doctor,schema,lock,sync,update}")
        return
    root = resolve_root_dir(args.path, respect_env=False)
    service = ExtensionProjectService(root)
    action = args.extension_action
    if action == "schema":
        from dialeng.extensions.contracts import contract_snapshot, write_contract_schema

        if args.output:
            output = write_contract_schema(Path(args.output).expanduser().resolve())
            result = {"ok": True, "output": str(output), **contract_snapshot()}
        else:
            result = {"ok": True, **contract_snapshot()}
    elif action == "create":
        result = service.create(args.name, template=args.template, extension_id=args.extension_id)
    elif action in {"lock", "sync", "update"} or (action == "check" and not args.extension_path):
        from dialeng.extensions.lockfile import ExtensionLockService

        lock_service = ExtensionLockService(root)
        if action == "lock":
            result = lock_service.write()
        elif action == "sync":
            result = lock_service.sync(offline=args.offline)
        elif action == "update":
            result = lock_service.update(args.extension_id)
        else:
            result = lock_service.check()
    elif action in {"build", "check", "validate", "test", "prepare", "package", "install", "migrate"}:
        source = Path(args.extension_path).resolve()
        if action == "build":
            result = service.build(source)
        elif action == "check":
            result = service.check(source)
        elif action == "validate":
            result = service.validate(source, isolated=True)
        elif action == "test":
            result = service.test(source, browser=args.browser)
        elif action == "package":
            result = service.package(source)
        elif action == "install":
            result = service.install_local(source)
        elif action == "migrate":
            result = service.migrate_autorun(source)
        else:
            result = service.prepare(source)
    else:
        discovery = discover_extensions(root)
        if action in {"list", "doctor"}:
            if action == "doctor":
                from dialeng.extensions.diagnostics import build_extension_doctor

                result = build_extension_doctor(
                    root,
                    discovery,
                    ExtensionTrustStore(),
                    ExtensionStateStore(),
                )
            else:
                result = discovery.public()
        else:
            source = discovery.active.get(args.extension_id)
            if source is None:
                raise SystemExit(f"Unknown extension: {args.extension_id}")
            if action == "inspect":
                result = source.public()
            elif action == "trust":
                ExtensionTrustStore().trust(source, "version")
                result = {"ok": True, "trusted": source.manifest.id}
            elif action == "remove":
                ExtensionStateStore().set_enabled(root, source.manifest.id, False)
                result = service.remove_local(source.manifest.id)
            elif action == "reload":
                result = service.prepare(source.root)
                result["message"] = "Validation passed. A running server reload must be requested from the Extension Manager."
            else:
                enabled = action == "enable"
                ExtensionStateStore().set_enabled(root, source.manifest.id, enabled)
                result = {"ok": True, "extension_id": source.manifest.id, "enabled": enabled}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if isinstance(result, dict) and result.get("ok") is False:
        raise SystemExit(1)


def _run_doctor(args):
    """Print one redacted project report without starting the web server."""
    import json
    import sys

    from dialeng.extensions.diagnostics import build_extension_doctor
    from dialeng.extensions.discovery import discover_extensions
    from dialeng.extensions.lockfile import ExtensionLockService
    from dialeng.extensions.state import ExtensionStateStore
    from dialeng.extensions.trust import ExtensionTrustStore
    from dialeng.services.credential_service import detect_credential_inventory
    from dialeng.services.dialeng_config import load_config

    root = resolve_root_dir(args.path, respect_env=False)
    config_path = root / "dialeng_config.json"
    config_error = ""
    try:
        config = load_config(config_path, force_reload=True, create_if_missing=False)
        config_summary = {
            "valid": True,
            "path": str(config_path),
            "models": len(config.available_models),
        }
    except Exception as exc:
        config_error = f"{type(exc).__name__}: {exc}"
        config_summary = {"valid": False, "path": str(config_path), "error": config_error}
    inventory = detect_credential_inventory(root / ".env", force=True)
    providers = [
        {
            "id": item.id,
            "authenticated": item.authenticated,
            "validation_state": item.validation_state,
            "source": item.source,
        }
        for item in inventory.providers.values()
    ]
    state = ExtensionStateStore()
    extension_report = build_extension_doctor(
        root, discover_extensions(root), ExtensionTrustStore(), state,
    )
    lock_report = ExtensionLockService(root, state_store=state).check()
    result = {
        "ok": bool(extension_report.get("ok") and config_summary["valid"]),
        "project_root": str(root),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "configuration": config_summary,
        "providers": providers,
        "kernels": ["local", "colab"],
        "extensions": extension_report,
        "extension_lock": lock_report,
        "generated_artifacts": {
            "extension_builds": str(root / ".dialeng" / "extension-builds"),
            "uv_lock": str(root / "uv.lock"),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)
