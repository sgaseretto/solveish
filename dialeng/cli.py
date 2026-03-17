"""CLI entry point for dialeng.

Usage:
    dialeng                    # Start server rooted at current directory
    dialeng ./my-project       # Start server rooted at ./my-project
    dialeng /abs/path          # Start server rooted at absolute path
    dialeng --port 9000        # Custom port
    dialeng --init             # Start server + create CRAFT.ipynb (auto-detect name)
    dialeng --init my_pkg      # Start server + create CRAFT.ipynb with package name
    dialeng new_folder --init  # Create folder, init CRAFT, start server
    dialeng package init       # Scaffold nbdev-compatible package
"""
import argparse
import os
import sys
from pathlib import Path

# Subcommand names that should not be treated as path arguments
_SUBCOMMANDS = {"package"}


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

    # --- `dialeng package init` ---
    pkg_parser = subparsers.add_parser("package", help="Package management commands")
    pkg_sub = pkg_parser.add_subparsers(dest="package_action")
    init_parser = pkg_sub.add_parser("init", help="Scaffold nbdev-compatible package from notebooks")
    init_parser.add_argument("--name", dest="package_name", default=None,
                             help="Package name (default: directory name)")

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

    if args.subcommand == "package":
        if args.package_action == "init":
            from dialeng.services.package_scaffold_service import scaffold_package
            root = resolve_root_dir(".", respect_env=False)
            scaffold_package(root, package_name=args.package_name)
        else:
            print("Usage: dialeng package init [--name NAME]")
        return

    # Default: start the server
    root_dir = resolve_root_dir(args.path)

    # --init: create CRAFT.ipynb before starting the server
    if args.init is not False:
        from dialeng.services.craft_init_service import resolve_and_init
        pkg_name = args.init if isinstance(args.init, str) else None
        print(f"Initializing reuse workflow in {root_dir}...")
        resolve_and_init(root_dir, pkg_name=pkg_name)
        print()

    from dialeng.app import main
    main(root_dir=root_dir, port=args.port)
