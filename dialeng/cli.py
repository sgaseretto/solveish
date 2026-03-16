"""CLI entry point for dialeng.

Usage:
    dialeng                    # Start server rooted at current directory
    dialeng ./my-project       # Start server rooted at ./my-project
    dialeng /abs/path          # Start server rooted at absolute path
    dialeng --port 9000        # Custom port
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
    """Parse args for the default serve mode: dialeng [path] [--port PORT]."""
    parser = argparse.ArgumentParser(
        prog="dialeng",
        description="Notebook-based dialog engine for exploration and development",
    )
    parser.add_argument("path", nargs="?", default=".",
                        help="Root directory for notebooks (default: current directory)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")

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
    from dialeng.app import main
    main(root_dir=root_dir, port=args.port)
