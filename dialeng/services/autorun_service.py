"""
AUTORUN service - processes the AUTORUN/ folder on server startup.

Two-phase loading:
1. Extract #| export cells from AUTORUN notebooks → .autorun_modules/ cache dir
   Load AUTORUN/*.py and .autorun_modules/*.py as extensions (main process)
2. Open AUTORUN/*.ipynb in their own kernels (idle-exempt background tasks)

Hot-reload:
    reload_autorun_extensions() re-runs Phase 1 only (extract + reimport),
    without restarting Phase 2 kernels. Used during development via
    POST /dialeng/reload-extensions or `from dialeng.dev import reload_extensions`.
"""
import asyncio
import importlib.util
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".autorun_modules")

# Set by process_autorun() at startup, used by reload_autorun_extensions()
_autorun_dir: Optional[Path] = None
_contribution_store = None
_legacy_modules: dict[str, str] = {}


def set_autorun_contribution_store(store) -> None:
    """Use the main extension ownership ledger for legacy registrations."""
    global _contribution_store
    _contribution_store = store


def _owner_for(path: Path, source_kind: str) -> str:
    return f"legacy.autorun.{source_kind}.{path.stem}"


def _load_owned(path: Path, source_kind: str) -> str:
    """Import one legacy module transactionally into a synthetic owner scope."""
    from dialeng.extensions.context import RegistrationScope, registration_scope
    from dialeng.extensions.contributions import ContributionStore

    global _contribution_store
    if _contribution_store is None:
        _contribution_store = ContributionStore()
    owner = _owner_for(path, source_kind)
    scope = RegistrationScope(owner)
    module_name = f"_dialeng_{owner.replace('.', '_')}_{uuid.uuid4().hex[:8]}"
    try:
        with registration_scope(scope):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load legacy extension: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        _contribution_store.replace(owner, scope.contributions)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    previous = _legacy_modules.get(owner)
    _legacy_modules[owner] = module_name
    if previous:
        sys.modules.pop(previous, None)
    return owner


def _sync_owned_modules(paths: list[tuple[Path, str]]) -> tuple[list[str], list[str]]:
    loaded: list[str] = []
    errors: list[str] = []
    expected = {_owner_for(path, kind) for path, kind in paths}
    for owner in set(_legacy_modules) - expected:
        if _contribution_store is not None:
            _contribution_store.remove(owner)
        sys.modules.pop(_legacy_modules.pop(owner), None)
    for path, kind in paths:
        try:
            _load_owned(path, kind)
            loaded.append(path.stem)
        except Exception as exc:
            errors.append(f"Load {path.name}: {exc}")
            logger.exception("AUTORUN legacy extension failed (path=%s)", path)
    return loaded, errors


async def process_autorun(kernel_service, notebooks_dir: Path = None) -> None:
    """Process AUTORUN folder on server startup.

    Args:
        kernel_service: KernelService instance for running background notebooks
        notebooks_dir: Root notebooks directory (for locating AUTORUN/)
    """
    global _autorun_dir
    _autorun_dir = (notebooks_dir / "AUTORUN") if notebooks_dir else Path("notebooks/AUTORUN")

    if not _autorun_dir.exists():
        logger.info("No AUTORUN/ directory found, skipping")
        return

    from dialeng.core.extensions import extract_extension

    # Phase 1b: Export #| export cells from notebooks → .autorun_modules/
    CACHE_DIR.mkdir(exist_ok=True)
    exported_count = 0
    for nb_path in sorted(_autorun_dir.glob("*.ipynb")):
        module_name = nb_path.stem
        output_path = CACHE_DIR / f"{module_name}.py"
        try:
            count = extract_extension(nb_path, output_path, marker="#| export")
            if count > 0:
                exported_count += count
                logger.info(f"AUTORUN: Exported {count} cells from {nb_path.name} → {output_path}")
        except Exception as e:
            logger.error(f"AUTORUN: Failed to extract from {nb_path.name}: {e}")

    # Phase 1c: Load .py extensions from AUTORUN/ and .autorun_modules/
    paths = [
        *((path, "source") for path in sorted(_autorun_dir.glob("*.py")) if not path.name.startswith("_")),
        *((path, "notebook") for path in sorted(CACHE_DIR.glob("*.py")) if not path.name.startswith("_")),
    ]
    loaded, errors = _sync_owned_modules(paths)
    autorun_exts = [path.stem for path, kind in paths if kind == "source" and path.stem in loaded]
    cache_exts = [path.stem for path, kind in paths if kind == "notebook" and path.stem in loaded]
    for error in errors:
        logger.error("AUTORUN: %s", error)
    if autorun_exts:
        logger.info(f"AUTORUN: Loaded {len(autorun_exts)} extension(s) from AUTORUN/: {autorun_exts}")
    if cache_exts:
        logger.info(f"AUTORUN: Loaded {len(cache_exts)} extension(s) from .autorun_modules/: {cache_exts}")

    # Phase 2: Open notebooks in their own kernels (background)
    for nb_path in sorted(_autorun_dir.glob("*.ipynb")):
        asyncio.create_task(_run_autorun_notebook(nb_path, kernel_service))


def reload_autorun_extensions() -> Dict[str, List[str]]:
    """Re-extract #| export cells and reload all AUTORUN extension modules.

    Re-runs Phase 1 only (extract + reimport). Does NOT restart Phase 2
    background kernels. Safe to call while the server is running.

    Returns:
        dict with 'extracted' (notebook names), 'loaded' (module names),
        and 'errors' (error descriptions).
    """
    from dialeng.core.extensions import extract_extension

    result: Dict[str, List[str]] = {"extracted": [], "loaded": [], "errors": []}

    if _autorun_dir is None or not _autorun_dir.exists():
        return result

    # Phase 1b: Re-extract #| export cells from notebooks
    CACHE_DIR.mkdir(exist_ok=True)
    for nb_path in sorted(_autorun_dir.glob("*.ipynb")):
        module_name = nb_path.stem
        output_path = CACHE_DIR / f"{module_name}.py"
        try:
            count = extract_extension(nb_path, output_path, marker="#| export")
            if count > 0:
                result["extracted"].append(nb_path.name)
                logger.info(f"AUTORUN reload: Exported {count} cells from {nb_path.name}")
        except Exception as e:
            result["errors"].append(f"Extract {nb_path.name}: {e}")
            logger.error(f"AUTORUN reload: Failed to extract from {nb_path.name}: {e}")

    # Phase 1c: Reload .py extensions from AUTORUN/ and .autorun_modules/
    paths = [
        *((path, "source") for path in sorted(_autorun_dir.glob("*.py")) if not path.name.startswith("_")),
        *((path, "notebook") for path in sorted(CACHE_DIR.glob("*.py")) if not path.name.startswith("_")),
    ]
    loaded, errors = _sync_owned_modules(paths)
    result["loaded"].extend(loaded)
    result["errors"].extend(errors)

    logger.info(f"AUTORUN reload: {len(result['loaded'])} loaded, {len(result['errors'])} errors")
    return result


async def _run_autorun_notebook(nb_path: Path, kernel_service) -> None:
    """Run an AUTORUN notebook in its own kernel (idle-exempt).

    Each notebook gets its own kernel. Execution errors are logged
    but don't affect other notebooks.
    """
    from dialeng.document.serialization import load_notebook
    from dialeng.document.cell import CellOutput

    notebook_id = f"autorun~{nb_path.stem}"
    logger.info(f"AUTORUN: Starting notebook {nb_path.name} as {notebook_id}")

    try:
        nb = load_notebook(nb_path)
        code_cells = [c for c in nb.cells
                      if (c.cell_type.value if hasattr(c.cell_type, 'value') else c.cell_type) == "code"
                      and c.source.strip()
                      and not c.source.lstrip().startswith("#| export")]

        if not code_cells:
            logger.info(f"AUTORUN: No code cells in {nb_path.name}, skipping")
            return

        # Execute each code cell
        for cell in code_cells:
            try:
                async for output in kernel_service.execute_cell(notebook_id, cell):
                    # Log errors but continue
                    if output.output_type == 'error':
                        logger.error(f"AUTORUN: Error in {nb_path.name} cell {cell.id}: "
                                     f"{output.ename}: {output.evalue}")
            except Exception as e:
                logger.error(f"AUTORUN: Failed to execute cell {cell.id} in {nb_path.name}: {e}")

        logger.info(f"AUTORUN: Completed {nb_path.name} ({len(code_cells)} cells)")

    except Exception as e:
        logger.error(f"AUTORUN: Failed to run notebook {nb_path.name}: {e}")
