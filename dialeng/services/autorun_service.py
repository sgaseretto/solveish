"""
AUTORUN service - processes the AUTORUN/ folder on server startup.

Two-phase loading:
1. Extract #| export cells from AUTORUN notebooks → .autorun_modules/ cache dir
   Load AUTORUN/*.py and .autorun_modules/*.py as extensions (main process)
2. Open AUTORUN/*.ipynb in their own kernels (idle-exempt background tasks)
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTORUN_DIR = Path("AUTORUN")
CACHE_DIR = Path(".autorun_modules")


async def process_autorun(kernel_service) -> None:
    """Process AUTORUN folder on server startup.

    Args:
        kernel_service: KernelService instance for running background notebooks
    """
    if not AUTORUN_DIR.exists():
        logger.info("No AUTORUN/ directory found, skipping")
        return

    from dialeng.core.extensions import load_extensions, extract_extension

    # Phase 1b: Export #| export cells from notebooks → .autorun_modules/
    CACHE_DIR.mkdir(exist_ok=True)
    exported_count = 0
    for nb_path in sorted(AUTORUN_DIR.glob("*.ipynb")):
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
    autorun_exts = load_extensions(AUTORUN_DIR, silent=True)
    cache_exts = load_extensions(CACHE_DIR, silent=True)
    if autorun_exts:
        logger.info(f"AUTORUN: Loaded {len(autorun_exts)} extension(s) from AUTORUN/: {autorun_exts}")
    if cache_exts:
        logger.info(f"AUTORUN: Loaded {len(cache_exts)} extension(s) from .autorun_modules/: {cache_exts}")

    # Phase 2: Open notebooks in their own kernels (background)
    for nb_path in sorted(AUTORUN_DIR.glob("*.ipynb")):
        asyncio.create_task(_run_autorun_notebook(nb_path, kernel_service))


async def _run_autorun_notebook(nb_path: Path, kernel_service) -> None:
    """Run an AUTORUN notebook in its own kernel (idle-exempt).

    Each notebook gets its own kernel. Execution errors are logged
    but don't affect other notebooks.
    """
    from dialeng.document.serialization import load_notebook
    from dialeng.document.cell import CellOutput

    notebook_id = f"autorun_{nb_path.stem}"
    logger.info(f"AUTORUN: Starting notebook {nb_path.name} as {notebook_id}")

    try:
        nb = load_notebook(nb_path)
        code_cells = [c for c in nb.cells
                      if (c.cell_type.value if hasattr(c.cell_type, 'value') else c.cell_type) == "code"
                      and c.source.strip()]

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
