"""Logging utilities for MadEvolve-Cosmo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def setup_logging(results_dir: Path, verbose: bool = True) -> Path:
    """Configure logging sinks."""
    log_file = Path(results_dir) / "madevolve_cosmo.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers = []
    if verbose:
        handlers.append(
            RichHandler(
                show_path=False,
                rich_tracebacks=True,
                console=get_console(),
            )
        )
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return log_file
