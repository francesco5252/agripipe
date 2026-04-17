"""Setup logging centralizzato."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    stream=sys.stdout,
) -> None:
    """Configura il logging globale per l'applicazione."""
    global _CONFIGURED
    
    handlers: list[logging.Handler] = [logging.StreamHandler(stream)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Ritorna un logger configurato."""
    global _CONFIGURED
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
