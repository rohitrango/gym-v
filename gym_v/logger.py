from __future__ import annotations

import sys

from loguru import logger
from loguru._logger import Logger

logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def set_level(level: str) -> None:
    """Set logging level."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level.upper(),
        colorize=True,
    )


def get_logger() -> Logger:
    """Get the logger."""
    return logger
