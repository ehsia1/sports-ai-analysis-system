"""Structured logging infrastructure.

Usage:
    from sports_betting.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import get_settings

# Track if logging has been set up
_logging_configured = False


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name for terminal
        color = self.COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "sports_betting",
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure structured logging with file and console output.

    Args:
        name: Logger name (usually module __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to settings.
        log_to_file: Whether to write to file. Defaults to settings.
        log_dir: Directory for log files. Defaults to settings.

    Returns:
        Configured logger instance
    """
    global _logging_configured

    settings = get_settings()

    # Use settings defaults if not specified
    level = level or settings.log_level
    log_to_file = log_to_file if log_to_file is not None else settings.log_to_file
    log_dir = log_dir or settings.logs_dir

    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure root logger once
    if not _logging_configured:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = ColoredFormatter(
            "%(asctime)s | %(levelname_colored)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

        # File handler (daily rotation by filename)
        if log_to_file:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"sports_betting_{datetime.now():%Y-%m-%d}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Capture everything to file
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)

        _logging_configured = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    This is the primary interface for getting loggers throughout the codebase.
    It ensures logging is set up before returning the logger.

    Args:
        name: Usually pass __name__ to get module-specific logger

    Returns:
        Configured logger instance

    Usage:
        from sports_betting.utils import get_logger
        logger = get_logger(__name__)
        logger.info("Hello world")
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """Decorator to log function entry and exit.

    Usage:
        @log_function_call(logger)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}(args={args}, kwargs={kwargs})")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} -> {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator
