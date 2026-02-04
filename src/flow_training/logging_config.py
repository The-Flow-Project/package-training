"""
Logger configuration for the FLOW Training Package with loguru.

Provides comprehensive logging setup with:
- Console output with colors and detailed formatting
- File logging with rotation and retention policies
- Separate error log tracking
- Thread-safe asynchronous logging
"""

import sys
from pathlib import Path
from typing import Literal
from loguru import logger

# Valid log levels for type safety
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logger(
    level: LogLevel = "DEBUG",
    log_dir: Path | str | None = None,
    in_production: bool = False,
) -> None:
    """
    Configure the Loguru logger for the application.

    Removes default handlers and adds:
    1. Console output with colors and detailed formatting
    2. File output with rotation and retention
    3. Separate error log for ERROR and CRITICAL levels

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to DEBUG.
        log_dir: Directory for log files. Defaults to ./logs. Can be absolute or relative path.
        in_production: If True, disables backtrace and diagnose for performance. Defaults to False.

    Raises:
        ValueError: If log_dir cannot be created or level is invalid.
    """
    # Validate log level
    valid_levels: set[LogLevel] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:  # type: ignore
        raise ValueError(f"Invalid log level '{level}'. Must be one of {sorted(valid_levels)}")

    # Set up log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    # Create logs directory with error handling
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot create log directory '{log_dir}': {e}") from e

    # Remove default handler to customize completely
    logger.remove()

    # Performance settings based on environment
    # Backtrace and diagnose are verbose, should be disabled in production
    use_backtrace = not in_production
    use_diagnose = not in_production

    # Console handler with colored output
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=use_backtrace,
        diagnose=use_diagnose,
    )

    # File handler for all logs with rotation and retention
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.add(
        log_dir / "app.log",
        format=file_format,
        level="DEBUG",
        rotation="5 MB",
        retention="10 days",
        backtrace=use_backtrace,
        diagnose=use_diagnose,
        enqueue=True,  # Thread-safe asynchronous logging
    )

    # Separate error log file for ERROR and CRITICAL levels
    logger.add(
        log_dir / "errors.log",
        format=file_format,
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        backtrace=use_backtrace,
        diagnose=use_diagnose,
        enqueue=True,  # Thread-safe asynchronous logging
    )

    logger.info(
        f"Logger initialized | Level: {level} | "
        f"Log directory: {log_dir.absolute()} | "
        f"Production mode: {in_production}"
    )
