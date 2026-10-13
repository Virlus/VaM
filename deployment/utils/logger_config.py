# Author: Jun Lv
"""
Unified logging configuration using loguru.
Compatible with both Windows (No rotation) and Ubuntu (With rotation).
"""

import sys
from pathlib import Path
from typing import Any
from loguru import logger


def setup_logger() -> Any:
    """
    Setup loguru logger with colored console output and file logging.
    Adapts configuration based on the Operating System.
    """
    # Remove default handler
    logger.remove()

    # Detect if current system is Windows
    # sys.platform returns 'win32' on Windows, 'linux' on Linux
    IS_WINDOWS = sys.platform == "win32"

    # ------------------------------------------------------------------
    # OS-specific configuration strategy
    # ------------------------------------------------------------------
    if IS_WINDOWS:
        # Windows: Disable rotation and compression to prevent [WinError 32]
        # Although automatic log rotation is disabled, this ensures stability in multi-process scenarios
        rotation_val = None
        retention_val = None
        compression_val = None
    else:
        # Ubuntu/Linux: Enable full features
        # Linux filesystem allows renaming files while open, so rotation is safe
        rotation_val = "1 day"  # Rotate once per day
        retention_val = "7 days"  # Retain for 7 days
        compression_val = "zip"  # Compress old logs

    # ------------------------------------------------------------------
    # 1. Console Handler
    # ------------------------------------------------------------------
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Enable queue on all systems to prevent console output chaos
    )

    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 2. File Handler (App Log - All logs)
    # ------------------------------------------------------------------
    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        enqueue=True,  # Must enable: async writing, foundation for multi-process safety
        rotation=rotation_val,  # Dynamically determined based on system
        retention=retention_val,  # Dynamically determined based on system
        compression=compression_val,  # Dynamically determined based on system
        backtrace=True,
        diagnose=True,
    )

    # ------------------------------------------------------------------
    # 3. File Handler (Error Log - Errors only)
    # ------------------------------------------------------------------
    logger.add(
        log_dir / "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        enqueue=True,
        rotation=rotation_val,
        retention="30 days" if not IS_WINDOWS else None,  # Error logs retained longer on Linux
        compression=compression_val,
        backtrace=True,
        diagnose=True,
    )

    return logger


# Initialize logger
setup_logger()

# Export logger
__all__ = ["logger"]
