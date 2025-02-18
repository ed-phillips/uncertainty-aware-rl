import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    logger_name: str = "app",
    log_file: str = "app.log",
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        logger_name: Name of the logger (default: app)
        log_file: Path to log file
        log_level: Logging level (default: INFO)
        log_format: Format string for log messages
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    # Get the logger
    logger = logging.getLogger(logger_name)

    # If handlers are already configured, return the logger
    if logger.hasHandlers():
        return logger

    # Set logger level
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create and configure file handler with rotation
    file_handler = RotatingFileHandler(
        log_path / log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
