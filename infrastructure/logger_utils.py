"""
Logging configuration and utilities.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry)


class StandardFormatter(logging.Formatter):
    """Standard text formatter for development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(level: str = "INFO", format_type: str = "json") -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'standard')
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter based on type
    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StandardFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs: Any) -> None:
    """
    Log a message with additional context data.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional context data
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "",
        0,
        message,
        (),
        None
    )
    record.extra_data = kwargs
    logger.handle(record)


if __name__ == "__main__":
    print("=" * 60)
    print("LOGGER TEST")
    print("=" * 60)

    # Test standard format
    setup_logging(level="DEBUG", format_type="standard")
    logger = get_logger("test")
    logger.info("Test message with standard format")
    logger.debug("Debug message")

    # Test JSON format
    setup_logging(level="INFO", format_type="json")
    logger = get_logger("test.json")
    logger.info("Test message with JSON format")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
