"""
config/logging.py

Centralised logging configuration for the H-CMAT engine.

Provides the `get_logger` utility used across all modules.
Ensures uniform formatting, timestamps, and prevents duplicate
log entries that often occur when mixing custom loggers with Uvicorn.
"""

import logging
import sys

# Define a strict format so logs are highly readable during the live demo
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger instance for the given module.
    
    Usage:
        from config.logging import get_logger
        logger = get_logger(__name__)
        
        logger.info("Encoders loaded.")
    """
    logger = logging.getLogger(module_name)

    # Set the baseline logging level. 
    # Use DEBUG for development, INFO for the live demo.
    logger.setLevel(logging.DEBUG)

    # Prevent logs from bubbling up to the root logger (prevents double-printing)
    logger.propagate = False

    # Only attach a handler if the logger doesn't already have one.
    # This prevents the "duplicate lines" bug when modules are re-imported.
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)

    return logger