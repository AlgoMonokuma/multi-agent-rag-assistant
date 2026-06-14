"""Shared logging utilities for the project."""

import logging


def setup_logger(name: str = "multi_agent_rag_assistant") -> logging.Logger:
    """Configure and return a standard project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()

# Alias for modules that prefer the get_logger(name) convention.
# Both functions are equivalent; setup_logger is the canonical name.
get_logger = setup_logger
