"""Test behavior."""

import logging

from core.log import logger, setup_logger


def test_setup_logger_creates_new_logger_with_handler():
    """Test behavior."""
    test_logger = setup_logger("test_target")
    assert test_logger.name == "test_target"
    assert test_logger.level == logging.INFO
    # Test note.
    assert len(test_logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in test_logger.handlers)


def test_setup_logger_does_not_duplicate_handlers():
    """Test behavior."""
    test_logger1 = setup_logger("test_duplicate")
    initial_handler_count = len(test_logger1.handlers)

    # Test note.
    test_logger2 = setup_logger("test_duplicate")
    assert test_logger1 is test_logger2
    assert len(test_logger2.handlers) == initial_handler_count


def test_default_logger():
    """Test behavior."""
    assert logger.name == "multi_agent_rag_assistant"
    assert logger.level == logging.INFO
