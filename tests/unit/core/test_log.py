"""日誌模組的單元測試。"""

import logging

from core.log import logger, setup_logger


def test_setup_logger_creates_new_logger_with_handler():
    """測試 setup_logger 能夠正確初始化並附加上 StreamHandler。"""
    test_logger = setup_logger("test_target")
    assert test_logger.name == "test_target"
    assert test_logger.level == logging.INFO
    # 確保自動附加了一個 Handler
    assert len(test_logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in test_logger.handlers)


def test_setup_logger_does_not_duplicate_handlers():
    """測試 setup_logger 不會重複附加 Handler。"""
    test_logger1 = setup_logger("test_duplicate")
    initial_handler_count = len(test_logger1.handlers)

    # 再次呼叫
    test_logger2 = setup_logger("test_duplicate")
    assert test_logger1 is test_logger2
    assert len(test_logger2.handlers) == initial_handler_count


def test_default_logger():
    """測試專案預設置 logger 能否正確取得。"""
    assert logger.name == "bmad"
    assert logger.level == logging.INFO
