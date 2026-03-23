"""專案基礎設定的單元測試。"""

from core.config import Settings


def test_settings_defaults() -> None:
    """確認設定物件具備可預期的本地開發預設值。"""
    config = Settings()

    assert config.app_env == "development"
    assert config.app_host == "127.0.0.1"
    assert config.app_port == 8000
    assert config.streamlit_port == 8501
