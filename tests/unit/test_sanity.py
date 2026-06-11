"""Test behavior."""

from core.config import Settings


def test_settings_defaults() -> None:
    """Test behavior."""
    config = Settings()

    assert config.app_env == "development"
    assert config.app_host == "127.0.0.1"
    assert config.app_port == 8000
    assert config.streamlit_port == 8501
