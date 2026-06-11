"""Test behavior."""

from pathlib import Path

import pytest

from core.config import Settings


def test_settings_default_values() -> None:
    """Test behavior."""
    config = Settings()

    assert config.app_env == "development"
    assert config.app_host == "127.0.0.1"
    assert config.app_port == 8000
    assert config.streamlit_port == 8501
    assert config.has_groq_api_key is False


def test_settings_read_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("STREAMLIT_PORT", "8601")
    monkeypatch.setenv("GROQ_API_KEY", "demo-secret")

    config = Settings()

    assert config.app_env == "test"
    assert config.app_host == "0.0.0.0"
    assert config.app_port == 9000
    assert config.streamlit_port == 8601
    assert config.has_groq_api_key is True
    assert config.require_groq_api_key() == "demo-secret"


def test_require_groq_api_key_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test behavior."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    config = Settings()

    with pytest.raises(ValueError, match='GROQ_API_KEY is missing'):
        config.require_groq_api_key()


def test_env_example_matches_settings_contract() -> None:
    """Test behavior."""
    env_example = Path(".env.example").read_text(encoding="utf-8")

    expected_keys = {
        "GROQ_API_KEY",
        "APP_ENV",
        "APP_HOST",
        "APP_PORT",
        "STREAMLIT_PORT",
    }
    actual_keys = {
        line.split("=", maxsplit=1)[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert expected_keys <= actual_keys
