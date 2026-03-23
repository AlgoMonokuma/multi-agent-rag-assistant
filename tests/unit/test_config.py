"""設定載入與安全取用的單元測試。"""

from pathlib import Path

import pytest

from core.config import Settings


def test_settings_default_values() -> None:
    """驗證預設設定可在未提供環境變數時正常載入。"""
    config = Settings()

    assert config.app_env == "development"
    assert config.app_host == "127.0.0.1"
    assert config.app_port == 8000
    assert config.streamlit_port == 8501
    assert config.has_groq_api_key is False


def test_settings_read_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證環境變數可覆寫預設值。"""
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
    """驗證缺少金鑰時會得到清楚的錯誤訊息。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    config = Settings()

    with pytest.raises(ValueError, match="缺少 GROQ_API_KEY"):
        config.require_groq_api_key()


def test_env_example_matches_settings_contract() -> None:
    """驗證範例環境變數檔案涵蓋目前設定契約。"""
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
