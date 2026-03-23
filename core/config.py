"""專案共享設定載入模組。"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理應用程式設定與敏感資訊。"""

    groq_api_key: SecretStr | None = Field(
        default=None,
        alias="GROQ_API_KEY",
    )
    app_env: Literal["development", "test", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def has_groq_api_key(self) -> bool:
        """回傳目前是否已提供 Groq API 金鑰。"""
        return self.groq_api_key is not None and bool(
            self.groq_api_key.get_secret_value().strip()
        )

    def require_groq_api_key(self) -> str:
        """在需要使用 Groq 時回傳金鑰，若缺少則拋出明確錯誤。"""
        if not self.has_groq_api_key:
            raise ValueError(
                "缺少 GROQ_API_KEY。請在 .env 設定後再啟用需要 Groq 的功能。"
            )
        return self.groq_api_key.get_secret_value().strip()


settings = Settings()
