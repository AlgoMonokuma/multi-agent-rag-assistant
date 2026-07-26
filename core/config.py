"""Shared application settings loader."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings and sensitive configuration."""

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
    TAVILY_API_KEY: str = Field(
        default="",
        alias="TAVILY_API_KEY",
    )
    """Tavily API key for web search fallback. Leave empty to disable web search."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def has_groq_api_key(self) -> bool:
        """Return whether a usable Groq API key is configured."""
        return self.groq_api_key is not None and bool(
            self.groq_api_key.get_secret_value().strip()
        )

    def require_groq_api_key(self) -> str:
        """Return the Groq API key or raise a clear configuration error."""
        if not self.has_groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in .env before enabling Groq-backed features."
            )
        return self.groq_api_key.get_secret_value().strip()


settings = Settings()
