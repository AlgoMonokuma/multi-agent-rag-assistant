"""集中管理專案設定。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """從環境變數載入應用程式設定。"""

    groq_api_key: str = ""
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    streamlit_port: int = 8501

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
