"""FastAPI 應用程式進入點。"""

from fastapi import FastAPI

from core.config import settings
from core.log import logger


def create_app() -> FastAPI:
    """建立 API 應用程式實體。"""
    app = FastAPI(
        title="AI Knowledge Work Assistant API",
        version="0.1.0",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """回傳簡單的存活狀態檢查。"""
        return {"status": "ok"}

    return app


app = create_app()


def run() -> None:
    """啟動本地端開發用 API 伺服器。"""
    logger.info(f"啟動 FastAPI 伺服器於 {settings.app_host}:{settings.app_port}")
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
