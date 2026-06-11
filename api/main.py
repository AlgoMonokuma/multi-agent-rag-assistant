"""FastAPI application entry point."""

import uvicorn
from fastapi import FastAPI

from core.config import settings
from core.log import logger


def create_app() -> FastAPI:
    """Create the API application instance."""
    app = FastAPI(
        title="Multi-Agent RAG Assistant API",
        version="0.1.0",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return a basic liveness response."""
        return {"status": "ok"}

    return app


def main() -> None:
    """Start the local development API server."""
    logger.info("Starting FastAPI server on %s:%s", settings.app_host, settings.app_port)
    uvicorn.run("api.main:create_app", host=settings.app_host, port=settings.app_port, factory=True)


app = create_app()


if __name__ == "__main__":
    main()
