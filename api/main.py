"""FastAPI application entrypoint."""

from fastapi import FastAPI

from core.config import settings


def create_app() -> FastAPI:
    """Build the API application instance."""
    app = FastAPI(
        title="AI Knowledge Work Assistant API",
        version="0.1.0",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return a lightweight readiness signal."""
        return {"status": "ok"}

    return app


app = create_app()


def run() -> None:
    """Run the local development API server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
