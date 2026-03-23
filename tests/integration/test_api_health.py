"""API 基本整合測試。"""

from fastapi.testclient import TestClient

from api.main import app


def test_health_check() -> None:
    """確認健康檢查端點可正常回應。"""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
