from fastapi.testclient import TestClient

from fast_rag.app import app


def test_engine_route_serves_search_page() -> None:
    client = TestClient(app)
    response = client.get("/engine?q=chromium")
    assert response.status_code == 200
    assert "SignalRAG Search" in response.text
    assert "./engine.js" in response.text
