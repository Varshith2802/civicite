from fastapi.testclient import TestClient

from civicite.agent.core import CiviCite
from civicite.api.server import create_app


def test_api_roundtrip(index):
    client = TestClient(create_app(CiviCite(index)))
    assert client.get("/api/health").json()["chunks"] == len(index.chunks)
    assert "CiviCite" in client.get("/").text
    r = client.post("/api/ask", json={"question": "Who pays sick pay during the first 14 days?"}).json()
    assert "employer" in r["answer"].lower() and r["verification"]["label"] == "verified"
    assert client.post("/api/ask", json={"question": "x", "mode": "llm"}).status_code == 422  # too short
    assert client.post("/api/ask", json={"question": "hello there", "mode": "llm"}).status_code == 400  # no LLM
