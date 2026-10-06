from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_home_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "dialogue-summarizer" in r.text


def test_empty_dialogue_is_rejected():
    r = client.post("/summarize", json={"dialogue": "   "})
    assert r.status_code == 200
    assert "paste a conversation" in r.json()["summary"].lower()


def test_too_long_is_rejected():
    r = client.post("/summarize", json={"dialogue": "x" * 7000})
    assert r.status_code == 200
    assert "too long" in r.json()["summary"].lower()