from fastapi.testclient import TestClient

from app.main import app
from app.rag_pipeline import RAGPipeline

client = TestClient(app)


def test_qa_upload_and_query():
    # Upload a small QA set
    with open("rag-perplexity-hackathon/tests/data/sample_qa.txt", "rb") as f:
        files = {"file": ("sample_qa.txt", f, "text/plain")}
        r = client.post("/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["type"] == "qa"

    # Ask a question that should be answered from QA directly
    r2 = client.post("/query", json={"query": "What is Foo?"})
    assert r2.status_code == 200
    ans = r2.json().get("answer", "")
    assert "Bar" in ans


def test_fallback_perplexity_stub(monkeypatch):
    # Stub the Perplexity call to avoid external dependency
    monkeypatch.setattr(RAGPipeline, "call_perplexity", staticmethod(lambda prompt: "STUBBED"))
    r = client.post("/query", json={"query": "This should go to LLM"})
    assert r.status_code == 200
    assert r.json().get("answer") == "STUBBED"
