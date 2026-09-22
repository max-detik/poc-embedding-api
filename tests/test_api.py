import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EMBED_DOCUMENT_TEMPLATE", "{title}\\n\\n{content}")
    with TestClient(app) as c:
        yield c


def encoded(client):
    return client.app.state.embedder.model.calls[-1]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "microsoft/harrier-oss-v1-0.6b"
    assert body["dimensions"] == 1024
    assert body["document_template"] == "{title}\n\n{content}"


def test_embed_query_echoes_prompt(client):
    body = client.post("/embed/query", json={"text": "harga bbm", "task": "Cari berita"}).json()
    assert body["dimensions"] == 1024 and len(body["embedding"]) == 1024
    assert body["prompt"] == "Instruct: Cari berita\nQuery: "


def test_embed_documents_with_and_without_titles(client):
    body = client.post("/embed/documents", json={"texts": ["isi"], "titles": ["Judul"]}).json()
    assert (body["count"], body["dimensions"]) == (1, 1024)
    assert encoded(client) == (None, ["Judul\n\nisi"])

    client.post("/embed/documents", json={"texts": ["isi"]})
    assert encoded(client) == (None, ["isi"])


@pytest.mark.parametrize("payload", [
    {"texts": []},
    {"texts": ["a", "b"], "titles": ["x"]},
])
def test_embed_documents_bad_requests(client, payload):
    assert client.post("/embed/documents", json=payload).status_code == 400


def test_removed_endpoints_are_gone(client):
    assert client.post("/rerank", json={}).status_code == 404
