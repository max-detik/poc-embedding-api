import numpy as np
import pytest

from embedder import STEmbeddings


def last_call(embedder):
    return embedder.model.calls[-1]


def test_harrier_defaults():
    e = STEmbeddings("harrier")
    assert (e.dim, e.model.max_seq_length, e.device, e.dtype) == (1024, 1024, "cpu", "float32")
    assert e.model.kwargs["tokenizer_kwargs"] == {"padding_side": "left"}
    assert e.query_prompt() == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    )


def test_gemma_defaults():
    e = STEmbeddings("gemma")
    assert (e.dim, e.model.max_seq_length) == (768, 2048)
    assert e.model.kwargs["tokenizer_kwargs"] is None
    e.embed_documents(["isi"])
    assert last_call(e) == (None, ["title: none | text: isi"])


def test_query_uses_prompt_and_task_override():
    e = STEmbeddings("harrier", task="Cari berita")
    e.embed_query("harga bbm")
    assert last_call(e) == ("Instruct: Cari berita\nQuery: ", ["harga bbm"])
    e.embed_query("harga bbm", task="Lain")
    assert last_call(e)[0] == "Instruct: Lain\nQuery: "


def test_documents_rendered_through_template_with_titles():
    e = STEmbeddings("harrier", document_template="{title}\n\n{content}")
    e.embed_documents(["satu", "dua"], titles=["T1", None])
    prompt, texts = last_call(e)
    assert prompt is None
    assert sorted(texts) == sorted(["T1\n\nsatu", "dua"])


def test_length_sorting_preserves_input_order():
    e = STEmbeddings("harrier")
    batched = e.embed_documents(["a much longer document here", "a"])
    alone = e.embed_documents(["a"])
    assert np.allclose(batched[1], alone[0])


def test_empty_inputs():
    e = STEmbeddings("harrier")
    assert e.embed_documents([]).shape == (0, 1024)
    assert e.embed_queries([]).shape == (0, 1024)


def test_invalid_template_fails_at_construction():
    with pytest.raises(ValueError):
        STEmbeddings("harrier", document_template="{title} only")


def test_float16_falls_back_on_cpu():
    assert STEmbeddings("harrier", dtype="float16").dtype == "float32"


def test_overrides():
    e = STEmbeddings("harrier", max_seq_length=2048, truncate_dim=512)
    assert e.model.max_seq_length == 2048
    assert e.dim == 512
    assert e.model.kwargs["truncate_dim"] == 512


def test_hf_token_passed_only_when_set(monkeypatch):
    assert "token" not in STEmbeddings("gemma").model.kwargs
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert STEmbeddings("gemma").model.kwargs["token"] == "hf_test"


def test_chunk_then_embed_respects_title_budget():
    e = STEmbeddings("harrier", document_template="{title}\n\n{content}", max_seq_length=60)
    title = "Harga BBM naik lagi mulai pekan depan"
    text = " ".join(f"w{i}" for i in range(200))
    chunks = e.chunk_text(text, title=title)
    assert len(chunks) > 1
    assert all(e.token_length(c, title=title) <= 60 for c in chunks)

    out = e.embed_documents_chunked([text], titles=[title])[0]
    assert [r["text"] for r in out] == chunks
    assert out[0]["embedded_text"] == f"{title}\n\n{chunks[0]}"
    assert len(out[0]["embedding"]) == 1024
