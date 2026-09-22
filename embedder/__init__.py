"""Sentence embeddings with per-model prompt handling.

    harrier : microsoft/harrier-oss-v1-0.6b      1024d, last-token pooling, Qwen3 arch
    gemma   : google/embeddinggemma-300m          768d, mean pooling

    from embedder import STEmbeddings

    embedder = STEmbeddings("harrier", document_template="{title}\\n\\n{content}")
    chunks = embedder.chunk_text(article, title=title)
    vectors = embedder.embed_documents(chunks, titles=[title] * len(chunks))
    query_vector = embedder.embed_query("berapa harga bbm hari ini")

Layout:
    specs.py     which models exist and how each is prompted
    prompts.py   rendering query prompts and document templates (pure)
    chunking.py  token-budgeted chunking (pure)
    runtime.py   device/dtype resolution, HF auth, model loading
    model.py     STEmbeddings, tying the above together
"""

from embedder.model import STEmbeddings
from embedder.specs import MODEL_SPECS, ModelSpec

__all__ = ["STEmbeddings", "MODEL_SPECS", "ModelSpec"]
