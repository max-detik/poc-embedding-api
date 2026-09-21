"""
FastAPI service exposing local embedding inference for
microsoft/harrier-oss-v1-0.6b through sentence-transformers / torch
(see st_embeddings.py).

Device and dtype default to CUDA + float16 when a GPU is present and CPU +
float32 otherwise; set DEVICE / DTYPE to pin them (see .env.example).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings-service")

from st_embeddings import STEmbeddings  # noqa: E402 -- logging must be configured first

# ---------------------------------------------------------------------------
# Config (all overridable via env vars — see .env.example)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE") or None  # None -> CUDA when available, else CPU
DTYPE = os.getenv("DTYPE") or None
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_SEQ_LENGTH = int(os.getenv("EMBED_MAX_SEQ_LENGTH") or 0) or None
EMBED_TRUNCATE_DIM = int(os.getenv("EMBED_TRUNCATE_DIM") or 0) or None
EMBED_TASK = os.getenv("EMBED_TASK") or None

# Created once at startup and reused across requests.
_model: Optional[STEmbeddings] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and warm before the healthcheck passes, so the first real request
    # doesn't pay the download/first-call cost.
    global _model
    _model = STEmbeddings(
        "harrier",
        device=DEVICE,
        dtype=DTYPE,
        batch_size=EMBED_BATCH_SIZE,
        max_seq_length=EMBED_MAX_SEQ_LENGTH,
        truncate_dim=EMBED_TRUNCATE_DIM,
        task=EMBED_TASK,
    )
    _model.warmup()
    yield
    _model = None


app = FastAPI(
    title="Embeddings Service",
    description="Local embedding inference for microsoft/harrier-oss-v1-0.6b (sentence-transformers on torch) wrapped in FastAPI.",
    version="3.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single query string (the query instruction is applied automatically).")
    task: Optional[str] = Field(
        None,
        description="Overrides the task description in the query instruction for this request.",
    )


class EmbedDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., description="Batch of documents/chunks to embed.")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    prompt: str


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if _model is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": _model.spec.model_id,
        "dimensions": _model.dim,
        "max_seq_length": _model.model.max_seq_length,
        "device": _model.device,
        "dtype": _model.dtype,
    }


@app.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(payload: EmbedQueryRequest):
    """Embed a single query string, with the query instruction applied automatically."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    try:
        vector = _model.embed_query(payload.text, task=payload.task)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_query failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingResponse(
        embedding=vector.tolist(),
        dimensions=len(vector),
        prompt=_model.query_prompt(payload.task),
    )


@app.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(payload: EmbedDocumentsRequest):
    """Embed a batch of documents/chunks (respects EMBED_BATCH_SIZE internally)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        vectors = _model.embed_documents(payload.texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_documents failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingsResponse(
        embeddings=vectors.tolist(),
        dimensions=int(vectors.shape[1]),
        count=int(vectors.shape[0]),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
