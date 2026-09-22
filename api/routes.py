"""HTTP endpoints. The loaded model lives on app.state.embedder."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    EmbeddingResponse,
    EmbeddingsResponse,
    EmbedDocumentsRequest,
    EmbedQueryRequest,
)
from embedder import STEmbeddings

logger = logging.getLogger("embeddings-service")

router = APIRouter()


def _embedder(request: Request) -> STEmbeddings:
    embedder: Optional[STEmbeddings] = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    return embedder


@router.get("/health")
def health(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": embedder.spec.model_id,
        "dimensions": embedder.dim,
        "max_seq_length": embedder.model.max_seq_length,
        "device": embedder.device,
        "dtype": embedder.dtype,
        "document_template": embedder.document_template,
    }


@router.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(payload: EmbedQueryRequest, request: Request):
    """Embed a single query string, with the query instruction applied automatically."""
    embedder = _embedder(request)
    try:
        vector = embedder.embed_query(payload.text, task=payload.task)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_query failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingResponse(
        embedding=vector.tolist(),
        dimensions=len(vector),
        prompt=embedder.query_prompt(payload.task),
    )


@router.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(payload: EmbedDocumentsRequest, request: Request):
    """Embed a batch of documents/chunks (respects EMBED_BATCH_SIZE internally)."""
    embedder = _embedder(request)
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    if payload.titles is not None and len(payload.titles) != len(payload.texts):
        raise HTTPException(status_code=400, detail="titles must have the same length as texts")

    try:
        vectors = embedder.embed_documents(payload.texts, titles=payload.titles)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_documents failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingsResponse(
        embeddings=vectors.tolist(),
        dimensions=int(vectors.shape[1]),
        count=int(vectors.shape[0]),
    )
