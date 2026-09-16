"""
FastAPI service exposing local ONNX embedding inference via plain ONNX
Runtime (no torch/sentence-transformers/optimum/llama-index in the request
path -- see onnx_embeddings.py for why).

IMPORTANT — Railway + GPU: Railway does not currently offer GPU-backed
services, so this deploys with device="cpu" and provider="CPUExecutionProvider"
by default. All of the above stays fully configurable via env vars, so the
exact same code runs unchanged with device="cuda" / CUDAExecutionProvider if
you deploy this on a GPU host instead (see .env.example).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from onnx_embeddings import ONNXEmbeddings
from onnx_reranker import ONNXReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx-embeddings-service")

# ---------------------------------------------------------------------------
# Config (all overridable via env vars — see .env.example)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "onnx-community/embeddinggemma-300m-ONNX")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" on Railway; "cuda" if deployed on a GPU host
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

ONNX_FILE_NAME = os.getenv("ONNX_FILE_NAME", "model_quantized.onnx")
ONNX_SUBFOLDER = os.getenv("ONNX_SUBFOLDER", "onnx")
# CPUExecutionProvider by default (Railway has no GPU). Set to
# CUDAExecutionProvider only when DEVICE=cuda on a GPU-enabled host.
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")

QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "task: search result | query: ")
# Some models (embeddinggemma included) also support/expect a distinct
# instruction prefix for documents being indexed. Leave unset to use the
# model's default.
TEXT_INSTRUCTION = os.getenv("TEXT_INSTRUCTION", "")

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "onnx-community/Qwen3-Reranker-0.6B-ONNX")
RERANKER_ONNX_FILE_NAME = os.getenv("RERANKER_ONNX_FILE_NAME", "model_quantized.onnx")
RERANKER_ONNX_SUBFOLDER = os.getenv("RERANKER_ONNX_SUBFOLDER", "onnx")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "8192"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
# Task description baked into Qwen3-Reranker's judging prompt. Ignored by
# cross-encoder rerankers (bge-reranker-v2-m3), which have no such prompt.
RERANKER_INSTRUCTION = os.getenv(
    "RERANKER_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)

if DEVICE == "cuda" and ONNX_PROVIDER == "CPUExecutionProvider":
    logger.warning(
        "DEVICE=cuda but ONNX_PROVIDER=CPUExecutionProvider — set "
        "ONNX_PROVIDER=CUDAExecutionProvider to actually use the GPU."
    )

# Embedding/reranker models are created once at startup and reused across requests.
_embed_model: Optional[ONNXEmbeddings] = None
_reranker_model: Optional[ONNXReranker] = None


def build_embed_model() -> ONNXEmbeddings:
    logger.info(
        "Loading ONNXEmbeddings: model=%s device=%s onnx_file=%s provider=%s",
        MODEL_NAME, DEVICE, ONNX_FILE_NAME, ONNX_PROVIDER,
    )
    return ONNXEmbeddings(
        model_id=MODEL_NAME,
        onnx_file_name=ONNX_FILE_NAME,
        onnx_subfolder=ONNX_SUBFOLDER,
        device=DEVICE,
        provider=ONNX_PROVIDER,
        max_length=MAX_LENGTH,
        batch_size=EMBED_BATCH_SIZE,
        query_instruction=QUERY_INSTRUCTION,
        text_instruction=TEXT_INSTRUCTION,
    )


def build_reranker_model() -> ONNXReranker:
    logger.info(
        "Loading ONNXReranker: model=%s device=%s onnx_file=%s provider=%s",
        RERANKER_MODEL_NAME, DEVICE, RERANKER_ONNX_FILE_NAME, ONNX_PROVIDER,
    )
    return ONNXReranker(
        model_id=RERANKER_MODEL_NAME,
        onnx_file_name=RERANKER_ONNX_FILE_NAME,
        onnx_subfolder=RERANKER_ONNX_SUBFOLDER,
        device=DEVICE,
        provider=ONNX_PROVIDER,
        max_length=RERANKER_MAX_LENGTH,
        batch_size=RERANKER_BATCH_SIZE,
        instruction=RERANKER_INSTRUCTION,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _reranker_model
    _embed_model = build_embed_model()
    _reranker_model = build_reranker_model()
    yield
    _embed_model = None
    _reranker_model = None


app = FastAPI(
    title="ONNX Embeddings Service",
    description="Local ONNX embedding inference (plain ONNX Runtime) wrapped in FastAPI, deployable on Railway.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single query string (query_instruction is applied automatically).")
    instruction: Optional[str] = Field(
        None,
        description=(
            "Overrides the configured query_instruction prefix for this request. "
            "Pass an empty string to embed the text with no prefix at all."
        ),
    )


class EmbedDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., description="Batch of documents/chunks to embed.")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query to rerank documents against.")
    documents: List[str] = Field(..., description="Candidate documents to score and rerank.")
    top_n: Optional[int] = Field(None, description="If set, only return the top N results.")
    instruction: Optional[str] = Field(
        None,
        description=(
            "Overrides the reranker's task description for this request. "
            "Only used by instruction-following rerankers (e.g. Qwen3-Reranker)."
        ),
    )


class RerankResultItem(BaseModel):
    index: int
    document: str
    score: float


class RerankResponse(BaseModel):
    results: List[RerankResultItem]
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "onnx_file_name": ONNX_FILE_NAME,
        "onnx_provider": ONNX_PROVIDER,
        "reranker_model": RERANKER_MODEL_NAME,
        "reranker_onnx_file_name": RERANKER_ONNX_FILE_NAME,
    }


@app.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(payload: EmbedQueryRequest):
    """Embed a single query string, with query_instruction applied automatically."""
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    try:
        vector = _embed_model.embed_query(payload.text, instruction=payload.instruction)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_query failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingResponse(embedding=vector, dimensions=len(vector))


@app.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(payload: EmbedDocumentsRequest):
    """Embed a batch of documents/chunks (respects embed_batch_size internally)."""
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        vectors = _embed_model.embed_documents(payload.texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_documents failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    dims = len(vectors[0]) if vectors else 0
    return EmbeddingsResponse(embeddings=vectors, dimensions=dims, count=len(vectors))


@app.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest):
    """Score and rerank documents against a query, highest relevance first."""
    if _reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not initialized")
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must be a non-empty list")

    try:
        results = _reranker_model.rerank(
            payload.query,
            payload.documents,
            top_n=payload.top_n,
            instruction=payload.instruction,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rerank failed")
        raise HTTPException(status_code=502, detail=f"Rerank request failed: {exc}") from exc

    return RerankResponse(results=results, count=len(results))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)