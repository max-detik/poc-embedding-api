"""App entry point: `uvicorn api.main:app` (or `python -m api.main`).

Device and dtype default to CUDA + float16 when a GPU is present and CPU +
float32 otherwise; set DEVICE / DTYPE to pin them (see .env.example).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)

from api.config import Settings  # noqa: E402 -- logging must be configured first
from api.routes import router  # noqa: E402
from embedder import STEmbeddings  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and warm before the healthcheck passes, so the first real request
    # doesn't pay the download/first-call cost.
    settings = Settings.from_env()
    embedder = STEmbeddings(
        settings.model_key,
        device=settings.device,
        dtype=settings.dtype,
        batch_size=settings.batch_size,
        max_seq_length=settings.max_seq_length,
        truncate_dim=settings.truncate_dim,
        task=settings.task,
        document_template=settings.document_template,
    )
    embedder.warmup()
    app.state.embedder = embedder
    yield
    app.state.embedder = None


app = FastAPI(
    title="Embeddings Service",
    description="Local embedding inference for microsoft/harrier-oss-v1-0.6b (sentence-transformers on torch) wrapped in FastAPI.",
    version="3.0.0",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=Settings.from_env().port, reload=False)
