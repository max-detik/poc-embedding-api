"""Request and response bodies."""

from typing import List, Optional

from pydantic import BaseModel, Field


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single query string (the query instruction is applied automatically).")
    task: Optional[str] = Field(
        None,
        description="Overrides the task description in the query instruction for this request.",
    )


class EmbedDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., description="Batch of documents/chunks to embed.")
    titles: Optional[List[Optional[str]]] = Field(
        None,
        description=(
            "One title per text, filled into the document template's {title}. "
            "Must match texts in length; null entries mean no title."
        ),
    )


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    prompt: str


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int
