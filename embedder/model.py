"""STEmbeddings: a SentenceTransformer with per-model prompt handling.

Callers just use embed_query / embed_documents; the right query instruction
and document template are applied for whichever model is loaded.
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from embedder import chunking, prompts
from embedder.runtime import load_sentence_transformer, resolve_device, resolve_dtype
from embedder.specs import get_spec

logger = logging.getLogger("embedder")

Titles = Optional[List[Optional[str]]]


class STEmbeddings:
    """Thin wrapper over SentenceTransformer with per-model prompt handling."""

    def __init__(
        self,
        model_key: str = "harrier",
        device: Optional[str] = None,  # None -> CUDA when available, else CPU
        dtype: Optional[str] = None,  # "float16" on Ampere+, "bfloat16", or "float32"
        truncate_dim: Optional[int] = None,  # Matryoshka; None keeps native_dim
        task: Optional[str] = None,  # overrides the spec's default_task
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        sort_by_length: bool = True,
        document_template: Optional[str] = None,  # overrides the spec's; {title} / {content}
    ):
        self.key = model_key
        self.spec = get_spec(model_key)
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.task = task or self.spec.default_task
        self.truncate_dim = truncate_dim
        self.dim = truncate_dim or self.spec.native_dim
        self.document_template = document_template or self.spec.document_template
        prompts.validate_document_template(self.document_template)

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.model = load_sentence_transformer(self.spec, self.device, self.dtype, truncate_dim)
        self.model.max_seq_length = max_seq_length or self.spec.max_seq_length
        self.model.eval()

        logger.info(
            "Loaded %s (%s) dim=%d max_seq_length=%d device=%s dtype=%s document_template=%r",
            model_key, self.spec.model_id, self.dim,
            self.model.max_seq_length, self.device, self.dtype, self.document_template,
        )

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    def query_prompt(self, task: Optional[str] = None) -> str:
        """The prefix applied to queries, for inspection or reuse."""
        return prompts.query_prompt(self.spec, task or self.task)

    def format_document(self, content: str, title: Optional[str] = None) -> str:
        """The exact string embedded for one document, for inspection or storage."""
        return prompts.format_document(self.spec, self.document_template, content, title)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _encode(self, texts: List[str], prompt: str, batch_size: Optional[int]) -> np.ndarray:
        return self.model.encode(
            texts,
            prompt=prompt or None,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,  # truncate-then-normalize is handled internally
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None, titles: Titles = None
    ) -> np.ndarray:
        """Embed documents, each rendered through `document_template`.
        `titles` pairs one title with each text (None/"" -> the model's empty
        title). Returns (n, dim) float32, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        docs = [
            self.format_document(text, title)
            for text, title in zip(texts, prompts.pair_titles(texts, titles))
        ]
        if not self.sort_by_length or len(docs) <= 1:
            return self._encode(docs, "", batch_size)

        # Group similar lengths together so one long article does not force
        # every other text in its batch to be padded up to that length.
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
        sorted_vecs = self._encode([docs[i] for i in order], "", batch_size)
        out = np.empty_like(sorted_vecs)
        out[order] = sorted_vecs
        return out

    def embed_query(self, text: str, task: Optional[str] = None) -> np.ndarray:
        """Embed a single query. Returns (dim,) float32, L2-normalized."""
        return self._encode([text], self.query_prompt(task), batch_size=1)[0]

    def embed_queries(
        self, texts: List[str], task: Optional[str] = None, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Embed many queries, e.g. when running a retrieval eval."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts, self.query_prompt(task), batch_size)

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    def token_length(self, text: str, is_query: bool = False, title: Optional[str] = None) -> int:
        """Token count of the exact string embedded, for chunking decisions."""
        full = self.query_prompt() + text if is_query else self.format_document(text, title)
        return len(self.model.tokenizer(full)["input_ids"])

    def chunk_text(
        self, text: str, chunk_overlap: int = 0, title: Optional[str] = None
    ) -> List[str]:
        """Split text so each piece, once rendered through `document_template`
        with `title`, still fits max_seq_length. Returns raw content chunks --
        pass the same title to embed_documents to embed them."""
        return chunking.chunk_text(
            self.model.tokenizer,
            text,
            max_seq_length=self.model.max_seq_length,
            reserved_text=prompts.template_overhead(self.spec, self.document_template, title),
            chunk_overlap=chunk_overlap,
        )

    def embed_documents_chunked(
        self, texts: List[str], chunk_overlap: int = 0, titles: Titles = None
    ) -> List[List[dict]]:
        """Chunk-then-embed. Each input becomes a list of
        {"text", "embedded_text", "embedding"}: the raw chunk, the exact string
        embedded (template + title applied), and its vector as a list."""
        results: List[List[dict]] = []
        for text, title in zip(texts, prompts.pair_titles(texts, titles)):
            chunk_texts = self.chunk_text(text, chunk_overlap=chunk_overlap, title=title)
            vectors = self.embed_documents(chunk_texts, titles=[title] * len(chunk_texts))
            results.append([
                {
                    "text": ct,
                    "embedded_text": self.format_document(ct, title),
                    "embedding": v.tolist(),
                }
                for ct, v in zip(chunk_texts, vectors)
            ])
        return results

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """Pay the first-call cost at startup instead of on a user query."""
        self.embed_query("pemanasan model")
        self.embed_documents(["pemanasan model"])
