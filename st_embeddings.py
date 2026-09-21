"""SentenceTransformer wrapper for microsoft/harrier-oss-v1-0.6b.

    harrier : microsoft/harrier-oss-v1-0.6b      1024d, last-token pooling, Qwen3 arch

Harrier needs asymmetric prompting: queries carry a task instruction,
documents carry no prefix. The wrapper handles that so callers just use
embed_query / embed_documents.

Harrier is ungated. HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is still honored if
set, e.g. for higher Hub rate limits or a private mirror.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("st-embeddings")

# Never hardcode this -- a token in source is a leaked token.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def resolve_device(device: Optional[str] = None) -> str:
    """Explicit device wins; otherwise CUDA if present, else CPU."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: Optional[str], device: str) -> str:
    """float16 is a GPU dtype -- on CPU it is unsupported by many kernels and
    slower where it works, so CPU falls back to float32."""
    if dtype:
        if device.startswith("cpu") and dtype == "float16":
            logger.warning("float16 is not usable on CPU; falling back to float32")
            return "float32"
        return dtype
    return "float16" if device.startswith("cuda") else "float32"


@dataclass
class ModelSpec:
    model_id: str
    native_dim: int
    max_seq_length: int
    # How the query instruction is formatted. None means the model uses fixed
    # prefixes instead of a free-text task instruction.
    instruct_template: Optional[str] = None
    # Fixed prefixes, used when instruct_template is None.
    query_prefix: str = ""
    document_prefix: str = ""
    # Decoder models pool the last token, so padding must be on the left.
    left_padding: bool = False
    default_task: str = ""
    extra_model_kwargs: Dict = field(default_factory=dict)


# Harrier's template ends in "Query: " with a trailing space -- keep it; it
# matches the model's training format.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "harrier": ModelSpec(
        model_id="microsoft/harrier-oss-v1-0.6b",
        native_dim=1024,
        max_seq_length=1024,  # supports 32k; cap it for throughput
        instruct_template="Instruct: {task}\nQuery: ",
        left_padding=True,
        default_task="Given a web search query, retrieve relevant passages that answer the query",
    ),
}


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
    ):
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model_key {model_key!r}. Options: {list(MODEL_SPECS)}")

        self.key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.task = task or self.spec.default_task
        self.truncate_dim = truncate_dim
        self.dim = truncate_dim or self.spec.native_dim

        device = resolve_device(device)
        dtype = resolve_dtype(dtype, device)
        self.device = device
        self.dtype = dtype

        model_kwargs = {"torch_dtype": getattr(torch, dtype)}
        model_kwargs.update(self.spec.extra_model_kwargs)
        # flash-attn is a CUDA-only kernel; asking for it on CPU is a hard error.
        if not device.startswith("cuda"):
            model_kwargs.pop("attn_implementation", None)

        tokenizer_kwargs = {}
        if self.spec.left_padding:
            tokenizer_kwargs["padding_side"] = "left"

        load_kwargs = dict(
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs or None,
            truncate_dim=truncate_dim,
        )
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN

        try:
            self.model = SentenceTransformer(self.spec.model_id, **load_kwargs)
        except Exception as exc:
            # flash-attn is optional; fall back to SDPA, which is fine on A30.
            if "flash" not in str(exc).lower():
                raise
            logger.warning("flash_attention_2 unavailable, falling back to sdpa: %s", exc)
            model_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["model_kwargs"] = model_kwargs
            self.model = SentenceTransformer(self.spec.model_id, **load_kwargs)

        self.model.max_seq_length = max_seq_length or self.spec.max_seq_length
        self.model.eval()

        logger.info(
            "Loaded %s (%s) dim=%d max_seq_length=%d device=%s dtype=%s",
            model_key, self.spec.model_id, self.dim,
            self.model.max_seq_length, device, dtype,
        )

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    def query_prompt(self, task: Optional[str] = None) -> str:
        """The prefix applied to queries, for inspection or reuse."""
        if self.spec.instruct_template is None:
            return self.spec.query_prefix
        return self.spec.instruct_template.format(task=task or self.task)

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
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Embed documents. Returns (n, dim) float32, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if not self.sort_by_length or len(texts) <= 1:
            return self._encode(texts, self.spec.document_prefix, batch_size)

        # Group similar lengths together so one long article does not force
        # every other text in its batch to be padded up to that length.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_vecs = self._encode(
            [texts[i] for i in order], self.spec.document_prefix, batch_size
        )
        out = np.empty_like(sorted_vecs)
        out[order] = sorted_vecs
        return out

    def embed_query(self, text: str, task: Optional[str] = None) -> np.ndarray:
        """Embed a single query. Returns (dim,) float32, L2-normalized."""
        return self._encode([text], self.query_prompt(task), batch_size=1)[0]

    def embed_queries(
        self, texts: List[str], task: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed many queries, e.g. when running a retrieval eval."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts, self.query_prompt(task), batch_size)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """Pay the first-call cost at startup instead of on a user query."""
        self.embed_query("pemanasan model")
        self.embed_documents(["pemanasan model"])

    def token_length(self, text: str, is_query: bool = False) -> int:
        """Token count including the model's prefix, for chunking decisions."""
        prefix = self.query_prompt() if is_query else self.spec.document_prefix
        return len(self.model.tokenizer(prefix + text)["input_ids"])

    def chunk_text(self, text: str, chunk_overlap: int = 0) -> List[str]:
        """Split text so each piece fits max_seq_length with the document prefix."""
        tok = self.model.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]

        prefix_len = (
            len(tok(self.spec.document_prefix, add_special_tokens=False)["input_ids"])
            if self.spec.document_prefix else 0
        )
        special_len = max(tok.num_special_tokens_to_add(pair=False), 1)
        budget = self.model.max_seq_length - prefix_len - special_len
        if budget <= 0:
            raise ValueError(f"max_seq_length={self.model.max_seq_length} too small for prefix")
        if len(ids) <= budget:
            return [text]
        if chunk_overlap >= budget:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk size ({budget})")

        step = budget - chunk_overlap
        chunks = []
        for start in range(0, len(ids), step):
            chunks.append(tok.decode(ids[start : start + budget], skip_special_tokens=True))
            if start + budget >= len(ids):
                break
        return chunks

    def embed_documents_chunked(
        self, texts: List[str], chunk_overlap: int = 0
    ) -> List[List[dict]]:
        """Chunk-then-embed. Each input becomes a list of {"text", "embedding"}."""
        results: List[List[dict]] = []
        for text in texts:
            chunk_texts = self.chunk_text(text, chunk_overlap=chunk_overlap)
            vectors = self.embed_documents(chunk_texts)
            results.append(
                [{"text": ct, "embedding": v.tolist()} for ct, v in zip(chunk_texts, vectors)]
            )
        return results
