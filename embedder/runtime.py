"""Where and how a model runs: device, dtype, Hub auth, and loading."""

import logging
import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from embedder.specs import ModelSpec

logger = logging.getLogger("embedder")


def hf_token() -> Optional[str]:
    # Read from the environment, never hardcoded -- a token in source is a
    # leaked token. Only gated models (gemma) need it.
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


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


def load_sentence_transformer(
    spec: ModelSpec, device: str, dtype: str, truncate_dim: Optional[int]
) -> SentenceTransformer:
    model_kwargs = {"torch_dtype": getattr(torch, dtype), **spec.extra_model_kwargs}
    # flash-attn is a CUDA-only kernel; asking for it on CPU is a hard error.
    if not device.startswith("cuda"):
        model_kwargs.pop("attn_implementation", None)

    load_kwargs = dict(
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"} if spec.left_padding else None,
        truncate_dim=truncate_dim,
    )
    token = hf_token()
    if token:
        load_kwargs["token"] = token

    try:
        return SentenceTransformer(spec.model_id, **load_kwargs)
    except Exception as exc:
        # flash-attn is optional; fall back to SDPA, which is fine on A30.
        if "flash" not in str(exc).lower():
            raise
        logger.warning("flash_attention_2 unavailable, falling back to sdpa: %s", exc)
        load_kwargs["model_kwargs"] = {**model_kwargs, "attn_implementation": "sdpa"}
        return SentenceTransformer(spec.model_id, **load_kwargs)
