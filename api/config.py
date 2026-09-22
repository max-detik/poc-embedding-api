"""Service settings, all read from the environment (see .env.example)."""

import os
from dataclasses import dataclass
from typing import Optional


def _int_or_none(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value else None


@dataclass(frozen=True)
class Settings:
    model_key: str = "harrier"
    device: Optional[str] = None  # None -> CUDA when available, else CPU
    dtype: Optional[str] = None
    batch_size: int = 64
    max_seq_length: Optional[int] = None
    truncate_dim: Optional[int] = None
    task: Optional[str] = None
    # {title} / {content}; None keeps the model's default.
    document_template: Optional[str] = None
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        template = os.getenv("EMBED_DOCUMENT_TEMPLATE") or None
        return cls(
            device=os.getenv("DEVICE") or None,
            dtype=os.getenv("DTYPE") or None,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
            max_seq_length=_int_or_none("EMBED_MAX_SEQ_LENGTH"),
            truncate_dim=_int_or_none("EMBED_TRUNCATE_DIM"),
            task=os.getenv("EMBED_TASK") or None,
            # A literal "\n" in the env value becomes a newline -- most .env
            # loaders and dashboards don't expand escapes themselves.
            document_template=template.replace("\\n", "\n") if template else None,
            port=int(os.getenv("PORT", "8000")),
        )
