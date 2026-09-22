"""Which embedding models exist, and how each one is prompted.

Adding a model is one entry in MODEL_SPECS; nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    native_dim: int
    max_seq_length: int
    # How the query instruction is formatted. None means the model uses a fixed
    # query_prefix instead of a free-text task instruction (Gemma).
    instruct_template: Optional[str] = None
    query_prefix: str = ""
    # How a document is rendered before embedding. Placeholders: {title},
    # {content}. "{content}" alone means documents carry no prefix.
    document_template: str = "{content}"
    # Filled into {title} when a document has no title.
    empty_title: str = ""
    # Decoder models pool the last token, so padding must be on the left.
    left_padding: bool = False
    default_task: str = ""
    extra_model_kwargs: Dict = field(default_factory=dict)


MODEL_SPECS: Dict[str, ModelSpec] = {
    # Harrier's template ends in "Query: " with a trailing space -- keep it; it
    # matches the model's training format.
    "harrier": ModelSpec(
        model_id="microsoft/harrier-oss-v1-0.6b",
        native_dim=1024,
        max_seq_length=1024,  # supports 32k; cap it for throughput
        instruct_template="Instruct: {task}\nQuery: ",
        left_padding=True,
        default_task="Given a web search query, retrieve relevant passages that answer the query",
    ),
    # Gated repo: needs HF_TOKEN for an account that accepted its license.
    "gemma": ModelSpec(
        model_id="google/embeddinggemma-300m",
        native_dim=768,
        max_seq_length=2048,
        query_prefix="task: search result | query: ",
        # Gemma's documented format; "none" is its own placeholder for no title.
        document_template="title: {title} | text: {content}",
        empty_title="none",
    ),
}


def get_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Options: {list(MODEL_SPECS)}"
        ) from None
