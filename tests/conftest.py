"""Test fixtures.

The tests never load real weights: a fake SentenceTransformer records what it
is asked to encode and returns deterministic vectors. torch and
sentence-transformers are replaced with minimal stand-ins when they aren't
installed (e.g. on a machine without a supported torch build), so the suite
runs anywhere.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest


def _install_stand_ins() -> None:
    if importlib.util.find_spec("torch") is None:
        torch = types.ModuleType("torch")
        torch.float16, torch.bfloat16, torch.float32 = "float16", "bfloat16", "float32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.inference_mode = lambda: (lambda fn: fn)
        sys.modules["torch"] = torch
    if importlib.util.find_spec("sentence_transformers") is None:
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = object
        sys.modules["sentence_transformers"] = st


_install_stand_ins()


class WhitespaceTokenizer:
    """One token per whitespace-separated word, plus one special token."""

    def __call__(self, text, add_special_tokens=True, **_):
        ids = text.split()
        return {"input_ids": ids + (["<eos>"] if add_special_tokens else [])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(i for i in ids if not (skip_special_tokens and i == "<eos>"))

    def num_special_tokens_to_add(self, pair=False):
        return 1


class FakeSentenceTransformer:
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.tokenizer = WhitespaceTokenizer()
        self.max_seq_length = 512
        self.calls = []  # (prompt, texts) per encode call

    def eval(self):
        return self

    def encode(self, texts, prompt=None, **_):
        self.calls.append((prompt, list(texts)))
        dim = 768 if "gemma" in self.model_id else 1024
        # Deterministic per text, so vectors can be compared across calls.
        vecs = np.stack([
            np.random.default_rng(abs(hash((prompt, t))) % 2**32).standard_normal(dim)
            for t in texts
        ]).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_sentence_transformer(monkeypatch):
    import embedder.runtime

    monkeypatch.setattr(embedder.runtime, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    return FakeSentenceTransformer


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()
