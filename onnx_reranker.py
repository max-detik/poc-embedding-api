"""
A minimal reranker class that runs an ONNX reranker model locally via plain
ONNX Runtime + a fast tokenizer -- no torch, sentence-transformers, or
optimum in the loop (same rationale as onnx_embeddings.py).

    reranker = ONNXReranker(
        model_id="onnx-community/Qwen3-Reranker-0.6B-ONNX",
        onnx_file_name="model_q4.onnx",
    )
    results = reranker.rerank("what is panda?", [
        "The giant panda is a bear species endemic to China.",
        "Paris is the capital of France.",
    ])

Two export shapes are supported, picked automatically:

  - Sequence-classification cross-encoders (e.g. bge-reranker-v2-m3): the
    (query, document) pair is fed as a token-type-separated pair and the
    single output logit is squashed with a sigmoid.
  - Causal-LM rerankers (e.g. Qwen3-Reranker): the pair is wrapped in the
    model's judging chat prompt and the score is the softmax probability of
    the "yes" token over the "no" token at the final position. These exports
    need left padding and empty KV-cache inputs, both handled here.

Either way the pair is scored jointly through the transformer rather than by
comparing independently-computed vectors -- slower per pair than embeddings,
but more accurate for top-k re-ranking of an initial retrieval result set.

Notes:
  - `device` is a convenience that maps to an ONNX Runtime execution provider
    ("cuda" -> CUDAExecutionProvider, "cpu" -> CPUExecutionProvider). Pass
    `provider=` directly if you need a specific provider (e.g. TensorrtExecutionProvider).
  - GPU inference (CUDAExecutionProvider) requires the `onnxruntime-gpu`
    package instead of plain `onnxruntime` -- see requirements.txt.
  - Scores are in [0, 1] for both export shapes, so callers and thresholds
    do not have to care which model is loaded.
"""

import logging
import os
from typing import List, Optional, TypedDict

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger("onnx-reranker")

_DEVICE_TO_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}

_ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

# Decoder-style rerankers that judge relevance by emitting "yes" / "no".
_CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "mistral", "llama"}

DEFAULT_RERANK_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Judging prompt Qwen3-Reranker was trained with; the score reads the token
# the assistant would emit right after this.
_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankResult(TypedDict):
    index: int
    document: str
    score: float


class ONNXReranker:
    """Relevance scoring of (query, document) pairs from a local ONNX model."""

    def __init__(
        self,
        model_id: str,
        onnx_file_name: str = "model_quantized.onnx",
        onnx_subfolder: str = "onnx",
        device: str = "cpu",
        provider: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 4,
        instruction: str = DEFAULT_RERANK_TASK,
        scoring: str = "auto",  # "auto" | "cross_encoder" | "causal_yes_no"
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction

        resolved_provider = provider or _DEVICE_TO_PROVIDER.get(
            device.lower(), "CPUExecutionProvider"
        )

        logger.info(
            "Loading ONNX reranker model=%s onnx_file_name=%s provider=%s",
            model_id, onnx_file_name, resolved_provider,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if scoring == "auto":
            try:
                config = AutoConfig.from_pretrained(model_id)
                model_type = config.model_type
                architectures = getattr(config, "architectures", None) or []
            except Exception:
                model_type, architectures = "", []
            is_causal = model_type in _CAUSAL_MODEL_TYPES or any(
                a.endswith("ForCausalLM") for a in architectures
            )
            scoring = "causal_yes_no" if is_causal else "cross_encoder"
        if scoring not in ("cross_encoder", "causal_yes_no"):
            raise ValueError(f"Unknown scoring mode: {scoring}")
        self.scoring = scoring

        if self.scoring == "causal_yes_no":
            self._yes_id = self.tokenizer.convert_tokens_to_ids("yes")
            self._no_id = self.tokenizer.convert_tokens_to_ids("no")
            unk = self.tokenizer.unk_token_id
            if self._yes_id is None or self._no_id is None or unk in (self._yes_id, self._no_id):
                raise ValueError(
                    "Tokenizer has no single-token 'yes'/'no' -- this model cannot be "
                    "scored in causal_yes_no mode."
                )
            self._pad_id = self.tokenizer.pad_token_id
            if self._pad_id is None:
                self._pad_id = self.tokenizer.eos_token_id or 0
            self._prefix_ids = self.tokenizer.encode(_QWEN_PREFIX, add_special_tokens=False)
            self._suffix_ids = self.tokenizer.encode(_QWEN_SUFFIX, add_special_tokens=False)
            self._pair_budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
            if self._pair_budget <= 0:
                raise ValueError(
                    f"max_length={self.max_length} is too small for the judging prompt "
                    f"({len(self._prefix_ids) + len(self._suffix_ids)} tokens)"
                )

        if onnx_subfolder == "":
            onnx_file_path = f"{onnx_file_name}"
        else:
            onnx_file_path = f"{onnx_subfolder}/{onnx_file_name}"

        onnx_path = hf_hub_download(
            repo_id=model_id,
            filename=onnx_file_path,
        )
        # Models in the ONNX external-data format keep large weights in a
        # sibling "<file>_data" blob that must sit next to the .onnx graph.
        try:
            hf_hub_download(
                repo_id=model_id, filename=f"{onnx_file_path}_data"
            )
        except Exception:
            pass

        sess_options = ort.SessionOptions()
        # Favor a smaller resident memory footprint over throughput:
        # Railway's small instances are memory-constrained, not CPU-constrained.
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = intra_op_num_threads or int(
            os.getenv("ORT_INTRA_OP_THREADS", "1")
        )
        sess_options.inter_op_num_threads = inter_op_num_threads or int(
            os.getenv("ORT_INTER_OP_THREADS", "1")
        )

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=[resolved_provider]
        )

        self._input_meta = {i.name: i for i in self.session.get_inputs()}
        self._past_inputs = [n for n in self._input_meta if n.startswith("past_key_values")]
        output_names = [o.name for o in self.session.get_outputs()]

        # Only fetch the tensor we need, so present KV caches are never returned.
        self._score_output = "logits" if "logits" in output_names else output_names[0]

        logger.info(
            "ONNX reranker model ready: %s (provider=%s, scoring=%s, output=%s, inputs=%s)",
            model_id, resolved_provider, self.scoring, self._score_output,
            [n for n in self._input_meta if not n.startswith("past_key_values")],
        )

    # ------------------------------------------------------------------ #
    # Tokenization and graph inputs
    # ------------------------------------------------------------------ #
    def _tokenize_cross_encoder(self, query: str, documents: List[str]):
        enc = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        tti = enc.get("token_type_ids")
        return (
            enc["input_ids"].astype(np.int64),
            enc["attention_mask"].astype(np.int64),
            None if tti is None else tti.astype(np.int64),
        )

    def _tokenize_causal(self, query: str, documents: List[str], instruction: str):
        """Wrap each pair in the judging prompt and pad on the left, so the
        final position is always the token the model would answer at."""
        pairs = [
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        enc = self.tokenizer(
            pairs,
            truncation=True,
            max_length=self._pair_budget,
            add_special_tokens=False,
        )
        seqs = [self._prefix_ids + ids + self._suffix_ids for ids in enc["input_ids"]]

        max_len = max(len(s) for s in seqs)
        input_ids = np.full((len(seqs), max_len), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            input_ids[i, max_len - len(s):] = s
            attention_mask[i, max_len - len(s):] = 1
        return input_ids, attention_mask, None

    def _build_feeds(self, input_ids, attention_mask, token_type_ids=None) -> dict:
        feeds = {}
        if "input_ids" in self._input_meta:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_meta:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_meta:
            feeds["token_type_ids"] = (
                token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)
            )
        if "position_ids" in self._input_meta:
            pos = np.cumsum(attention_mask, axis=1) - 1
            feeds["position_ids"] = np.clip(pos, 0, None).astype(np.int64)

        # Exports "with past" need empty KV caches:
        # [batch, num_kv_heads, past_seq_len=0, head_dim]
        batch = input_ids.shape[0]
        for name in self._past_inputs:
            meta = self._input_meta[name]
            shape = []
            for idx, dim in enumerate(meta.shape):
                if isinstance(dim, int):
                    shape.append(dim)
                elif idx == 0:
                    shape.append(batch)
                else:
                    shape.append(0)
            dtype = _ORT_TYPE_TO_NUMPY.get(meta.type, np.float32)
            feeds[name] = np.zeros(shape, dtype=dtype)
        return feeds

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _score_batch(
        self, query: str, documents: List[str], instruction: str
    ) -> List[float]:
        if self.scoring == "causal_yes_no":
            input_ids, attention_mask, tti = self._tokenize_causal(
                query, documents, instruction
            )
        else:
            input_ids, attention_mask, tti = self._tokenize_cross_encoder(query, documents)

        feeds = self._build_feeds(input_ids, attention_mask, tti)
        (out,) = self.session.run([self._score_output], feeds)
        out = out.astype(np.float32)

        if self.scoring == "cross_encoder":
            # Some exports emit (batch, 1), others (batch,).
            return self._sigmoid(out.reshape(-1)).tolist()

        # Causal: softmax over the "yes" / "no" logits at the final position,
        # which is the real last token because the batch is left-padded.
        last = out[:, -1, :]
        pair = np.stack([last[:, self._no_id], last[:, self._yes_id]], axis=1)
        pair -= pair.max(axis=1, keepdims=True)
        probs = np.exp(pair)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1].tolist()

    def compute_scores(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> List[float]:
        """Score each document against `query`, in the given order.

        `instruction` overrides the task description baked into the judging
        prompt; it is ignored by cross-encoder models, which have no such prompt.
        """
        if not documents:
            return []

        task = self.instruction if instruction is None else instruction
        scores: List[float] = []
        for i in range(0, len(documents), self.batch_size):
            chunk = documents[i : i + self.batch_size]
            scores.extend(self._score_batch(query, chunk, task))
        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        """Score `documents` against `query` and return them sorted by
        relevance (highest first), each tagged with its original index.

        Pass `top_n` to truncate to the top N results.
        """
        scores = self.compute_scores(query, documents, instruction=instruction)
        results: List[RerankResult] = [
            {"index": i, "document": doc, "score": score}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n] if top_n is not None else results
