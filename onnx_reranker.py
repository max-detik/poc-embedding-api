"""
A minimal cross-encoder reranker class that runs an ONNX reranker model
locally via plain ONNX Runtime + a fast tokenizer -- no torch,
sentence-transformers, or optimum in the loop (same rationale as
onnx_embeddings.py).

    reranker = ONNXReranker(
        model_id="onnx-community/bge-reranker-v2-m3-ONNX",
        onnx_file_name="model_quantized.onnx",
    )
    results = reranker.rerank("what is panda?", [
        "The giant panda is a bear species endemic to China.",
        "Paris is the capital of France.",
    ])

Unlike an embedding model, a reranker is a cross-encoder: it scores each
(query, document) pair jointly through the transformer rather than
comparing independently-computed vectors, which is slower per-pair but
more accurate for top-k re-ranking of an initial retrieval result set.

Notes:
  - `device` is a convenience that maps to an ONNX Runtime execution provider
    ("cuda" -> CUDAExecutionProvider, "cpu" -> CPUExecutionProvider). Pass
    `provider=` directly if you need a specific provider (e.g. TensorrtExecutionProvider).
  - GPU inference (CUDAExecutionProvider) requires the `onnxruntime-gpu`
    package instead of plain `onnxruntime` -- see requirements.txt.
  - Raw logits are passed through a sigmoid to produce a relevance score in
    [0, 1], matching how BGE reranker scores are conventionally interpreted.
"""

import logging
import os
from typing import List, Optional, TypedDict

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

logger = logging.getLogger("onnx-reranker")

_DEVICE_TO_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


class RerankResult(TypedDict):
    index: int
    document: str
    score: float


class ONNXReranker:
    """Cross-encoder relevance scoring from a local ONNX reranker model."""

    def __init__(
        self,
        model_id: str,
        onnx_file_name: str = "model_quantized.onnx",
        onnx_subfolder: str = "onnx",
        device: str = "cpu",
        provider: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 4,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size

        resolved_provider = provider or _DEVICE_TO_PROVIDER.get(
            device.lower(), "CPUExecutionProvider"
        )

        logger.info(
            "Loading ONNX reranker model=%s onnx_file_name=%s provider=%s",
            model_id, onnx_file_name, resolved_provider,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        onnx_path = hf_hub_download(
            repo_id=model_id, filename=f"{onnx_subfolder}/{onnx_file_name}"
        )
        # Models that use the ONNX external-data format split large weight
        # tensors into a sibling "<file>_data" blob that must sit next to the
        # .onnx graph on disk -- pull it down too when the repo has one.
        try:
            hf_hub_download(
                repo_id=model_id, filename=f"{onnx_subfolder}/{onnx_file_name}_data"
            )
        except Exception:
            pass

        sess_options = ort.SessionOptions()
        # Trade a bit of latency for a smaller resident memory footprint --
        # the default arena/thread settings are tuned for throughput, not
        # RAM, and Railway's small instances are memory- not CPU-constrained.
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
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._input_names = {i.name for i in self.session.get_inputs()}

        logger.info("ONNX reranker model ready: %s (provider=%s)", model_id, resolved_provider)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _score_batch(self, query: str, documents: List[str]) -> List[float]:
        encoded = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {name: encoded[name] for name in encoded if name in self._input_names}
        outputs = self.session.run(self._output_names, feed)
        logits = outputs[0]
        # Some reranker exports emit shape (batch, 1), others (batch,).
        scores = logits.reshape(-1).astype(np.float32)
        return self._sigmoid(scores).tolist()

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Score each document against `query`, in the given order."""
        if not documents:
            return []

        scores: List[float] = []
        for i in range(0, len(documents), self.batch_size):
            chunk = documents[i : i + self.batch_size]
            scores.extend(self._score_batch(query, chunk))
        return scores

    def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Score `documents` against `query` and return them sorted by
        relevance (highest first), each tagged with its original index.

        Pass `top_n` to truncate to the top N results.
        """
        scores = self.compute_scores(query, documents)
        results: List[RerankResult] = [
            {"index": i, "document": doc, "score": score}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n] if top_n is not None else results
