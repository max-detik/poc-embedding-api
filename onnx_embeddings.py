"""
A minimal embeddings class that runs an ONNX embedding model locally via
plain ONNX Runtime + a fast tokenizer -- no torch, sentence-transformers, or
optimum in the loop.

That stack is skipped on purpose: torch alone adds a few hundred MB of
resident RAM just from being imported, which matters on small Railway
instances where inference itself runs entirely through ONNX Runtime anyway.
transformers' AutoTokenizer does not require torch as long as a fast
(Rust-backed) tokenizer is available for the model, which is the case for
onnx-community/embeddinggemma-300m-ONNX.

    embeddings = ONNXEmbeddings(
        model_id="onnx-community/embeddinggemma-300m-ONNX",
        onnx_file_name="model_quantized.onnx",
        query_instruction="task: search result | query: ",
    )

Notes:
  - `device` is a convenience that maps to an ONNX Runtime execution provider
    ("cuda" -> CUDAExecutionProvider, "cpu" -> CPUExecutionProvider). Pass
    `provider=` directly if you need a specific provider (e.g. TensorrtExecutionProvider).
  - GPU inference (CUDAExecutionProvider) requires the `onnxruntime-gpu`
    package instead of plain `onnxruntime` -- see requirements.txt.
  - `query_instruction` / `text_instruction` prepend a task prefix before
    embedding, matching how asymmetric-retrieval models like EmbeddingGemma
    and BGE are meant to be used (different prefix for queries vs. documents).
  - If the ONNX graph exposes a `sentence_embedding` output (as
    embeddinggemma-300m-ONNX does), that's used directly instead of manually
    mean-pooling `last_hidden_state`.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

logger = logging.getLogger("onnx-embeddings")

_DEVICE_TO_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


class ONNXEmbeddings:
    """Mean-pooled (or model-provided) sentence embeddings from a local ONNX model."""

    def __init__(
        self,
        model_id: str,
        onnx_file_name: str = "model_quantized.onnx",
        onnx_subfolder: str = "onnx",
        device: str = "cpu",
        provider: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruction: str = "",
        text_instruction: str = "",
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.text_instruction = text_instruction

        resolved_provider = provider or _DEVICE_TO_PROVIDER.get(
            device.lower(), "CPUExecutionProvider"
        )

        logger.info(
            "Loading ONNX embeddings model=%s onnx_file_name=%s provider=%s",
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

        logger.info("ONNX embeddings model ready: %s (provider=%s)", model_id, resolved_provider)

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {name: encoded[name] for name in encoded if name in self._input_names}
        outputs = dict(zip(self._output_names, self.session.run(self._output_names, feed)))

        if "sentence_embedding" in outputs:
            pooled = outputs["sentence_embedding"]
        else:
            pooled = self._mean_pool(outputs["last_hidden_state"], encoded["attention_mask"])

        if self.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            pooled = pooled / norms

        return pooled.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents/chunks, in batch_size-sized groups.

        Applies `text_instruction` as a prefix to each text.
        """
        if not texts:
            return []

        prefixed = [f"{self.text_instruction}{t}" for t in texts]

        all_vectors: List[List[float]] = []
        for i in range(0, len(prefixed), self.batch_size):
            chunk = prefixed[i : i + self.batch_size]
            all_vectors.extend(self._embed_batch(chunk))
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string, prefixed with `query_instruction`."""
        return self._embed_batch([f"{self.query_instruction}{text}"])[0]
