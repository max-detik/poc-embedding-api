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
  - Decoder-style embedders (Qwen3-Embedding, Mistral, Llama) pool on the
    final EOS token instead; `pooling="auto"` picks that from the model config,
    and such exports get left padding plus empty KV-cache inputs.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger("onnx-embeddings")

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

# Decoder-style embedders that pool on the final (EOS) token.
_LAST_TOKEN_MODEL_TYPES = {"qwen2", "qwen3", "mistral", "llama"}

DEFAULT_QWEN3_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def qwen3_query_instruction(task: str = DEFAULT_QWEN3_TASK) -> str:
    """Query prefix format expected by Qwen3-Embedding. Documents get no prefix."""
    return f"Instruct: {task}\nQuery:"


class ONNXEmbeddings:
    """Sentence embeddings from a local ONNX model.

    Supports encoder models (mean pooling or a model-provided
    `sentence_embedding` output, e.g. EmbeddingGemma) and decoder models
    with last-token pooling (e.g. Qwen3-Embedding).
    """

    def __init__(
        self,
        model_id: str,
        onnx_file_name: str = "model_quantized.onnx",
        onnx_subfolder: str = "onnx",
        device: str = "cpu",
        provider: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 16,
        query_instruction: str = "",
        text_instruction: str = "",
        pooling: str = "auto",  # "auto" | "mean" | "last_token"
        truncate_dim: Optional[int] = None,  # Matryoshka truncation
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.text_instruction = text_instruction
        self.truncate_dim = truncate_dim

        resolved_provider = provider or _DEVICE_TO_PROVIDER.get(
            device.lower(), "CPUExecutionProvider"
        )

        logger.info(
            "Loading ONNX embeddings model=%s onnx_file_name=%s provider=%s",
            model_id, onnx_file_name, resolved_provider,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if pooling == "auto":
            try:
                model_type = AutoConfig.from_pretrained(model_id).model_type
            except Exception:
                model_type = ""
            pooling = "last_token" if model_type in _LAST_TOKEN_MODEL_TYPES else "mean"
        if pooling not in ("mean", "last_token"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling

        # Token that last-token pooling reads from. Qwen3-Embedding uses <|endoftext|>.
        self._eod_id = None
        self._pad_id = self.tokenizer.pad_token_id
        if self.pooling == "last_token":
            eod = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if eod is None or eod == self.tokenizer.unk_token_id:
                eod = self.tokenizer.eos_token_id
            if eod is None:
                raise ValueError("Tokenizer has no EOS/<|endoftext|> token for last-token pooling")
            self._eod_id = eod
            if self._pad_id is None:
                self._pad_id = eod

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
        if "sentence_embedding" in output_names and self.pooling == "mean":
            self._embedding_output = "sentence_embedding"
        elif "last_hidden_state" in output_names:
            self._embedding_output = "last_hidden_state"
        elif "sentence_embedding" in output_names:
            self._embedding_output = "sentence_embedding"
        else:
            raise ValueError(
                f"ONNX model has no usable embedding output. Outputs: {output_names}. "
                "Make sure you exported a feature-extraction graph, not a text-generation one."
            )

        logger.info(
            "ONNX embeddings model ready: %s (provider=%s, pooling=%s, output=%s, inputs=%s)",
            model_id, resolved_provider, self.pooling, self._embedding_output,
            [n for n in self._input_meta if not n.startswith("past_key_values")],
        )

    # ------------------------------------------------------------------ #
    # Tokenization and graph inputs
    # ------------------------------------------------------------------ #
    def _tokenize(self, texts: List[str]):
        """Returns (input_ids, attention_mask, token_type_ids or None) as int64 arrays."""
        if self.pooling != "last_token":
            enc = self.tokenizer(
                texts,
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

        # Last-token pooling: guarantee the sequence ends with <|endoftext|>
        # and pad on the left so position -1 is always that token.
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length - 1,
            add_special_tokens=True,
        )
        seqs = []
        for ids in enc["input_ids"]:
            if not ids or ids[-1] != self._eod_id:
                ids = list(ids) + [self._eod_id]
            seqs.append(ids)

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
    # Pooling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        input_ids, attention_mask, token_type_ids = self._tokenize(texts)
        feeds = self._build_feeds(input_ids, attention_mask, token_type_ids)
        (out,) = self.session.run([self._embedding_output], feeds)
        out = out.astype(np.float32)

        if self._embedding_output == "sentence_embedding":
            pooled = out
        elif self.pooling == "last_token":
            pooled = out[:, -1, :]  # left-padded, so last position is <|endoftext|>
        else:
            pooled = self._mean_pool(out, attention_mask)

        if self.truncate_dim:
            pooled = pooled[:, : self.truncate_dim]

        if self.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            pooled = pooled / norms

        return pooled.tolist()

    # ------------------------------------------------------------------ #
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------ #
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in batch_size groups, prefixed with `text_instruction`."""
        if not texts:
            return []

        prefixed = [f"{self.text_instruction}{t}" for t in texts]

        all_vectors: List[List[float]] = []
        for i in range(0, len(prefixed), self.batch_size):
            chunk = prefixed[i : i + self.batch_size]
            all_vectors.extend(self._embed_batch(chunk))
        return all_vectors

    def embed_query(self, text: str, instruction: Optional[str] = None) -> List[float]:
        """Embed a single query string, prefixed with `query_instruction`.

        `instruction` overrides that prefix for this call; pass "" for no prefix.
        """
        prefix = self.query_instruction if instruction is None else instruction
        return self._embed_batch([f"{prefix}{text}"])[0]

    def chunk_text(self, text: str, chunk_overlap: int = 0) -> List[str]:
        """Split `text` into pieces that fit within max_length (leaving room for
        `text_instruction` and special/EOS tokens). Returns [text] if it fits.
        """
        ids = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

        prefix_len = 0
        if self.text_instruction:
            prefix_len = len(
                self.tokenizer(self.text_instruction, add_special_tokens=False)["input_ids"]
            )
        special_len = self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.pooling == "last_token":
            special_len = max(special_len, 1)  # reserved for <|endoftext|>
        budget = self.max_length - prefix_len - special_len
        if budget <= 0:
            raise ValueError(
                f"max_length={self.max_length} is too small to fit text_instruction "
                f"({prefix_len} tokens) plus special tokens ({special_len} tokens)"
            )

        if len(ids) <= budget:
            return [text]

        if chunk_overlap >= budget:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than the effective "
                f"chunk size ({budget} tokens)"
            )

        step = budget - chunk_overlap
        chunks = []
        for start in range(0, len(ids), step):
            chunk_ids = ids[start : start + budget]
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if start + budget >= len(ids):
                break
        return chunks

    def embed_documents_chunked(
        self, texts: List[str], chunk_overlap: int = 0
    ) -> List[List[dict]]:
        """Chunk-then-embed. Each input becomes a list of {"text", "embedding"} dicts."""
        results: List[List[dict]] = []
        for text in texts:
            chunk_texts = self.chunk_text(text, chunk_overlap=chunk_overlap)
            vectors = self.embed_documents(chunk_texts)
            results.append(
                [{"text": ct, "embedding": vec} for ct, vec in zip(chunk_texts, vectors)]
            )
        return results
