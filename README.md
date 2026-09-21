# Harrier Embeddings Service

A small FastAPI service that serves embeddings from
[`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
through [sentence-transformers](https://sbert.net) on torch, loading the model
straight from its Hugging Face repo (see `st_embeddings.py`).

## The model

| Repo | Dim | `max_seq_length` | Pooling | Query prompt | Document prompt |
|---|---|---|---|---|---|
| `microsoft/harrier-oss-v1-0.6b` | 1024 | 1024 | last token | `Instruct: {task}\nQuery: ` | *(none)* |

Harrier is **asymmetric**: queries carry a task instruction, documents carry
no prefix. The service applies the right one on each side, so callers just use
`/embed/query` and `/embed/documents`.

- The default task is `Given a web search query, retrieve relevant passages
  that answer the query`. Change it service-wide with `EMBED_TASK`, or per
  request with the `task` field.
- The template ends in `Query: ` **with** a trailing space. That matches how
  harrier was trained, so leave it as is.
- Harrier supports 32k context. `max_seq_length` is capped at 1024 for
  throughput; raise it with `EMBED_MAX_SEQ_LENGTH` if you index long documents.
- Embeddings come back L2-normalized, so cosine similarity is a plain dot
  product.

The model is loaded and warmed at startup, before `/health` reports `ok`.

### Using the class directly

```python
from st_embeddings import STEmbeddings

embeddings = STEmbeddings("harrier", batch_size=64)   # device/dtype auto-detected
qv = embeddings.embed_query("berapa harga bbm hari ini")
dv = embeddings.embed_documents(["chunk pertama", "chunk kedua"])  # (n, 1024) float32, L2-normalized
```

`STEmbeddings` also offers:

- `embed_queries()` for encoding many queries at once (e.g. retrieval evals)
- `truncate_dim` for Matryoshka truncation
- `token_length()`
- `chunk_text()` / `embed_documents_chunked()`, which split over-long inputs
  on token boundaries (leaving room for special tokens) before embedding each
  piece.

## Endpoints

- `GET /health`: the model, dimensions, sequence length, device, and dtype.
- `POST /embed/query`: embed a single query. The query instruction is applied
  automatically; `task` overrides it for this request.
  ```json
  { "text": "berapa harga bbm hari ini", "task": "Cari berita yang relevan" }
  ```
  ```json
  {
    "embedding": [0.013, ...],
    "dimensions": 1024,
    "prompt": "Instruct: Cari berita yang relevan\nQuery: "
  }
  ```
  The echoed `prompt` is the exact prefix that was applied, which helps when
  debugging retrieval quality.
- `POST /embed/documents`: embed a batch. Texts are grouped by length
  internally so one long article doesn't pad out everything batched with it.
  Results come back in the order you sent them.
  ```json
  { "texts": ["chunk 1 ...", "chunk 2 ..."] }
  ```
  ```json
  { "embeddings": [[0.01, ...], [0.02, ...]], "dimensions": 1024, "count": 2 }
  ```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

The first run downloads the weights from the Hub, so expect a slow start and
fast responses after that. On a CPU-only machine, lower `EMBED_BATCH_SIZE` to
about 8 and expect seconds rather than milliseconds per request; this is a
0.6B model.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "contoh teks berita dalam bahasa Indonesia"}'

curl -X POST http://localhost:8000/embed/documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["chunk pertama", "chunk kedua"]}'
```

## Deploying

This stack is meant for a **GPU host** (a VM with an NVIDIA card, RunPod,
Fly.io GPU, Northflank BYOC, …). Device and dtype are detected automatically:
CUDA + float16 when a GPU is visible, CPU + float32 otherwise. Pin them with
`DEVICE` / `DTYPE`.

> **Railway has no GPU support.** It can still run this on CPU, but torch plus
> a 0.6B model means a multi-GB image, slow cold starts, and seconds-per-request
> latency.

Sizing: harrier is about 1.2GB of weights in float16, plus activation memory
that grows with `EMBED_BATCH_SIZE` × `max_seq_length`. The startup download
has to finish within your platform's healthcheck window (`railway.json`
allows 180s).

**Environment variables:** see `.env.example`. All of them are optional:

```bash
DEVICE=cuda
DTYPE=float16
EMBED_BATCH_SIZE=64
# EMBED_MAX_SEQ_LENGTH=2048
# EMBED_TRUNCATE_DIM=512
# EMBED_TASK=Given a news search query, retrieve relevant articles
```

### Persisting weights across deploys

On an ephemeral filesystem, the weights are re-downloaded on every fresh
boot. To avoid that, mount a volume and point `HF_HOME` at it (e.g.
`HF_HOME=/data/hf-cache`). `HF_HOME` is the standard `huggingface_hub` cache
variable, so no code changes are needed. The first boot after attaching the
volume still downloads once; later boots reuse the cached files.

## Calling it from other code

```python
import requests

BASE_URL = "https://<your-host>"

def embed_query(text: str, task: str | None = None) -> list[float]:
    r = requests.post(f"{BASE_URL}/embed/query", json={"text": text, "task": task})
    r.raise_for_status()
    return r.json()["embedding"]

def embed_documents(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{BASE_URL}/embed/documents", json={"texts": texts})
    r.raise_for_status()
    return r.json()["embeddings"]
```

## Notes

- Harrier is a Qwen3-architecture decoder that pools the **last** token, so
  its tokenizer is loaded with `padding_side="left"`.
- **Changing models or settings means re-indexing.** Vectors produced with a
  different model, `EMBED_TRUNCATE_DIM`, or task instruction are not
  comparable with the ones already in your index.
- Earlier versions of this service ran ONNX Runtime, then multiple
  sentence-transformers models plus a Qwen3 reranker behind `/rerank`. Both
  were removed; see git history. Old env vars (`MODEL_NAME`, `ONNX_PROVIDER`,
  `DEFAULT_EMBED_MODEL`, `PRELOAD_*`, `RERANKER_*`, …) are no longer read.
