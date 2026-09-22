# Harrier Embeddings Service

A small FastAPI service that serves embeddings from
[`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
through [sentence-transformers](https://sbert.net) on torch, loading the model
straight from its Hugging Face repo.

## Project layout

```
embedder/            the embedding library (no FastAPI) -- import this from notebooks
  specs.py           which models exist and how each is prompted
  prompts.py         query prompts and document templates (pure functions)
  chunking.py        token-budgeted chunking (pure function)
  runtime.py         device/dtype resolution, HF auth, model loading
  model.py           STEmbeddings, tying the above together
api/                 the FastAPI service
  config.py          settings, read from env vars
  schemas.py         request/response bodies
  routes.py          /health, /embed/query, /embed/documents
  main.py            app + startup (loads and warms the model)
tests/               runs without GPU or real weights (fake SentenceTransformer)
```

Adding a model means adding one entry to `embedder/specs.py`.

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
- Documents are rendered through a **document template** with `{title}` and
  `{content}` placeholders before embedding. Harrier's default is `{content}`
  (no prefix). Set `EMBED_DOCUMENT_TEMPLATE`, e.g. `{title}\n\n{content}`, to
  include the article title, and send titles with `/embed/documents`. When a
  document has no title, the result is stripped so no stray separator is left.
- The template ends in `Query: ` **with** a trailing space. That matches how
  harrier was trained, so leave it as is.
- Harrier supports 32k context. `max_seq_length` is capped at 1024 for
  throughput; raise it with `EMBED_MAX_SEQ_LENGTH` if you index long documents.
- Embeddings come back L2-normalized, so cosine similarity is a plain dot
  product.

The model is loaded and warmed at startup, before `/health` reports `ok`.

### Using the class directly

```python
from embedder import STEmbeddings

embeddings = STEmbeddings("harrier", batch_size=64)   # device/dtype auto-detected
qv = embeddings.embed_query("berapa harga bbm hari ini")
dv = embeddings.embed_documents(["chunk pertama", "chunk kedua"])  # (n, 1024) float32, L2-normalized

# Custom document template, filled per document:
titled = STEmbeddings("harrier", document_template="{title}\n\n{content}")
chunks = titled.chunk_text(article, title=title)       # reserves room for the title
vectors = titled.embed_documents(chunks, titles=[title] * len(chunks))
titled.format_document(chunks[0], title)                # the exact string embedded
```

The class also still supports `STEmbeddings("gemma")` for
`google/embeddinggemma-300m` (768d; gated, so it needs `HF_TOKEN`). Its default
template is `title: {title} | text: {content}`, with `none` used when a
document has no title. The API service itself loads harrier only.

`STEmbeddings` also offers:

- `embed_queries()` for encoding many queries at once (e.g. retrieval evals)
- `truncate_dim` for Matryoshka truncation
- `token_length()`
- `chunk_text()` / `embed_documents_chunked()`, which split over-long inputs
  on token boundaries before embedding each piece. The token budget accounts
  for the rendered template, including the title, so no chunk gets truncated
  after the title is added. `embed_documents_chunked()` returns the raw
  `text`, the `embedded_text`, and the `embedding` for each chunk.

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
  { "texts": ["chunk 1 ...", "chunk 2 ..."], "titles": ["Judul artikel", "Judul artikel"] }
  ```
  `titles` is optional. When given, it must match `texts` in length and fills
  the document template's `{title}`; `null` entries mean no title.
  ```json
  { "embeddings": [[0.01, ...], [0.02, ...]], "dimensions": 1024, "count": 2 }
  ```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8000
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

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite swaps in a fake SentenceTransformer, so it needs no GPU and
downloads no weights. If torch or sentence-transformers aren't installed at
all, it replaces them with minimal stand-ins. It covers prompt formats,
document templates, chunk budgets, and the API. It does **not** check the real
model's output, so do one live call on the target machine after
dependency upgrades.

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
# EMBED_DOCUMENT_TEMPLATE={title}\n\n{content}
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
  different model, `EMBED_TRUNCATE_DIM`, document template, or task instruction are not
  comparable with the ones already in your index.
- Earlier versions of this service ran ONNX Runtime, then multiple
  sentence-transformers models plus a Qwen3 reranker behind `/rerank`. Both
  were removed; see git history. Old env vars (`MODEL_NAME`, `ONNX_PROVIDER`,
  `DEFAULT_EMBED_MODEL`, `PRELOAD_*`, `RERANKER_*`, …) are no longer read.
