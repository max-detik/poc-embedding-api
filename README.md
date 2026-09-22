# ONNX Embeddings + Reranker Service (Railway-ready)

A minimal FastAPI service that runs local embedding **and reranking**
inference directly through **ONNX Runtime** — no torch, sentence-transformers,
optimum, or llama-index in the request path (see `onnx_embeddings.py` /
`onnx_reranker.py`). That stack is skipped on purpose to keep the container's
resident RAM small: torch alone adds a few hundred MB just from being
imported, which matters on Railway's smaller instance tiers.

```python
embeddings = ONNXEmbeddings(
    model_id="onnx-community/embeddinggemma-300m-ONNX",
    device="cpu",
    onnx_file_name="model_quantized.onnx",
    provider="CPUExecutionProvider",
    batch_size=8,
    query_instruction="task: search result | query: ",
)

reranker = ONNXReranker(
    model_id="onnx-community/bge-reranker-v2-m3-ONNX",
    device="cpu",
    onnx_file_name="model_quantized.onnx",
    provider="CPUExecutionProvider",
    batch_size=4,
    max_length=8192,
)
```

## ⚠️ Railway has no GPU support

As of this writing, **Railway does not offer GPU-backed services** — so
`device="cuda"` / `provider="CUDAExecutionProvider"` from your original
snippet won't run there. This service defaults to `DEVICE=cpu` and
`ONNX_PROVIDER=CPUExecutionProvider` instead, which works fine for a
300M-parameter model at moderate request volume. Every setting is still
env-driven, so if you later deploy this same code on a GPU host (a VM,
RunPod, Fly.io GPU, Northflank BYOC, etc.) you just flip `DEVICE=cuda`,
`ONNX_PROVIDER=CUDAExecutionProvider`, and swap `onnxruntime` for
`onnxruntime-gpu` in `requirements.txt` — no code changes needed.

## Endpoints

- `GET /health` — check the service is up and see the active embedding/reranker
  model, device, and provider
- `POST /embed/query` — embed a single query string. **`query_instruction` is
  applied automatically**, distinguishing queries from documents.
  ```json
  { "text": "berapa harga bbm hari ini" }
  ```
- `POST /embed/documents` — embed a batch of chunks, respecting
  `EMBED_BATCH_SIZE` internally.
  ```json
  { "texts": ["chunk 1 ...", "chunk 2 ..."] }
  ```
- `POST /rerank` — score and rerank documents against a query using the
  cross-encoder reranker (`onnx-community/bge-reranker-v2-m3-ONNX` by
  default), highest relevance first. Optional `top_n` truncates the results.
  ```json
  {
    "query": "what is a panda?",
    "documents": [
      "The giant panda is a bear species endemic to China.",
      "Paris is the capital of France."
    ],
    "top_n": 1
  }
  ```
  ```json
  {
    "results": [
      { "index": 0, "document": "The giant panda is a bear species endemic to China.", "score": 0.98 }
    ],
    "count": 1
  }
  ```

## 1. Local development

```bash
cd railway-llamaindex-embeddings
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults already match your original snippet (minus device/provider)
uvicorn main:app --reload --port 8000
```

First request will download the model + ONNX weights from the Hub — expect a
slower first call, then fast responses after.

Test it:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "contoh teks berita dalam bahasa Indonesia"}'

curl -X POST http://localhost:8000/embed/documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["chunk pertama", "chunk kedua"]}'

curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "apa itu panda?", "documents": ["Panda adalah beruang dari China.", "Paris adalah ibu kota Prancis."]}'
```

## 2. Deploy to Railway

**Option A — Railway CLI**
```bash
npm install -g @railway/cli
railway login
railway init      # or `railway link` to an existing project
railway up
```

**Option B — GitHub deploy**
1. Push this directory to a GitHub repo.
2. Railway dashboard → New Project → Deploy from GitHub repo.
3. Railway auto-detects Python via Nixpacks and uses `railway.json` /
   `Procfile` for the start command.

**Environment variables** (Railway dashboard → your service → Variables, or
via CLI) — the defaults in `.env.example` already match your snippet aside
from device/provider:
```bash
railway variables --set "MODEL_NAME=onnx-community/embeddinggemma-300m-ONNX" \
                   --set "DEVICE=cpu" \
                   --set "ONNX_FILE_NAME=model_quantized.onnx" \
                   --set "ONNX_PROVIDER=CPUExecutionProvider" \
                   --set "EMBED_BATCH_SIZE=8" \
                   --set "QUERY_INSTRUCTION=task: search result | query: " \
                   --set "RERANKER_MODEL_NAME=onnx-community/bge-reranker-v2-m3-ONNX" \
                   --set "RERANKER_ONNX_FILE_NAME=model_quantized.onnx" \
                   --set "RERANKER_MAX_LENGTH=8192" \
                   --set "RERANKER_BATCH_SIZE=4"
```

Railway provides `$PORT` automatically; the start command in `railway.json`
and `Procfile` already binds to it. The healthcheck timeout is set to 180s to
allow for the model download on first boot.

**Resource sizing:** embeddinggemma-300m is small and this container no
longer loads torch — just `transformers` (tokenizer only), `onnxruntime`, and
the quantized ONNX weights (~300MB). 512MB–1GB RAM should comfortably fit the
model plus a request or two in flight; size up from there if you raise
`EMBED_BATCH_SIZE` or run many requests concurrently.

The reranker (bge-reranker-v2-m3, a much larger cross-encoder than
embeddinggemma) adds noticeably more RAM and per-request compute, especially
at long `RERANKER_MAX_LENGTH` — size the instance up accordingly if you use
`/rerank` under real load, and keep `RERANKER_BATCH_SIZE` low unless you have
headroom to spare.

### Persisting the model across deploys (Railway volumes)

By default the ONNX weights + tokenizer are downloaded from the Hub into the
Hugging Face cache (`~/.cache/huggingface`) on every fresh boot, since
Railway's filesystem is ephemeral. To avoid re-downloading ~300MB on every
deploy/restart (which briefly spikes memory and CPU right as the healthcheck
is waiting on you):

1. Railway dashboard → your service → **Volumes** → add a volume, mount path
   e.g. `/data`.
2. Set `HF_HOME=/data/hf-cache` as an environment variable (this is the
   standard `huggingface_hub` cache-location variable — no code changes
   needed, `AutoTokenizer`/`hf_hub_download` both honor it automatically).
3. Redeploy. The first boot after attaching the volume still downloads the
   model once; every boot after that reuses the cached files from the volume.

Note volumes only attach to a single service replica, so this doesn't help if
you're running multiple replicas of this service — each would need its own
volume (or you skip this and accept the download on every cold start).

## 3. Calling it from other code

```python
import requests

BASE_URL = "https://<your-app>.up.railway.app"

def embed_query(text: str) -> list[float]:
    r = requests.post(f"{BASE_URL}/embed/query", json={"text": text})
    r.raise_for_status()
    return r.json()["embedding"]

def embed_documents(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{BASE_URL}/embed/documents", json={"texts": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    r = requests.post(f"{BASE_URL}/rerank", json={"query": query, "documents": documents, "top_n": top_n})
    r.raise_for_status()
    return r.json()["results"]
```

This works as a drop-in remote embedding + reranking source for any RAG
pipeline (e.g. feeding vectors into Elasticsearch/Qdrant, then reranking the
top-k retrieved candidates before passing them to an LLM), without needing
LlamaIndex, sentence-transformers, or the ONNX runtime installed in the
calling service.

## Running with a GPU (elsewhere)

If you move this off Railway to a GPU host later:

```bash
DEVICE=cuda
ONNX_PROVIDER=CUDAExecutionProvider
```
and in `requirements.txt`, replace:
```
onnxruntime==1.20.1
```
with:
```
onnxruntime-gpu==1.20.1
```
(and use a CUDA-enabled base image appropriate to the host's driver version).

## Notes

- `text_instruction` is supported alongside `query_instruction` if your model
  expects a different prefix for indexed documents vs. queries — set
  `TEXT_INSTRUCTION` if needed (left unset by default, matching your snippet).
- `onnx-community/*` models ship pre-exported ONNX weights, so there's no
  on-the-fly conversion step here — `onnx_embeddings.py` / `onnx_reranker.py`
  download `onnx/model_quantized.onnx` (+ its external-data sibling file, if
  present) straight from the Hub and run it directly via
  `onnxruntime.InferenceSession`.
- If the ONNX graph exposes a `sentence_embedding` output (as
  embeddinggemma-300m-ONNX does), it's used directly instead of manually
  mean-pooling `last_hidden_state`.
- `ORT_INTRA_OP_THREADS` / `ORT_INTER_OP_THREADS` (default `1` each) cap ONNX
  Runtime's thread pools — kept low by default to favor a small memory
  footprint over max throughput on Railway's smaller instance tiers; raise
  them if you have CPU headroom and want more throughput.
- The reranker is a cross-encoder: it scores each `(query, document)` pair
  jointly through the transformer (slower per-pair than embedding similarity,
  more accurate for reranking) and returns a `[0, 1]` relevance score via a
  sigmoid over the model's raw logit. `RERANKER_MAX_LENGTH` defaults to 8192
  (bge-reranker-v2-m3's full context window) to support long-document
  reranking — lower it if you only rerank short passages and want less
  memory/latency per request.