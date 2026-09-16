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
    model_id="onnx-community/Qwen3-Reranker-0.6B-ONNX",
    device="cpu",
    onnx_file_name="model_quantized.onnx",
    provider="CPUExecutionProvider",
    batch_size=4,
    max_length=8192,
    instruction="Given a web search query, retrieve relevant passages that answer the query",
)
```

Both classes adapt to the export they are handed. `ONNXEmbeddings` covers
encoder models (mean pooling, or a model-provided `sentence_embedding` output
like embeddinggemma's) and decoder models that pool on the final EOS token
(Qwen3-Embedding, Mistral, Llama). `ONNXReranker` covers sequence-classification
cross-encoders (bge-reranker-v2-m3) and causal-LM rerankers that answer
"yes"/"no" (Qwen3-Reranker). Detection is automatic from the model config;
override it with `pooling=` / `scoring=` if you need to.

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
  Pass `instruction` to override the server's configured prefix for a single
  request — useful for task-specific retrieval, or for models like
  Qwen3-Embedding whose query prefix encodes the task. Pass `""` to embed the
  text with no prefix at all; omit the field to keep `QUERY_INSTRUCTION`.
  ```json
  {
    "text": "berapa harga bbm hari ini",
    "instruction": "Instruct: Given a news search query, retrieve relevant articles\nQuery:"
  }
  ```
- `POST /embed/documents` — embed a batch of chunks, respecting
  `EMBED_BATCH_SIZE` internally.
  ```json
  { "texts": ["chunk 1 ...", "chunk 2 ..."] }
  ```
- `POST /rerank` — score and rerank documents against a query using the
  reranker model (`onnx-community/Qwen3-Reranker-0.6B-ONNX` by default),
  highest relevance first. Optional `top_n` truncates the results; optional
  `instruction` overrides the task description in the judging prompt for this
  request (instruction-following rerankers only — ignored by bge-style
  cross-encoders).
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
                   --set "RERANKER_MODEL_NAME=onnx-community/Qwen3-Reranker-0.6B-ONNX" \
                   --set "RERANKER_ONNX_FILE_NAME=model_quantized.onnx" \
                   --set "RERANKER_MAX_LENGTH=2048" \
                   --set "RERANKER_BATCH_SIZE=4" \
                   --set "RERANKER_INSTRUCTION=Given a web search query, retrieve relevant passages that answer the query"
```

Railway provides `$PORT` automatically; the start command in `railway.json`
and `Procfile` already binds to it. The healthcheck timeout is set to 180s to
allow for the model download on first boot.

**Resource sizing:** embeddinggemma-300m is small and this container no
longer loads torch — just `transformers` (tokenizer only), `onnxruntime`, and
the quantized ONNX weights (~300MB). 512MB–1GB RAM should comfortably fit the
model plus a request or two in flight; size up from there if you raise
`EMBED_BATCH_SIZE` or run many requests concurrently.

The reranker is the expensive half of this service. Qwen3-Reranker-0.6B is a
0.6B-parameter decoder run once per `(query, document)` pair with no KV-cache
reuse, so both RAM and latency scale with `RERANKER_MAX_LENGTH` × batch size.
The 8192 default is the model's full context window; **drop
`RERANKER_MAX_LENGTH` to 1024–2048 unless you actually rerank long
documents** — an 8k-token forward pass per document is slow and
memory-hungry on a small instance. Keep `RERANKER_BATCH_SIZE` low unless you
have headroom to spare, and size the instance up if `/rerank` sees real load.

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
- The reranker scores each `(query, document)` pair jointly through the
  transformer — slower per pair than embedding similarity, more accurate for
  reranking. Qwen3-Reranker is a causal LM rather than a classifier: each pair
  is wrapped in the judging prompt it was trained with (`<Instruct>` /
  `<Query>` / `<Document>`, asking for a yes/no verdict) and the score is the
  softmax probability of the `yes` token over the `no` token at the final
  position. bge-style cross-encoders take a sigmoid over the single output
  logit instead. Either way scores land in `[0, 1]`, so callers and thresholds
  do not have to know which model is loaded.
- Decoder exports (Qwen3 embedding and reranker alike) are fed left-padded so
  the final position is always a real token, are given `position_ids`, and get
  empty `past_key_values` inputs built from the graph's own input metadata.
  Only the tensor actually needed is fetched from each run, so present-KV
  outputs are never materialized.
- `ONNXEmbeddings` also offers `truncate_dim` (Matryoshka truncation) and
  `chunk_text()` / `embed_documents_chunked()`, which split over-long inputs on
  token boundaries — accounting for `text_instruction` and special/EOS tokens —
  before embedding each piece.