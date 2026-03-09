# Local AI Assistant with RAG and Benchmarking

A fully offline AI assistant running 3B–7B LLMs locally via Ollama.
Supports conversational chat, document-based question answering (RAG), performance benchmarking, and AI evaluation.
No cloud APIs. No internet dependency.

---

## Features

- **Chat** — Conversational interface with streaming token output
- **RAG** — Upload PDF, TXT, or Markdown documents and ask questions grounded in their content
- **Hybrid retrieval** — FAISS vector search + BM25 keyword search fused with Reciprocal Rank Fusion (RRF)
- **Query rewriting** — LLM-powered query reformulation for improved retrieval
- **Semantic cache** — Redis-backed cache that returns cached responses for semantically similar queries
- **Benchmarking** — TTFT, tokens/sec, and latency comparison across models
- **Evaluation** — Recall@k, faithfulness scoring, and response quality assessment (LLM-as-judge)

---

## Prerequisites

### Required

| Requirement | Version | Notes |
|-------------|---------|-------|
| [Ollama](https://ollama.com) | latest | Must be running before the app starts |
| `mistral` model | — | `ollama pull mistral` |
| `llama3` model | — | `ollama pull llama3` |

### Docker quickstart (recommended)

| Requirement | Version |
|-------------|---------|
| Docker Desktop | 24+ |
| Docker Compose | v2+ |

### Local (no Docker)

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| Redis | 7+ |

---

## Quickstart

### Option A — Docker (recommended)

```bash
# 1. Start Ollama on your host machine and pull models
ollama pull mistral
ollama pull llama3

# 2. Clone the repo and enter the directory
git clone <repo-url> local-ai-assistant
cd local-ai-assistant

# 3. Copy the example env file and edit if needed
cp docker/.env.example .env

# 4. Build and start the stack
docker compose up --build

# 5. Open the app
open http://localhost:8501
```

> **Note (Linux):** `host.docker.internal` resolves via the `extra_hosts` entry in `docker-compose.yml`.
> On macOS/Windows Docker Desktop resolves it automatically.

### Option B — Local (no Docker)

```bash
# 1. Start Ollama
ollama serve &
ollama pull mistral && ollama pull llama3

# 2. Start Redis
redis-server --daemonize yes

# 3. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy and configure environment variables
cp docker/.env.example .env
# Edit .env if you need non-default ports or model names

# 6. Launch the app
streamlit run app/ui.py
```

---

## Folder Structure

```
local-ai-assistant/
├── app/
│   ├── config.py      # Single source of truth for all configuration constants
│   ├── ui.py          # Streamlit entry point and UI layout
│   ├── chat.py        # OllamaClient (streaming), SemanticCache
│   ├── rag.py         # Document ingestion, FAISS index, RAG pipeline, query rewriting
│   ├── retriever.py   # FAISS vector search, BM25, RRF hybrid search
│   ├── embedder.py    # Sentence Transformers embedding wrapper
│   └── metrics.py     # MetricsCollector — TTFT, tokens/sec, latency
├── benchmarks/
│   ├── benchmark.py   # run_benchmark(), run_multi_model_benchmark(), export_results()
│   └── results/       # JSON benchmark output files (gitignored)
├── evaluation/
│   ├── dataset.json   # Evaluation Q&A pairs
│   ├── evaluator.py   # recall_at_k(), evaluate_faithfulness(), evaluate_response_quality()
│   ├── run_eval.py    # CLI runner for the full evaluation suite
│   └── results/       # JSON evaluation output files (gitignored)
├── data/
│   └── documents/     # Uploaded user documents (gitignored)
├── vector_store/      # Persisted FAISS index and BM25 corpus (gitignored)
├── docker/
│   └── .env.example   # All supported environment variables with defaults
├── docs/
│   ├── benchmark_report.md   # mistral vs llama3 benchmark results
│   └── evaluation_report.md  # Recall@k, faithfulness, and quality scores
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Architecture

```
User
 │
 ▼
┌─────────────────────────────┐
│     Streamlit Chat UI       │  app/ui.py
│  (model selector, sidebar,  │
│   file uploader, metrics)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Semantic Cache         │  app/chat.py — SemanticCache
│  Redis cosine-sim lookup    │
│  (threshold: 0.92, TTL: 1h) │
└──────┬──────────────────────┘
       │ cache miss
       ▼
┌─────────────────────────────┐
│      Query Rewriter         │  app/rag.py — rewrite_query()
│  LLM reformulates question  │
│  for better retrieval       │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│    FAISS    │  │    BM25     │  app/retriever.py
│  (semantic) │  │  (keyword)  │
└──────┬──────┘  └──────┬──────┘
       └────────┬────────┘
                │  RRF fusion  (k=60)
                ▼
┌─────────────────────────────┐
│      Context Builder        │  app/rag.py — build_context_prompt()
│  Top-k chunks → prompt      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Ollama (local LLM)      │  app/chat.py — OllamaClient.stream()
│  mistral / llama3 / phi3    │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌──────────────┐
│ Streaming  │  │   Metrics    │  app/metrics.py — MetricsCollector
│  Response  │  │  Collector   │
└────────────┘  └──────────────┘
```

---

## Configuration Reference

All variables are read from `.env` (or `docker/.env.example`) via `app/config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama REST API endpoint. Set to `http://host.docker.internal:11434` inside Docker. |
| `DEFAULT_MODEL` | `mistral` | Model selected on startup. Must be available in Ollama. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model for embeddings. Must match between indexing and querying. |
| `CHUNK_SIZE` | `512` | Document chunk size in characters. |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks in characters. |
| `TOP_K` | `5` | Number of chunks retrieved per query. |
| `RRF_K` | `60` | RRF constant. Higher values reduce the impact of high-rank results. |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for a semantic cache hit (0–1). |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string. Set to `redis://redis:6379` inside Docker. |
| `CACHE_TTL` | `3600` | Cache entry time-to-live in seconds. |
| `VECTOR_STORE_PATH` | `vector_store/` | Directory for persisted FAISS index and BM25 corpus. |
| `DOCUMENTS_DIR` | `data/documents` | Directory for uploaded source documents. |

---

## Running the Benchmark Suite

```bash
# Run 5 iterations on both mistral and llama3 (default)
python benchmarks/benchmark.py

# Specify models and run count
python benchmarks/benchmark.py --models mistral llama3 --runs 10

# Save results to a specific path
python benchmarks/benchmark.py --output benchmarks/results/my_run.json
```

Results are written to `benchmarks/results/benchmark_<timestamp>.json`.
See [docs/benchmark_report.md](docs/benchmark_report.md) for reference results.

---

## Running the Evaluation Suite

```bash
# Retrieval evaluation — Recall@k
python evaluation/evaluator.py --mode retrieval

# Faithfulness evaluation (LLM-as-judge)
python evaluation/evaluator.py --mode faithfulness

# Response quality evaluation (LLM-as-judge)
python evaluation/evaluator.py --mode quality

# Full suite via run_eval.py
python evaluation/run_eval.py
```

Results are written to `evaluation/results/`.
See [docs/evaluation_report.md](docs/evaluation_report.md) for reference results.

---

## Performance Targets

| Metric | Target | Hardware assumption |
|--------|--------|---------------------|
| Time To First Token (TTFT) | < 2 s | Apple Silicon / NVIDIA GPU |
| Tokens/sec | > 20 | Apple Silicon / NVIDIA GPU |
| End-to-end latency | < 5 s | Apple Silicon / NVIDIA GPU |

CPU-only machines may not meet these targets at 7B model scale.

---

## Supported Models

| Model | Minimum | Notes |
|-------|---------|-------|
| `mistral` | Yes | Default model |
| `llama3` | Yes | |
| `phi3` | Optional | |
| `gemma` | Optional | |

Any model available in your local Ollama installation can be selected from the UI dropdown at runtime.

---

## Hardware Requirements

- **Minimum:** 16 GB RAM
- **Recommended:** Apple Silicon (M1/M2/M3) or NVIDIA GPU with 8+ GB VRAM
- Ollama must be running as a local service before starting the app
