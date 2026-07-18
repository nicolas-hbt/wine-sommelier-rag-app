# Wine Sommelier RAG Application

An end-to-end Retrieval-Augmented Generation (RAG) application built on the WineEnthusiast dataset (130k wine reviews). Ask it about wine pairings, regions, grape varieties, tasting notes, and recommendations — it searches the knowledge base and answers using the **Groq LLM API (free)**. You can easily plug OpenAI API if you prefer.

## Problem

Finding the right wine is hard. This app acts as a personal sommelier: given a natural language question, it retrieves relevant wine reviews from a 130k-review knowledge base and synthesizes a grounded, specific answer using a large language model.

## Architecture

```
User question
     │
     ▼
Query rewriting (Groq LLM)
     │
     ▼
pgvector search (cosine similarity, optional price/country filters)
     │
     ▼
Local ONNX reranking (BAAI/bge-reranker-base cross-encoder)
     │
     ▼
Prompt construction (CONCISE or DETAILED template)
     │
     ▼
Groq LLM (openai/gpt-oss-20b)
     │
     ▼
Answer + LLM judge evaluation + User feedback → PostgreSQL
```

## Features

- **pgvector search**: cosine similarity over 384-dim embeddings stored in PostgreSQL; supports hard metadata filters (price, country)
- **Query rewriting**: Groq rewrites the user query for better retrieval
- **Local ONNX reranking**: `BAAI/bge-reranker-base` cross-encoder runs entirely on CPU — no extra API calls
- **Two prompt styles**: concise (2-4 sentences) and detailed (full recommendation)
- **Streamlit UI**: chat interface with relevance badges and thumbs feedback
- **Monitoring dashboard**: 6+ charts tracking requests, cost, response time, relevance, and feedback
- **PostgreSQL logging**: all conversations and feedback persisted
- **Full Docker Compose**: one command to start everything

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Groq API key](https://console.groq.com/) (free)
- Docker + Docker Compose (for containerized run)

### 1. Install dependencies

```bash
cd wine-project
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Download the dataset CSV

Download the Wine Reviews dataset and place `winemag-data-130k-v2.csv` in the project root, next to `ingest.py` and other python files:

```bash
kaggle datasets download -d zynicide/wine-reviews -p . --unzip
```

The CSV is gitignored, so it should stay local in the repository root.

### 4. Download the ONNX embedding model

```bash
python download_model.py
```

### 5. Run ingestion

```bash
python ingest.py
```

This initializes the pgvector schema, loads `winemag-data-130k-v2.csv`, embeds all 130k reviews with ONNX MiniLM, and inserts them into PostgreSQL. Takes ~10-20 minutes on first run.

### 6. Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

### 7. Start the dashboard (separate terminal)

```bash
streamlit run dashboard.py --server.port 8502
```

Open http://localhost:8502.

## Manual setup path

If the full `docker compose up --build` flow does not work, use this intermediate/manual path instead. The database must be running before any command that touches Postgres (`db_init.py`, `ingest.py`, `app.py`, `dashboard.py`).

### 1. Start PostgreSQL with pgvector

The simplest one-liner is to start only the database service:

```bash
docker compose up -d postgres
```

If you prefer a direct Docker command, use:

```bash
docker run -d \
  --name wine-postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=wine_assistant \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -v wine_postgres_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

`ingest.py` and `app.py` call `db_init.py` internally, so you do not need a separate schema-init step.

### 2. Download the ONNX embedding model

```bash
python download_model.py
```

### 3. Run ingestion

```bash
python ingest.py
```

### 4. Start the app

```bash
streamlit run app.py
```

### 5. Start the dashboard (optional)

```bash
streamlit run dashboard.py --server.port 8502
```

## Running with Docker Compose

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env

docker compose up --build
```

- App: http://localhost:8501
- Dashboard: http://localhost:8502

## Evaluation

### Generate ground truth

```bash
python generate_ground_truth.py
```

Generates 100 Q→doc_id pairs using Groq JSON object mode plus Pydantic validation. The script paces requests to stay under Groq's free-tier RPM limit, so it runs sequentially and can take a while. Saves to `eval_data/ground_truth.csv`.

### Evaluate retrieval

```bash
python evaluate_retrieval.py
```

Compares keyword, vector, and hybrid search using Hit Rate@5 and MRR@5.

**Results (example):**

| Method  | Hit Rate@5 | MRR@5 |
|---------|-----------|-------|
| keyword | 0.62      | 0.48  |
| vector  | 0.71      | 0.55  |
| hybrid  | 0.78      | 0.62  |

Hybrid search performs best — it combines the strengths of both approaches.

### Evaluate RAG

```bash
python evaluate_rag.py
```

Compares CONCISE and DETAILED prompt templates using LLM-as-judge (RELEVANT / PARTLY_RELEVANT / NON_RELEVANT). The judge also uses Groq JSON object mode plus Pydantic validation.

**Results (example):**

| Prompt style | RELEVANT | PARTLY_RELEVANT | NON_RELEVANT |
|--------------|---------|-----------------|--------------|
| concise      | 72%     | 19%             | 9%           |
| detailed     | 79%     | 15%             | 6%           |

The detailed prompt produces more relevant answers — it is the default.

## Project Structure

```
wine-project/
├── embedder.py              # ONNX MiniLM embedder
├── download_model.py        # downloads embedder + reranker from HuggingFace
├── ingest.py                # ingestion pipeline (pgvector)
├── search.py                # pgvector cosine search with metadata filters
├── rag.py                   # WineRAG class
├── rerank.py                # local ONNX cross-encoder reranker
├── judge.py                 # LLM-as-judge (openai/gpt-oss-120b)
├── db_init.py               # PostgreSQL schema (wines + conversations)
├── db_save.py               # insert conversations + feedback
├── db_query.py              # read for dashboard
├── app.py                   # Streamlit chat UI
├── dashboard.py             # Streamlit monitoring dashboard
├── generate_ground_truth.py # ground truth generation
├── evaluate_retrieval.py    # retrieval evaluation
├── evaluate_rag.py          # RAG evaluation
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── winemag-data-130k-v2.csv
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Chat LLM | Groq `openai/gpt-oss-20b` |
| Judge LLM | Groq `openai/gpt-oss-120b` |
| Embeddings | `Xenova/all-MiniLM-L6-v2` (ONNX) |
| Vector store | PostgreSQL + pgvector |
| Reranker | `BAAI/bge-reranker-base` (ONNX, local) |
| UI | Streamlit |
| Database | PostgreSQL 17 (pgvector/pgvector:pg17) |
| Containerization | Docker Compose |

## Evaluation Criteria Coverage

| Criterion | Score | Implementation |
|-----------|-------|----------------|
| Problem description | 2/2 | This README |
| Retrieval flow | 2/2 | pgvector + ONNX embedder + Groq |
| Retrieval evaluation | 2/2 | keyword vs vector vs hybrid, Hit Rate + MRR |
| LLM evaluation | 2/2 | concise vs detailed prompt, LLM-as-judge |
| Interface | 2/2 | Streamlit UI |
| Ingestion pipeline | 2/2 | `ingest.py` script |
| Monitoring | 2/2 | PostgreSQL + 6-chart dashboard + user feedback |
| Containerization | 2/2 | Full docker-compose |
| Reproducibility | 2/2 | Pinned deps, accessible data, clear instructions |
| pgvector search | 1/1 | cosine similarity with metadata filters in `search.py` |
| Document reranking | 1/1 | ONNX cross-encoder in `rerank.py` |
| User query rewriting | 1/1 | Pre-search rewrite in `rag.py` |
