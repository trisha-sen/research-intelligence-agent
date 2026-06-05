# Research Intelligence Agent

A research Q&A system built on 21K ML journal abstracts. The agent retrieves relevant papers and synthesises cited answers using a LangGraph pipeline backed by Postgres + pgvector. Retrieval quality is evaluated with a principled metric: weighted Precision@10 using NMF topic weights as continuous relevance scores - comparing TF-IDF keyword search, sentence-transformer embeddings, and a hybrid of both.

---

## NMF topic model

The 20 topics were fit on the full 21K abstract corpus using scikit-learn's NMF with a TF-IDF input matrix. Each paper stores:

- `topics` - top 3 topic IDs ordered by weight
- `all_topic_prop` - full 20-dimensional weight vector

Selected topic labels:

| ID | Label |
|---|---|
| 2 | graph_neural_networks |
| 3 | attention_mechanisms |
| 10 | medical_image_segmentation |
| 12 | federated_learning |
| 17 | knowledge_distillation |
| 18 | recommender_systems |

The agent's `search_by_topic` tool runs NMF inference on the incoming query at request time to map it to the closest topic cluster - no keyword match required.

---

## Retrieval evaluation

### Methodology

Standard IR metrics (binary hit/miss) don't fit here because every paper has a continuous NMF topic weight between 0 and 1 - a paper is not simply "relevant" or "not relevant" to a topic. The evaluation uses **weighted Precision@10**: for each query about topic *X*, retrieve the top 10 results and compute the mean `all_topic_prop[X]` score across them. This treats the NMF weight as a graded relevance signal rather than a threshold.

Two query types are tested against all 20 NMF topics:

- **Exact** - precise technical terms matching the topic label (e.g. `"federated learning"`, `"graph neural network"`)
- **Conceptual** - paraphrased descriptions with no keywords from the topic label (e.g. `"training across devices without sharing data"`, `"learning from graph structured data"`)

The conceptual set is deliberately adversarial for keyword search.

### Results

| Eval Type | Method | Mean P@10 | Min | Max |
|---|---|---|---|---|
| conceptual | keyword | 0.051 | 0.001 | 0.172 |
| conceptual | semantic | **0.580** | 0.129 | 0.894 |
| conceptual | hybrid | 0.577 | 0.129 | 0.894 |
| exact | keyword | 0.477 | 0.207 | 0.798 |
| exact | semantic | **0.648** | 0.252 | 0.884 |
| exact | hybrid | 0.643 | 0.209 | 0.846 |

**Key findings:**

- Keyword search collapses on conceptual queries (P@10 = 0.051 vs 0.580 for semantic) - it can only match what it can literally see in the text.
- Semantic search dominates across both query types, with a 10× improvement on conceptual queries.
- Hybrid search tracks semantic closely but does not improve over pure semantic here. The binary keyword score (1.0 / 0.0 ILIKE match) contributes noise rather than signal when the query vocabulary differs from the abstract.

---

## Architecture

```
User query
    |
FastAPI  (/research, /search, /chat)
    │
LangGraph pipeline
 ┌--------------------------------------------------------------┐
 │  search_node                                                 │
 │  LLM chooses tool:                                           │
 │   - search_by_topic --> NMF classify --> topic cluster lookup│
 │   - search_by_content --> keyword / semantic / hybrid        │
 └--------------------------------------------------------------┘
    │ top-k abstracts
 summarise_node
  Streams a cited answer (3-5 paragraphs, inline DOI refs)
    │
 MLflow  (query, method, tokens, cost logged per run)
```

**Stack**

| Layer | Technology |
|---|---|
| API | FastAPI + asyncpg (async Postgres driver) |
| LLM & agent | Anthropic Claude Haiku 4.5, LangGraph |
| Database | Postgres 17 + pgvector extension |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Topic model | NMF (20 topics) + TF-IDF vectoriser |
| Experiment tracking | MLflow 2.17 |
| Container | Docker Compose (3 services: postgres, app, mlflow) |

---

## Project layout

```
research_agent/
├-- main.py               # FastAPI app - all HTTP endpoints
├-- agent_graph.py        # LangGraph pipeline (search --> summarise)
├-- agent_config.py       # State definition, model constants, tool wrappers
├-- agent_tools.py        # DB queries: keyword, semantic, hybrid, NMF, trend
├-- mlflow_tracking.py    # Logs every research run to MLflow
├-- load_data.py          # Loads abstracts.csv --> Postgres on startup
├-- embed_corpus.py       # Generates pgvector embeddings for all rows
├-- evals/
│   ├-- eval_retrieval.py # Weighted P@10 eval across 3 retrieval methods
│   └-- eval_queries.py   # 40 benchmark queries (20 conceptual + 20 exact)
├-- models/
│   ├-- nmf_model.pkl     # Trained NMF model (20 topics)
│   ├-- tfidf_vectorizer.pkl
│   └-- topic_labels.json # topic_id --> human-readable label
├-- data/
│   └-- abstracts.csv     # 21K abstracts with pre-computed NMF topic columns
├-- Dockerfile
└-- docker-compose.yml
```

---

## Running locally

### Prerequisites

- Docker Desktop
- An Anthropic API key

### Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd research_agent

# 2. Create .env
cat > .env <<EOF
POSTGRES_PASSWORD=your_postgres_password
ANTHROPIC_API_KEY=sk-ant-...
EOF

# Windows: if you edit .env in a Windows text editor, fix line endings before running in WSL:
#   sed -i 's/\r//' .env

# 3. Start all services
docker compose up --build
```

On first boot the entrypoint runs `load_data.py` (CSV --> Postgres) then `embed_corpus.py` (pgvector embeddings). This takes a few minutes. Subsequent starts are instant because data is persisted in Docker volumes.

**Endpoints after boot:**

| URL | Description |
|---|---|
| `GET  localhost:8000/health` | Row count sanity check |
| `GET  localhost:8000/search?q=federated+learning` | Keyword search with pagination |
| `GET  localhost:8000/search/hybrid?q=...&alpha=0.5` | Hybrid search (`alpha`: 1=keyword, 0=semantic) |
| `POST localhost:8000/research` | Agent Q&A - streams a cited answer |
| `POST localhost:8000/chat` | Stateful multi-turn chat (session_id required) |
| `GET  localhost:5001` | MLflow UI - experiment runs, token costs |

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Requires a running Postgres instance with pgvector installed.
# Set DATABASE_URL or POSTGRES_PASSWORD in .env, then:
python load_data.py
python embed_corpus.py
uvicorn main:app --reload
```

---

## API usage examples

**Agent Q&A (streaming):**
```bash
curl -X POST localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What privacy techniques are used in federated learning?"}'
```

**Hybrid search:**
```bash
# alpha=0.3 --> mostly semantic
curl "localhost:8000/search/hybrid?q=graph+neural+network&alpha=0.3&limit=5"
```

**Multi-turn chat:**
```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "message": "What is knowledge distillation?"}'
```
---

## MLflow experiment tracking

Every `/research` request logs: query text, search method chosen, retrieved paper DOIs, full answer, input/output token counts, and wall-clock latency. Access the UI at `localhost:5001` when running via Docker Compose.

To browse runs from the command line:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```
