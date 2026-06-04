# What this project is
A research intelligence agent built on 21K ML journal abstracts.
The agent can search the corpus, classify papers into 20 NMF topic clusters,
look up topic trends by year, and answer research questions with citations.
Key differentiator: retrieval quality is evaluated using NMF cluster labels
as ground truth (Precision@10 per topic), comparing TF-IDF vs embedding-based retrieval.

## Data

- **data/abstracts.csv** - 21K rows. Columns:
  - `doi` - unique paper identifier
  - `title` - paper title
  - `abstract` - full abstract text
  - `author_keywords` - keywords provided by author
  - `authors` - paper authors
  - `cited_by` - number of citations the paper received
  - `journal` - publication journal
  - `year` - publication year
  - `topics` - top 3 NMF topic IDs as a Python list, e.g. `[4, 11, 2]`
  - `all_topic_prop` - weight vector across all 20 NMF topics, e.g. `[0.42, 0.01, 0.0, ...]`

       'DOI': 'doi',
     'Title': 'title',
     'Abstract': 'abstract',
     'Author Keywords': 'author keywords',
     'Authors': 'authors',
     'Cited by': 'cited by',
     'Source title': 'journal',
     'Year': 'year',
     'topics': 'topics',
     'all_topic_prop': 'all_topic_prop'

- **Parsing** - both list columns are stored as stringified Python lists.
  Always parse on load:
  ```python
  import ast
  df["topics"] = df["topics"].apply(ast.literal_eval)
  df["all_topic_prop"] = df["all_topic_prop"].apply(ast.literal_eval)
  ```
  After parsing: `topics` --> `list[int]`, `all_topic_prop`--> `list[float]` (length 20)

- **topic_labels.json** - maps topic_id (0–19) --> human-readable label
  e.g. `{"0": "attention_mechanisms", "1": "graph_neural_networks", ...}`

- **Primary topic** - to get a paper's dominant topic:
  ```python
  primary_topic = df["topics"].apply(lambda x: x[0])
  ```

- **Topic affinity score** - to get a paper's weight for a specific topic:
  ```python
  def topic_weight(row, topic_id: int) -> float:
      return row["all_topic_prop"][topic_id]
  ```

## Weekly Plan

[DONE] Week 1 · infrastructure & serving
New tooling only - Docker, FastAPI, first LLM call - applied to abstracts corpus from day one.
- Docker basics [DONE]
  Dockerfile, image, container. Wrap a script that loads abstracts.csv.
- FastAPI + search [DONE]
  Serve keyword search over 33K abstracts. Run locally in Docker.
- Compose + Postgres [DONE]
  Move abstracts into Postgres. API queries DB instead of CSV.
- Streaming LLM [DONE]
  Streaming research Q&A via OpenAI. Understand token costs.
Ship: FastAPI app serving 33K abstracts from Postgres, with a streaming LLM Q&A endpoint, running in Docker.
corpus is live and queryable


Week 2 · agents & tool use
Your NMF model becomes an agent tool. The LLM can now classify papers into your 20 topics on demand and reason across the corpus.
- ReAct loop [DONE]
  Hand-code the agent. Tools: abstract search + NMF classifier.
- LangGraph basics [DONE]
  Nodes, edges, state. Replaces your hand-coded loop.
- 3 LangGraph research agent tools [DONE]
  Search + NMF classify + trend lookup. Your model is callable.
- MCP server [DONE]
  -- nice to have, but not part of final workflow --
  Expose your 3 research tools as an MCP server.
- Agent as API [DONE]
  FastAPI wraps the agent. Streams answers with citations.
Ship: LangGraph agent with 3 tools (search, NMF classify, trend lookup) served via FastAPI with streaming citations.
agent is reasoning over your corpus

Week 3 · pgvector, MLflow & cloud
Replace TF-IDF with embeddings and measure the difference.
End the week with a real public URL on your existing AWS account.

- pgvector setup
  Add vector column to abstracts table. Embed corpus with sentence-transformers.
- Hybrid search
  Keyword + vector similarity in one SQL query. No separate service.
- Semantic tool
  Agent now chooses: keyword, semantic, or hybrid per query.
- MLflow tracking
  Log every run: query, docs, response, cost, method.
- Deploy to AWS
  ECR + App Runner. Real public URL on your existing AWS account.
Ship: hybrid search (keyword + pgvector) in one Postgres query, MLflow tracking,
deployed to AWS. Keyword vs semantic comparison documented with numbers.
agent is live in the cloud