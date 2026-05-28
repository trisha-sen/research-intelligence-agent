# What this project is
A research intelligence agent built on 21K ML journal abstracts.
The agent can search the corpus, classify papers into 20 NMF topic clusters,
look up topic trends by year, and answer research questions with citations.
Key differentiator: retrieval quality is evaluated using NMF cluster labels
as ground truth (Precision@10 per topic), comparing TF-IDF vs embedding-based retrieval.

## Data

- **data/abstracts.csv** — 21K rows. Columns:
  - `doi` — unique paper identifier
  - `title` — paper title
  - `abstract` — full abstract text
  - `journal` — publication journal
  - `year` — publication year
  - `topics` — top 3 NMF topic IDs as a Python list, e.g. `[4, 11, 2]`
  - `all_topic_prop` — weight vector across all 20 NMF topics, e.g. `[0.42, 0.01, 0.0, ...]`

- **Parsing** — both list columns are stored as stringified Python lists.
  Always parse on load:
  ```python
  import ast
  df["topics"] = df["topics"].apply(ast.literal_eval)
  df["all_topic_prop"] = df["all_topic_prop"].apply(ast.literal_eval)
  ```
  After parsing: `topics` --> `list[int]`, `all_topic_prop`--> `list[float]` (length 20)

- **topic_labels.json** — maps topic_id (0–19) --> human-readable label
  e.g. `{"0": "attention_mechanisms", "1": "graph_neural_networks", ...}`

- **Primary topic** — to get a paper's dominant topic:
  ```python
  primary_topic = df["topics"].apply(lambda x: x[0])
  ```

- **Topic affinity score** — to get a paper's weight for a specific topic:
  ```python
  def topic_weight(row, topic_id: int) -> float:
      return row["all_topic_prop"][topic_id]
  ```



