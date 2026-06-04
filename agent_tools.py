import os
from dotenv import load_dotenv
import asyncpg
import numpy as np

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_URL = f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/research_db"

import joblib, json
from sentence_transformers import SentenceTransformer

_nmf_model      = None
_tfidf_vec      = None
_topic_labels   = None
_embed_model    = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def _load_models():
    global _nmf_model, _tfidf_vec, _topic_labels
    if _nmf_model is None:
        _nmf_model    = joblib.load("models/nmf_model.pkl")
        _tfidf_vec    = joblib.load("models/tfidf_vectorizer.pkl")
        _topic_labels = json.load(open("models/topic_labels.json"))

# -- keyword search over Postgres -------------------------------------

async def search_abstracts(query: str, limit: int = 5) -> list[dict]:
    """Keyword search over the abstracts table."""
    conn = await asyncpg.connect(DATABASE_URL)
    conditions = ["(title ILIKE $1 OR abstract ILIKE $1)"]
    args: list = [f"%{query}%"]
    where = " AND ".join(conditions)
    
    rows = await conn.fetch(
        f"""
        SELECT doi, title, abstract, year, topics, cited_by
        FROM papers
        WHERE {where}
        ORDER BY cited_by DESC NULLS LAST, year DESC
        LIMIT ${len(args) + 1}
        """,
        *args, limit
    )
    await conn.close()
    return [
        {
            "doi":      r["doi"],
            "title":    r["title"],
            "abstract": r["abstract"][:300] + "...",  # trim for context window
            "year":     r["year"],
            "topics":   list(r["topics"]) if r["topics"] else [],
        }
        for r in rows
    ]


async def search_abstracts_by_topic(topic_id: int, limit: int = 10) -> list[dict]:
    """Find papers by NMF topic ID, ranked by topic position, citation count, then year."""
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(
        """
        SELECT doi, title, abstract, year, topics, cited_by
        FROM papers
        WHERE $1 = ANY(topics)
        ORDER BY
            array_position(topics, $1) ASC,
            cited_by DESC NULLS LAST,
            year DESC
        LIMIT $2
        """,
        topic_id, limit,
    )
    await conn.close()
    return [
        {
            "doi":            r["doi"],
            "title":          r["title"],
            "abstract":       r["abstract"][:300] + "...",
            "year":           r["year"],
            "topics":         list(r["topics"]) if r["topics"] else [],
            "topic_position": list(r["topics"]).index(topic_id) + 1 if r["topics"] else None,
            "cited_by":       r["cited_by"],
        }
        for r in rows
    ]


async def search_abstracts_hybrid(query: str, limit: int = 8, alpha: float = 0.5) -> list[dict]:
    """Hybrid search: combines keyword (ILIKE) and vector cosine similarity in one SQL query.
    alpha=1.0 → pure keyword, alpha=0.0 → pure semantic, alpha=0.5 → equal weight."""
    emb = _get_embed_model().encode([query])[0].tolist()
    emb_str = "[" + ",".join(str(x) for x in emb) + "]"

    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(
        """
        SELECT doi, title, abstract, year, topics, cited_by,
               (1 - (embedding <=> $2::vector)) AS vec_score,
               CASE WHEN title ILIKE $1 OR abstract ILIKE $1 THEN 1.0 ELSE 0.0 END AS kw_score
        FROM papers
        ORDER BY (
            $3 * (CASE WHEN title ILIKE $1 OR abstract ILIKE $1 THEN 1.0 ELSE 0.0 END)
            + (1 - $3) * (1 - (embedding <=> $2::vector))
        ) DESC
        LIMIT $4
        """,
        f"%{query}%", emb_str, alpha, limit,
    )
    await conn.close()
    return [
        {
            "doi":       r["doi"],
            "title":     r["title"],
            "abstract":  r["abstract"][:300] + "...",
            "year":      r["year"],
            "topics":    list(r["topics"]) if r["topics"] else [],
            "cited_by":  r["cited_by"],
            "vec_score": round(float(r["vec_score"]), 4),
            "kw_score":  round(float(r["kw_score"]), 4),
        }
        for r in rows
    ]


# -- NMF topic inference -----------------------------------------------

def classify_topic(text: str) -> dict:
    """Run NMF inference on a text string. Returns top topic label + all weights."""
    _load_models()
    vec     = _tfidf_vec.transform([text])
    weights = _nmf_model.transform(vec)[0]          # shape: (20,)
    top_id  = int(np.argmax(weights))
    return {
        "top_topic_id":    top_id,
        "top_topic_label": _topic_labels[str(top_id)],
        "confidence":      round(float(weights[top_id]), 4),
        "all_weights":     [round(float(w), 4) for w in weights],
    }


# -- consolidated agent-facing wrappers -------------------------------

async def search_by_topic(query: str, limit: int = 8) -> list[dict]:
    """Classify the query using NMF and retrieve papers assigned to that
    topic cluster. Best for broad thematic questions like 'what research
    exists on federated learning' or 'papers about graph neural networks'.
    Uses pre-computed NMF topic assignments — fast and topic-coherent."""
    result = classify_topic(query)
    return await search_abstracts_by_topic(topic_id=result["top_topic_id"], limit=limit)


async def search_by_content(query: str, mode: str = "hybrid", limit: int = 8) -> list[dict]:
    """Search abstract text directly using keyword matching, pgvector semantic
    similarity, or a hybrid of both. Best for specific method names, authors,
    conceptual questions that cross topic boundaries, or queries where exact
    wording matters. Mode: 'keyword', 'semantic', or 'hybrid' (default)."""
    alpha_map = {"keyword": 1.0, "semantic": 0.0, "hybrid": 0.5}
    alpha = alpha_map.get(mode, 0.5)
    return await search_abstracts_hybrid(query=query, limit=limit, alpha=alpha)


# -- topic trend by year -----------------------------------------------

async def topic_trend(topic_id: int) -> dict:
    """Annual paper fraction for an NMF topic cluster (topic_id 0-19).
    Returns topic_id, topic_label, and trend {year: fraction_of_all_papers}."""
    if not (0 <= topic_id <= 19):
        raise ValueError(f"topic_id must be 0-19, got {topic_id}")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT
                year,
                COUNT(*) FILTER (WHERE $1 = ANY(topics)) AS topic_count,
                COUNT(*)                                  AS year_total
            FROM papers
            GROUP BY year
            HAVING COUNT(*) FILTER (WHERE $1 = ANY(topics)) > 0
            ORDER BY year ASC
            """,
            topic_id,
        )
    finally:
        await conn.close()
    _load_models()
    return {
        "topic_id":    topic_id,
        "topic_label": _topic_labels[str(topic_id)],
        "trend":       {r["year"]: round(r["topic_count"] / r["year_total"], 4) for r in rows},
    }