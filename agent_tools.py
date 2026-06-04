import os
from dotenv import load_dotenv
import asyncpg
import numpy as np

load_dotenv()

# -- Tool 1: keyword search over Postgres -------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_URL = f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/research_db"

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


# -- Tool 2: NMF topic inference -----------------------------------------------

import joblib, json

_nmf_model      = None
_tfidf_vec      = None
_topic_labels   = None

def _load_models():
    global _nmf_model, _tfidf_vec, _topic_labels
    if _nmf_model is None:
        _nmf_model    = joblib.load("models/nmf_model.pkl")
        _tfidf_vec    = joblib.load("models/tfidf_vectorizer.pkl")
        _topic_labels = json.load(open("models/topic_labels.json"))

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


# -- Tool 3: topic trend by year -----------------------------------------------

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