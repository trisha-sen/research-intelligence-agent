import csv
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query

CSV_PATH = Path(os.environ.get("CSV_PATH", "data/abstracts.csv"))
SEARCH_FIELDS = ("title", "abstract", "author keywords")

papers: list[dict] = []


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global papers
    csv.field_size_limit(10_000_000)
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        papers.extend(csv.DictReader(f))
    yield


app = FastAPI(title="Abstract Search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": len(papers)}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Case-insensitive substring"),
    year: int | None = Query(None, description="Filter by exact year"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    needle = q.casefold()
    results = []
    for row in papers:
        if year is not None and row.get("year") != str(year):
            continue
        if any(needle in (row.get(f) or "").casefold() for f in SEARCH_FIELDS):
            results.append(row)

    results.sort(key=lambda r: int(r.get("cited by") or 0), reverse=True)
    page = results[offset : offset + limit]
    return {
        "query": q,
        "year": year,
        "total": len(results),
        "returned": len(page),
        "results": [
            {
                "title": r.get("title"),
                # "authors": r.get("authors"),
                "year": int(r.get("year") or 0),
                "cited by": int(r.get("cited by") or 0),
                "journal": r.get("journal"),
                "doi": r.get("doi"),
                "keywords": r.get("author keywords"),
            }
            for r in page
        ],
    }
