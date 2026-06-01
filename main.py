import os
from collections import defaultdict
from contextlib import asynccontextmanager

import anthropic
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/research_db")

# --- Chatbot ---
_anthropic = anthropic.AsyncAnthropic()
CHAT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = """
You are a helpful assistant specializing in ML research.
You are an expert ML research assistant with specializing in
ML research. You help researchers navigate the academic
literature by answering questions clearly and precisely.
 
Your responses should:
- Be technically accurate and use correct ML terminology
- Distinguish between established results and open research questions
- Mention relevant sub-fields, methods, or landmark papers where appropriate
- Be concise — prefer 1-3 focused paragraphs over exhaustive lists
- Acknowledge uncertainty rather than confabulate
"""

conversations: dict[str, list[dict]] = defaultdict(list)

class ChatRequest(BaseModel):
    session_id: str
    message: str

pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)
    yield
    await pool.close()


app = FastAPI(title="Abstract Search", lifespan=lifespan)


@app.get("/health")
async def health():
    count = await pool.fetchval("SELECT COUNT(*) FROM papers")
    return {"status": "ok", "rows_loaded": count}


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Case-insensitive substring"),
    year: int | None = Query(None, description="Filter by exact year"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conditions = ["(title ILIKE $1 OR abstract ILIKE $1)"]
    args: list = [f"%{q}%"]

    if year is not None:
        conditions.append(f"year = ${len(args) + 1}")
        args.append(year)

    where = " AND ".join(conditions)
    total = await pool.fetchval(f"SELECT COUNT(*) FROM papers WHERE {where}", *args)

    rows = await pool.fetch(
        f"""
        SELECT doi, title, authors, author_keywords, journal, year, cited_by
        FROM papers
        WHERE {where}
        ORDER BY cited_by DESC NULLS LAST, year DESC
        LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
        """,
        *args, limit, offset,
    )

    return {
        "query": q,
        "year": year,
        "total": total,
        "returned": len(rows),
        "results": [
            {
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "cited_by": r["cited_by"],
                "journal": r["journal"],
                "doi": r["doi"],
                "author_keywords": r["author_keywords"],
            }
            for r in rows
        ],
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    history = conversations[request.session_id]
    history.append({"role": "user", "content": request.message})

    async def event_stream():
        chunks: list[str] = []
        async with _anthropic.messages.stream(
            model=CHAT_MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=history,
        ) as stream:
            async for text in stream.text_stream:
                chunks.append(text)
                yield text
        history.append({"role": "assistant", "content": "".join(chunks)})

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    return {"session_id": session_id, "messages": conversations.get(session_id, [])}


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    conversations.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
