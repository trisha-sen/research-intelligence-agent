import os
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 512
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_URL = f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/research_db"

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
register_vector(conn)

cur.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS embedding vector(384)")
conn.commit()

cur.execute("SELECT COUNT(*) FROM papers WHERE embedding IS NOT NULL")
if cur.fetchone()[0] > 0:
    print("Already embedded, skipping.")
    cur.close()
    conn.close()
    raise SystemExit(0)

print("Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

cur.execute("SELECT doi, title, abstract FROM papers")
rows = cur.fetchall()
total = len(rows)
print(f"Embedding {total} papers...")

for i in range(0, total, BATCH_SIZE):
    batch = rows[i : i + BATCH_SIZE]
    texts = [f"{title}. {abstract}" for _, title, abstract in batch]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    execute_values(
        cur,
        """
        UPDATE papers SET embedding = data.emb
        FROM (VALUES %s) AS data(doi, emb)
        WHERE papers.doi = data.doi
        """,
        [(doi, emb.tolist()) for (doi, _, _), emb in zip(batch, embeddings)],
    )
    conn.commit()
    print(f"  {min(i + BATCH_SIZE, total)}/{total} embedded")

print("Creating HNSW index...")
cur.execute("CREATE INDEX ON papers USING hnsw (embedding vector_cosine_ops)")
conn.commit()

cur.close()
conn.close()
print("Done.")
