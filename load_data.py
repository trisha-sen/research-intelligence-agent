import ast
import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_values

CSV_PATH = Path(os.environ.get("CSV_PATH", "data/abstracts.csv"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/research_db")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        doi TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        author_keywords TEXT,
        authors TEXT,
        cited_by INTEGER,
        journal TEXT,
        year INTEGER,
        topics JSONB,
        all_topic_prop JSONB
    )
""")

cur.execute("SELECT COUNT(*) FROM papers")
if cur.fetchone()[0] > 0:
    print("Data already loaded, skipping.")
    cur.close()
    conn.close()
    raise SystemExit(0)

csv.field_size_limit(10_000_000)

with CSV_PATH.open(encoding="utf-8") as f:
    rows = [
        (
            row["doi"],
            row["title"],
            row["abstract"],
            row.get("author_keywords"),
            row.get("authors"),
            row.get("cited_by"),
            row["journal"],
            int(row["year"]),
            json.dumps(ast.literal_eval(row["topics"])),
            json.dumps(ast.literal_eval(row["all_topic_prop"])),
        )
        for row in csv.DictReader(f)
    ]

execute_values(
    cur,
    """
    INSERT INTO papers (doi, title, abstract, author_keywords, authors, cited_by, journal, year, topics, all_topic_prop)
    VALUES %s
    ON CONFLICT (doi) DO NOTHING
    """,
    rows,
)
inserted = cur.rowcount

conn.commit()
cur.close()
conn.close()

print(f"Done - {inserted} rows inserted.")