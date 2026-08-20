"""
load_blog_posts.py — loads data/blog_posts.csv into pipeline.db and
embeds each post's body text.

SETUP
  python load_blog_posts.py
  (sentence-transformers already installed from embed_images.py earlier)

WHAT IT DOES
- Creates the `blog_posts` table if it doesn't exist yet
- Reads data/blog_posts.csv (id, title, body columns)
- Skips any id already in the DB — resumable, safe to rerun after adding
  more posts to the CSV later without re-embedding everything
- Embeds `body` using the same model as embed_images.py, so image
  captions and blog post text land in the same 384-dim vector space —
  that's what makes cosine similarity between them meaningful
"""

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from embeddings import embed_text

DB_PATH = Path("data/pipeline.db")
CSV_PATH = Path("data/blog_posts.csv")


def init_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blog_posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            embedding TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def already_loaded(conn, post_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM blog_posts WHERE id = ?", (post_id,)
    ).fetchone() is not None


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"No CSV found at {CSV_PATH}. Put blog_posts.csv in data/ first."
        )

    conn = sqlite3.connect(DB_PATH)
    init_table(conn)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} post(s) in {CSV_PATH}")

    loaded, skipped = 0, 0
    for row in rows:
        if already_loaded(conn, row["id"]):
            skipped += 1
            print(f"  SKIP (already loaded): {row['title']}")
            continue

        vec = embed_text(row["body"])
        conn.execute(
            "INSERT INTO blog_posts (id, title, body, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["title"], row["body"], json.dumps(vec), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        loaded += 1
        print(f"  OK: {row['title']}  ({len(vec)} dims)")

    print(f"\nDone. Loaded: {loaded}  Skipped: {skipped}")
    total = conn.execute("SELECT COUNT(*) FROM blog_posts").fetchone()[0]
    print(f"Total blog posts in DB: {total}")
    conn.close()


if __name__ == "__main__":
    main()
