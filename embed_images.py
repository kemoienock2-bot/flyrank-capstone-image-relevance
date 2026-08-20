"""
embed_images.py — fills in the `embedding` column for every image in
pipeline.db that was successfully tagged in Phase 2 but doesn't have one yet.

SETUP
  pip install sentence-transformers
  python embed_images.py

WHAT IT DOES
- Selects rows from `images` where embedding IS NULL (i.e. every row that
  passed Phase 2 tagging but hasn't been embedded yet)
- Embeds the `caption` field (not category — caption is the natural-language
  text that's actually comparable to blog post prose, per your Phase 1
  design decision)
- Writes the embedding back as a JSON-encoded list of floats, matching
  your "JSON text column, not BLOB" schema choice
- Resumable: only touches rows with a NULL embedding, so reruns are safe
  and free (no API calls to worry about wasting here anyway)
"""

import json
import sqlite3
from pathlib import Path

from embeddings import embed_text

DB_PATH = Path("data/pipeline.db")


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"No DB found at {DB_PATH}. Run run_batch.py (Phase 2) first.")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, filepath, caption FROM images WHERE embedding IS NULL"
    ).fetchall()

    print(f"Found {len(rows)} image(s) needing embeddings.")

    for i, (row_id, filepath, caption) in enumerate(rows, start=1):
        vec = embed_text(caption)
        conn.execute(
            "UPDATE images SET embedding = ? WHERE id = ?",
            (json.dumps(vec), row_id),
        )
        conn.commit()
        print(f"  [{i}/{len(rows)}] embedded {filepath}  ({len(vec)} dims)")

    total = conn.execute(
        "SELECT COUNT(*) FROM images WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    print(f"\nDone. {total} image(s) now have embeddings in {DB_PATH.resolve()}")
    conn.close()


if __name__ == "__main__":
    main()
