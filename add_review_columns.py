"""
add_review_columns.py — one-time migration.

Adds columns to `matches` for recording a human reviewer's decision,
separate from (and never overwriting) the automated guard's decision —
same "don't silently mutate, keep both visible" principle as everywhere
else in this pipeline.

Run this once, before starting review_api.py.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/pipeline.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()]

    added = []
    if "human_decision" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN human_decision TEXT")  # NULL until reviewed
        added.append("human_decision")
    if "human_note" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN human_note TEXT")
        added.append("human_note")
    if "reviewed_at" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN reviewed_at TEXT")
        added.append("reviewed_at")

    conn.commit()
    if added:
        print(f"Added columns: {', '.join(added)}")
    else:
        print("Review columns already exist — nothing to do.")
    conn.close()


if __name__ == "__main__":
    main()
