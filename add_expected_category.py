"""
add_expected_category.py — one-time migration.

Adds an `expected_category` column to blog_posts and backfills it from
the updated data/blog_posts.csv. Doesn't touch embeddings — those are
already computed and don't need to change.

Run this once, before rerunning compute_matches.py.
"""

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("data/pipeline.db")
CSV_PATH = Path("data/blog_posts.csv")


def main():
    conn = sqlite3.connect(DB_PATH)

    cols = [r[1] for r in conn.execute("PRAGMA table_info(blog_posts)").fetchall()]
    if "expected_category" not in cols:
        conn.execute("ALTER TABLE blog_posts ADD COLUMN expected_category TEXT")
        conn.commit()
        print("Added expected_category column to blog_posts.")
    else:
        print("expected_category column already exists — skipping ALTER.")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        conn.execute(
            "UPDATE blog_posts SET expected_category = ? WHERE id = ?",
            (row["expected_category"], row["id"]),
        )
        updated += 1

    conn.commit()
    print(f"Backfilled expected_category for {updated} post(s).")

    check = conn.execute("SELECT title, expected_category FROM blog_posts").fetchall()
    for title, cat in check:
        print(f"  {cat:<8} {title}")

    conn.close()


if __name__ == "__main__":
    main()
