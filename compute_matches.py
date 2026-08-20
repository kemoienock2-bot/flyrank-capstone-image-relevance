"""
compute_matches.py — scores every image x blog post pair through the
full guard decision order and writes results to the `matches` table.

Applies real per-category similarity thresholds (see SIMILARITY_THRESHOLDS
below), calibrated from the actual score distribution observed across
your 43 images x 8 posts. Every pair gets a final decision: "approved"
or "rejected", each with a specific, auditable reason.

GUARD DECISION ORDER (from Phase 1 design, + category-mismatch step
added after seeing real similarity data — see note below):
  1. category == "uncertain"           -> reject, similarity not computed
  2. conf_category < CAT_THRESHOLD     -> reject, similarity not computed
  3. conf_caption < CAP_THRESHOLD      -> reject, similarity not computed
  4. category == "other"               -> reject, similarity not computed
  5. category != post.expected_category -> reject, similarity not computed
  6. otherwise                         -> compute cosine similarity,
                                           compare against SIMILARITY_THRESHOLDS[category],
                                           decision = "approved" or "rejected"

WHY STEP 5 EXISTS: the first matching pass (no category check) showed
wolf images scoring HIGHER against fox blog posts than against real wolf
posts (0.4858 vs 0.4315) — the fox posts' physical/descriptive language
overlapped more with image captions than the wolf posts' behavior-focused
writing did. No similarity threshold alone can fix that; a wide-enough
cutoff to accept correct wolf matches would also accept the wrong-species
one. Checking category match explicitly, before similarity, closes this
gap — matches your capstone brief's own example guard reason ("Animal
category mismatch: expected fox, detected wolf").

Every image x post pair gets a row in `matches`, per your DB design —
not just the top match per image.

Tune CAT_THRESHOLD / CAP_THRESHOLD below if needed; these weren't pinned
to specific numbers in your Phase 1 design, just to "independent checks",
so 0.6 is a reasonable starting default given your actual confidence
scores are mostly clustering in the 0.7-1.0 range.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from embeddings import cosine_similarity
from guard import guard_decision, SIMILARITY_THRESHOLDS, DEFAULT_THRESHOLD, CAT_THRESHOLD, CAP_THRESHOLD

DB_PATH = Path("data/pipeline.db")


def init_matches_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id TEXT PRIMARY KEY,
            image_id TEXT,
            post_id TEXT,
            similarity REAL,
            guard_decision TEXT,
            guard_reason TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    init_matches_table(conn)

    # Wipe and recompute each run — cheap at this scale (344 pairs), and
    # keeps this exploratory pass simple. Once you've picked a real
    # threshold, we'll switch this to resumable/incremental.
    conn.execute("DELETE FROM matches")
    conn.commit()

    images = conn.execute(
        "SELECT id, filepath, category, conf_category, conf_caption, embedding "
        "FROM images WHERE embedding IS NOT NULL"
    ).fetchall()
    posts = conn.execute(
        "SELECT id, title, embedding, expected_category FROM blog_posts WHERE embedding IS NOT NULL"
    ).fetchall()

    print(f"Scoring {len(images)} images x {len(posts)} posts = {len(images) * len(posts)} pairs\n")

    pre_check_rejects = {}   # rejected before similarity was even computed
    scored_rows = []         # every pair that reached similarity scoring

    for img_id, filepath, category, conf_cat, conf_cap, img_emb_json in images:
        img_vec = json.loads(img_emb_json)

        for post_id, title, post_emb_json, expected_category in posts:
            # Compute similarity only when the guard will actually need it
            # (category match already confirmed) — avoids wasted cosine
            # calls on pairs that get rejected before that point anyway.
            if (
                category not in ("uncertain", "other")
                and conf_cat >= CAT_THRESHOLD
                and conf_cap >= CAP_THRESHOLD
                and category == expected_category
            ):
                similarity = cosine_similarity(img_vec, json.loads(post_emb_json))
            else:
                similarity = None

            decision, reason = guard_decision(
                image_category=category,
                conf_category=conf_cat,
                conf_caption=conf_cap,
                expected_category=expected_category,
                similarity=similarity,
            )

            if similarity is not None:
                scored_rows.append((filepath, category, title, similarity, decision))

            if similarity is None:
                # Rejected at a pre-check step, before similarity was computed.
                # Bucket by which step fired, not the full per-row reason text.
                if category == "uncertain":
                    key = "uncertain category"
                elif conf_cat < CAT_THRESHOLD:
                    key = "low category confidence"
                elif conf_cap < CAP_THRESHOLD:
                    key = "low caption confidence"
                elif category == "other":
                    key = "category out of scope (other)"
                else:
                    key = "category mismatch"
                pre_check_rejects[key] = pre_check_rejects.get(key, 0) + 1

            conn.execute(
                "INSERT INTO matches (id, image_id, post_id, similarity, guard_decision, guard_reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), img_id, post_id, similarity, decision, reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    conn.commit()

    print("=== Rejected before similarity check ===")
    for reason_key, count in pre_check_rejects.items():
        print(f"  {count:4d}  {reason_key}")

    approved = [r for r in scored_rows if r[4] == "approved"]
    rejected_by_threshold = [r for r in scored_rows if r[4] == "rejected"]

    print(f"\n=== Scored pairs: {len(scored_rows)} total "
          f"({len(approved)} approved, {len(rejected_by_threshold)} rejected by threshold) ===\n")

    print(f"--- Approved matches (sorted by similarity) ---")
    print(f"{'IMAGE':<20} {'CATEGORY':<8} {'BLOG POST':<45} {'SIMILARITY':>10}")
    for filepath, category, title, sim, _ in sorted(approved, key=lambda r: r[3], reverse=True):
        print(f"{filepath:<20} {category:<8} {title[:43]:<45} {sim:>10.4f}")

    print(f"\n--- Rejected by threshold (closest misses first) ---")
    for filepath, category, title, sim, _ in sorted(rejected_by_threshold, key=lambda r: r[3], reverse=True):
        threshold = SIMILARITY_THRESHOLDS.get(category, DEFAULT_THRESHOLD)
        print(f"{filepath:<20} {category:<8} {title[:43]:<45} {sim:>10.4f}  (needed {threshold})")

    print(f"\n=== Approved matches per category ===")
    for cat in SIMILARITY_THRESHOLDS:
        n = len([r for r in approved if r[1] == cat])
        total_cat = len([r for r in scored_rows if r[1] == cat])
        print(f"  {cat:<8} {n}/{total_cat} scored pairs approved")

    print(f"\nTotal rows written to matches: {len(images) * len(posts)}")
    conn.close()


if __name__ == "__main__":
    main()
