"""
find_demo_examples.py — pulls real IDs and numbers from your own database
for each beat of the Phase 5 demo, so you're not hunting through Swagger
UI live in front of an audience.

Run once before your demo, save the output, and use it as your script.

Run:
  python find_demo_examples.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/pipeline.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("DEMO SCRIPT — pull these up live in the order below")
    print("=" * 70)

    # --- Beat 1: tagging in action ---
    print("\n[BEAT 1] Show tagging happening")
    print("  Say: 'Every image gets run through Gemini Flash and comes back")
    print("        with a category, a caption, and two confidence scores.'")
    sample = conn.execute(
        "SELECT filepath, category, caption, conf_category, conf_caption "
        "FROM images ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    if sample:
        print(f"  Show:  {sample['filepath']}")
        print(f"         category={sample['category']}  "
              f"conf_cat={sample['conf_category']:.2f}  conf_cap={sample['conf_caption']:.2f}")
        print(f"         \"{sample['caption']}\"")

    # --- Beat 2: a good match, ranked correctly ---
    print("\n[BEAT 2] The fox post surfaces the fox photo on top")
    row = conn.execute(
        """
        SELECT i.filepath, p.title, m.similarity, m.guard_decision
        FROM matches m JOIN images i ON m.image_id = i.id
        JOIN blog_posts p ON m.post_id = p.id
        WHERE m.guard_decision = 'approved' AND i.category = 'fox'
        ORDER BY m.similarity DESC LIMIT 1
        """
    ).fetchone()
    if row:
        print(f"  Query: GET /images/{{fox_image_id}}/matches")
        print(f"  Show:  {row['filepath']} -> \"{row['title']}\"  "
              f"(similarity={row['similarity']:.3f}, {row['guard_decision']})")

    # --- Beat 3: forced mismatch, guard refuses ---
    print("\n[BEAT 3] Force a wrong-species candidate — guard refuses it")
    mismatch = conn.execute(
        """
        SELECT i.filepath AS image_file, i.category AS image_category,
               p.title, m.guard_reason, m.similarity
        FROM matches m JOIN images i ON m.image_id = i.id
        JOIN blog_posts p ON m.post_id = p.id
        WHERE m.guard_decision = 'rejected' AND m.guard_reason LIKE 'category mismatch%'
        ORDER BY m.similarity DESC
        LIMIT 1
        """
    ).fetchone()
    if mismatch:
        print(f"  Show:  {mismatch['image_file']} (category={mismatch['image_category']}) "
              f"vs. \"{mismatch['title']}\"")
        print(f"  Reason shown on screen: \"{mismatch['guard_reason']}\"")
        if mismatch["similarity"] is not None:
            print(f"  (Worth mentioning: raw similarity was {mismatch['similarity']:.3f} — "
                  f"high enough to fool similarity alone, which is exactly why the "
                  f"category check exists as a separate guard step.)")
    else:
        print("  No category-mismatch example found in current data — rerun compute_matches.py")

    # --- Beat 4: no confident match ---
    print("\n[BEAT 4] An image with no good match — guard says so honestly")
    orphan = conn.execute(
        """
        SELECT i.filepath, i.category
        FROM images i
        WHERE i.category NOT IN ('other', 'uncertain')
        AND i.id NOT IN (
            SELECT image_id FROM matches WHERE guard_decision = 'approved'
        )
        LIMIT 1
        """
    ).fetchone()
    if orphan:
        print(f"  Show:  {orphan['filepath']} (category={orphan['category']})")
        near_misses = conn.execute(
            """
            SELECT p.title, m.similarity, m.guard_reason
            FROM matches m JOIN images i ON m.image_id = i.id
            JOIN blog_posts p ON m.post_id = p.id
            WHERE i.filepath = ? AND m.guard_decision = 'rejected'
            ORDER BY m.similarity DESC
            """,
            (orphan["filepath"],),
        ).fetchall()
        for nm in near_misses:
            sim_str = f"{nm['similarity']:.3f}" if nm["similarity"] is not None else "n/a"
            print(f"    - vs \"{nm['title']}\": similarity={sim_str} -> {nm['guard_reason']}")
        print("  Say: 'Every candidate was checked. None cleared the bar. "
              "The system says so instead of guessing.'")
    else:
        print("  No orphaned in-scope image found — every image has an approved match.")

    # --- Beat 5: review trail ---
    print("\n[BEAT 5] Approve one, reject another — show the review trail")
    print("  Live actions:")
    print("    POST /matches/{approved_match_id}/review  {\"decision\": \"approved\", \"note\": \"confirmed by eye\"}")
    print("    POST /matches/{rejected_match_id}/review  {\"decision\": \"rejected\", \"note\": \"...\"}")
    print("  Then: GET /matches?reviewed=true")
    print("  Point out: guard_decision never changes — human_decision sits next to it.")

    # --- Beat 6: eval number ---
    print("\n[BEAT 6] Close with the eval number")
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    correct = 0
    rows = conn.execute("SELECT filepath, category FROM images").fetchall()
    import re
    for r in rows:
        gt = re.split(r"[\\/]", r["filepath"])[0]
        if gt == r["category"]:
            correct += 1
    if total:
        print(f"  Say: 'Tagging accuracy: {correct}/{total} = {correct/total:.1%}'")
    print("  Say: 'Good suggestions when confident, safe rejection when uncertain —")
    print("        that's production AI.'")

    conn.close()


if __name__ == "__main__":
    main()
