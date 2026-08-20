"""
eval_report.py — Phase 4 evaluation numbers for the capstone writeup.

Two separate metrics, kept separate on purpose (same "distinct failure
modes deserve distinct representations" principle as the rest of this
project — conflating tagging accuracy with matching quality would hide
which part of the pipeline is actually underperforming):

1. TAGGING ACCURACY — how often did Gemini's predicted `category` match
   the ground-truth label baked into your file path convention
   (data/raw/{category}/{id}.jpg)? This evaluates Phase 2 (the vision
   model) in isolation, independent of matching/guard logic entirely.

2. MATCHING COVERAGE — of your tagged images, how many ended up with at
   least one guard-approved blog post match, vs. how many are "orphaned"
   (correctly tagged, but no post scored high enough to pass the
   similarity threshold)? This evaluates Phase 3 (the guard + threshold
   choices) — orphaned images aren't bugs, they're the guard correctly
   refusing to force a weak match, but worth surfacing explicitly for
   your writeup rather than leaving it implicit.

Run:
  python eval_report.py
"""

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("data/pipeline.db")


def ground_truth_category(filepath: str) -> str:
    """Folder name is the ground truth, per the Phase 1 naming convention.
    Handles both '/' and '\\' since paths were written on Windows."""
    return re.split(r"[\\/]", filepath)[0]


def tagging_accuracy(conn):
    rows = conn.execute("SELECT filepath, category FROM images").fetchall()

    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[ground_truth][predicted] = count
    misclassified = []

    for filepath, predicted in rows:
        gt = ground_truth_category(filepath)
        confusion[gt][predicted] += 1
        if predicted == gt:
            correct += 1
        else:
            misclassified.append((filepath, gt, predicted))

    total = len(rows)
    accuracy = correct / total if total else 0.0

    print("=== Tagging Accuracy (Phase 2 eval) ===")
    print(f"Total tagged images: {total}")
    print(f"Correct (predicted == ground truth folder): {correct}")
    print(f"Accuracy: {accuracy:.1%}\n")

    print("Confusion matrix (rows = ground truth folder, columns = model prediction):")
    all_categories = sorted(set(confusion.keys()) | {p for d in confusion.values() for p in d})
    header = "ground truth".ljust(14) + "".join(c.ljust(10) for c in all_categories)
    print(header)
    for gt in sorted(confusion.keys()):
        row = gt.ljust(14)
        for pred in all_categories:
            row += str(confusion[gt].get(pred, 0)).ljust(10)
        print(row)

    if misclassified:
        print(f"\nMisclassified images ({len(misclassified)}):")
        for filepath, gt, pred in misclassified:
            print(f"  {filepath:<20} ground_truth={gt:<8} predicted={pred}")
    else:
        print("\nNo misclassifications.")

    return {"total": total, "correct": correct, "accuracy": accuracy, "misclassified": misclassified}


def matching_coverage(conn):
    images = conn.execute("SELECT id, filepath, category FROM images").fetchall()

    orphaned = []
    covered = []

    for img_id, filepath, category in images:
        approved = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE image_id = ? AND guard_decision = 'approved'",
            (img_id,),
        ).fetchone()[0]
        if approved > 0:
            covered.append((filepath, category, approved))
        else:
            orphaned.append((filepath, category))

    total = len(images)
    coverage_rate = len(covered) / total if total else 0.0

    print("\n=== Matching Coverage (Phase 3 eval) ===")
    print(f"Total tagged images: {total}")
    print(f"Images with >=1 approved match: {len(covered)}  ({coverage_rate:.1%})")
    print(f"Orphaned images (0 approved matches): {len(orphaned)}\n")

    if orphaned:
        print("Orphaned images (correctly tagged, but no post cleared the similarity threshold):")
        for filepath, category in orphaned:
            print(f"  {filepath:<20} category={category}")
    else:
        print("No orphaned images — every tagged image found at least one approved match.")

    print("\nCoverage by category:")
    by_cat = defaultdict(lambda: {"covered": 0, "total": 0})
    for filepath, category, _ in covered:
        by_cat[category]["covered"] += 1
        by_cat[category]["total"] += 1
    for filepath, category in orphaned:
        by_cat[category]["total"] += 1

    for cat in sorted(by_cat.keys()):
        c, t = by_cat[cat]["covered"], by_cat[cat]["total"]
        print(f"  {cat:<10} {c}/{t}  ({c/t:.0%})")

    return {"total": total, "covered": len(covered), "orphaned": orphaned, "coverage_rate": coverage_rate}


def main():
    conn = sqlite3.connect(DB_PATH)
    tagging_accuracy(conn)
    matching_coverage(conn)
    conn.close()


if __name__ == "__main__":
    main()
