"""
run_batch.py — sequential batch tagging of your full dataset.

SETUP
1. pip uninstall google-generativeai -y   (old, deprecated package — remove
   it to avoid confusion, though having both installed doesn't break this)
2. pip install google-genai
3. Get a free Gemini API key: https://aistudio.google.com/apikey
4. PowerShell: $env:GEMINI_API_KEY="your_key_here"
   Mac/Linux:  export GEMINI_API_KEY="your_key_here"
5. python run_batch.py

WHAT IT DOES
- Walks data/raw/{category}/*.jpg (and .webp) in sorted order
- Calls Gemini Flash (via the "gemini-flash-latest" alias) with structured
  output for each image, sequentially
- On a transient error (rate limit, network, timeout): retries up to 3x,
  then logs as "api_error" and moves on — no image is silently dropped
  without a reason in the log
- On a successful call: runs tag_schema.validate_row(). If it fails
  (bad caption length, etc.) logs as "validation_error" — no retry,
  matching the hard-fail rule you set for content quality
- On success: inserts into data/pipeline.db -> images table
- Resumable: if the script crashes or you stop it, rerunning skips images
  already present in the DB, so you don't reprocess (and re-spend calls on)
  images that already passed
- Logs every API call's token usage to data/cost_log.csv, separately from
  pass/fail outcome — the call cost was incurred whether or not the
  content later passed validation

Rate limit note: Gemini's free tier RPM limits change over time — check
https://ai.google.dev/gemini-api/docs/rate-limits for the current number
and adjust DELAY_BETWEEN_CALLS below if you hit repeated 429s even with
the retry logic.
"""

import os
import csv
import time
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

from google import genai

from gemini_tagger import call_gemini
from tag_schema import validate_row

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Missing GEMINI_API_KEY. Set it with:\n"
        "  PowerShell:  $env:GEMINI_API_KEY=\"your_key_here\"\n"
        "  Mac/Linux:   export GEMINI_API_KEY='your_key_here'\n"
        "Get a free key at https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=API_KEY)

DATA_ROOT = Path("data/raw")
DB_PATH = Path("data/pipeline.db")
FAILURE_LOG = Path("data/tagging_failures.csv")
COST_LOG = Path("data/cost_log.csv")

DELAY_BETWEEN_CALLS = 4.0  # seconds between calls, stay comfortably under free-tier RPM


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filepath TEXT UNIQUE,
            category TEXT,
            caption TEXT,
            conf_category REAL,
            conf_caption REAL,
            embedding TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def already_processed(conn, filepath: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM images WHERE filepath = ?", (filepath,)
    ).fetchone() is not None


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    fail_write_header = not FAILURE_LOG.exists()
    cost_write_header = not COST_LOG.exists()

    image_paths = sorted(DATA_ROOT.glob("*/*.jpg")) + sorted(DATA_ROOT.glob("*/*.webp"))
    print(f"Found {len(image_paths)} images under {DATA_ROOT}/")

    with open(FAILURE_LOG, "a", newline="") as flog, open(COST_LOG, "a", newline="") as clog:
        fwriter = csv.writer(flog)
        cwriter = csv.writer(clog)
        if fail_write_header:
            fwriter.writerow(["filepath", "error_type", "detail", "timestamp"])
        if cost_write_header:
            cwriter.writerow(["filepath", "prompt_tokens", "output_tokens", "total_tokens", "timestamp"])

        passed, failed, skipped = 0, 0, 0

        for path in image_paths:
            rel_path = str(path.relative_to(DATA_ROOT))

            if already_processed(conn, rel_path):
                skipped += 1
                print(f"  SKIP (already tagged): {rel_path}")
                continue

            print(f"Processing {rel_path} ...")
            result, error_type, detail = call_gemini(client, path)

            if error_type:
                failed += 1
                fwriter.writerow([rel_path, error_type, detail, datetime.now(timezone.utc).isoformat()])
                flog.flush()
                print(f"  FAILED ({error_type}): {detail}")
                time.sleep(DELAY_BETWEEN_CALLS)
                continue

            # Log token usage regardless of validation outcome below —
            # the API call happened and cost tokens either way.
            tokens = result.get("tokens", {})
            cwriter.writerow([
                rel_path,
                tokens.get("prompt_tokens"),
                tokens.get("output_tokens"),
                tokens.get("total_tokens"),
                datetime.now(timezone.utc).isoformat(),
            ])
            clog.flush()

            validation = validate_row(result["raw"])
            if validation["status"] == "FAIL":
                failed += 1
                fwriter.writerow([rel_path, "validation_error", validation["reason"], datetime.now(timezone.utc).isoformat()])
                flog.flush()
                print(f"  REJECTED (validation): {validation['reason']}")
                time.sleep(DELAY_BETWEEN_CALLS)
                continue

            row = validation["row"]
            conn.execute(
                """INSERT INTO images
                   (id, filepath, category, caption, conf_category, conf_caption, embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    rel_path,
                    row["category"],
                    row["caption"],
                    row["confidence"]["category"],
                    row["confidence"]["caption"],
                    None,  # embedding filled in during Phase 3
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            passed += 1
            print(f"  OK: category={row['category']}  "
                  f"conf_cat={row['confidence']['category']:.2f}  "
                  f"conf_cap={row['confidence']['caption']:.2f}")

            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\nDone. Passed: {passed}  Failed: {failed}  Skipped (already done): {skipped}")
    print(f"DB:           {DB_PATH.resolve()}")
    print(f"Failure log:  {FAILURE_LOG.resolve()}")
    print(f"Cost log:     {COST_LOG.resolve()}")
    conn.close()


if __name__ == "__main__":
    main()