"""
Phase 1 dataset collector — AI Image Understanding & Content Matching Engine

WHAT THIS DOES
Downloads images from Unsplash (free API, no credit card) into your
`data/raw/{category}/{id}.jpg` convention, and writes `data/provenance.csv`
with source URL + license info as it goes — captured at collection time,
not reconstructed after.

SETUP (2 minutes)
1. Go to https://unsplash.com/developers -> "Register as a developer"
   (free, Google/GitHub login is fine, no card requested).
2. Create a new application -> copy the "Access Key".
3. Set it as an environment variable before running:
     export UNSPLASH_ACCESS_KEY="your_key_here"
4. pip install requests
5. python collect_dataset.py

WHAT YOU GET
data/raw/fox/0001.jpg ... data/raw/fox/0013.jpg
data/raw/wolf/0001.jpg ...
data/raw/dog/0001.jpg ...
data/raw/other/0001.jpg ...   (cats, birds — deliberately out-of-scope
                                images to exercise the "other" guard path)
data/provenance.csv           (filename, category, source_url, photographer,
                                license, downloaded_at)

Adjust CATEGORY_QUERIES below to match your real category enum before running.
"""

import os
import csv
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
if not ACCESS_KEY:
    raise SystemExit(
        "Missing UNSPLASH_ACCESS_KEY. Set it with:\n"
        "  export UNSPLASH_ACCESS_KEY='your_key_here'\n"
        "Get a free key at https://unsplash.com/developers"
    )

# category -> (search query, how many images to pull)
# "other" uses real species outside your enum, so the guard's "other" path
# actually gets exercised in eval, not just theorized about.
CATEGORY_QUERIES = {
    "fox": ("red fox wildlife", 13),
    "wolf": ("gray wolf wildlife", 13),
    "dog": ("dog breed portrait", 13),
    "other": ("cat OR bird wildlife", 5),
}

OUTPUT_ROOT = Path("data/raw")
PROVENANCE_PATH = Path("data/provenance.csv")
API_URL = "https://api.unsplash.com/search/photos"

HEADERS = {"Authorization": f"Client-ID {ACCESS_KEY}"}


def fetch_category(query: str, count: int, page: int = 1):
    """Fetch one page of search results from Unsplash."""
    resp = requests.get(
        API_URL,
        headers=HEADERS,
        params={"query": query, "per_page": min(count, 30), "page": page},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def download_image(url: str, dest: Path):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not PROVENANCE_PATH.exists()
    with open(PROVENANCE_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["filename", "category", "source_url", "photographer",
                 "license", "downloaded_at"]
            )

        for category, (query, count) in CATEGORY_QUERIES.items():
            print(f"\n=== {category}: fetching '{query}' (target {count}) ===")
            cat_dir = OUTPUT_ROOT / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            results = fetch_category(query, count)
            if not results:
                print(f"  WARNING: no results for '{query}' — check query or key")
                continue

            for i, photo in enumerate(results[:count], start=1):
                image_url = photo["urls"]["regular"]
                photo_page_url = photo["links"]["html"]
                photographer = photo["user"]["name"]
                filename = f"{i:04d}.jpg"
                dest_path = cat_dir / filename

                try:
                    download_image(image_url, dest_path)
                except requests.RequestException as e:
                    print(f"  FAILED: {filename} ({e}) — skipping, not logged")
                    continue

                writer.writerow([
                    f"{category}/{filename}",
                    category,
                    photo_page_url,
                    photographer,
                    "Unsplash License (free use, attribution appreciated)",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ])
                f.flush()  # write provenance immediately, not after the fact
                print(f"  saved {dest_path}  (by {photographer})")

                time.sleep(0.3)  # stay well under Unsplash's free-tier rate limit

    print(f"\nDone. Provenance log: {PROVENANCE_PATH.resolve()}")


if __name__ == "__main__":
    main()
