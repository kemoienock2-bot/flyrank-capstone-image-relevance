# AI Image Understanding & Content Matching Engine

A capstone project that looks at a photo, figures out what animal is in it, and
matches it to a blog post that actually makes sense to pair it with — while
being willing to say "no good match" instead of forcing a bad one.

Built as part of a backend AI engineering internship. Free-tier only: Gemini
Flash for the vision step, a local embedding model for everything else.

## What it actually does

1. **Look at a photo and describe it.** Gemini Flash looks at each image and
   returns a category (fox, wolf, dog, or "other"/"uncertain" if it's not
   sure), a one-sentence caption, and two separate confidence scores.
2. **Check that the output is actually usable.** Every response gets checked
   against strict rules — right category, a real one-sentence caption, valid
   confidence numbers. If something fails, it's rejected and logged with a
   reason. Nothing gets silently patched up.
3. **Turn text into numbers.** Both the image captions and the blog posts get
   converted into embeddings (vectors) using a small local model, so they can
   be compared mathematically.
4. **Score every image against every blog post.** Not just the best match —
   every possible pairing gets a similarity score.
5. **Decide what's actually a good match.** This is the guard. It rejects
   pairings step by step — wrong category, low confidence, out of scope, or
   just not similar enough — and only approves what's left. Every rejection
   comes with a plain-English reason.
6. **Let a human double-check.** A small API lets someone look at what the
   guard approved or rejected and override it if needed, without ever
   erasing the original automated decision.

## Why it's built this way

The one rule that shows up everywhere in this project: **when something's
wrong, say so clearly — don't guess, don't quietly fix it, don't blend
signals together.** A caption that's too long gets rejected, not trimmed. A
low-confidence tag gets flagged, not averaged into a "good enough" score. A
human's override doesn't erase the guard's original reasoning — both stay
visible.

## Where this deviates from the brief

The brief's example tag schema uses two separate fields — `subject` (e.g.
"red fox") and `category` (a broader class like "animal"). This project
merges those into a single `category` field, using a closed set of specific
species (fox, wolf, dog) plus two fallback values: `other` (confidently
out of scope) and `uncertain` (not confident enough to say). This was a
deliberate simplification: for a guard that needs to compare an image's
category against a blog post's expected category, one specific, closed
value is easier to validate and reason about than two fields at different
levels of abstraction. `EVIDENCE.md` covers the reasoning in full, including
one guard step that isn't in the brief's original example at all — a direct
category-match check between image and post, added after real similarity
scores showed a wolf photo scoring *higher* against a fox blog post than
against an actual wolf post. That's documented in detail in `EVIDENCE.md`
under "A real problem the guard caught mid-build."

Confidence is also split into two independent scores (`category` and
`caption`) rather than the brief's single confidence number, since a model
can be sure what an animal is while being unsure its caption describes it
well, or the reverse — collapsing that into one number would hide which
part of the output to actually distrust.

Everything else — the guard's core purpose, the matching approach, the
review workflow — follows the brief as written.

## Using it from an AI agent (MCP server)

`mcp_server.py` exposes the matching engine over the
[Model Context Protocol](https://modelcontextprotocol.io), so an AI agent
(Claude Desktop, an IDE agent, the MCP Inspector) can query and review
matches directly.

| Tool | What it does | Writes? |
|---|---|---|
| `check_pair` | Runs the guard on a hypothetical image/post pair and explains the decision | No |
| `list_matches` | Lists stored matches, filterable by guard decision, category, review status | No |
| `get_match` | Full detail for one match | No |
| `matches_for_image` | Every post scored against one image, best first | No |
| `review_match` | Records a human decision; requires a note; never overwrites the guard's decision | **Yes** |

**Run it**

```bash
pip install "mcp[cli]>=2,<3"
mcp dev mcp_server.py      # opens the MCP Inspector in your browser
pytest -q test_mcp_server.py
```

**Connect it to Claude Desktop** by adding this to `claude_desktop_config.json`
(use absolute paths):

```json
{
  "mcpServers": {
    "image-relevance": {
      "command": "python",
      "args": ["C:/path/to/flyrank-capstone-image-relevance/mcp_server.py"]
    }
  }
}
```

**Design choices**
- The database path is resolved relative to the script, not the working
  directory, because MCP hosts launch servers from an arbitrary folder.
- Expected failures (unknown id, bad input) come back as an `error` field
  the agent can read and recover from, not a crash.
- Only one tool writes, it demands an explanation, and it leaves the guard's
  original reasoning untouched — the same audit rule as `review_api.py`.
- Nothing is printed to stdout: over stdio, stdout is the protocol channel.


## Project structure

```
flyrank-capstone/
├── data/
│   ├── raw/{category}/{id}.jpg     # source images, sorted by true category
│   ├── provenance.csv              # where each image came from + license
│   ├── blog_posts.csv              # sample blog posts used for matching
│   ├── pipeline.db                 # SQLite database — everything lives here
│   ├── tagging_failures.csv        # every rejected caption/validation, with reason
│   └── cost_log.csv                # token usage per API call
├── collect_dataset.py              # pulls the image dataset from Unsplash
├── tag_schema.py                   # validation rules for vision model output
├── gemini_tagger.py                # the actual Gemini API call, with retry logic
├── run_batch.py                    # runs tagging across the whole dataset
├── embeddings.py                   # shared embedding + similarity code
├── embed_images.py                 # generates embeddings for tagged images
├── load_blog_posts.py              # loads and embeds the blog post dataset
├── add_expected_category.py        # one-time migration: adds category labels to posts
├── compute_matches.py              # scores every image x post pair, runs the guard
├── add_review_columns.py           # one-time migration: adds human review fields
├── review_api.py                   # FastAPI service for human review
└── eval_report.py                  # accuracy + coverage numbers for evaluation
```

## How to run it, start to finish

```bash
# 1. Get the dataset
export UNSPLASH_ACCESS_KEY="your_key"
python collect_dataset.py

# 2. Tag every image with Gemini
export GEMINI_API_KEY="your_key"
python run_batch.py

# 3. Generate embeddings (local, no API key needed)
pip install sentence-transformers
python embed_images.py

# 4. Load and embed the sample blog posts
python load_blog_posts.py
python add_expected_category.py

# 5. Score every image against every post, run the guard
python compute_matches.py

# 6. Start the review API
pip install fastapi uvicorn
python add_review_columns.py
uvicorn review_api:app --reload
# then open http://127.0.0.1:8000/docs

# 7. See how well it actually did
python eval_report.py
```

## The guard, in plain terms

For every image/post pairing, the system checks things in this order and
stops at the first one that applies:

1. Is the image's category "uncertain"? → reject, model wasn't confident enough
2. Is the category confidence too low? → reject, even though it did pick a category
3. Is the caption confidence too low? → reject
4. Is the category "other" (a cat, a bird, something out of scope)? → reject
5. Does the image's category not match what this blog post is about? → reject
6. Otherwise — how similar is the caption to the post, actually? Compare
   against a threshold and either approve or reject.

Every one of these produces a specific, readable reason that gets stored
alongside the decision. Nothing is a mystery "no."

## Results

See `EVIDENCE.md` for the actual numbers, what went wrong along the way, and
what was learned from it.

## Known limitations

- The dataset is small (43 images, 8 blog posts) — good enough to prove the
  pipeline works end to end, not enough to trust the exact threshold numbers
  as production-ready.
- The similarity thresholds were hand-picked by looking at real score
  distributions, not learned from a large labeled set. They're a reasonable
  starting point, not a finished calibration.
- Blog post content here is written for testing, not pulled from a real
  content library.
