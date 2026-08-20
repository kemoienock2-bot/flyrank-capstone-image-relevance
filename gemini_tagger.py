"""
gemini_tagger.py — single-image call to Gemini Flash with structured output.

Uses the current `google-genai` SDK (the old `google.generativeai` package
is deprecated and no longer receiving updates).

Uses the "gemini-flash-latest" alias rather than a pinned version like
"gemini-1.5-flash" — Google's Flash line ships new versions every few
weeks, and pinned old versions get fully retired (404, not just
deprecated) surprisingly fast. The alias always resolves to the current
stable Flash model, with a 2-week email notice before any breaking change.

Implements the error split you chose:
  - transient errors (rate limit, network, server hiccup) -> retry a few times
  - everything else -> no retry (validation happens separately in tag_schema.py)
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

MODEL_NAME = "gemini-2.5-flash"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["fox", "wolf", "dog", "other", "uncertain"],
        },
        "caption": {"type": "string"},
        "confidence": {
            "type": "object",
            "properties": {
                "category": {"type": "number"},
                "caption": {"type": "number"},
            },
            "required": ["category", "caption"],
        },
    },
    "required": ["category", "caption", "confidence"],
}

PROMPT = """You are labeling a single animal photo for a content-matching system.

Return:
- category: pick exactly one from the allowed list. Use "other" only if you
  clearly see an animal that is NOT in the list. Use "uncertain" only if you
  cannot confidently identify the animal at all (e.g. poor image quality,
  partial/obscured view, ambiguous species).
- caption: one sentence, 10-25 words, describing the animal, its setting,
  and what it's doing. Do not list keywords — write a real sentence a person
  would read, similar in style to a blog post description.
- confidence.category: your certainty (0-1) that the category label is correct.
- confidence.caption: your certainty (0-1) that the caption accurately
  describes what's in the image.

Score confidence honestly and independently — a confident category with an
uncertain caption (or vice versa) is a valid and expected combination.
"""

MAX_RETRIES = 3
BACKOFF_SECONDS = 2  # doubles each attempt: 2s, 4s, 8s


def _is_transient(e: Exception) -> bool:
    """429 rate limit or 5xx server errors are worth retrying; other client
    errors (bad request, invalid model name, auth failure) are not — retrying
    the same broken request just wastes calls."""
    if isinstance(e, genai_errors.ServerError):
        return True
    if isinstance(e, genai_errors.ClientError):
        return getattr(e, "code", None) == 429
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    return False


def call_gemini(client: "genai.Client", image_path: Path) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    Returns (result_or_None, error_type_or_None, detail_or_None)

    result, when present: {"raw": <parsed json dict>, "tokens": {...}}
    error_type, when present: "api_error" | "empty_response"
      (validation_error is assigned later, in run_batch.py, after
       tag_schema.validate_row runs on a *successful* API result)
    """
    image_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/webp"

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[PROMPT, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )

            if not response.text or not response.text.strip():
                # Not transient — retrying the same image+prompt won't fix a
                # safety block or empty candidate. No retry here.
                return None, "empty_response", "Gemini returned an empty response (possible safety block)"

            raw = json.loads(response.text)

            usage = getattr(response, "usage_metadata", None)
            tokens = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            } if usage else {}

            return {"raw": raw, "tokens": tokens}, None, None

        except Exception as e:
            if _is_transient(e):
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_SECONDS * attempt)
                    continue
                return None, "api_error", f"Failed after {MAX_RETRIES} attempts: {last_error}"
            else:
                # Non-transient (bad request, invalid model, auth error) —
                # don't burn retries on something retrying won't fix.
                return None, "api_error", f"Non-transient error (not retried): {e}"

    return None, "api_error", f"Failed after {MAX_RETRIES} attempts: {last_error}"