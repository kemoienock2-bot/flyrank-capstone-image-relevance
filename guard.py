"""
guard.py — the mismatch guard, as a standalone function.

Extracted out of compute_matches.py so it can be unit tested in isolation
(see test_pipeline.py) without needing a real database or embeddings.

Same decision order as designed in Phase 1, with the category-match step
added after real data showed pure similarity wasn't reliable across
species (see EVIDENCE.md for the full story):

  1. category == "uncertain"            -> reject, similarity not computed
  2. conf_category < CAT_THRESHOLD      -> reject, similarity not computed
  3. conf_caption < CAP_THRESHOLD       -> reject, similarity not computed
  4. category == "other"                -> reject, similarity not computed
  5. category != post.expected_category -> reject, similarity not computed
  6. otherwise                          -> compare similarity against the
                                            category's threshold, approve
                                            or reject
"""

CAT_THRESHOLD = 0.6
CAP_THRESHOLD = 0.6

SIMILARITY_THRESHOLDS = {
    "fox": 0.40,
    "wolf": 0.25,
    "dog": 0.20,
}
DEFAULT_THRESHOLD = 0.30


def guard_decision(
    image_category: str,
    conf_category: float,
    conf_caption: float,
    expected_category: str,
    similarity: float | None,
) -> tuple[str, str]:
    """
    Runs one image/post pair through the full guard decision order.

    `similarity` should be the precomputed cosine similarity if you have
    it, or None if you want the guard to short-circuit before needing it
    (the guard only actually uses `similarity` at step 6 — the caller is
    responsible for computing it, this function doesn't touch embeddings).

    Returns (decision, reason) where decision is "approved" or "rejected".
    """
    if image_category == "uncertain":
        return "rejected", "category uncertain — model could not confidently identify the animal"

    if conf_category < CAT_THRESHOLD:
        return "rejected", f"category confidence {conf_category:.2f} below threshold {CAT_THRESHOLD}"

    if conf_caption < CAP_THRESHOLD:
        return "rejected", f"caption confidence {conf_caption:.2f} below threshold {CAP_THRESHOLD}"

    if image_category == "other":
        return "rejected", "category out of scope (other)"

    if image_category != expected_category:
        return "rejected", f"category mismatch: image={image_category}, post expects={expected_category}"

    threshold = SIMILARITY_THRESHOLDS.get(image_category, DEFAULT_THRESHOLD)
    if similarity is None:
        raise ValueError("similarity must be provided once a pair reaches the threshold check")

    if similarity >= threshold:
        return "approved", f"similarity={similarity:.3f} >= {image_category} threshold {threshold}"
    else:
        return "rejected", f"similarity={similarity:.3f} below {image_category} threshold {threshold}"
