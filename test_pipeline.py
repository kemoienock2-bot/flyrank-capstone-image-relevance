"""
test_pipeline.py — automated tests for the Definition of Done checkbox:
"Automated tests cover schema validation, mismatch rejection, and
matching accuracy."

Run:
  pip install pytest
  pytest test_pipeline.py -v

These test the pure logic (tag_schema.py, guard.py, embeddings.py)
directly — no live API calls, no real database needed, so they run
instantly and don't burn any API quota.
"""

import math

import pytest

from tag_schema import validate_row
from guard import guard_decision, CAT_THRESHOLD, CAP_THRESHOLD, SIMILARITY_THRESHOLDS
from embeddings import cosine_similarity


# ---------------------------------------------------------------------
# Schema validation (Phase 1/2 rules)
# ---------------------------------------------------------------------

def test_valid_row_passes():
    row = {
        "category": "fox",
        "caption": "A red fox stands alert in a snowy forest clearing at dusk.",
        "confidence": {"category": 0.95, "caption": 0.9},
    }
    result = validate_row(row)
    assert result["status"] == "PASS"
    assert result["row"]["category"] == "fox"


def test_caption_too_short_fails():
    row = {
        "category": "dog",
        "caption": "A dog in a field.",  # 5 words, needs 10-25
        "confidence": {"category": 0.9, "caption": 0.8},
    }
    result = validate_row(row)
    assert result["status"] == "FAIL"
    assert "10-25 words" in result["reason"]


def test_caption_too_long_fails():
    row = {
        "category": "wolf",
        "caption": " ".join(["word"] * 30),  # 30 words, over the limit
        "confidence": {"category": 0.9, "caption": 0.8},
    }
    result = validate_row(row)
    assert result["status"] == "FAIL"
    assert "10-25 words" in result["reason"]


def test_multi_sentence_caption_fails():
    row = {
        "category": "dog",
        "caption": "A dog sits in a field. It looks happy and calm today.",
        "confidence": {"category": 0.9, "caption": 0.8},
    }
    result = validate_row(row)
    assert result["status"] == "FAIL"
    assert "one sentence" in result["reason"]


def test_invalid_category_fails():
    row = {
        "category": "cat",  # not in the enum
        "caption": "A cat sits calmly on a windowsill in the afternoon sun.",
        "confidence": {"category": 0.9, "caption": 0.8},
    }
    result = validate_row(row)
    assert result["status"] == "FAIL"


def test_confidence_out_of_range_fails():
    row = {
        "category": "fox",
        "caption": "A red fox stands alert in a snowy forest clearing at dusk.",
        "confidence": {"category": 1.5, "caption": 0.8},  # out of 0-1 range
    }
    result = validate_row(row)
    assert result["status"] == "FAIL"


def test_uncertain_and_other_are_both_valid_categories():
    for cat in ("uncertain", "other"):
        row = {
            "category": cat,
            "caption": "The animal in this photo could not be clearly identified here.",
            "confidence": {"category": 0.3, "caption": 0.3},
        }
        result = validate_row(row)
        assert result["status"] == "PASS", f"category={cat} should be a valid enum value"


# ---------------------------------------------------------------------
# Guard mismatch rejection (Phase 3 guard logic)
# ---------------------------------------------------------------------

def test_uncertain_category_rejected_before_similarity():
    decision, reason = guard_decision(
        image_category="uncertain", conf_category=0.9, conf_caption=0.9,
        expected_category="fox", similarity=None,
    )
    assert decision == "rejected"
    assert "uncertain" in reason


def test_low_category_confidence_rejected():
    decision, reason = guard_decision(
        image_category="fox", conf_category=0.3, conf_caption=0.9,
        expected_category="fox", similarity=None,
    )
    assert decision == "rejected"
    assert "category confidence" in reason


def test_low_caption_confidence_rejected():
    decision, reason = guard_decision(
        image_category="fox", conf_category=0.9, conf_caption=0.3,
        expected_category="fox", similarity=None,
    )
    assert decision == "rejected"
    assert "caption confidence" in reason


def test_other_category_rejected():
    decision, reason = guard_decision(
        image_category="other", conf_category=0.9, conf_caption=0.9,
        expected_category="fox", similarity=None,
    )
    assert decision == "rejected"
    assert "out of scope" in reason


def test_category_mismatch_rejected_the_fox_wolf_case():
    """The exact real-world case from the brief: a wolf photo must not
    be approved for a fox post, even if raw similarity would say yes."""
    decision, reason = guard_decision(
        image_category="wolf", conf_category=0.9, conf_caption=0.9,
        expected_category="fox", similarity=0.95,  # deliberately high —
        # must still be rejected on category alone, before this number matters
    )
    assert decision == "rejected"
    assert "category mismatch" in reason
    assert "wolf" in reason and "fox" in reason


def test_matching_category_and_high_similarity_is_approved():
    threshold = SIMILARITY_THRESHOLDS["fox"]
    decision, reason = guard_decision(
        image_category="fox", conf_category=0.9, conf_caption=0.9,
        expected_category="fox", similarity=threshold + 0.1,
    )
    assert decision == "approved"


def test_matching_category_but_low_similarity_is_rejected():
    threshold = SIMILARITY_THRESHOLDS["dog"]
    decision, reason = guard_decision(
        image_category="dog", conf_category=0.9, conf_caption=0.9,
        expected_category="dog", similarity=threshold - 0.05,
    )
    assert decision == "rejected"
    assert "below" in reason


def test_missing_similarity_at_threshold_step_raises():
    """If a pair reaches the final check, similarity must have been
    computed — this should never happen in practice, and a silent
    None here would be a bug worth catching loudly, not swallowing."""
    with pytest.raises(ValueError):
        guard_decision(
            image_category="fox", conf_category=0.9, conf_caption=0.9,
            expected_category="fox", similarity=None,
        )


# ---------------------------------------------------------------------
# Matching accuracy (embedding similarity math)
# ---------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_crash():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    # Should return 0.0, not raise a ZeroDivisionError
    assert cosine_similarity(a, b) == 0.0


def test_more_similar_captions_score_higher():
    """Sanity check on the actual similarity ordering: a caption closer
    in meaning to the reference should score higher than an unrelated one."""
    reference = [0.9, 0.1, 0.0]
    similar = [0.85, 0.15, 0.05]
    unrelated = [0.0, 0.1, 0.95]

    sim_similar = cosine_similarity(reference, similar)
    sim_unrelated = cosine_similarity(reference, unrelated)

    assert sim_similar > sim_unrelated
