"""
tag_schema.py — app-side validation for vision model output.

This is the ground-truth validator. Gemini's response_schema constrains the
*shape* of what comes back; this module enforces the *content rules* (word
count, punctuation, confidence range) that JSON Schema can't express.
Never trust the API's schema constraint as your only validation layer.
"""

from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal

# Swap these to match your real dataset categories if they differ.
CATEGORIES = ["fox", "wolf", "dog", "other", "uncertain"]


class Confidence(BaseModel):
    category: float
    caption: float

    @field_validator("category", "caption")
    @classmethod
    def in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be 0-1, got {v}")
        return v


class ImageTag(BaseModel):
    category: Literal["fox", "wolf", "dog", "other", "uncertain"]
    caption: str
    confidence: Confidence

    @field_validator("caption")
    @classmethod
    def caption_rules(cls, v):
        stripped = v.strip()
        if not stripped:
            raise ValueError("caption is empty")
        word_count = len(stripped.split())
        if not (10 <= word_count <= 25):
            raise ValueError(f"caption must be 10-25 words, got {word_count}")
        end_punct_count = sum(stripped.count(p) for p in ".!?")
        if end_punct_count > 1:
            raise ValueError(f"caption must be one sentence, found {end_punct_count} sentence-end marks")
        return stripped


def validate_row(raw_json: dict):
    """Returns {'status': 'PASS'|'FAIL', 'row': dict|None, 'reason': str|None}"""
    try:
        row = ImageTag.model_validate(raw_json)
        return {"status": "PASS", "row": row.model_dump(), "reason": None}
    except ValidationError as e:
        return {"status": "FAIL", "row": None, "reason": str(e)}
