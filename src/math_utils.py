"""Numerical operations shared by the patent formulas."""

from collections import Counter
from collections.abc import Sequence
from math import exp, fsum, hypot, isfinite
from unicodedata import category


def sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1 / (1 + exp(-logit))
    exponential = exp(logit)
    return exponential / (1 + exponential)


def cosine_similarity(
    left: Sequence[float], right: Sequence[float], *, norm_floor: float = 0.0
) -> float | None:
    """Return signed cosine, or None for an empty or effectively zero vector."""
    if len(left) != len(right):
        raise ValueError("cosine vectors must have the same dimension")
    if not all(isfinite(value) for vector in (left, right) for value in vector):
        raise ValueError("cosine vectors must be finite")

    left_scale = max(map(abs, left), default=0.0)
    right_scale = max(map(abs, right), default=0.0)
    if not left_scale or not right_scale:
        return None
    scaled_left = [value / left_scale for value in left]
    scaled_right = [value / right_scale for value in right]
    left_norm = hypot(*scaled_left)
    right_norm = hypot(*scaled_right)
    if left_scale <= norm_floor / left_norm or right_scale <= norm_floor / right_norm:
        return None
    dot = fsum(x * y for x, y in zip(scaled_left, scaled_right, strict=True))
    return max(-1.0, min(1.0, dot / left_norm / right_norm))


def _character_counts(text: str) -> Counter[str]:
    chars = [char for char in text.casefold() if char.isalnum() or category(char) == "So"]
    counts = Counter(chars)
    counts.update(left + right for left, right in zip(chars, chars[1:]))
    return counts


def demo_lexical_cosine(left: str, right: str) -> float | None:
    """Deterministic character unigram/bigram baseline, independent of Sentence-BERT."""
    left_counts = _character_counts(left)
    right_counts = _character_counts(right)
    vocabulary = sorted(left_counts.keys() | right_counts.keys())
    return cosine_similarity(
        [left_counts[term] for term in vocabulary],
        [right_counts[term] for term in vocabulary],
    )
