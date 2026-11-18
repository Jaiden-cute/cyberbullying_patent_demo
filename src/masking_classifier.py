"""S3 linear-head math and a separately labeled Phase A masking baseline."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite

from .config import validate_unit_weights
from .math_utils import sigmoid
from .schemas import MASKING_LABELS

MASKING_RULES = (
    ("没事", "我很好", "没有不开心"),
    ("没关系", "不严重", "小事", "开个玩笑"),
    ("他们只是", "他们就是", "本来是玩笑", "因为他们"),
    ("我本来就", "我真笨", "怪我太菜"),
    ("你们说得对", "都听你们的", "就按你们说的"),
    ("😂", "🤣", "😄", "😁"),
    ("别问了", "算了", "不想说"),
    ("换个话题", "说点别的", "对了，"),
)


@dataclass(frozen=True)
class MaskingResult:
    labels: tuple[str, ...]
    p: tuple[float, ...] | None
    non_model_score: tuple[float, ...] | None
    M: float | None
    inference_source: str
    model_status: str
    matched_phrases: dict[str, tuple[str, ...]]
    model_version: str | None = None
    training_data_source: str | None = None
    unavailable_reason: str | None = None


def _validate_values(values: Sequence[float]) -> None:
    if len(values) != len(MASKING_LABELS) or any(not 0 <= value <= 1 for value in values):
        raise ValueError("eight masking values in [0, 1] are required")


def joint_representation(event_vector: Sequence[float], reply_vector: Sequence[float]) -> tuple[float, ...]:
    if not len(event_vector) or len(event_vector) != len(reply_vector):
        raise ValueError("the two embeddings must have the same positive dimension")
    difference = (abs(left - right) for left, right in zip(event_vector, reply_vector, strict=True))
    joint = (*event_vector, *reply_vector, *difference)
    if any(not isfinite(value) for value in joint):
        raise ValueError("embeddings and their differences must be finite")
    return joint


def linear_probabilities(
    joint: Sequence[float], weight_rows: Sequence[Sequence[float]], biases: Sequence[float]
) -> tuple[float, ...]:
    """Evaluate p from supplied W/b; model loading and provenance checks belong to Stage C."""
    if (not len(joint) or len(weight_rows) != len(MASKING_LABELS) or len(biases) != len(MASKING_LABELS)
            or any(len(row) != len(joint) for row in weight_rows)):
        raise ValueError("the linear head requires eight weight rows matching the joint dimension")
    if any(not isfinite(value) for row in (joint, biases, *weight_rows) for value in row):
        raise ValueError("linear head inputs must be finite")
    return tuple(
        sigmoid(fsum((bias, *(value * weight for value, weight in zip(joint, row, strict=True)))))
        for row, bias in zip(weight_rows, biases, strict=True)
    )


def masking_score(values: Sequence[float], weights: tuple[float, ...]) -> float:
    _validate_values(values)
    if len(weights) != len(MASKING_LABELS):
        raise ValueError("eight masking weights are required")
    validate_unit_weights(weights, "masking_weights")
    return fsum(value * weight for value, weight in zip(values, weights, strict=True))


def evaluate_demo_masking(reply_text: str, weights: tuple[float, ...]) -> MaskingResult:
    matched = {
        label: tuple(phrase for phrase in phrases if phrase in reply_text)
        for label, phrases in zip(MASKING_LABELS, MASKING_RULES, strict=True)
    }
    scores = tuple(float(bool(phrases)) for phrases in matched.values())
    return MaskingResult(
        MASKING_LABELS, None, scores, masking_score(scores, weights),
        "rule_baseline", "trained_head_not_connected", matched,
    )
