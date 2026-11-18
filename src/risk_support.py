"""S6 risk calculation and a deliberately separate review trigger."""

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum

from .config import FourWeights, Settings, validate_unit_weights
from .schemas import Message, MessageKey

LEVELS = ("一般关注", "重点关注", "较高风险", "高风险")
STRATEGIES = (
    ("共情回应", "温和询问"),
    ("渐进式提问", "鼓励表达"),
    ("证据保存", "屏蔽或举报", "可信成年人求助引导"),
    ("及时求助", "人工复核"),
)


@dataclass(frozen=True)
class SafetySignal:
    message_key: MessageKey
    category: str
    matched_phrase: str


@dataclass(frozen=True)
class RiskResult:
    U: float | None
    score_level: str | None
    risk_level: str | None
    status: str
    safety_override: bool
    human_review_required: bool
    safety_source: str
    safety_signals: tuple[SafetySignal, ...]
    allowed_strategies: tuple[str, ...]
    support_source: str
    empathy_reply: str | None = None
    support_strategy: dict | None = None
    support_status: str = "unavailable"


def risk_score(B: float, H: float, P: float, F: float, weights: FourWeights) -> float:
    values = (B, H, P, F)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("B, H, P and F must lie in [0, 1]")
    validate_unit_weights(weights, "risk_weights")
    return fsum(value * weight for value, weight in zip(values, weights, strict=True))


def risk_level_index(U: float, thresholds: tuple[float, float, float]) -> int:
    return bisect_right(thresholds, U)


def check_safety(messages: Iterable[Message], rules: dict[str, tuple[str, ...]]) -> tuple[SafetySignal, ...]:
    return tuple(
        SafetySignal(message.key, category, phrase)
        for message in messages
        for category, phrases in rules.items()
        for phrase in phrases if phrase in message.text
    )


def evaluate_risk(
    B: float, H: float | None, P: float | None, F: float, messages: Iterable[Message], settings: Settings
) -> RiskResult:
    demo = settings.engineering_demo_default
    signals = check_safety(messages, demo.safety_phrases)
    U = risk_score(B, H, P, F, demo.risk_weights) if H is not None and P is not None else None
    score_index = risk_level_index(U, settings.patent_example.risk_thresholds) if U is not None else None
    final_index = 3 if signals else score_index
    return RiskResult(
        U=U,
        score_level=LEVELS[score_index] if score_index is not None else None,
        risk_level=LEVELS[final_index] if final_index is not None else None,
        status="calculated" if U is not None else "score_unavailable",
        safety_override=bool(signals),
        human_review_required=final_index == 3,
        safety_source="engineering_keyword_baseline",
        safety_signals=signals,
        allowed_strategies=STRATEGIES[final_index] if final_index is not None else (),
        support_source="policy_mapping_only" if final_index is not None else "unavailable",
    )
