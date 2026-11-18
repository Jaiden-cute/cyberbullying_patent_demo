"""S1 Table 1 and the weighted event severity B."""

from dataclasses import dataclass
from datetime import timedelta
from math import fsum

from .config import SeverityWeights, Settings
from .event_builder import Event
from .math_utils import demo_lexical_cosine
from .schemas import MessageKey


@dataclass(frozen=True)
class SeverityResult:
    g: float
    a: float
    r: float
    h: float
    B: float
    repetition_basis: str
    repeat_count: int
    repetition_message_keys: tuple[MessageKey, ...]
    weights: tuple[float, ...]


def severity_score(g: float, a: float, r: float, h: float, weights: SeverityWeights) -> float:
    scores = (g, a, r, h)
    if any(score not in (0, 0.5, 1) for score in scores):
        raise ValueError("each Table 1 dimension must be 0, 0.5 or 1")
    return fsum(score * weight for score, weight in zip(scores, weights.values, strict=True))


def repetition_score(attack_count: int) -> float:
    if attack_count < 1:
        raise ValueError("an event must contain at least one attack")
    return min(attack_count - 1, 2) / 2


def _similar_attack_keys(
    event: Event, earlier_events: tuple[Event, ...], settings: Settings
) -> tuple[MessageKey, ...]:
    max_age = timedelta(days=settings.engineering_demo_default.similar_attack_window_days)
    threshold = settings.patent_example.attack_similarity_tau_s
    previous = [
        attack
        for prior in earlier_events
        if prior.platform == event.platform and prior.victim_id == event.victim_id and prior.synthetic == event.synthetic
        for attack in prior.attacks
        if attack.message.utc_time < event.start_time
    ]
    strongest: tuple[MessageKey, ...] = ()
    for attack in event.attacks:
        anchor = attack.message
        keys = [anchor.key]
        for old_attack in previous:
            old = old_attack.message
            if old_attack.initiator_id == attack.initiator_id and anchor.utc_time - old.utc_time <= max_age:
                similarity = demo_lexical_cosine(anchor.text, old.text)
                if similarity is not None and similarity >= threshold:
                    keys.append(old.key)
        if len(keys) > len(strongest):
            strongest = tuple(keys)
    return strongest


def evaluate_severity(
    event: Event, earlier_events: tuple[Event, ...], settings: Settings
) -> SeverityResult:
    default = settings.engineering_demo_default
    if default.repetition_basis == "within_event":
        evidence = tuple(attack.message.key for attack in event.attacks)
    else:
        evidence = _similar_attack_keys(event, earlier_events, settings)
    g = max(attack.finding.g for attack in event.attacks)
    a = max(attack.finding.a for attack in event.attacks)
    h = max(attack.finding.h for attack in event.attacks)
    r = repetition_score(len(evidence))
    B = severity_score(g, a, r, h, default.severity_weights)
    return SeverityResult(g, a, r, h, B, default.repetition_basis,
                          len(evidence), evidence, default.severity_weights.values)
