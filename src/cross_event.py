"""S5 event recurrence F and time-weighted cross-event masking recurrence P."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, expm1, fsum, isfinite

from .config import Settings
from .event_builder import Event
from .math_utils import cosine_similarity
from .schemas import MASKING_LABELS


@dataclass(frozen=True)
class Pattern:
    values: tuple[float, ...]
    N: float | None
    H: float
    source: str
    reply_time: datetime

    @property
    def vector(self) -> tuple[float, ...]:
        return self.values if self.N is None else (*self.values, self.N)


@dataclass(frozen=True)
class HistoryEvent:
    event: Event
    pattern: Pattern | None


@dataclass(frozen=True)
class Comparison:
    event_id: str
    dimension: int
    delta_days: float
    Sim: float | None
    H_m: float
    indicator: int
    decay_weight: float
    weighted_contribution: float
    status: str


@dataclass(frozen=True)
class CrossEventResult:
    P: float | None
    P_status: str
    F: float
    n_i: int
    recent_event_ids: tuple[str, ...]
    qualifying_pattern_count_including_current: int
    comparisons: tuple[Comparison, ...]
    numerator: float | None
    denominator: float | None


def event_recurrence(n_i: int, n0: float) -> float:
    if n_i < 0 or not isfinite(n0) or n0 <= 0:
        raise ValueError("event count must be nonnegative and n0 must be finite and positive")
    return 0.0 if n_i == 0 else -expm1(-n_i / n0)


def compare_patterns(current: Pattern, previous: Pattern, *, norm_floor: float = 0.0) -> tuple[float | None, int]:
    if len(current.values) != len(MASKING_LABELS) or len(previous.values) != len(MASKING_LABELS):
        raise ValueError("an event pattern needs eight masking values")
    if current.N is not None and previous.N is not None:
        return cosine_similarity(current.vector, previous.vector, norm_floor=norm_floor), len(current.vector)
    return cosine_similarity(current.values, previous.values, norm_floor=norm_floor), len(current.values)


def _compare_event(current: HistoryEvent, prior: HistoryEvent, settings: Settings) -> Comparison:
    patent = settings.patent_example
    delta_days = (current.event.start_time - prior.event.start_time).total_seconds() / 86400
    H_m = prior.pattern.H
    similarity, dimension = compare_patterns(
        current.pattern, prior.pattern, norm_floor=settings.engineering_demo_default.cosine_norm_floor
    )
    weight = exp(-delta_days / patent.history_tau_p)
    indicator = int(H_m >= patent.history_mask_tau_H)
    contribution = weight * indicator * max(0.0, similarity if similarity is not None else 0.0)
    return Comparison(
        prior.event.event_id, dimension, delta_days, similarity, H_m, indicator,
        weight, contribution, "available" if similarity is not None else "zero_norm",
    )


def cross_event_recurrence(
    current: HistoryEvent, history: Sequence[HistoryEvent], settings: Settings
) -> CrossEventResult:
    event = current.event
    patent, demo = settings.patent_example, settings.engineering_demo_default
    earlier = [
        prior for prior in history
        if prior.event.platform == event.platform
        and prior.event.victim_id == event.victim_id
        and prior.event.synthetic == event.synthetic
        and prior.event.start_time < event.start_time
        and prior.event.end_time <= event.start_time
    ]
    limit = timedelta(days=demo.second_history_window_days)
    recent = tuple(prior.event.event_id for prior in earlier if event.start_time - prior.event.start_time <= limit)
    F = event_recurrence(len(recent), demo.history_n0)
    if current.pattern is None:
        return CrossEventResult(None, "current_pattern_unavailable", F, len(recent), recent, 0, (), None, None)

    comparable = [
        prior for prior in earlier if prior.pattern is not None
        and prior.pattern.reply_time < event.start_time
        and prior.pattern.source == current.pattern.source
    ]
    pattern_count = len(comparable) + 1
    if pattern_count < patent.min_events_for_cross_event:
        return CrossEventResult(0.0, "insufficient_history", F, len(recent), recent, pattern_count, (), None, None)

    comparisons = tuple(_compare_event(current, prior, settings) for prior in comparable)
    numerator = fsum(item.weighted_contribution for item in comparisons)
    denominator = fsum(item.decay_weight for item in comparisons) + demo.epsilon
    status = "partial_similarity" if any(item.Sim is None for item in comparisons) else "available"
    return CrossEventResult(numerator / denominator, status, F, len(recent), recent,
                            pattern_count, comparisons, numerator, denominator)
