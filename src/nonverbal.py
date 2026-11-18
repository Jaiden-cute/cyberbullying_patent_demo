"""S3 nonverbal formulas over timestamped structured test observations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import expm1, fsum
from statistics import fmean, pstdev

from .config import Settings
from .event_builder import Event
from .schemas import Message, Observation, ReplyFeatureTests, ReplyMediaFeatures


@dataclass(frozen=True)
class ReferenceTrace:
    observation: Observation
    included: bool
    status: str


@dataclass(frozen=True)
class FeatureTrace:
    feature_name: str
    modality: str
    unit: str
    feature_value: float | None
    window_start: datetime | None
    window_end: datetime | None
    reference_values: tuple[float, ...]
    reference_observations: tuple[ReferenceTrace, ...]
    mu: float | None
    sigma: float | None
    d: int
    q: int
    eta: float
    z: float | None
    u: float | None
    status: str
    current_unavailable_reason: str | None


@dataclass(frozen=True)
class NonverbalResult:
    N: float | None
    status: str
    inference_source: str
    weighted_deviation_sum: float
    available_weight_sum: float
    features: tuple[FeatureTrace, ...]
    extractor_id: str | None = None


def _reference_status(observation: Observation, event_start: datetime, earliest: datetime) -> str:
    if observation.value is None:
        return "extraction_unavailable"
    if observation.window_end > event_start:
        return "not_before_event"
    if observation.window_start < earliest:
        return "outside_local_baseline"
    return "valid"


def evaluate_nonverbal(
    event: Event, reply: Message, bundle: ReplyFeatureTests | ReplyMediaFeatures | None, settings: Settings
) -> NonverbalResult:
    patent, demo = settings.patent_example, settings.engineering_demo_default
    if bundle is not None and (bundle.reply.key != reply.key or bundle.subject_id != event.victim_id):
        raise ValueError("structured feature input must refer to the same reply and affected subject")
    supplied = {entry.feature_name: entry for entry in bundle.features} if bundle else {}
    unknown_names = supplied.keys() - demo.nonverbal_features.keys()
    if unknown_names:
        raise ValueError(f"unknown nonverbal features: {sorted(unknown_names)}")

    traces = []
    earliest = event.start_time - timedelta(minutes=demo.baseline_window_minutes)
    for name, definition in demo.nonverbal_features.items():
        input_feature = supplied.get(name)
        current = input_feature.current if input_feature is not None else None
        references = []
        values = []
        for reference in input_feature.references if input_feature is not None else ():
            status = _reference_status(reference, event.start_time, earliest)
            references.append(ReferenceTrace(reference, status == "valid", status))
            if status == "valid":
                values.append(reference.value)

        value = current.value if current is not None else None
        mu = sigma = z = u = None
        q = 0
        if len(values) >= patent.baseline_min_valid_observations:
            mu, sigma = fmean(values), pstdev(values)
        if current is None:
            status = "missing_current_observation"
        elif not current.window_start <= reply.utc_time <= current.window_end:
            status = "not_synchronized_to_reply"
        elif value is None:
            status = "extraction_unavailable"
        elif sigma is None:
            status = "insufficient_reference_observations"
        elif sigma < demo.baseline_std_floor:
            status = "reference_std_too_small"
        else:
            status, q = "available", 1
            z = definition.direction * (value - mu) / (sigma + demo.epsilon)
            u = max(0.0, z)
        traces.append(FeatureTrace(
            name, definition.modality, definition.unit, value,
            current.window_start if current else None, current.window_end if current else None,
            tuple(values), tuple(references), mu, sigma, definition.direction, q,
            definition.weight, z, u, status, current.unavailable_reason if current else None,
        ))

    numerator = fsum(row.eta * row.u for row in traces if row.q and row.eta)
    weight_sum = fsum(row.eta for row in traces if row.q)
    N = -expm1(-numerator / (weight_sum + demo.epsilon)) if weight_sum else None
    return NonverbalResult(
        N, "available" if N is not None else "unavailable",
        bundle.source if bundle else "no_nonverbal_input", numerator, weight_sum, tuple(traces),
        bundle.extractor_id if isinstance(bundle, ReplyMediaFeatures) else None,
    )
