"""S2 candidate qualification, relation evidence, and the association formula."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from math import exp, fsum, isfinite
from typing import Literal

from .config import AssociationWeights, Settings
from .event_builder import Event
from .llm_adapter import RelationEvidence
from .math_utils import demo_lexical_cosine, sigmoid
from .schemas import Message

Binary = Literal[0, 1]
REFERENCE_CUES = ("刚才", "那句", "这句话", "他们", "这件事", "那件事")


@dataclass(frozen=True)
class AssociationResult:
    message: Message
    L: Binary
    Q: Binary
    S: float | None
    T: float
    D: Binary
    G: Binary
    delta_t_minutes: float
    logit: float | None
    A: float | None
    tau_A: float
    effective: bool
    reason: str
    feature_sources: dict[str, str]
    matched_reference_ids: tuple[str, ...]
    matched_mentions: tuple[str, ...]
    matched_reference_cues: tuple[str, ...]
    demo_topic_cosine: float | None
    llm_status: str
    llm_reasons: tuple[str, ...]


def interaction_flags(event: Event, reply: Message) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if reply.scope != event.scope:
        return (), ()
    referenced = (reply.reply_to_message_id, reply.quoted_message_id)
    links = tuple(dict.fromkeys(key for key in referenced if key in event.message_ids))
    mentions = tuple(dict.fromkeys(key for key in reply.mention_ids if key in event.initiators))
    return links, mentions


def time_proximity(delta_minutes: float, tau_minutes: float) -> float:
    if not isfinite(delta_minutes) or delta_minutes < 0:
        raise ValueError("delta_minutes must be finite and nonnegative")
    if not isfinite(tau_minutes) or tau_minutes <= 0:
        raise ValueError("tau_minutes must be finite and positive")
    return exp(-delta_minutes / tau_minutes)


def association_logit(
    weights: AssociationWeights, *, L: Binary, Q: Binary, S: float, T: float, D: Binary, G: Binary
) -> float:
    if any(flag not in (0, 1) for flag in (L, Q, D, G)) or not -1 <= S <= 1 or not 0 <= T <= 1:
        raise ValueError("invalid association feature value")
    terms = (value * weight for value, weight in zip((L, Q, S, T, D, G), weights.slopes, strict=True))
    return fsum((weights.b0, *terms))


def is_effective(A: float | None, threshold: float) -> bool:
    return A is not None and A >= threshold


def classify_candidate(event: Event, reply: Message, settings: Settings) -> str:
    if reply.scope != event.scope or reply.sender_id != event.victim_id:
        return "different_scope_or_sender"
    if reply.utc_time < event.start_time:
        return "before_event"
    if reply.utc_time < event.end_time:
        return "during_event"
    window = timedelta(minutes=settings.engineering_demo_default.candidate_window_minutes)
    if reply.utc_time - event.end_time > window:
        return "after_candidate_window"
    return "eligible"


def evaluate_reply(
    event: Event, reply: Message, settings: Settings, *,
    semantic_similarity: Callable[[str, str, float], float | None] | None = None,
    require_model: bool = False, relation: RelationEvidence | None = None,
) -> AssociationResult:
    qualification = classify_candidate(event, reply, settings)
    if qualification != "eligible":
        raise ValueError(f"cannot score an ineligible reply: {qualification}")
    patent, demo = settings.patent_example, settings.engineering_demo_default
    links, mentions = interaction_flags(event, reply)
    L = 1 if links else 0
    Q = 1 if mentions else 0
    if semantic_similarity is not None:
        S = semantic_similarity(event.summary, reply.text, demo.cosine_norm_floor) if reply.text.strip() else None
        semantic_source = "sentence_bert"
    elif require_model:
        S, semantic_source = None, "sentence_bert_unavailable"
    else:
        S, semantic_source = demo_lexical_cosine(event.summary, reply.text), "character_ngram_cosine_baseline"
    delta_minutes = (reply.utc_time - event.end_time).total_seconds() / 60
    T = time_proximity(delta_minutes, patent.time_tau_minutes)

    topic_score, cues = None, ()
    if relation is None:
        topic_score = demo_lexical_cosine(event.content, reply.text)
        shares_topic = topic_score is not None and topic_score >= demo.demo_topic_continuity_threshold
        cues = tuple(cue for cue in REFERENCE_CUES if cue in reply.text)
        D = 1 if cues and (L or Q or shares_topic) else 0
        G = 1 if shares_topic or (L and D) else 0
        relation = RelationEvidence(D, G, "heuristic_fallback", "demo_rule")
    D, G = relation.coreference, relation.topic_continuity
    source_d, source_g = relation.field_sources or (relation.source, relation.source)
    logit = A = None
    if S is not None:
        logit = association_logit(demo.association_weights, L=L, Q=Q, S=S, T=T, D=D, G=G)
        A = sigmoid(logit)
    effective = is_effective(A, patent.association_tau_A)
    if A is None:
        reason = "semantic_representation_unavailable"
    else:
        reason = "at_or_above_threshold" if effective else "below_threshold"
    sources = {
        "L": "scoped_reply_or_quote_metadata",
        "Q": "scoped_mention_metadata",
        "S": semantic_source,
        "T": "utc_elapsed_minutes",
        "D": source_d,
        "G": source_g,
    }
    return AssociationResult(
        reply, L, Q, S, T, D, G, delta_minutes, logit, A, patent.association_tau_A,
        effective, reason, sources, links, mentions, cues, topic_score, relation.status, relation.reasons,
    )
