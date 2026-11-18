"""Replay the patent calculations and expose their inputs, provenance and status."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .config import Settings
from .cross_event import HistoryEvent, Pattern, cross_event_recurrence
from .event_builder import Event, build_events
from .event_masking import HResult, event_masking_score, representative_reply
from .llm_adapter import SOURCE as QWEN_SOURCE
from .masking_classifier import MaskingResult
from .media_interface import LocalMedia, MediaFeatureExtractor, resolve_media
from .nonverbal import NonverbalResult, evaluate_nonverbal
from .reply_linker import AssociationResult, classify_candidate, evaluate_reply
from .risk_support import evaluate_risk
from .runtime import Runtime
from .schemas import AttackReview, Message, MessageKey, ReplyFeatureTests, ReplyMediaFeatures
from .severity import evaluate_severity
from .support import generate_support


@dataclass(frozen=True)
class CandidateResult:
    S2: AssociationResult
    S3_masking: MaskingResult | None = None
    S3_nonverbal: NonverbalResult | None = None
    S4: HResult | None = None


def _score_candidate(
    event: Event, link: AssociationResult, feature_input: ReplyFeatureTests | ReplyMediaFeatures | None, settings: Settings,
    runtime: Runtime,
) -> CandidateResult:
    if not link.effective:
        return CandidateResult(link)
    reply = link.message
    demo, patent = settings.engineering_demo_default, settings.patent_example
    masking = runtime.masking(event.summary, reply.text, demo.masking_weights)
    nonverbal = evaluate_nonverbal(event, reply, feature_input, settings)
    if masking.M is None:
        return CandidateResult(link, masking, nonverbal)
    H = event_masking_score(link.A, masking.M, nonverbal.N, patent.fusion_weights, demo.kappa_text)
    return CandidateResult(link, masking, nonverbal, H)


def _index_features(
    bundles: Sequence[ReplyFeatureTests], messages: dict[MessageKey, Message], settings: Settings
) -> dict[MessageKey, ReplyFeatureTests]:
    indexed: dict[MessageKey, ReplyFeatureTests] = {}
    for bundle in bundles:
        key = bundle.reply.key
        if key in indexed or key not in messages:
            raise ValueError(f"structured features require a unique, existing scoped reply ID: {key}")
        if messages[key].sender_id != bundle.subject_id:
            raise ValueError(f"structured features subject must match reply sender: {key}")
        for feature in bundle.features:
            if feature.feature_name not in settings.engineering_demo_default.nonverbal_features:
                raise ValueError(f"unknown nonverbal feature: {feature.feature_name}")
        indexed[key] = bundle
    return indexed


def run_pipeline(
    messages: Sequence[Message],
    settings: Settings,
    *,
    features: Sequence[ReplyFeatureTests] = (),
    reviews: Sequence[AttackReview] = (),
    previous_history: Sequence[HistoryEvent] = (),
    runtime: Runtime | None = None,
    media_extractor: MediaFeatureExtractor | None = None,
    allowed_media_dir: Path | None = None,
    affected_subject: tuple[str, str] | None = None,
) -> dict:
    runtime = runtime or Runtime()
    built = build_events(messages, settings.engineering_demo_default, reviews)
    ordered = sorted(messages, key=lambda message: (message.utc_time, message.key))
    feature_index = _index_features(features, {message.key: message for message in ordered}, settings)
    by_sender: dict[MessageKey, list[Message]] = defaultdict(list)
    for message in ordered:
        by_sender[(*message.scope, message.sender_id)].append(message)

    events = [event for event in built.events
              if affected_subject is None or (event.platform, event.victim_id) == affected_subject]
    current_ids = {event.event_id for event in events}
    history = [item for item in previous_history if item.event.event_id not in current_ids]
    local_media: dict[MessageKey, LocalMedia] = {}
    invalid_media = []
    for message in ordered:
        if message.audio_path is None and message.video_path is None:
            continue
        try:
            local_media[message.key] = resolve_media(message, allowed_media_dir)
        except (ValueError, OSError) as error:
            invalid_media.append({"message_key": message.key, "reason": str(error)})
    results = []
    used_feature_keys = set()
    matches_llm = []
    support_sources = []
    trained_output = False
    for event in events:
        severity = evaluate_severity(event, tuple(record.event for record in history), settings)
        candidates = []
        during_event = []
        safety_messages = {attack.message.key: attack.message for attack in event.attacks}
        for reply in by_sender[(*event.scope, event.victim_id)]:
            qualification = classify_candidate(event, reply, settings)
            if qualification == "during_event":
                during_event.append(reply.key)
            if qualification in ("during_event", "eligible"):
                safety_messages[reply.key] = reply
            if qualification == "eligible":
                link = evaluate_reply(
                    event, reply, settings, semantic_similarity=runtime.similarity if runtime.encoder else None,
                    require_model=runtime.strict, relation=runtime.relation(event, reply),
                )
                source = feature_index.get(reply.key)
                if link.effective and source is None and media_extractor is not None and reply.key in local_media:
                    baseline_start = event.start_time - timedelta(minutes=settings.engineering_demo_default.baseline_window_minutes)
                    references = [
                        (sample, local_media[sample.key])
                        for sample in by_sender[(*event.scope, event.victim_id)]
                        if sample.key in local_media and baseline_start <= sample.utc_time < event.start_time
                    ]
                    source = media_extractor.extract_reply(event, reply, local_media[reply.key], references)
                    if source is not None and not isinstance(source, ReplyMediaFeatures):
                        raise ValueError("the real extractor must return ReplyMediaFeatures")
                item = _score_candidate(event, link, source, settings, runtime)
                candidates.append(item)
                matches_llm.append(item.S2.llm_status == "available")
                if item.S3_masking is not None and item.S3_masking.p is not None:
                    trained_output = True
                if item.S3_nonverbal is not None and reply.key in feature_index:
                    used_feature_keys.add(reply.key)

        representative = representative_reply(
            (item.S2.message, item.S4) for item in candidates if item.S4 is not None
        )
        pattern = None
        if representative is not None:
            selected = next(item for item in candidates if item.S2.message.key == representative.key)
            classification = selected.S3_masking
            pattern_source = classification.model_version or classification.inference_source
            pattern = Pattern(
                classification.p if classification.p is not None else classification.non_model_score,
                selected.S3_nonverbal.N, selected.S4.H, pattern_source, representative.utc_time,
            )
        H = pattern.H if pattern is not None else None
        current = HistoryEvent(event, pattern)
        recurrence = cross_event_recurrence(current, history, settings)
        risk = evaluate_risk(severity.B, H, recurrence.P, recurrence.F, safety_messages.values(), settings)
        support_context = {
            "event_summary": event.summary,
            "effective_replies": [item.S2.message.text for item in candidates if item.S2.effective],
            "masking_result": (
                {"labels": classification.labels, "values": pattern.values, "source": pattern.source}
                if pattern is not None else None
            ),
        }
        risk = generate_support(risk, runtime.qwen, support_context, unavailable_reason=runtime.qwen_status)
        if risk.risk_level is not None:
            support_sources.append(risk.support_source)
        masking_status = "calculated" if pattern is not None else "no_effective_reply"
        if pattern is None and any(item.S2.effective for item in candidates):
            masking_status = "masking_model_unavailable"
        results.append({
            "event": event,
            "S1": severity,
            "event_detector": event.source,
            "event_quality": "reviewed" if event.source == "manual_review" else "rule_baseline_needs_review",
            "candidates": candidates,
            "excluded_during_event_message_keys": during_event,
            "S4": {
                "H": H,
                "status": masking_status,
                "representative_reply_id": representative.message_id if representative is not None else None,
                "representative_reply_key": representative.key if representative is not None else None,
                "pattern_vector": pattern.vector if pattern is not None else None,
                "pattern_source": pattern.source if pattern is not None else None,
                "pattern_n": pattern.N if pattern is not None else None,
                "representative_reply_time": pattern.reply_time if pattern is not None else None,
            },
            "S5": recurrence,
            "S6": risk,
            "S7": {"status": "within_run_history_only", "persisted": False},
        })
        history.append(current)

    real_dg = any(matches_llm)
    qwen_support = QWEN_SOURCE in support_sources
    if trained_output and real_dg and all(matches_llm) and support_sources and all(
        source == QWEN_SOURCE for source in support_sources
    ):
        mode = "MODEL"
    elif runtime.encoder or real_dg or qwen_support:
        mode = "HYBRID"
    elif runtime.strict:
        mode = "UNAVAILABLE"
    else:
        mode = "DEMO"
    return {
        "report_schema_version": 2,
        "run_mode": mode,
        "requested_runtime": runtime.requested,
        "demo_mode": mode in ("DEMO", "HYBRID") or trained_output,
        "inference_source": {
            "event_detection": "rule_baseline_with_optional_manual_review",
            "similarity": "sentence_bert" if runtime.encoder else "sentence_bert_unavailable" if runtime.strict
                          else "character_ngram_cosine_baseline",
            "D_and_G": sorted({item.S2.feature_sources[key] for row in results for item in row["candidates"]
                               for key in ("D", "G")}),
            "masking": sorted({item.S3_masking.inference_source for row in results for item in row["candidates"]
                               if item.S3_masking}),
            "nonverbal": sorted({item.S3_nonverbal.inference_source for row in results for item in row["candidates"]
                                 if item.S3_nonverbal}),
            "support": sorted(set(support_sources)),
        },
        "model_status": {
            "sentence_bert": runtime.encoder_status,
            "trained_mask_head": runtime.head_status,
            "qwen_2_5_7b_instruct": runtime.qwen_status,
            "audio_video_extraction": "external_extractor_attached" if media_extractor else "interface_only",
        },
        "model_sources": runtime.model_sources,
        "input_sources": {
            "messages": sorted({message.source_type for message in messages}),
            "consent_status": sorted({message.consent_status for message in messages}),
            "nonverbal": sorted({item.source for item in features} | {
                item.S3_nonverbal.inference_source for row in results for item in row["candidates"]
                if item.S3_nonverbal and item.S3_nonverbal.extractor_id}),
            "event_reviews": "manual_review" if reviews else None,
        },
        "input_quality": {
            "unresolved_attack_target_message_keys": built.unresolved_targets,
            "unused_test_feature_reply_keys": sorted(feature_index.keys() - used_feature_keys),
            "media_path_message_keys": [message.key for message in ordered if message.audio_path or message.video_path],
            "invalid_media_paths": invalid_media,
        },
        "limitations": [
            "事件识别使用规则基线和可选人工核对。",
            "八类模型的训练数据为 synthetic_training_demo，仅用于核对原型流程。",
            "结构化非语言测试观测用于公式验算。报告按候选回应记录实际来源。",
        ],
        "settings": settings,
        "inputs": {"messages": ordered, "test_features": features, "attack_reviews": reviews},
        "affected_subject": affected_subject,
        "events": results,
    }
