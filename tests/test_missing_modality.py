from datetime import timedelta
from math import sqrt

import pytest

from src.event_builder import build_events
from src.nonverbal import evaluate_nonverbal


@pytest.fixture
def first_event_and_reply(settings, messages):
    event = build_events(messages, settings.engineering_demo_default).events[0]
    reply = next(message for message in messages if message.message_id == "oct01-reply-1")
    return event, reply


def test_reference_mean_population_std_and_n_with_missing_video(settings, features, first_event_and_reply):
    result = evaluate_nonverbal(*first_event_and_reply, features[0], settings)
    records = {row.feature_name: row for row in result.features}
    audio = records["voice_rms"]
    video = records["head_down_ratio"]
    assert result.inference_source == "structured_test_features"
    assert (audio.q, video.q, result.available_weight_sum) == (1, 0, 1.0)
    assert video.current_unavailable_reason == "simulated_no_face"
    assert audio.reference_values == (50, 55, 60)
    assert audio.mu == 55
    assert audio.sigma == pytest.approx(sqrt(50 / 3))
    assert audio.z == pytest.approx(2.449489736783178)
    assert result.N == pytest.approx(0.9136623677067808)


def test_missing_n_is_null_and_valid_zero_remains_available(settings, features, first_event_and_reply):
    empty = evaluate_nonverbal(*first_event_and_reply, None, settings)
    assert (empty.N, empty.status) == (None, "unavailable")
    bundle = features[0]
    feature = bundle.features[0]
    zero = feature.model_copy(update={"current": feature.current.model_copy(update={"value": 60})})
    result = evaluate_nonverbal(*first_event_and_reply, bundle.model_copy(update={"features": (zero,)}), settings)
    assert (result.N, result.status, result.features[0].q) == (0, "available", 1)


@pytest.mark.parametrize(
    ("change", "expected"),
    [("two_refs", "insufficient_reference_observations"),
     ("flat", "reference_std_too_small"),
     ("unsynced", "not_synchronized_to_reply"),
     ("late_reference", "insufficient_reference_observations")],
)
def test_nonverbal_quality_gates(settings, features, first_event_and_reply, change, expected):
    event, reply = first_event_and_reply
    bundle = features[0]
    feature = bundle.features[0]
    refs = feature.references
    if change == "two_refs":
        feature = feature.model_copy(update={"references": refs[:2]})
    elif change == "flat":
        feature = feature.model_copy(update={"references": tuple(ref.model_copy(update={"value": 55}) for ref in refs)})
    elif change == "unsynced":
        current = feature.current.model_copy(update={
            "window_start": reply.timestamp + timedelta(minutes=1),
            "window_end": reply.timestamp + timedelta(minutes=2),
        })
        feature = feature.model_copy(update={"current": current})
    elif change == "late_reference":
        ref = refs[2].model_copy(update={
            "window_start": event.start_time + timedelta(seconds=1),
            "window_end": event.start_time + timedelta(seconds=2),
        })
        feature = feature.model_copy(update={"references": (*refs[:2], ref)})
    result = evaluate_nonverbal(event, reply, bundle.model_copy(update={"features": (feature,)}), settings)
    row = result.features[0]
    assert (row.status, row.q, row.z, row.u, result.N) == (expected, 0, None, None, None)


def test_feature_bundle_identity_mismatch(settings, features, first_event_and_reply):
    wrong_subject = features[0].model_copy(update={"subject_id": "另一个人"})
    with pytest.raises(ValueError, match="same reply and affected subject"):
        evaluate_nonverbal(*first_event_and_reply, wrong_subject, settings)
