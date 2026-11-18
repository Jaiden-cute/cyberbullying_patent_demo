from datetime import timedelta
from math import exp, sqrt

from pydantic import ValidationError
import pytest

from src.config import Settings, SeverityWeights
from src.cross_event import HistoryEvent, Pattern, compare_patterns, cross_event_recurrence, event_recurrence
from src.event_builder import build_events
from src.event_masking import event_masking_score, representative_reply
from src.masking_classifier import evaluate_demo_masking, joint_representation, linear_probabilities, masking_score
from src.math_utils import cosine_similarity, sigmoid
from src.reply_linker import association_logit, is_effective, time_proximity
from src.risk_support import LEVELS, risk_level_index, risk_score
from src.severity import repetition_score, severity_score


@pytest.mark.parametrize(("count", "expected"), [(1, 0), (2, 0.5), (3, 1), (4, 1)])
def test_table_one_repetition(count, expected):
    assert repetition_score(count) == expected


def test_weighted_severity_and_validation():
    weights = SeverityWeights(targeting=0.1, aggression=0.2, repetition=0.3, harm=0.4)
    assert severity_score(1, 0.5, 0, 1, weights) == pytest.approx(0.6)
    assert severity_score(1, 1, 1, 1, weights) == 1.0
    with pytest.raises(ValueError, match="Table 1"):
        severity_score(0.25, 1, 1, 1, weights)
    with pytest.raises(ValidationError, match="summing to 1"):
        SeverityWeights(targeting=0.2, aggression=0.2, repetition=0.3, harm=0.4)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("engineering_demo_default", "first_event_window_minutes", 50000),
        ("engineering_demo_default", "risk_weights", [0.25, 0.25, 0.25, 0.5]),
        ("engineering_demo_default", "masking_weights", [0.1] * 8),
        ("engineering_demo_default", "kappa_text", 1),
        ("engineering_demo_default", "epsilon", 0),
        ("patent_example", "fusion_weights", {"alpha": 0, "beta": 0, "gamma": 0}),
        ("patent_example", "risk_thresholds", [0.5, 0.5, 0.75]),
        ("patent_example", "baseline_min_valid_observations", 2),
    ],
)
def test_invalid_configuration(settings, section, key, value):
    config = settings.model_dump()
    config[section][key] = value
    with pytest.raises(ValidationError):
        Settings.model_validate(config)


def test_signed_cosine_and_zero_vector():
    assert cosine_similarity((1, 0), (-1, 0)) == -1
    assert cosine_similarity((1, 0), (1, 1)) == pytest.approx(1 / sqrt(2))
    assert cosine_similarity((0, 0), (1, 1)) is None
    assert cosine_similarity((1e-16, 0), (1, 1), norm_floor=1e-12) is None
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity((1,), (1, 1))


def test_time_proximity_minutes_and_stable_sigmoid():
    assert time_proximity(10, 10) == pytest.approx(0.36787944117144233)
    assert time_proximity(0, 10) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        time_proximity(-1, 10)
    assert sigmoid(-1000) == 0
    assert sigmoid(1000) == 1


def test_association_formula_all_six_features(settings):
    weights = settings.engineering_demo_default.association_weights
    logit = association_logit(weights, L=1, Q=1, S=0.25, T=0.5, D=1, G=1)
    assert logit == pytest.approx(2.4)
    assert sigmoid(logit) == pytest.approx(0.9168273035060777)
    assert association_logit(weights, L=0, Q=0, S=-0.5, T=0, D=0, G=0) == pytest.approx(-2.6)
    assert is_effective(0.7, 0.7)
    assert not is_effective(0.699999, 0.7)


def test_linear_head_and_weighted_masking(settings):
    joint = joint_representation((2, -1), (1, 3))
    assert joint == (2, -1, 1, 3, 1, 4)
    # Hand-selected W and b define a formula test, independent of model artifacts.
    p = linear_probabilities(joint, [(0.5, 0, 0, 0, 0, 0)] * 8, (0, -1, *([0] * 6)))
    assert p == pytest.approx((0.7310585786300049, 0.5, *([0.7310585786300049] * 6)))
    assert sum(p) > 1
    assert masking_score(p, settings.engineering_demo_default.masking_weights) == pytest.approx(0.7021762563012543)
    with pytest.raises(ValueError, match="eight masking"):
        masking_score([0.3] * 7, settings.engineering_demo_default.masking_weights)
    with pytest.raises(ValueError, match="eight weight rows"):
        linear_probabilities(joint, [(1,)] * 8, (0,) * 8)


def test_demo_masking_has_no_model_probabilities(settings):
    result = evaluate_demo_masking("没事，他们就是开个玩笑😂", settings.engineering_demo_default.masking_weights)
    assert result.p is None
    assert result.non_model_score == (1, 1, 1, 0, 0, 1, 0, 0)
    assert result.M == 0.5
    assert result.inference_source == "rule_baseline"


def test_multimodal_and_text_h_from_requirements(settings):
    weights = settings.patent_example.fusion_weights
    assert event_masking_score(0.8, 0.6, 0.4, weights, 0.5).H == pytest.approx(0.3744)
    text = event_masking_score(0.8, 0.6, None, weights, 0.5)
    assert text.branch == "text"
    assert text.H == pytest.approx(0.24)
    zero = event_masking_score(0.8, 0.6, 0.0, weights, 0.5)
    assert zero.branch == "multimodal"


def test_representative_tie_is_deterministic(settings, messages):
    replies = [message for message in messages if message.message_id in ("oct01-reply-1", "oct01-reply-2")]
    weights = settings.patent_example.fusion_weights
    H = event_masking_score(0.8, 0.6, None, weights, 0.5)
    assert representative_reply((message, H) for message in reversed(replies)) == replies[0]
    assert representative_reply([]) is None


def _pattern(event, values, H, N=None):
    return Pattern(tuple(values), N, H, "arithmetic_fixture", event.end_time + timedelta(minutes=1))


def test_p_f_u_independent_values_from_requirements(settings, messages):
    old, recent, current = build_events(messages, settings.engineering_demo_default).events
    old_pattern = _pattern(old, [0.3, sqrt(0.91), *([0] * 6)], 0.2)
    new_pattern = _pattern(recent, [0.8, 0.6, *([0] * 6)], 0.4)
    current_pattern = _pattern(current, [1, *([0] * 7)], 0.3744)
    history = [HistoryEvent(old, old_pattern), HistoryEvent(recent, new_pattern)]
    result = cross_event_recurrence(HistoryEvent(current, current_pattern), history, settings)
    assert [c.delta_days for c in result.comparisons] == [20, 10]
    assert [c.indicator for c in result.comparisons] == [0, 1]
    assert [c.Sim for c in result.comparisons] == pytest.approx([0.3, 0.8])
    assert result.denominator == pytest.approx(exp(-20 / 30) + exp(-10 / 30) + 1e-8)
    assert result.P == pytest.approx(0.4660561613806184)
    assert result.F == pytest.approx(0.486582880967408)
    U = risk_score(0.7, 0.3744, result.P, result.F, settings.engineering_demo_default.risk_weights)
    assert U == pytest.approx(0.5067597605870067)
    assert LEVELS[risk_level_index(U, settings.patent_example.risk_thresholds)] == "较高风险"
    assert event_recurrence(0, 3) == 0


def test_pattern_alignment_and_zero_norm(settings, messages):
    event = build_events(messages, settings.engineering_demo_default).events[0]
    text = _pattern(event, [1] + [0] * 7, 0.3)
    multimodal = _pattern(event, [1] + [0] * 7, 0.5, N=1)
    zero = _pattern(event, [0] * 8, 0)
    assert compare_patterns(text, multimodal) == (1.0, 8)
    similarity, width = compare_patterns(multimodal, multimodal)
    assert similarity == pytest.approx(1)
    assert width == 9
    assert compare_patterns(text, zero) == (None, 8)


@pytest.mark.parametrize(("U", "level"), [(0.249999, 0), (0.25, 1), (0.5, 2), (0.75, 3)])
def test_risk_threshold_inclusivity(settings, U, level):
    assert risk_level_index(U, settings.patent_example.risk_thresholds) == level
