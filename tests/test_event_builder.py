from datetime import timedelta

from pydantic import ValidationError
import pytest

from src.config import Settings
from src.event_builder import build_events
from src.reply_linker import evaluate_reply, interaction_flags
from src.schemas import AttackReview, Message
from src.severity import evaluate_severity


def test_aggregation_and_exact_summary_template(settings, messages):
    events = build_events(reversed(messages), settings.engineering_demo_default).events
    assert len(events) == 3
    event = events[0]
    assert event.message_ids == ("oct01-attack-1", "oct01-attack-2")
    assert event.initiators == ("同学甲", "同学乙")
    assert event.types == ("辱骂", "嘲讽", "群体排斥")
    assert event.attacks[1].matched_topic_message_key == messages[0].key
    assert event.attacks[1].topic_similarity >= settings.engineering_demo_default.event_topic_similarity_threshold
    assert event.summary == (
        "2026-10-01T19:00:00+08:00 至 2026-10-01T19:01:00+08:00，"
        "同学甲、同学乙针对同学丙实施了辱骂、嘲讽、群体排斥行为，"
        "事件主要内容为：你真是个笨蛋，连小组练习都不会；"
        "你真是个笨蛋，连小组练习都不会，别参加小组。"
    )
    severity = evaluate_severity(event, (), settings)
    assert (severity.g, severity.a, severity.r, severity.h, severity.B) == (1, 1, 0.5, 0.5, 0.75)


def test_scoped_reply_quote_and_mentions(settings, messages):
    event = build_events(messages, settings.engineering_demo_default).events[0]
    explicit, quoted = messages[3:5]
    assert interaction_flags(event, explicit) == (("oct01-attack-1",), ("同学甲",))
    assert interaction_flags(event, quoted) == (("oct01-attack-2",), ("同学乙",))
    other_conversation = explicit.model_copy(update={"conversation_id": "other"})
    other_platform = explicit.model_copy(update={"platform": "other"})
    assert interaction_flags(event, other_conversation) == ((), ())
    assert interaction_flags(event, other_platform) == ((), ())
    with pytest.raises(ValueError, match="ineligible"):
        evaluate_reply(event, other_conversation, settings)


def test_contextual_target_and_unresolved_attack(settings, messages):
    target = messages[3].model_copy(update={"timestamp": messages[0].timestamp - timedelta(minutes=1)})
    attack = messages[0].model_copy(update={"target_user_ids": (), "reply_to_message_id": target.message_id})
    event = build_events([target, attack], settings.engineering_demo_default).events[0]
    assert event.victim_id == target.sender_id
    assert event.attacks[0].finding.g == 0.5
    unresolved = build_events([attack], settings.engineering_demo_default)
    assert unresolved.events == ()
    assert unresolved.unresolved_targets == (attack.key,)


def test_duplicate_id_in_scope_is_rejected(settings, messages):
    with pytest.raises(ValueError, match="duplicate scoped message ID"):
        build_events([messages[0], messages[0]], settings.engineering_demo_default)
    other = messages[0].model_copy(update={"conversation_id": "other"})
    assert len(build_events([messages[0], other], settings.engineering_demo_default).events) == 2


def test_timestamp_requires_timezone_and_elapsed_uses_utc(settings, messages):
    reply = messages[3]
    with pytest.raises(ValidationError, match="timezone"):
        Message.model_validate({**reply.model_dump(), "timestamp": "2026-10-01T19:02:00"})
    with pytest.raises(ValidationError, match="ISO 8601"):
        Message.model_validate({**reply.model_dump(), "timestamp": 1790852520})
    event = build_events(messages, settings.engineering_demo_default).events[0]
    offset_reply = reply.model_copy(update={"timestamp": reply.utc_time})
    assert evaluate_reply(event, offset_reply, settings).delta_t_minutes == 1


def test_human_review_replaces_rule_finding(settings, messages):
    attack = messages[0]
    review = AttackReview.model_validate({
        "message": {"platform": attack.platform, "conversation_id": attack.conversation_id, "message_id": attack.message_id},
        "findings": [{"victim_id": "人工核对对象", "types": ["嘲讽"], "g": 0.5, "a": 0.5, "h": 0.5}],
    })
    event = build_events([attack], settings.engineering_demo_default, [review]).events[0]
    assert (event.victim_id, event.types, event.source) == ("人工核对对象", ("嘲讽",), "manual_review")
    rejected = review.model_copy(update={"findings": ()})
    assert not build_events([attack], settings.engineering_demo_default, [rejected]).events


def test_similar_attack_repetition_requires_all_four_conditions(settings, messages):
    raw_config = settings.model_dump()
    raw_config["engineering_demo_default"]["repetition_basis"] = "similar_attacks"
    demo = Settings.model_validate(raw_config)
    attack = messages[0]
    current = attack.model_copy(update={"message_id": "current", "timestamp": attack.timestamp + timedelta(days=10)})
    within = attack.model_copy(update={"message_id": "same-event", "timestamp": current.timestamp + timedelta(minutes=1)})
    wrong_actor = attack.model_copy(update={"message_id": "wrong-actor", "sender_id": "别的发起者"})
    wrong_victim = attack.model_copy(update={"message_id": "wrong-victim", "target_user_ids": ("另一个人",)})
    old = attack.model_copy(update={"message_id": "old", "timestamp": current.timestamp - timedelta(days=31)})
    wrong_topic = attack.model_copy(update={"message_id": "wrong-topic", "text": "把你的住址发到群里"})
    # Separate conversations keep the historical evidence as distinct attack units.
    historical = [attack, wrong_actor, wrong_victim, old, wrong_topic]
    historical = [item.model_copy(update={"conversation_id": item.message_id}) for item in historical]
    events = build_events([current, within, *historical], demo.engineering_demo_default).events
    result = evaluate_severity(events[-1], events[:-1], demo)
    assert [key[-1] for key in result.repetition_message_keys] == ["current", attack.message_id]
    assert result.repeat_count == 2
    assert result.r == 0.5
