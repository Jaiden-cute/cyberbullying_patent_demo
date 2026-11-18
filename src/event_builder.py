"""S1 rule-based attack findings, review overrides, and topic/time aggregation."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

from .config import EngineeringDefaults
from .math_utils import demo_lexical_cosine
from .schemas import AttackFinding, AttackReview, EventType, Message, MessageKey, TableScore

DISPLAY_ZONE = timezone(timedelta(hours=8))

# Patterns and table scores are the Phase A engineering rule baseline.
ATTACK_RULES: tuple[tuple[EventType, tuple[str, ...], TableScore, TableScore], ...] = (
    ("辱骂", ("笨蛋", "废物", "蠢货"), 1, 0.5),
    ("嘲讽", ("什么都不会", "连这都不会", "真丢人", "都不会"), 0.5, 0.5),
    ("恶意玩梗", ("笑话素材", "做成表情包嘲笑"), 0.5, 0.5),
    ("群体排斥", ("别参加小组", "别来我们组", "踢出群"), 0.5, 0.5),
    ("隐私传播", ("把你的住址发到群里", "公开你的电话号码"), 0.5, 1),
    ("威胁", ("我要打你", "要杀了你", "打死你"), 1, 1),
    ("关系孤立", ("别跟他说话", "别和她说话", "谁都别理你"), 0.5, 0.5),
)


@dataclass(frozen=True)
class Attack:
    message: Message
    finding: AttackFinding
    source: str
    matched_phrases: tuple[str, ...] = ()
    matched_topic_message_key: MessageKey | None = None
    topic_similarity: float | None = None

    @property
    def initiator_id(self) -> str:
        return self.finding.initiator_id or self.message.sender_id


@dataclass(frozen=True)
class Event:
    event_id: str
    platform: str
    conversation_id: str
    victim_id: str
    initiators: tuple[str, ...]
    types: tuple[EventType, ...]
    start_time: datetime
    end_time: datetime
    message_ids: tuple[str, ...]
    content: str
    summary: str
    attacks: tuple[Attack, ...]
    source: str

    @property
    def scope(self) -> tuple[str, str]:
        return self.platform, self.conversation_id

    @property
    def synthetic(self) -> bool:
        return all(attack.message.consent_status == "synthetic" for attack in self.attacks)


@dataclass(frozen=True)
class BuildResult:
    events: tuple[Event, ...]
    unresolved_targets: tuple[MessageKey, ...]


def index_messages(messages: Iterable[Message]) -> dict[MessageKey, Message]:
    indexed: dict[MessageKey, Message] = {}
    for message in messages:
        if message.key in indexed:
            raise ValueError(f"duplicate scoped message ID {message.key}")
        indexed[message.key] = message
    return indexed


def _resolve_targets(
    message: Message, by_key: dict[MessageKey, Message]
) -> tuple[tuple[str, ...], TableScore]:
    explicit = tuple(dict.fromkeys(
        target for target in (message.target_user_ids or message.mention_ids)
        if target != message.sender_id
    ))
    if explicit:
        return explicit, 1
    contextual = []
    for linked_id in (message.reply_to_message_id, message.quoted_message_id):
        if linked_id is None:
            continue
        linked = by_key.get((*message.scope, linked_id))
        if linked and linked.utc_time <= message.utc_time and linked.sender_id != message.sender_id:
            contextual.append(linked.sender_id)
    return tuple(dict.fromkeys(contextual)), 0.5 if contextual else 0


def format_event_summary(
    when: str, initiators: tuple[str, ...], victim: str, types: tuple[str, ...], content: str
) -> str:
    return f"{when}，{'、'.join(initiators)}针对{victim}实施了{'、'.join(types)}行为，事件主要内容为：{content}。"


def _make_event(attacks: list[Attack]) -> Event:
    first = attacks[0].message
    victim = attacks[0].finding.victim_id
    starts, ends = first.utc_time, attacks[-1].message.utc_time
    initiators = tuple(dict.fromkeys(attack.initiator_id for attack in attacks))
    types = tuple(dict.fromkeys(kind for attack in attacks for kind in attack.finding.types))
    message_ids = tuple(attack.message.message_id for attack in attacks)
    identity = (*first.scope, victim, sorted(message_ids))
    event_id = "evt-" + sha256(json.dumps(identity).encode("ascii")).hexdigest()
    when = starts.astimezone(DISPLAY_ZONE).isoformat()
    if ends != starts:
        when += " 至 " + ends.astimezone(DISPLAY_ZONE).isoformat()
    content = "；".join(attack.message.text.strip() for attack in attacks)
    summary = format_event_summary(when, initiators, victim, types, content)
    sources = {attack.source for attack in attacks}
    source = attacks[0].source if len(sources) == 1 else "mixed_review_and_rule_baseline"
    return Event(
        event_id, first.platform, first.conversation_id, victim, initiators, types,
        starts, ends, message_ids, content, summary, tuple(attacks), source,
    )


def build_events(
    messages: Iterable[Message], settings: EngineeringDefaults, reviews: Iterable[AttackReview] = ()
) -> BuildResult:
    by_key = index_messages(messages)
    reviewed: dict[MessageKey, AttackReview] = {}
    for review in reviews:
        key = review.message.key
        if key not in by_key or key in reviewed:
            raise ValueError(f"review requires a unique, existing scoped message ID: {key}")
        reviewed[key] = review

    groups: dict[MessageKey, list[list[Attack]]] = defaultdict(list)
    unresolved = []
    event_window = timedelta(minutes=settings.first_event_window_minutes)
    for message in sorted(by_key.values(), key=lambda record: (record.utc_time, record.key)):
        review = reviewed.get(message.key)
        if review is not None:
            attacks = [Attack(message, finding, "manual_review") for finding in review.findings]
        else:
            matched = [rule for rule in ATTACK_RULES if any(term in message.text for term in rule[1])]
            if not matched:
                continue
            victims, g = _resolve_targets(message, by_key)
            if not victims:
                unresolved.append(message.key)
                continue
            types = tuple(rule[0] for rule in matched)
            phrases = tuple(term for rule in matched for term in rule[1] if term in message.text)
            a = max(rule[2] for rule in matched)
            h = max(rule[3] for rule in matched)
            attacks = [
                Attack(message, AttackFinding(victim_id=victim, types=types, g=g, a=a, h=h),
                       "rule_baseline", phrases)
                for victim in victims
            ]

        for attack in attacks:
            related_groups = groups[(*message.scope, attack.finding.victim_id)]
            best_group = None
            best_anchor = None
            best_score = -1.0
            for group in reversed(related_groups):
                if message.utc_time - group[0].message.utc_time > event_window:
                    break
                for anchor in group:
                    score = demo_lexical_cosine(message.text, anchor.message.text)
                    if score is not None and score >= settings.event_topic_similarity_threshold and score > best_score:
                        best_group, best_anchor, best_score = group, anchor, score
            if best_group is None:
                related_groups.append([attack])
            else:
                best_group.append(replace(
                    attack, matched_topic_message_key=best_anchor.message.key, topic_similarity=best_score
                ))

    events = [_make_event(group) for scoped in groups.values() for group in scoped]
    events.sort(key=lambda event: (event.start_time, event.event_id))
    return BuildResult(tuple(events), tuple(unresolved))
