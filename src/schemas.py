"""Validated input records and the scoped identifiers used throughout the pipeline."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
UnitValue = Annotated[Finite, Field(ge=0, le=1)]
TableScore = Literal[0, 0.5, 1]
EventType = Literal["辱骂", "嘲讽", "恶意玩梗", "群体排斥", "隐私传播", "威胁", "关系孤立"]
MessageKey = tuple[str, str, str]

MASKING_LABELS = ("否认", "弱化", "合理化", "自嘲", "迎合", "反向积极Emoji", "回避", "话题转移")


def _parse_iso_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("expected an ISO 8601 timestamp") from error
    raise ValueError("expected an ISO 8601 timestamp")


Timestamp = Annotated[AwareDatetime, BeforeValidator(_parse_iso_datetime)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MessageRef(InputModel):
    platform: Identifier
    conversation_id: Identifier
    message_id: Identifier

    @property
    def scope(self) -> tuple[str, str]:
        return self.platform, self.conversation_id

    @property
    def key(self) -> MessageKey:
        return *self.scope, self.message_id


class Message(MessageRef):
    sender_id: Identifier
    timestamp: Timestamp
    text: str
    reply_to_message_id: Identifier | None = None
    quoted_message_id: Identifier | None = None
    mention_ids: tuple[Identifier, ...] = ()
    target_user_ids: tuple[Identifier, ...] = ()
    audio_path: Identifier | None = None
    video_path: Identifier | None = None
    media_start: Timestamp | None = None
    media_end: Timestamp | None = None
    source_type: Identifier
    consent_status: Literal["synthetic", "authorized"]

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if (self.source_type == "synthetic") != (self.consent_status == "synthetic"):
            raise ValueError("source_type and consent_status must agree on synthetic input")
        if (self.media_start is None) != (self.media_end is None):
            raise ValueError("media_start and media_end must be supplied together")
        if self.media_start is not None and self.media_start >= self.media_end:
            raise ValueError("media_start must precede media_end")
        return self

    @property
    def utc_time(self) -> datetime:
        return self.timestamp.astimezone(UTC)


class AttackFinding(InputModel):
    victim_id: Identifier
    initiator_id: Identifier | None = None
    types: Annotated[tuple[EventType, ...], Field(min_length=1)]
    g: TableScore
    a: TableScore
    h: TableScore


class AttackReview(InputModel):
    message: MessageRef
    # A reviewed message with no findings is an explicit rejection.
    findings: tuple[AttackFinding, ...]

    @model_validator(mode="after")
    def distinct_victims(self) -> Self:
        if len({finding.victim_id for finding in self.findings}) != len(self.findings):
            raise ValueError("a reviewed message can have only one finding per victim")
        return self


class Observation(InputModel):
    window_start: Timestamp
    window_end: Timestamp
    value: Finite | None
    unavailable_reason: Identifier | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.window_start >= self.window_end:
            raise ValueError("observation window_start must precede window_end")
        if (self.value is None) != (self.unavailable_reason is not None):
            raise ValueError("an unavailable observation requires a reason and a null value")
        return self


class FeatureTest(InputModel):
    feature_name: Identifier
    current: Observation
    references: tuple[Observation, ...] = ()

    @model_validator(mode="after")
    def distinct_reference_windows(self) -> Self:
        ordered = sorted(self.references, key=lambda item: item.window_start)
        if any(left.window_end > right.window_start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("reference observation windows must be distinct and non-overlapping")
        return self


class ReplyObservations(InputModel):
    reply: MessageRef
    subject_id: Identifier
    features: tuple[FeatureTest, ...]

    @model_validator(mode="after")
    def distinct_feature_names(self) -> Self:
        if len({feature.feature_name for feature in self.features}) != len(self.features):
            raise ValueError("feature names must be unique per reply")
        return self


class ReplyFeatureTests(ReplyObservations):
    source: Literal["structured_test_features"]


class ReplyMediaFeatures(ReplyObservations):
    source: Literal["extracted_local_media"]
    extractor_id: Identifier
