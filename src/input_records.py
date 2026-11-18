"""Load JSON/CSV messages and the separately labeled review or feature inputs."""

import csv
from io import StringIO
import json
from pathlib import Path

from pydantic import TypeAdapter

from .model_runtime import ROOT
from .schemas import AttackReview, Message, ReplyFeatureTests

DEMO_MESSAGES = ROOT / "data/synthetic_messages.json"
DEMO_FEATURES = ROOT / "data/synthetic_reference_features.json"
_MESSAGES = TypeAdapter(list[Message])
_FEATURES = TypeAdapter(list[ReplyFeatureTests])
_REVIEWS = TypeAdapter(list[AttackReview])
_LIST_FIELDS = {"mention_ids", "target_user_ids"}
_OPTIONAL_FIELDS = {
    "reply_to_message_id", "quoted_message_id", "audio_path", "video_path", "media_start", "media_end",
}


def parse_messages(content: bytes, filename: str) -> list[Message]:
    text = content.decode("utf-8-sig")
    suffix = Path(filename).suffix.casefold()
    if suffix == ".json":
        return _MESSAGES.validate_json(text)
    if suffix != ".csv":
        raise ValueError("消息文件需为 JSON 或 CSV")
    reader = csv.DictReader(StringIO(text), strict=True)
    headings = reader.fieldnames or []
    required = {name for name, field in Message.model_fields.items() if field.is_required()}
    if not required.issubset(headings) or len(headings) != len(set(headings)):
        raise ValueError(f"CSV 标题缺字段或有重复字段，必需字段为 {', '.join(sorted(required))}")
    if set(headings) - Message.model_fields.keys():
        raise ValueError("CSV 包含无法识别的列")
    result = []
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV 第 {reader.line_num} 行的列数不匹配")
            try:
                data = dict(row)
                for field in _LIST_FIELDS & data.keys():
                    data[field] = json.loads(data[field]) if data[field] else []
                for field in _OPTIONAL_FIELDS & data.keys():
                    data[field] = data[field] or None
                result.append(Message.model_validate(data))
            except ValueError as error:
                raise ValueError(f"CSV 第 {reader.line_num} 行无效：{error}") from error
    except csv.Error as error:
        raise ValueError(f"CSV 第 {reader.line_num} 行格式有误：{error}") from error
    return result


def parse_features(content: bytes) -> list[ReplyFeatureTests]:
    return _FEATURES.validate_json(content.decode("utf-8-sig"))


def parse_reviews(content: bytes) -> list[AttackReview]:
    return _REVIEWS.validate_json(content.decode("utf-8-sig"))
