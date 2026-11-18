"""Adapt the user demo's pasted or imported records to the existing message schema."""

import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import Message

SCENE_FILES = {
    "A / B 第 1 次 · 单次事件与无关话题": "synthetic_scene_A_B01.json",
    "B 第 2 次 · 无回复 ID 的回应": "synthetic_scene_B02.json",
    "B 第 3 次 · 跨事件历史比较": "synthetic_scene_B03.json",
    "C · 严重威胁与安全优先": "synthetic_scene_C.json",
}

ALIASES = {
    "sender": "sender_id",
    "reply_to_id": "reply_to_message_id",
    "mentions": "mention_ids",
    "target": "target_user_ids",
}
LIST_FIELDS = {"mention_ids", "target_user_ids"}
OPTIONAL_TEXT_FIELDS = {
    "reply_to_message_id", "quoted_message_id", "audio_path", "video_path", "media_start", "media_end",
}
FIELD_LABELS = {
    "sender_id": "发言人",
    "timestamp": "时间",
    "message_id": "消息 ID",
    "text": "消息内容",
    "consent_status": "数据来源或授权状态",
}


def _objects(text: str, file_format: str) -> list[dict]:
    if file_format == "json" or (file_format == "paste" and text.lstrip().startswith("[")):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON 第 {error.lineno} 行格式有误，请检查逗号、引号和括号。") from error
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("JSON 应为消息数组，每条消息用一个对象表示。")
        return rows
    if file_format == "csv":
        reader = csv.DictReader(StringIO(text), strict=True)
        try:
            headings = reader.fieldnames or []
            if not headings or len(headings) != len(set(headings)):
                raise ValueError("CSV 需要表头，列名须唯一。")
            rows = list(reader)
        except csv.Error as error:
            raise ValueError(f"CSV 第 {reader.line_num} 行格式有误。") from error
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV 某行的列数与表头不同，请检查分隔符和引号。")
        return rows
    rows = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 2)
        if len(fields) != 3:
            raise ValueError(f"第 {index} 行需要「时间 | 发言人 | 消息内容」。")
        rows.append(dict(zip(("timestamp", "sender_id", "text"), fields, strict=True)))
    return rows


def _normalize(row: dict, *, consent: str, conversation: str) -> tuple[dict, bool]:
    item = {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}
    for alias, field in ALIASES.items():
        if alias in item:
            value = item.pop(alias)
            if field in item and item[field] != value:
                raise ValueError(f"{field} 与 {alias} 的填写内容不一致。")
            item[field] = value
    for field in LIST_FIELDS & item.keys():
        value = item[field]
        if isinstance(value, str):
            if value.startswith("["):
                try:
                    item[field] = json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{field} 的 JSON 数组格式有误。") from error
            else:
                item[field] = [value] if value else []
    for field in OPTIONAL_TEXT_FIELDS & item.keys():
        if item[field] == "":
            item[field] = None
    if not isinstance(item.get("text"), str) or not item["text"]:
        raise ValueError("消息内容不能为空。")
    if not item.get("consent_status"):
        item["consent_status"] = "synthetic" if item.get("source_type") == "synthetic" else consent
    synthetic = item["consent_status"] == "synthetic"
    if not item.get("source_type"):
        item["source_type"] = "synthetic" if synthetic else "authorized_chat"
    if not item.get("platform"):
        item["platform"] = "local-synthetic-chat" if synthetic else "local-chat"
    if not item.get("conversation_id"):
        item["conversation_id"] = conversation
    generated = item.get("message_id") in (None, "")
    if generated:
        item["message_id"] = "pending"
    return item, generated


def parse_chat_input(content: bytes, filename: str | None = None, *, consent: str, conversation: str) -> list[Message]:
    """Support the full schema plus lightweight JSON, CSV and line-based chat."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("请使用 UTF-8 编码的聊天文件。") from error
    file_format = Path(filename).suffix.lower().lstrip(".") if filename else "paste"
    if file_format not in {"json", "csv", "paste"}:
        raise ValueError("请选择 JSON 或 CSV 聊天文件。")
    rows = _objects(text, file_format) if text.strip() else []
    if not rows:
        raise ValueError("请至少提供一条聊天消息。")
    messages = []
    seen = set()
    for index, row in enumerate(rows, 1):
        try:
            item, generated = _normalize(row, consent=consent, conversation=conversation)
            message = Message.model_validate(item)
        except ValidationError as error:
            field = error.errors(include_input=False)[0]["loc"]
            if field and field[0] == "timestamp":
                raise ValueError(f"第 {index} 条消息的时间无效，请填写带时区的时间，例如 2026-10-01T19:00:00+08:00。") from error
            label = FIELD_LABELS.get(field[0], field[0]) if field else "字段或来源标记"
            raise ValueError(f"第 {index} 条消息的「{label}」缺失或格式不正确。") from error
        except ValueError as error:
            raise ValueError(f"第 {index} 条消息有误。{error}") from error
        if generated:
            identity = (*message.scope, message.utc_time.isoformat(), message.sender_id, message.text)
            digest = sha256(json.dumps(identity).encode()).hexdigest()
            message = message.model_copy(update={"message_id": f"auto-{digest}"})
        if message.key in seen:
            raise ValueError(f"第 {index} 条消息在同一平台和会话中重复使用了消息 ID。")
        seen.add(message.key)
        messages.append(message)
    return messages
