"""Serialize typed calculations for the CLI, Streamlit and SQLite."""

from dataclasses import fields, is_dataclass
from datetime import datetime
import json

from pydantic import BaseModel


def _json_record(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def encode_record(value, *, indent=None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent, default=_json_record)


def render_report(report: dict) -> str:
    return encode_record(report, indent=2) + "\n"
