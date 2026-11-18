import csv
from datetime import timedelta
from io import StringIO
import json

import pytest

from src.demo_inputs import parse_chat_input
from src.input_records import parse_messages
from src.media_interface import resolve_media
from src.pipeline import run_pipeline


def test_user_chat_import_accepts_compact_json_and_uses_stable_scoped_ids():
    rows = [
        {"timestamp": "2026-10-01T19:00:00+08:00", "sender": "甲", "text": "你好", "message_id": "m1"},
        {"timestamp": "2026-10-01T19:01:00+08:00", "sender": "乙", "text": "收到",
         "reply_to_id": "m1", "mentions": ["甲"]},
    ]
    messages = parse_chat_input(json.dumps(rows).encode(), "chat.json", consent="synthetic", conversation="课堂")
    reordered = parse_chat_input(json.dumps(rows[::-1]).encode(), "chat.json", consent="synthetic", conversation="课堂")
    assert messages[0].message_id == "m1"
    assert messages[1].message_id == reordered[0].message_id
    assert messages[1].reply_to_message_id == "m1" and messages[1].mention_ids == ("甲",)
    assert messages[0].reply_to_message_id is None


def test_user_chat_csv_source_and_duplicate_display_error():
    text = "timestamp,sender,text,message_id,platform,conversation_id,consent_status\n"
    text += "2026-10-01T19:00:00+08:00,甲,你好,m1,local-test,c1,authorized\n"
    text += "2026-10-01T19:01:00+08:00,乙,收到,m1,local-test,c2,authorized\n"
    parsed = parse_chat_input(text.encode(), "chat.csv", consent="synthetic", conversation="未标注")
    assert [message.conversation_id for message in parsed] == ["c1", "c2"]
    assert all(message.consent_status == "authorized" for message in parsed)
    with pytest.raises(ValueError, match="重复使用"):
        parse_chat_input(text.replace(",c2,", ",c1,").encode(), "chat.csv", consent="synthetic", conversation="未标注")


def test_message_csv_preserves_lists_ids_and_timestamp(messages):
    message = messages[3]
    source = StringIO(newline="")
    row = message.model_dump(mode="json")
    writer = csv.DictWriter(source, fieldnames=list(row))
    writer.writeheader()
    for key in ("mention_ids", "target_user_ids"):
        row[key] = json.dumps(row[key], ensure_ascii=False)
    writer.writerow(row)
    assert parse_messages(source.getvalue().encode("utf-8"), "messages.csv") == [message]


def test_media_path_rejects_escape_and_reports_quality(tmp_path, messages, settings):
    root = tmp_path / "allowed"
    root.mkdir()
    message = messages[3].model_copy(update={
        "audio_path": "../elsewhere.wav", "media_start": messages[3].utc_time,
        "media_end": messages[3].utc_time + timedelta(seconds=4),
        "consent_status": "authorized", "source_type": "local_test",
    })
    with pytest.raises(ValueError, match="outside_allowed"):
        resolve_media(message, root)
    report = run_pipeline([messages[0], message], settings, allowed_media_dir=root)
    assert report["input_quality"]["invalid_media_paths"] == [
        {"message_key": message.key, "reason": "media_outside_allowed_directory"},
    ]
    candidate = report["events"][0]["candidates"][0]
    assert candidate.S3_nonverbal.N is None and candidate.S4.branch == "text"


def test_extractor_is_called_only_after_association_succeeds(tmp_path, messages, settings):
    # These files exercise path and gating behavior. The spy returns no observations.
    (tmp_path / "test.wav").write_bytes(b"test path fixture")
    attack = messages[0]
    sample_items = [
        reply.model_copy(update={
            "audio_path": "test.wav", "media_start": reply.utc_time,
            "media_end": reply.utc_time + timedelta(seconds=4),
            "source_type": "local_test", "consent_status": "authorized",
        })
        for reply in (messages[3], messages[5])
    ]
    calls = []

    class ExtractorSpy:
        def extract_reply(self, event, reply, current, references):
            calls.append(reply.message_id)
            return None

    report = run_pipeline(
        [attack, *sample_items], settings, media_extractor=ExtractorSpy(), allowed_media_dir=tmp_path
    )
    assert calls == [sample_items[0].message_id]
    assert report["input_sources"]["nonverbal"] == []
