"""S7 local SQLite history, deduplicated events, and transactional follow-up records."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3

from .cross_event import HistoryEvent, Pattern
from .event_builder import Attack, Event
from .pipeline import run_pipeline
from .reporting import encode_record
from .schemas import AttackFinding, Message

DDL = """
CREATE TABLE IF NOT EXISTS history_revision (
    id INTEGER PRIMARY KEY CHECK (id = 1), revision INTEGER NOT NULL
);
INSERT OR IGNORE INTO history_revision VALUES (1, 0);
CREATE TABLE IF NOT EXISTS messages (
    platform TEXT NOT NULL, conversation_id TEXT NOT NULL, message_id TEXT NOT NULL,
    synthetic INTEGER NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (platform, conversation_id, message_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY, platform TEXT NOT NULL, victim_id TEXT NOT NULL,
    started_utc TEXT NOT NULL, summary TEXT NOT NULL, synthetic INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_subject_time ON events(platform, victim_id, started_utc);
CREATE TABLE IF NOT EXISTS event_results (
    event_id TEXT PRIMARY KEY REFERENCES events ON DELETE CASCADE,
    processed_utc TEXT NOT NULL, pattern TEXT, report TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_replies (
    event_id TEXT REFERENCES events ON DELETE CASCADE,
    reply_id TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (event_id, reply_id)
);
CREATE TABLE IF NOT EXISTS feature_records (
    event_id TEXT NOT NULL, reply_id TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (event_id, reply_id, name),
    FOREIGN KEY (event_id, reply_id) REFERENCES candidate_replies ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS support_records (
    event_id TEXT PRIMARY KEY REFERENCES events ON DELETE CASCADE,
    payload TEXT NOT NULL, follow_up TEXT, follow_up_utc TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    source TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (source, version)
);
"""


class HistoryChanged(ValueError):
    pass


@dataclass(frozen=True)
class Snapshot:
    revision: int
    events: tuple[HistoryEvent, ...]


def _restore_event(payload: str) -> Event:
    stored = json.loads(payload)
    stored["attacks"] = tuple(Attack(
        Message.model_validate(row["message"]), AttackFinding.model_validate(row["finding"]), row["source"],
        tuple(row["matched_phrases"]),
        tuple(row["matched_topic_message_key"]) if row["matched_topic_message_key"] else None,
        row["topic_similarity"],
    ) for row in stored["attacks"])
    for key in ("start_time", "end_time"):
        stored[key] = datetime.fromisoformat(stored[key])
    for key in ("types", "initiators", "message_ids"):
        stored[key] = tuple(stored[key])
    return Event(**stored)


def _restore_pattern(payload: str | None) -> Pattern | None:
    if payload is None:
        return None
    row = json.loads(payload)
    return Pattern(tuple(row["values"]), row["N"], row["H"], row["source"], datetime.fromisoformat(row["reply_time"]))


def _stored_pattern(row: dict) -> str | None:
    result = row["S4"]
    if result["H"] is None:
        return None
    return encode_record(Pattern(
        tuple(result["pattern_vector"][:8]), result["pattern_n"], result["H"],
        result["pattern_source"], result["representative_reply_time"],
    ))


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript(DDL)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self) -> Snapshot:
        with self._connection() as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM history_revision WHERE id = 1").fetchone()[0]
            prior = db.execute("""
                SELECT e.payload, r.pattern FROM events e JOIN event_results r USING(event_id)
                ORDER BY e.started_utc, e.event_id
            """).fetchall()
        return Snapshot(revision, tuple(HistoryEvent(_restore_event(row["payload"]), _restore_pattern(row["pattern"]))
                                        for row in prior))

    def analyze(self, messages, settings, **options) -> dict:
        snapshot = self.snapshot()
        report = run_pipeline(messages, settings, previous_history=snapshot.events, **options)
        current_ids = {row["event"].event_id for row in report["events"]}
        saved_ids = {item.event.event_id for item in snapshot.events} - current_ids
        for row in report["events"]:
            recurrence = row["S5"]
            used_ids = dict.fromkeys((*recurrence.recent_event_ids, *(item.event_id for item in recurrence.comparisons)))
            row["S7"]["history_read_event_ids"] = [event_id for event_id in used_ids if event_id in saved_ids]
            row["S7"]["same_run_history_event_ids"] = [event_id for event_id in used_ids if event_id not in saved_ids]
        processed = self.save(report, expected_revision=snapshot.revision)
        for row in report["events"]:
            row["S7"].update(status="sqlite_saved", persisted=True, processed_utc=processed)
        return report

    def save(self, report: dict, *, expected_revision: int) -> str:
        processed = datetime.now(UTC).isoformat()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT revision FROM history_revision WHERE id = 1").fetchone()[0]
            if current != expected_revision:
                raise HistoryChanged("历史记录已被另一运行更新，请重跑本次分析")
            db.executemany("""
                INSERT INTO messages VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(platform, conversation_id, message_id) DO UPDATE SET
                synthetic=excluded.synthetic, payload=excluded.payload
            """, [(*message.key, message.consent_status == "synthetic", encode_record(message))
                  for message in report["inputs"]["messages"]])
            for row in report["events"]:
                self._save_event(db, row, report, processed)
            db.executemany("""
                INSERT INTO model_versions VALUES (?, ?, ?)
                ON CONFLICT(source, version) DO UPDATE SET payload=excluded.payload
            """, [
                (name, info.get("model_version") or info.get("revision") or info["model_id"], encode_record(info))
                for name, info in report["model_sources"].items()
            ])
            db.execute("UPDATE history_revision SET revision = revision + 1 WHERE id = 1")
        return processed

    @staticmethod
    def _save_event(db: sqlite3.Connection, row: dict, report: dict, processed: str) -> None:
        event = row["event"]
        event_id = event.event_id
        db.execute("""
            INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET started_utc=excluded.started_utc, summary=excluded.summary,
            synthetic=excluded.synthetic, payload=excluded.payload
        """, (event_id, event.platform, event.victim_id, event.start_time.isoformat(),
              event.summary, event.synthetic, encode_record(event)))
        record = {
            **row,
            "S7": {**row["S7"], "status": "sqlite_saved", "persisted": True, "processed_utc": processed},
            "settings": report["settings"], "run_mode": report["run_mode"],
            "model_sources": report["model_sources"], "model_status": report["model_status"],
        }
        db.execute("""
            INSERT INTO event_results VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET processed_utc=excluded.processed_utc,
            pattern=excluded.pattern, report=excluded.report
        """, (event_id, processed, _stored_pattern(row), encode_record(record)))
        db.execute("DELETE FROM candidate_replies WHERE event_id = ?", (event_id,))
        for candidate in row["candidates"]:
            reply_id = candidate.S2.message.message_id
            db.execute("INSERT INTO candidate_replies VALUES (?, ?, ?)",
                       (event_id, reply_id, encode_record(candidate)))
            if candidate.S3_nonverbal:
                db.executemany("INSERT INTO feature_records VALUES (?, ?, ?, ?)", [
                    (event_id, reply_id, feature.feature_name, encode_record(feature))
                    for feature in candidate.S3_nonverbal.features
                ])
        db.execute("""
            INSERT INTO support_records (event_id, payload) VALUES (?, ?)
            ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload
        """, (event_id, encode_record(row["S6"])))

    def list_events(self, *, platform: str | None = None, victim_id: str | None = None) -> list[dict]:
        with self._connection() as db:
            rows = db.execute("""
                SELECT e.event_id, e.platform, e.victim_id, e.started_utc, e.synthetic, e.summary,
                       r.processed_utc, json_extract(s.payload, '$.risk_level') AS risk_level,
                       json_extract(s.payload, '$.support_source') AS support_source, s.follow_up
                FROM events e JOIN event_results r USING(event_id) JOIN support_records s USING(event_id)
                WHERE (? IS NULL OR e.platform = ?) AND (? IS NULL OR e.victim_id = ?)
                ORDER BY e.started_utc DESC, e.event_id
            """, (platform, platform, victim_id, victim_id)).fetchall()
        return [dict(row) for row in rows]

    def load_event(self, event_id: str) -> dict | None:
        with self._connection() as db:
            found = db.execute("""
                SELECT r.report, s.follow_up, s.follow_up_utc
                FROM event_results r JOIN support_records s USING(event_id) WHERE r.event_id = ?
            """, (event_id,)).fetchone()
        if found is None:
            return None
        record = json.loads(found["report"])
        record["S7"].update(follow_up=found["follow_up"], follow_up_utc=found["follow_up_utc"])
        return record

    def save_follow_up(self, event_id: str, note: str) -> None:
        cleaned = note.strip()
        if not cleaned or len(cleaned) > 2000:
            raise ValueError("后续支持记录需填写 1 至 2000 字")
        with self._connection() as db:
            result = db.execute("UPDATE support_records SET follow_up=?, follow_up_utc=? WHERE event_id=?",
                                (cleaned, datetime.now(UTC).isoformat(), event_id))
            if not result.rowcount:
                raise ValueError("历史事件已不存在")

    def clear_synthetic(self) -> int:
        with self._connection() as db:
            count = db.execute("DELETE FROM events WHERE synthetic=1").rowcount
            db.execute("DELETE FROM messages WHERE synthetic=1")
            db.execute("UPDATE history_revision SET revision = revision + 1 WHERE id = 1")
        return count
