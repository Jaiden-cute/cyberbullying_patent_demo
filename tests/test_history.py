import sqlite3

import pytest

from src.history_store import HistoryChanged, HistoryStore
from src.pipeline import run_pipeline


def test_separate_submissions_use_recent_persisted_events_and_deduplicate(settings, messages, features, tmp_path):
    db = tmp_path / "history.sqlite3"
    store = HistoryStore(db)
    subsets = [[message for message in messages if message.message_id.startswith(prefix)]
               for prefix in ("oct01", "oct11", "oct21")]
    store.analyze(subsets[0], settings, features=features)
    store.analyze(subsets[1], settings)
    later = store.analyze(subsets[2], settings)["events"][0]
    assert later["S5"].n_i == 2
    assert later["S5"].P == pytest.approx(0.373361, abs=1e-6)
    assert later["S7"]["persisted"]
    event_id = later["event"].event_id
    store.save_follow_up(event_id, "演示记录：教师复核仍待完成。")
    repeated = store.analyze(subsets[2], settings)["events"][0]
    assert repeated["S5"] == later["S5"]
    assert len(store.list_events(platform="synthetic-chat", victim_id="同学丙")) == 3
    assert store.load_event(event_id)["S7"]["follow_up"] == "演示记录：教师复核仍待完成。"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert connection.execute("SELECT COUNT(*) FROM candidate_replies").fetchone()[0] == 5
        assert connection.execute("SELECT COUNT(*) FROM feature_records WHERE name='voice_rms'").fetchone()[0] == 4


def test_synthetic_clear_preserves_other_consent_status(settings, messages, tmp_path):
    store = HistoryStore(tmp_path / "both.sqlite3")
    synthetic = [messages[0]]
    separate = messages[-2].model_copy(update={"consent_status": "authorized", "source_type": "local_test",
                                                "platform": "local"})
    store.analyze([*synthetic, separate], settings)
    assert store.clear_synthetic() == 1
    left = store.list_events()
    assert len(left) == 1 and not left[0]["synthetic"]


def test_stale_computation_rejected_without_partial_update(settings, messages, tmp_path):
    store = HistoryStore(tmp_path / "race.sqlite3")
    old = store.snapshot()
    first = run_pipeline(messages[:1], settings)
    second = run_pipeline(messages[-2:], settings)
    store.save(first, expected_revision=old.revision)
    with pytest.raises(HistoryChanged):
        store.save(second, expected_revision=old.revision)
    assert len(store.list_events()) == 1
    assert store.load_event(second["events"][0]["event"].event_id) is None
