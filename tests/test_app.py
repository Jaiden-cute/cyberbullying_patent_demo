import json

from streamlit.testing.v1 import AppTest

from src.history_store import HistoryStore
from src.model_runtime import ROOT


def _widget(group, label):
    return next(item for item in group if item.label == label)


def test_streamlit_load_run_history_and_report(monkeypatch, tmp_path):
    database = tmp_path / "ui.sqlite3"
    monkeypatch.setenv("PATENT_HISTORY_DB", str(database))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    _widget(app.radio, "推理方式").set_value("规则演示").run()
    _widget(app.checkbox, "加入结构化模拟非语言特征").set_value(True).run()
    _widget(app.button, "运行 S1–S7").click().run()
    assert not app.exception
    assert not app.error
    report = json.loads(app.session_state["analysis_json"])
    assert report["run_mode"] == "DEMO"
    assert len(report["events"]) == 3
    assert report["events"][0]["candidates"][0]["S3_nonverbal"]["status"] == "available"
    assert report["events"][2]["S6"]["human_review_required"]
    assert all(item["S7"]["persisted"] for item in report["events"])
    assert len(app.tabs) == 8
    _widget(app.sidebar.radio, "页面").set_value("历史记录").run()
    assert not app.exception
    _widget(app.text_area, "人工跟进记录").set_value("验证 SQLite 跟进.")
    _widget(app.button, "保存跟进").click().run()
    assert not app.exception
    assert HistoryStore(database).list_events()[0]["follow_up"] == "验证 SQLite 跟进."


def test_streamlit_review_updates_roles_and_event_type(monkeypatch, tmp_path, messages):
    monkeypatch.setenv("PATENT_HISTORY_DB", str(tmp_path / "review.sqlite3"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    _widget(app.selectbox, "选择要核对的消息").set_value(messages[0].key).run()
    _widget(app.text_input, "核对后的发起者 ID").set_value("核对发起者")
    _widget(app.multiselect, "事件类型").select("嘲讽")
    _widget(app.button, "保存人工核对").click().run()
    assert not app.exception and not app.error
    _widget(app.radio, "推理方式").set_value("规则演示").run()
    _widget(app.button, "运行 S1–S7").click().run()
    assert not app.exception and not app.error
    report = json.loads(app.session_state["analysis_json"])
    assert report["events"][0]["event"]["initiators"][0] == "核对发起者"
    assert report["events"][0]["event_detector"] == "mixed_review_and_rule_baseline"
