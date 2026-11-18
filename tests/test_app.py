import json

from streamlit.testing.v1 import AppTest

from src.demo_inputs import SCENE_FILES, parse_chat_input
from src.history_store import HistoryStore
from src.model_runtime import ROOT


def _widget(group, label):
    return next(item for item in group if item.label == label)


def test_streamlit_load_run_history_and_report(monkeypatch, tmp_path):
    database = tmp_path / "ui.sqlite3"
    monkeypatch.setenv("PATENT_HISTORY_DB", str(database))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert _widget(app.sidebar.radio, "页面").value == "用户交互"
    assert _widget(app.sidebar.radio, "页面").options == ["用户交互", "历史记录"]
    assert not app.caption
    assert "同学丙" in _widget(app.selectbox, "当前受影响对象").options
    runs = []
    for scenario in SCENE_FILES:
        _widget(app.selectbox, "演示场景").set_value(scenario).run()
        _widget(app.button, "开始智能分析").click().run()
        assert not app.exception and not app.error
        assert not app.tabs and not app.json
        assert [item.value for item in app.caption] == ["预设支持回复"]
        assert any(item.value == "下一步怎么做" for item in app.subheader)
        report = json.loads(app.session_state["analysis_json"])
        assert report["run_mode"] == "DEMO" and len(report["events"]) == 1
        row = report["events"][0]
        assert row["S7"]["persisted"] and row["S6"]["empathy_reply"]
        assert row["S6"]["support_source"] == "safe_template"
        runs.append(row)
    first, second, third, threat = runs
    assert [(item["S2"]["message"]["message_id"], item["S2"]["effective"])
            for item in first["candidates"]][-1] == ("a-unrelated", False)
    assert first["candidates"][0]["S3_nonverbal"]["N"] is None
    assert first["candidates"][0]["S4"]["branch"] == "text"
    assert second["candidates"][0]["S2"]["L"] == 0 and second["candidates"][0]["S2"]["effective"]
    assert first["S5"]["P_status"] == second["S5"]["P_status"] == "insufficient_history"
    assert third["S7"]["history_read_event_ids"] == [row["event"]["event_id"] for row in runs[:2]]
    assert third["S7"]["same_run_history_event_ids"] == []
    assert third["S5"]["P_status"] == "available" and third["S5"]["P"] > 0
    assert all(item["dimension"] == 8 for item in third["S5"]["comparisons"])
    assert threat["S6"]["U"] is None and threat["S6"]["human_review_required"]
    assert any("外部通知未发送" in item.value for item in app.warning)
    assert any("现在就找成年人陪同" in item.value for item in app.markdown)

    reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    _widget(reopened.sidebar.radio, "页面").set_value("历史记录").run()
    assert not reopened.exception
    assert len(HistoryStore(database).list_events()) == 4
    assert any("给你的回应" == element.value for element in reopened.subheader)
    assert not reopened.tabs and not reopened.json


def test_streamlit_paste_and_confirm_target_without_inventing_a_reply_chain(monkeypatch, tmp_path):
    database = tmp_path / "pasted.sqlite3"
    monkeypatch.setenv("PATENT_HISTORY_DB", str(database))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    _widget(app.radio, "消息来源").set_value("粘贴 / 编辑").run()
    editor = _widget(app.text_area, "聊天记录")
    editor.set_value("昨天 | 同学甲 | 你真是个笨蛋").run()
    assert any("时间无效" in element.value for element in app.error)
    pasted = (
        "2026-10-01T19:00:00+08:00 | 同学甲 | 你真是个笨蛋\n"
        "2026-10-01T19:01:00+08:00 | 同学丙 | 没事。"
    )
    _widget(app.text_area, "聊天记录").set_value(pasted).run()
    subject = ("local-synthetic-chat", "同学丙")
    _widget(app.selectbox, "当前受影响对象").set_value(subject).run()
    first_message = parse_chat_input(pasted.encode(), consent="synthetic", conversation="本地会话")[0]
    _widget(app.multiselect, "我确认下列消息指向当前对象").set_value([first_message.key]).run()
    _widget(app.button, "开始智能分析").click().run()
    assert not app.exception and not app.error
    report = json.loads(app.session_state["analysis_json"])
    assert report["affected_subject"] == list(subject)
    row = report["events"][0]
    assert row["candidates"][0]["S2"]["L"] == 0
    assert row["S4"]["status"] == "no_effective_reply"
    assert any("没有找到足够相关" in element.value for element in app.info)


def test_ai_support_separates_empathy_from_actionable_steps():
    row = {"S6": {
        "human_review_required": False,
        "empathy_reply": "这段经历听起来让人不好受。",
        "support_source": "qwen_2_5_7b_instruct",
        "support_strategy": {"steps": [
            {"action": "共情回应", "text": "我理解你现在可能感到很受伤。"},
            {"action": "温和询问", "text": "你能告诉我具体发生了什么吗？"},
        ]},
    }}
    app = AppTest.from_string(f"from app import render_user_support\nrender_user_support({row!r})").run()
    assert not app.exception
    assert [item.value for item in app.caption] == ["AI 生成"]
    text = "\n".join(item.value for item in app.markdown)
    assert "1. 找一个你信任的人" in text and "给对方发一句" in text
    assert "2. 从一条消息说起" in text
    assert "我理解你现在可能感到很受伤" not in text
