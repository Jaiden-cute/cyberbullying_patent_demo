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
    assert _widget(app.sidebar.radio, "页面").value == "事件研判"
    assert _widget(app.sidebar.radio, "页面").options == ["事件研判", "历史与跟进"]
    assert [item.value for item in app.caption] == ["学校心理咨询中心"]
    assert "同学丙" in _widget(app.sidebar.selectbox, "受影响学生").options
    runs = []
    for scenario in SCENE_FILES:
        _widget(app.sidebar.selectbox, "演示场景").set_value(scenario).run()
        _widget(app.sidebar.button, "研判本次事件").click().run()
        assert not app.exception and not app.error
        assert not app.tabs and not app.json
        assert [item.value for item in app.main.header] == ["发生了什么"]
        assert [item.value for item in app.main.subheader] == ["为什么值得关注", "接下来怎么跟进"]
        assert not app.main.selectbox
        strategy = _widget(app.main.expander, "查看风险评分、支持策略与沟通话术")
        assert not strategy.proto.expanded
        assert [item.value for item in strategy.caption] == ["预设支持回复"]
        assert not _widget(app.main.expander, "查看聊天原文").proto.expanded
        report = json.loads(app.session_state["analysis_json"])
        assert report["run_mode"] == "DEMO" and len(report["events"]) == 1
        row = report["events"][0]
        assert row["S7"]["persisted"] and row["S6"]["empathy_reply"]
        assert row["S6"]["support_source"] == "safe_template"
        result_text = "\n".join(item.value for item in app.main.markdown)
        assert row["S6"]["risk_level"] in result_text
        if not runs:
            assert "2 条回应与事件有关，1 条未纳入" in result_text
            assert "可能是掩饰感受的线索" in result_text
            assert any("明天几点交数学作业" in item.value for item in app.main.text)
            assert any("没事，刚才小组练习那句话" in item.value for item in app.main.text)
        if len(runs) == 2:
            assert "2 次来自已保存的记录（10月01日、10月11日）" in result_text
            assert "此前的相似表达也纳入了本次研判" in result_text
        if row["S6"]["risk_level"] == "较高风险":
            assert "协助保存聊天记录" in result_text and "协助屏蔽或举报" in result_text
            assert "协助联系可信任的成年人" in result_text
        runs.append(row)
    first, second, third, elevated, threat = runs
    assert [row["S6"]["risk_level"] for row in runs] == ["重点关注", "一般关注", "重点关注", "较高风险", "高风险"]
    assert not elevated["S6"]["safety_override"]
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
    assert "优先确认学生安全" in result_text and "安排人工复核" in result_text
    assert "安全信号" in result_text and "事件风险评分" not in result_text

    _widget(app.sidebar.radio, "页面").set_value("历史与跟进").run()
    event_id = third["event"]["event_id"]
    _widget(app.selectbox, "查看历史事件").set_value(event_id).run()
    note = "已与学生核对群聊记录，约好明天再次沟通。"
    _widget(app.text_area, "已采取的措施与下一步安排").set_value(note)
    _widget(app.button, "保存跟进记录").click().run()
    assert not app.exception and any(item.value == "跟进记录已保存。" for item in app.success)

    reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    _widget(reopened.sidebar.radio, "页面").set_value("历史与跟进").run()
    _widget(reopened.selectbox, "查看历史事件").set_value(event_id).run()
    assert not reopened.exception
    assert len(HistoryStore(database).list_events()) == 5
    assert _widget(reopened.text_area, "已采取的措施与下一步安排").value == note
    assert any("跟进记录" == element.value for element in reopened.subheader)
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
    _widget(app.selectbox, "受影响学生").set_value(subject).run()
    first_message = parse_chat_input(pasted.encode(), consent="synthetic", conversation="本地会话")[0]
    _widget(app.multiselect, "我确认下列消息指向当前对象").set_value([first_message.key]).run()
    _widget(app.button, "研判本次事件").click().run()
    assert not app.exception and not app.error
    report = json.loads(app.session_state["analysis_json"])
    assert report["affected_subject"] == list(subject)
    row = report["events"][0]
    assert row["candidates"][0]["S2"]["L"] == 0
    assert row["S4"]["status"] == "no_effective_reply"
    assert any("建议先补充事发后的对话" in element.value for element in app.info)


def test_ai_support_separates_empathy_from_actionable_steps():
    row = {"S6": {
        "U": 0.172,
        "safety_override": False,
        "human_review_required": False,
        "empathy_reply": "这段经历听起来让人不好受。",
        "support_source": "qwen_2_5_7b_instruct",
        "support_strategy": {"approach": ["共情回应", "温和询问"], "steps": [
            {"action": "共情回应", "text": "我理解你现在可能感到很受伤。"},
            {"action": "温和询问", "text": "你能告诉我具体发生了什么吗？"},
        ]},
    }}
    app = AppTest.from_string(f"from app import render_follow_up\nrender_follow_up({row!r})").run()
    assert not app.exception
    assert [item.value for item in app.caption] == ["AI 生成"]
    assert "1. 先倾听学生的感受" in app.markdown[0].value
    assert "2. 从一条消息问起" in app.markdown[1].value
    details = "\n".join(item.value for item in app.expander[0].markdown)
    assert "事件风险评分 0.172 / 1" in details
    assert "采用的支持策略**　共情回应、温和询问" in details
    assert "与学生沟通时可参考的话" in details and row["S6"]["empathy_reply"] in details
