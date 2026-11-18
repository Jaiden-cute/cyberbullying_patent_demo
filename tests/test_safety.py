from src.pipeline import run_pipeline


def test_safety_changes_final_level_but_keeps_numeric_result(settings, messages, features):
    severe = run_pipeline(messages, settings, features=features)["events"][-1]["S6"]
    assert severe.U < 0.5
    assert severe.score_level == "重点关注"
    assert severe.risk_level == "高风险"
    assert severe.safety_override and severe.human_review_required
    assert severe.safety_signals[0].message_key[-1] == "oct21-attack"
    assert severe.support_source == "safe_template"
    assert "可信任的成年人" in severe.empathy_reply
    assert "已通知" not in severe.empathy_reply and "已联系" not in severe.empathy_reply


def test_safety_override_works_without_associated_reply_or_u(settings, messages):
    threat = next(message for message in messages if message.message_id == "oct21-attack")
    result = run_pipeline([threat], settings)["events"][0]
    assert result["S4"]["status"] == "no_effective_reply"
    risk = result["S6"]
    assert risk.U is None
    assert risk.score_level is None
    assert risk.risk_level == "高风险"
    assert risk.safety_override and risk.human_review_required
    assert risk.support_source == "safe_template"
    assert risk.empathy_reply
