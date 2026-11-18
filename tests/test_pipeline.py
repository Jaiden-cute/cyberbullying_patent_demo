import json

import pytest

from src.cli import DEMO_MESSAGES, main, render_report
from src.pipeline import run_pipeline


def test_selected_subject_filters_events_without_changing_its_candidates(settings, messages):
    original = messages[0]
    second_attack = original.model_copy(update={
        "message_id": "attack-other-person",
        "target_user_ids": ("另一位同学",),
    })
    subject = (original.platform, original.target_user_ids[0])
    baseline = run_pipeline(messages, settings)["events"][0]
    result = run_pipeline([*messages, second_attack], settings, affected_subject=subject)
    assert all(row["event"].victim_id == subject[1] for row in result["events"])
    assert result["events"][0]["candidates"] == baseline["candidates"]


def test_effective_explicit_and_implicit_replies_and_unrelated_message(settings, messages, features):
    report = run_pipeline(messages, settings, features=features)
    first, second, _ = report["events"]
    explicit, quoted, unrelated = first["candidates"]
    implicit = second["candidates"][0]
    assert (explicit.S2.L, explicit.S2.Q, explicit.S2.effective) == (1, 1, True)
    assert (implicit.S2.L, implicit.S2.Q, implicit.S2.D, implicit.S2.G) == (0, 1, 1, 1)
    assert implicit.S2.effective
    assert implicit.S2.A == pytest.approx(0.730927, abs=1e-6)
    assert unrelated.S2.A < 0.7
    assert unrelated.S3_masking is None
    assert unrelated.S4 is None
    assert quoted.S4.H < explicit.S4.H
    assert first["S4"]["representative_reply_id"] == explicit.S2.message.message_id
    assert first["excluded_during_event_message_keys"][0][-1] == "oct01-during-event"


def test_explicit_relationship_does_not_skip_association_threshold(settings, messages):
    attack = messages[0]
    unrelated = messages[5].model_copy(update={"reply_to_message_id": attack.message_id})
    row = run_pipeline([attack, unrelated], settings)["events"][0]["candidates"][0]
    assert row.S2.L == 1
    assert row.S2.A < 0.7
    assert row.S3_masking is None


def test_empty_semantic_representation_preserves_unavailability(settings, messages):
    attack, reply = messages[0], messages[3].model_copy(update={"text": "？"})
    event = run_pipeline([attack, reply], settings)["events"][0]
    S2 = event["candidates"][0].S2
    assert (S2.L, S2.S, S2.A, S2.reason) == (1, None, None, "semantic_representation_unavailable")
    assert event["S4"]["status"] == "no_effective_reply"
    assert event["S6"].U is None


def test_text_and_structured_runs_and_cross_event_history(settings, messages, features):
    text = run_pipeline(messages, settings)["events"]
    structured = run_pipeline(messages, settings, features=features)["events"]
    assert text[0]["candidates"][0].S4.branch == "text"
    assert text[0]["candidates"][0].S3_nonverbal.N is None
    assert structured[0]["candidates"][0].S4.branch == "multimodal"
    assert structured[0]["S4"]["H"] == pytest.approx(0.646471, abs=1e-6)
    assert [row["S5"].n_i for row in structured] == [0, 1, 2]
    assert structured[0]["S5"].P_status == structured[1]["S5"].P_status == "insufficient_history"
    assert structured[2]["S5"].P_status == "available"
    assert structured[2]["S5"].P == pytest.approx(0.373361, abs=1e-6)
    assert all(item.dimension == 8 for item in structured[2]["S5"].comparisons)
    assert text[2]["S5"].P_status == "available" and text[2]["S5"].P == 0
    assert all(row["S7"] == {"status": "within_run_history_only", "persisted": False} for row in structured)


def test_normal_no_effective_reply_does_not_produce_risk_score(settings, messages):
    result = run_pipeline([messages[0]], settings)["events"][0]
    assert result["S4"]["H"] is None
    assert result["S4"]["representative_reply_id"] is None
    assert result["S5"].P is None
    assert result["S6"].U is None
    assert result["S6"].risk_level is None


def test_report_is_deterministic_with_out_of_order_messages(settings, messages, features):
    forward = render_report(run_pipeline(messages, settings, features=features))
    backward = render_report(run_pipeline(list(reversed(messages)), settings, features=features))
    assert forward == backward
    report = json.loads(forward)
    assert report["run_mode"] == "DEMO"
    first = report["events"][0]["candidates"][0]
    assert first["S3_masking"]["p"] is None
    assert first["S3_nonverbal"]["inference_source"] == "structured_test_features"
    assert report["model_sources"] == {}


def test_cli_exports_reproducible_json(tmp_path):
    destination = tmp_path / "run.json"
    assert main(["--output", str(destination)]) == 0
    content = destination.read_bytes()
    assert main(["--output", str(destination)]) == 0
    assert destination.read_bytes() == content
    assert len(json.loads(content)["events"]) == 3


def test_cli_refuses_to_overwrite_input(capsys):
    assert main(["--output", str(DEMO_MESSAGES)]) == 2
    assert "必须与输入文件路径不同" in capsys.readouterr().err
