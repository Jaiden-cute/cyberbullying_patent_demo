import json

from src.event_builder import build_events
from src.llm_adapter import QwenAdapter, SOURCE
from src.pipeline import run_pipeline
from src.risk_support import evaluate_risk
from src.runtime import Runtime
from src.support import generate_support


def _fake_transport(monkeypatch, responses):
    adapter = QwenAdapter(base_url="http://127.0.0.1:8080/v1", served_id="Qwen2.5-7B-Instruct",
                          expected_model="Qwen2.5-7B-Instruct")
    requests = []
    contents = iter(responses)

    def request(route, body=None):
        requests.append((route, body))
        if route == "models":
            return {"data": [{"id": adapter.served_id}]}
        return {"model": adapter.served_id, "choices": [{"message": {"content": next(contents)}}]}

    monkeypatch.setattr(adapter, "_request", request)
    return adapter, requests


def test_joint_d_g_one_request_and_invalid_field_falls_to_zero(monkeypatch, settings, messages):
    adapter, calls = _fake_transport(monkeypatch, ['{"coreference":1,"topic_continuity":true}'])
    adapter.probe()
    event = build_events(messages, settings.engineering_demo_default).events[0]
    response = adapter.judge_relations(event, messages[3])
    assert (response.coreference, response.topic_continuity) == (1, 0)
    assert response.status == "llm_unavailable"
    assert response.field_sources == (SOURCE, "llm_unavailable")
    assert [route for route, _ in calls] == ["models", "chat/completions"]
    context = json.loads(calls[1][1]["messages"][1]["content"])
    assert context["event_summary"] == event.summary and len(context["event_messages"]) == 2


def test_two_support_calls_and_allowlist_validation(monkeypatch, settings):
    plan = {"goal": "留出表达空间", "approach": ["共情回应", "温和询问"],
            "steps": [{"action": "温和询问", "text": "问问哪段经历想先说。"}]}
    adapter, calls = _fake_transport(monkeypatch, [json.dumps(plan, ensure_ascii=False), "你可以按自己的节奏说。"])
    calculated = evaluate_risk(0.1, 0.1, 0.0, 0.0, [], settings)
    result = generate_support(calculated, adapter, {"event_summary": "合成事件"})
    assert result.support_source == SOURCE
    assert result.support_strategy["goal"] == "留出表达空间"
    assert len(calls) == 2
    assert "allowed_actions" in json.loads(calls[0][1]["messages"][1]["content"])
    assert "strategy" in json.loads(calls[1][1]["messages"][1]["content"])


def test_unsafe_qwen_plan_falls_back_without_second_call(monkeypatch, settings):
    plan = {"goal": "替你联系管理员", "approach": ["共情回应"],
            "steps": [{"action": "共情回应", "text": "听你说。"}]}
    adapter, calls = _fake_transport(monkeypatch, [json.dumps(plan, ensure_ascii=False)])
    calculated = evaluate_risk(0.1, 0.1, 0.0, 0.0, [], settings)
    result = generate_support(calculated, adapter, {})
    assert result.support_source == "safe_template"
    assert result.support_status == "strategy_goal_or_fields_rejected"
    assert len(calls) == 1


def test_requested_qwen_unavailable_d_g_are_zero_and_marked(settings, messages):
    runtime = Runtime(requested="trained", qwen_requested=True, qwen_status="endpoint_connection_error")
    report = run_pipeline(messages[:1] + messages[3:4], settings, runtime=runtime)
    link = report["events"][0]["candidates"][0].S2
    assert (link.D, link.G) == (0, 0)
    assert link.llm_status == "llm_unavailable"
    assert link.llm_reasons == ("endpoint_connection_error",)
