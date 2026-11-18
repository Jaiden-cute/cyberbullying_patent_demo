"""Interactive S1–S7 analysis and local history for the patent prototype."""

import json
import os
from pathlib import Path
import sqlite3
from typing import get_args

import altair as alt
import streamlit as st

from src.cli import media_directory
from src.config import load_settings
from src.event_builder import DISPLAY_ZONE, build_events
from src.history_store import HistoryStore
from src.input_records import DEMO_FEATURES, DEMO_MESSAGES, parse_features, parse_messages
from src.model_runtime import ROOT
from src.reporting import render_report
from src.runtime import load_runtime
from src.schemas import AttackReview, EventType

SCENARIOS = {
    "三期合成事件（同一对象）": None,
    "10 月 1 日 · 多条回应与无关话题": "oct01",
    "10 月 11 日 · 无回复或引用 ID": "oct11",
    "10 月 21 日 · 独立安全规则": "oct21",
}
STAGE_TABS = (
    "S1 事件", "S2 回应关联", "S3 八类掩饰", "S3 非语言", "S4–S5 融合与历史",
    "S6 风险与支持", "S7 保存状态", "参数与诊断",
)
METRIC_TEXT = {
    "g": "针对性", "a": "攻击性", "r": "重复性", "h": "伤害性",
    "L": "显式回复或引用", "Q": "@ 发起者", "S": "文本相似度",
    "T": "时间接近度", "D": "指代关系", "G": "主题连续性",
}


def _show_number(value) -> str:
    return f"{value:.6f}" if value is not None else "未计算"


def _record_time(value: str) -> str:
    from datetime import datetime

    return datetime.fromisoformat(value).astimezone(DISPLAY_ZONE).isoformat(timespec="seconds")


def _message_rows(messages):
    return [{
        "时间（UTC+08:00）": message.timestamp.astimezone(DISPLAY_ZONE).isoformat(timespec="seconds"),
        "会话": message.conversation_id, "消息 ID": message.message_id, "发送者": message.sender_id,
        "消息": message.text, "目标": "、".join(message.target_user_ids),
        "回复 ID": message.reply_to_message_id, "引用 ID": message.quoted_message_id,
        "@": "、".join(message.mention_ids), "来源": message.source_type, "许可": message.consent_status,
    } for message in messages]


def edit_findings(messages, settings) -> list[AttackReview]:
    by_key = {message.key: message for message in messages}
    current = st.session_state.setdefault("manual_findings", {})
    with st.expander("人工核对事件类别与参与者"):
        message_key = st.selectbox(
            "选择要核对的消息", list(by_key), index=None,
            format_func=lambda key: f"{key[2]} · {by_key[key].sender_id} · {by_key[key].text[:48]}",
        )
        if message_key is not None:
            selected = by_key[message_key]
            original = current.get(message_key)
            finding = original.findings[0] if original and original.findings else None
            people = sorted({message.sender_id for message in messages} |
                            {person for message in messages for person in message.target_user_ids})
            target = finding.victim_id if finding else next(iter(selected.target_user_ids), "")
            actor = finding.initiator_id if finding and finding.initiator_id else selected.sender_id
            with st.form(f"review-{message_key}"):
                status = st.radio("动作", ("确认或修正", "排除该消息", "恢复规则识别"), horizontal=True)
                victim_id = st.text_input("受影响对象 ID", value=target, help="可用标识：" + "、".join(people))
                initiator = st.text_input("核对后的发起者 ID", value=actor)
                types = st.multiselect("事件类型", get_args(EventType), default=finding.types if finding else ())
                boxes = st.columns(3)
                scores = {
                    key: column.selectbox(METRIC_TEXT[key], (0.0, 0.5, 1.0), index=1 if finding is None else
                                          (0.0, 0.5, 1.0).index(getattr(finding, key)))
                    for column, key in zip(boxes, ("g", "a", "h"), strict=True)
                }
                submitted = st.form_submit_button("保存人工核对")
            if submitted:
                try:
                    if status == "恢复规则识别":
                        current.pop(message_key, None)
                    else:
                        findings = [] if status == "排除该消息" else [
                            {"victim_id": victim_id, "initiator_id": initiator, "types": types, **scores}
                        ]
                        current[message_key] = AttackReview.model_validate({
                            "message": {field: getattr(selected, field) for field in ("platform", "conversation_id", "message_id")},
                            "findings": findings,
                        })
                    st.session_state.pop("analysis_json", None)
                    st.success("已更新核对项。下次运行将应用。")
                except ValueError as error:
                    st.error(str(error))
        reviewed = [value for key, value in current.items() if key in by_key]
        preview = build_events(messages, settings.engineering_demo_default, reviewed)
        st.caption(f"当前预览有 {len(preview.events)} 个事件、{len(reviewed)} 条人工核对。")
        if preview.events:
            st.dataframe([{"摘要": event.summary, "来源": event.source} for event in preview.events], hide_index=True)
    return reviewed


def render_stages(report: dict) -> None:
    st.subheader("计算明细")
    st.write("运行模式", report["run_mode"])
    if report["model_sources"].get("trained_mask_head"):
        st.info("synthetic_training_demo · 分类头由合成标注训练。用于验证原型流程，未在真实青少年数据上验证。")
    events = report["events"]
    if not events:
        st.warning("此次输入未形成可分析事件。")
        st.json(report["input_quality"])
        return
    selection = st.selectbox("事件", range(len(events)), format_func=lambda i: (
        f"{_record_time(events[i]['event']['start_time'])} · {events[i]['event']['victim_id']} · "
        f"{'、'.join(events[i]['event']['types'])}"
    ))
    row = events[selection]
    event, scores = row["event"], row["S1"]
    tabs = st.tabs(STAGE_TABS)
    with tabs[0]:
        st.text(event["summary"])
        st.write(f"来源 {row['event_detector']} · 质量状态 {row['event_quality']}")
        st.dataframe([{
            "消息": item["message"]["text"], "发起者": item["finding"].get("initiator_id") or item["message"]["sender_id"],
            "受影响对象": item["finding"]["victim_id"], "类型": "、".join(item["finding"]["types"]),
            "证据来源": item["source"], "与组内原消息主题余弦": item["topic_similarity"],
        } for item in event["attacks"]], hide_index=True)
        for column, key in zip(st.columns(5), ("g", "a", "r", "h", "B"), strict=True):
            column.metric(METRIC_TEXT.get(key, "严重程度 B"), _show_number(scores[key]))
        st.latex(r"B_i=w_g g_i+w_a a_i+w_r r_i+w_h h_i")
        st.json({"weights": scores["weights"], "repetition_basis": scores["repetition_basis"],
                 "repetition_message_keys": scores["repetition_message_keys"]})
    rows = row["candidates"]
    with tabs[1]:
        st.latex(r"A_{ij}=\sigma(b_0+b_1L_{ij}+b_2Q_{ij}+b_3S_{ij}+b_4T_{ij}+b_5D_{ij}+b_6G_{ij})")
        st.latex(r"S_{ij}=\cos(e_i,r_j),\qquad T_{ij}=\exp(-\Delta t_{ij}/\tau_t)")
        threshold = report["settings"]["patent_example"]["association_tau_A"]
        st.caption(f"有效阈值 A ≥ {threshold}。"
                   "L、Q、S、T、D、G 分别表示" + "、".join(METRIC_TEXT[key] for key in "LQSTDG") + "。")
        associations = [{
            "回应 ID": item["S2"]["message"]["message_id"], "原文": item["S2"]["message"]["text"],
            **{key: item["S2"][key] for key in "LQSTDGA"},
            "筛选": "通过" if item["S2"]["effective"] else "未通过",
            "原因": item["S2"]["reason"],
        } for item in rows]
        st.dataframe(associations, hide_index=True)
        plotted = [item for item in associations if item["A"] is not None]
        if plotted:
            bars = alt.Chart(alt.Data(values=plotted)).mark_bar().encode(
                x=alt.X("A:Q", scale=alt.Scale(domain=[0, 1]), title="关联程度 A"),
                y=alt.Y("回应 ID:N", sort=None, title=None),
                color=alt.Color("筛选:N", scale=alt.Scale(domain=["通过", "未通过"], range=["#387466", "#9d6b49"])),
                tooltip=["回应 ID:N", "A:Q", "筛选:N"],
            )
            line = alt.Chart(alt.Data(values=[{"threshold": threshold}])).mark_rule(
                color="#424955", strokeDash=[5, 4]
            ).encode(x="threshold:Q")
            st.altair_chart(bars + line, width="stretch")
        if row["excluded_during_event_message_keys"]:
            st.write("事件尚未结束时发出的回应", row["excluded_during_event_message_keys"])
        with st.expander("关联证据来源和加权明细"):
            st.json([{key: value for key, value in item["S2"].items() if key != "message"}
                     | {"reply_id": item["S2"]["message"]["message_id"]} for item in rows])
    effective = [item for item in rows if item["S2"]["effective"]]
    selected = None
    if effective:
        with tabs[2]:
            choice = st.selectbox("查看有效回应", range(len(effective)),
                                  format_func=lambda i: effective[i]["S2"]["message"]["message_id"])
            selected = effective[choice]
            result = selected["S3_masking"]
            st.text(selected["S2"]["message"]["text"])
            st.write("推理来源", result["inference_source"], "模型状态", result["model_status"])
            if result["training_data_source"]:
                st.write("训练来源", result["training_data_source"], "模型版本", result["model_version"])
            values = result["p"] if result["p"] is not None else result["non_model_score"]
            if values is not None:
                label = "Sigmoid 模型概率" if result["p"] is not None else "规则分数 non_model_score"
                weights = report["settings"]["engineering_demo_default"]["masking_weights"]
                st.dataframe([{"标签": kind, label: value, "ω": weight}
                              for kind, value, weight in zip(result["labels"], values, weights, strict=True)],
                             hide_index=True)
            else:
                st.warning("八类输出当前不可用。")
            st.metric("综合表达程度 M", _show_number(result["M"]))
            st.latex(r"p_{ijk}=\sigma(W_kh_{ij}+b_k),\qquad M_{ij}=\sum_{k=1}^{8}\omega_k p_{ijk}")
    else:
        with tabs[2]:
            st.info("当前事件没有通过关联筛选的回应。")
    with tabs[3]:
        st.latex(r"z_{ijf}=d_f\frac{x_{ijf}-\mu_{if}}{\sigma_{if}+\epsilon},\qquad u_{ijf}=\max(0,z_{ijf})")
        st.latex(r"N_{ij}=1-\exp\left(-\frac{\sum_f\eta_f q_{ijf}u_{ijf}}{\sum_f\eta_f q_{ijf}+\epsilon}\right)")
        if selected:
            result = selected["S3_nonverbal"]
            st.caption("展示与 S3 八类掩饰页选中的同一回应。")
            st.write("特征来源", result["inference_source"], "可用状态", result["status"])
            if result["inference_source"] == "structured_test_features":
                st.info("结构化模拟观测 · 这些数值用于复算公式。")
            st.metric("同步非语言异常程度 N", _show_number(result["N"]))
            st.dataframe([{"特征": item["feature_name"], **{key: item[key] for key in
                           ("feature_value", "reference_values", "mu", "sigma", "d", "q", "eta", "z", "u", "status")}}
                          for item in result["features"]], hide_index=True)
            with st.expander("当前及参考时间窗、观测质量"):
                st.json(result["features"])
        else:
            st.info("没有有效回应可用于计算 N。")
    event_masking, history = row["S4"], row["S5"]
    with tabs[4]:
        st.latex(r"H_{ij}=A_{ij}\frac{\alpha M_{ij}+\beta N_{ij}+\gamma M_{ij}N_{ij}}{\alpha+\beta+\gamma},\qquad H^{text}_{ij}=\kappa A_{ij}M_{ij}")
        st.latex(r"H_i=\max_{j\in R_i} H_{ij}")
        st.dataframe([{"回应 ID": item["S2"]["message"]["message_id"], "H": item["S4"]["H"] if item["S4"] else None,
                       "分支": item["S4"]["branch"] if item["S4"] else "未计算"} for item in effective], hide_index=True)
        st.write("代表回应", event_masking["representative_reply_id"] or "未选出", "模式来源", event_masking["pattern_source"])
        for column, label, value in zip(st.columns(3), ("H", "P", "F"),
                                        (event_masking["H"], history["P"], history["F"]), strict=True):
            column.metric(label, _show_number(value))
        st.latex(r"P_i=\frac{\sum_{m<i}e^{-\Delta t_{im}/\tau_p}I(H_m\ge\tau_H)\max(0,Sim_{im})}{\sum_{m<i}e^{-\Delta t_{im}/\tau_p}+\epsilon},\qquad F_i=1-e^{-n_i/n_0}")
        st.write("P 状态", history["P_status"], "窗口内较早事件数", history["n_i"])
        if history["P_status"] == "insufficient_history":
            st.caption("可比较的有效事件数不足，P=0 为计算回退值。")
        if history["comparisons"]:
            st.dataframe(history["comparisons"], hide_index=True)
        with st.expander("代表回应向量与历史计算分项"):
            st.json({"S4": event_masking, "S5": history})
    with tabs[5]:
        risk = row["S6"]
        st.latex(r"U_i=\theta_1 B_i+\theta_2 H_i+\theta_3 P_i+\theta_4 F_i")
        st.metric("辅助风险分数 U", _show_number(risk["U"]))
        st.write("分数等级", risk["score_level"] or "未分级", "最终等级", risk["risk_level"] or "未分级")
        st.write("风险阈值", report["settings"]["patent_example"]["risk_thresholds"])
        if risk["human_review_required"]:
            st.warning("需要人工复核。")
        st.write("安全优先规则触发", "是" if risk["safety_override"] else "否")
        if risk["safety_signals"]:
            st.json(risk["safety_signals"])
        st.write("支持来源", risk["support_source"], "状态", risk["support_status"])
        if risk["empathy_reply"]:
            st.text(risk["empathy_reply"])
            st.json(risk["support_strategy"])
    with tabs[6]:
        st.json(row["S7"])
        st.caption("可在历史记录页查询、回看和填写人工跟进。")
    with tabs[7]:
        st.json({key: report[key] for key in ("inference_source", "model_status", "model_sources",
                                             "input_sources", "input_quality", "demo_mode", "limitations")})
        st.json(report["settings"])


def history_page(store: HistoryStore) -> None:
    st.subheader("历史记录")
    records = store.list_events()
    if not records:
        st.info("暂无历史记录。")
        return
    people = sorted({(item["platform"], item["victim_id"]) for item in records})
    subject = st.selectbox("受影响对象", [None, *people],
                           format_func=lambda value: "全部" if value is None else " · ".join(value))
    if subject:
        records = [item for item in records if (item["platform"], item["victim_id"]) == subject]
    st.dataframe([{"时间（UTC+08:00）": _record_time(row["started_utc"]), "受影响对象": row["victim_id"],
                   "事件摘要": row["summary"], "风险": row["risk_level"], "支持来源": row["support_source"],
                   "数据": "合成" if row["synthetic"] else "授权本地"} for row in records], hide_index=True)
    item = st.selectbox("历史事件回看", records, format_func=lambda entry: entry["summary"])
    record = store.load_event(item["event_id"])
    if record is not None:
        with st.expander("保存的计算明细"):
            st.json(record)
        with st.form(f"follow-up-{item['event_id']}"):
            note = st.text_area("人工跟进记录", value=record["S7"]["follow_up"] or "", max_chars=2000)
            if st.form_submit_button("保存跟进"):
                try:
                    store.save_follow_up(item["event_id"], note)
                    st.success("跟进已保存。")
                except (ValueError, sqlite3.Error) as error:
                    st.error(str(error))
    with st.expander("清空合成历史"):
        confirmed = st.checkbox("确认清空合成事件及其计算结果")
        if st.button("清空合成记录", disabled=not confirmed):
            count = store.clear_synthetic()
            st.session_state.pop("analysis_json", None)
            st.session_state["history_notice"] = f"已清空 {count} 个合成事件。"
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="网络欺凌事件与回应分析", layout="wide")
    st.title("网络欺凌事件与回应分析")
    st.caption("青少年网络欺凌回应关联与分级支持 · 专利原型")
    config = load_settings(ROOT / "config.yaml")
    store = HistoryStore(Path(os.environ.get("PATENT_HISTORY_DB", str(ROOT / "state/history.sqlite3"))))
    navigation = st.sidebar.radio("页面", ("分析与计算明细", "历史记录"))
    if navigation == "历史记录":
        if notice := st.session_state.pop("history_notice", None):
            st.success(notice)
        history_page(store)
        return

    st.subheader("数据导入")
    source = st.radio("消息来源", ("加载合成示例", "上传本地 JSON/CSV"), horizontal=True)
    features = []
    try:
        if source == "加载合成示例":
            scene = st.selectbox("合成场景", list(SCENARIOS))
            prefix = SCENARIOS[scene]
            messages = parse_messages(DEMO_MESSAGES.read_bytes(), DEMO_MESSAGES.name)
            if prefix:
                messages = [message for message in messages if message.message_id.startswith(prefix)]
            if st.checkbox("加入结构化模拟非语言特征"):
                keys = {message.key for message in messages}
                features = [item for item in parse_features(DEMO_FEATURES.read_bytes()) if item.reply.key in keys]
        else:
            uploaded = st.file_uploader("带来源与许可标记的消息文件", type=["json", "csv"])
            if uploaded is None:
                st.info("上传消息文件后可以核对事件并运行分析。")
                return
            messages = parse_messages(uploaded.getvalue(), uploaded.name)
            observed = st.file_uploader("可选的结构化模拟特征 JSON", type=["json"])
            if observed:
                features = parse_features(observed.getvalue())
        if not messages:
            st.warning("消息文件为空。")
            return
        st.dataframe(_message_rows(messages), hide_index=True)
        reviews = edit_findings(messages, config)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    trained = st.radio(
        "推理方式", ("Sentence-BERT + 已训练分类头", "规则演示"), horizontal=True
    ) == "Sentence-BERT + 已训练分类头"
    if trained:
        st.caption("synthetic_training_demo · 合成训练。模型概率仅用于验证流程。")
        use_qwen = st.checkbox("使用已配置的 Qwen 推理服务")
        strict = st.checkbox("模型缺失时保留未计算状态")
    else:
        use_qwen = strict = False
    if st.button("运行 S1–S7", type="primary"):
        try:
            with st.spinner("正在计算关联特征、跨事件指标和风险等级"):
                runtime = load_runtime(config, requested="trained" if trained else "demo", strict=strict, use_qwen=use_qwen)
                report = store.analyze(messages, config, reviews=reviews, features=features, runtime=runtime,
                                       allowed_media_dir=media_directory())
                st.session_state["analysis_json"] = render_report(report)
        except (OSError, ValueError, sqlite3.Error, RuntimeError) as error:
            st.session_state.pop("analysis_json", None)
            st.error(str(error))
    if exported := st.session_state.get("analysis_json"):
        st.caption("显示最近一次运行结果。修改输入后重新运行即可更新。")
        st.download_button("下载完整计算报告", exported, file_name="patent_analysis.json", mime="application/json")
        render_stages(json.loads(exported))


if __name__ == "__main__":
    main()
