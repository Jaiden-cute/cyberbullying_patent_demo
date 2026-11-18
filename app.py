"""Interactive S1–S7 analysis and local history for the patent prototype."""

from collections import Counter
from datetime import datetime, timedelta
import json
from hashlib import sha256
import os
from pathlib import Path
import sqlite3

import streamlit as st

from src.cli import media_directory
from src.config import load_settings
from src.demo_inputs import SCENE_FILES, parse_chat_input
from src.event_builder import DISPLAY_ZONE
from src.history_store import HistoryStore
from src.input_records import parse_messages
from src.model_runtime import ROOT
from src.reporting import encode_record, render_report
from src.runtime import load_runtime

CASE_PAGE = "事件研判"
HISTORY_PAGE = "历史与跟进"
LEVEL_COLORS = {
    "一般关注": "blue",
    "重点关注": "violet",
    "较高风险": "orange",
    "高风险": "red",
}
SOURCE_NAMES = {
    "qwen_2_5_7b_instruct": "AI 生成",
    "safe_template": "预设支持回复",
}
FOLLOW_UP_ACTIONS = {
    "共情回应": ("先倾听学生的感受", "表达理解，给学生按自己的节奏讲述的空间。"),
    "温和询问": ("从一条消息问起", "请学生选一句最在意的话，说说当时发生了什么。"),
    "渐进式提问": ("按事情经过逐步询问", "先了解发生的时间和场景，再请学生说说当时的回应和后续情况。"),
    "鼓励表达": ("确认学生希望得到的帮助", "询问学生希望先有人倾听，还是希望中心协助处理。"),
    "证据保存": ("协助保存聊天记录", "保留发送者、时间、原话和前后对话，便于后续核对。"),
    "屏蔽或举报": ("协助屏蔽或举报", "保存记录后，和学生一起查看平台的屏蔽或举报入口。"),
    "可信成年人求助引导": ("协助联系可信任的成年人", "和学生确认可以请哪位家长、老师或其他可信任的成年人协助。"),
    "及时求助": ("优先确认学生安全", "尽快确认学生是否安全，并安排可信任的成年人到场陪同和协助。"),
    "人工复核": ("安排人工复核", "按中心流程复核事件，并记录已采取的措施和后续安排。"),
}


def _record_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(DISPLAY_ZONE).strftime("%Y-%m-%d %H:%M")


def _message_rows(messages):
    return [{
        "时间（UTC+08:00）": message.timestamp.astimezone(DISPLAY_ZONE).isoformat(timespec="seconds"),
        "发送者": message.sender_id,
        "消息": message.text,
    } for message in messages]


def _subject_labels(subjects: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    counts = Counter(person for _, person in subjects)
    return {(platform, person): person if counts[person] == 1 else f"{person}（{platform}）"
            for platform, person in subjects}


def render_follow_up(row: dict) -> None:
    st.subheader("接下来怎么跟进")
    risk = row["S6"]
    if risk["empathy_reply"] is None:
        if row["S4"]["status"] == "masking_model_unavailable":
            st.info("模型暂未给出支持建议。请检查分析设置后重试。")
        else:
            st.info("建议先补充事发后的对话，再研判需要怎样的支持。")
        return

    plan = risk["support_strategy"]
    actions = ("及时求助", "人工复核") if risk["human_review_required"] else (
        dict.fromkeys(step["action"] for step in plan["steps"])
    )
    for index, action in enumerate(actions, 1):
        title, instruction = FOLLOW_UP_ACTIONS[action]
        with st.container(key=f"support_action_{index}"):
            st.markdown(f"**{index}. {title}**  \n{instruction}")

    with st.expander("查看风险评分、支持策略与沟通话术"):
        if risk["U"] is not None:
            st.write(f"事件风险评分 {risk['U']:.3f} / 1")
        if risk["safety_override"]:
            st.write("本次由安全优先规则触发人工复核。")
        st.write("**采用的支持策略**　" + "、".join(plan["approach"]))
        st.write("**与学生沟通时可参考的话**")
        st.write(risk["empathy_reply"])
        st.caption(SOURCE_NAMES[risk["support_source"]])


def render_analysis_basis(row: dict) -> None:
    st.subheader("为什么值得关注")
    event, risk = row["event"], row["S6"]
    candidates = row["candidates"]
    accepted = sum(item["S2"]["effective"] for item in candidates)
    representative = row["S4"]["representative_reply_id"]
    history = row["S5"]
    saved_ids = row["S7"].get("history_read_event_ids", [])
    if risk["safety_override"]:
        signals = dict.fromkeys(item["matched_phrase"] for item in risk["safety_signals"])
        st.write("**安全信号**　" + "、".join(f"「{phrase}」" for phrase in signals))

    if candidates:
        st.write(f"**回应筛选**　{accepted} 条回应与事件有关，{len(candidates) - accepted} 条未纳入分析。")
        excluded = next((item["S2"]["message"]["text"] for item in candidates
                         if item["S2"]["reason"] == "below_threshold"), None)
        if excluded:
            st.text(f"一条未纳入的消息　{excluded}")
    else:
        st.write("**回应筛选**　事件之后暂无可分析的回应。")

    if representative:
        selected = next(item for item in candidates if item["S2"]["message"]["message_id"] == representative)
        st.write("**重点参考的回应**")
        st.text(selected["S2"]["message"]["text"])
        matches = selected["S3_masking"]["matched_phrases"]
        phrases = list(dict.fromkeys(phrase for hits in matches.values() for phrase in hits))[:2]
        if phrases:
            st.write("结合事发情境，" + "、".join(f"「{phrase}」" for phrase in phrases)
                     + "可能是掩饰感受的线索，值得进一步了解。")

    prior = f"**过往事件**　同一学生近期有 {history['n_i']} 次更早的事件。"
    comparisons = [item for item in history["comparisons"] if item["Sim"] is not None]
    if comparisons:
        start = datetime.fromisoformat(event["start_time"]).astimezone(DISPLAY_ZONE)
        saved_dates = [(start - timedelta(days=item["delta_days"])).strftime("%m月%d日")
                       for item in comparisons if item["event_id"] in saved_ids]
        detail = f"已比较本次回应与 {len(comparisons)} 次过往事件的回应"
        if saved_dates:
            detail += f"，其中 {len(saved_dates)} 次来自已保存的记录（{'、'.join(saved_dates)}）"
        if history["P"]:
            detail += "。此前的相似表达也纳入了本次研判"
        else:
            detail += "。本次比较的回应均未计入重复表达"
    elif history["P_status"] == "insufficient_history":
        detail = "可比较的回应记录还不足"
    else:
        detail = "本次暂无可比较的回应"
    st.markdown(f"{prior}  \n**跨事件比较**　{detail}。")


def render_case(row: dict) -> None:
    event, risk = row["event"], row["S6"]
    with st.container(key="event_overview"):
        st.header("发生了什么", anchor=False)
        level = risk["risk_level"] or "待补充材料"
        color = LEVEL_COLORS.get(level, "gray")
        st.markdown(f"**{event['victim_id']}**　·　{_record_time(event['start_time'])}　:{color}-badge[{level}]")
        st.write(f"{'、'.join(event['initiators'])}针对该学生的 {len(event['message_ids'])} 条消息涉及"
                 f"{'、'.join(event['types'])}。")
        st.text(event["attacks"][0]["message"]["text"])
    if risk["human_review_required"]:
        st.warning("需要人工复核。请优先确认学生安全，按中心流程安排复核。外部通知未发送。")
    with st.container(key="case_results"):
        basis, follow_up = st.columns((3, 2), gap="small")
        with basis, st.container(key="case_evidence"):
            render_analysis_basis(row)
        with follow_up, st.container(key="case_actions"):
            render_follow_up(row)


def history_page(store: HistoryStore) -> None:
    records = store.list_events()
    with st.container(key="history_filters"):
        st.header("历史与跟进", anchor=False)
        if not records:
            st.info("暂无历史记录。")
            return
        people = sorted({(item["platform"], item["victim_id"]) for item in records})
        labels = _subject_labels(people)
        student_filter, event_picker = st.columns((1, 2), gap="small")
        subject = student_filter.selectbox("筛选学生", [None, *people],
                                           format_func=lambda value: "全部" if value is None else labels[value])
        if subject:
            records = [item for item in records if (item["platform"], item["victim_id"]) == subject]
        events = {item["event_id"]: f"{_record_time(item['started_utc'])} · "
                  f"{labels[(item['platform'], item['victim_id'])]} · {item['risk_level'] or '待补充材料'}"
                  for item in records}
        event_id = event_picker.selectbox("查看历史事件", list(events), format_func=events.__getitem__)
    record = store.load_event(event_id)
    if record is not None:
        render_case(record)
        with st.container(key="history_follow_up"):
            st.subheader("跟进记录", anchor=False)
            with st.form(f"follow-up-{event_id}", border=False):
                note = st.text_area("已采取的措施与下一步安排", value=record["S7"]["follow_up"] or "",
                                    max_chars=2000, placeholder="填写已完成的沟通或处理，以及计划何时再跟进。")
                if st.form_submit_button("保存跟进记录", type="primary"):
                    try:
                        store.save_follow_up(event_id, note)
                    except (ValueError, sqlite3.Error) as error:
                        st.error(str(error))
                    else:
                        st.success("跟进记录已保存。")


def _load_user_messages() -> list | None:
    source = st.radio("消息来源", ("虚构演示案例", "粘贴 / 编辑", "导入 JSON / CSV"))
    if source == "虚构演示案例":
        scene = st.selectbox("演示场景", list(SCENE_FILES))
        sample = ROOT / "data" / SCENE_FILES[scene]
        return parse_messages(sample.read_bytes(), sample.name)

    with st.expander("输入格式与来源设置"):
        st.write("聊天记录需要包含时间、发言人和消息正文。")
        consent = st.radio("文件或文本未标注来源时", ("虚构测试数据", "已授权的本地记录"), horizontal=True)
        conversation = st.text_input("会话标识", value="本地会话", help="文件中已有的会话标识优先使用。")
    consent = "synthetic" if consent == "虚构测试数据" else "authorized"
    if source == "粘贴 / 编辑":
        if st.button("载入一段可编辑的虚构示例"):
            sample = ROOT / "data" / next(iter(SCENE_FILES.values()))
            st.session_state["chat_editor"] = sample.read_text(encoding="utf-8")
        text = st.text_area(
            "聊天记录", key="chat_editor", height=200,
            placeholder="2026-10-01T19:00:00+08:00 | 同学甲 | 第一条消息",
            help="可粘贴 JSON 数组，或每行填写「时间 | 发言人 | 消息内容」。时间需要带时区。",
        )
        if not text.strip():
            st.info("请粘贴聊天记录，或载入一段可编辑的虚构示例。")
            return None
        return parse_chat_input(text.encode(), consent=consent, conversation=conversation)

    uploaded = st.file_uploader("聊天记录文件", type=["json", "csv"])
    if uploaded is None:
        st.info("请选择 UTF-8 编码的 JSON 或 CSV 文件。")
        return None
    return parse_chat_input(uploaded.getvalue(), uploaded.name, consent=consent, conversation=conversation)


def _subject_input(messages):
    stated_targets = {(message.platform, target) for message in messages for target in message.target_user_ids}
    everyone = {(message.platform, person) for message in messages for person in (message.sender_id, *message.mention_ids)}
    subjects = sorted(stated_targets) + sorted(everyone - stated_targets)
    subject = st.selectbox("受影响学生", subjects, format_func=_subject_labels(subjects).__getitem__)
    unaddressed = {message.key: message for message in messages if message.platform == subject[0]
                   and message.sender_id != subject[1] and not (message.target_user_ids or message.mention_ids)
                   and not (message.reply_to_message_id or message.quoted_message_id)}
    if unaddressed:
        with st.expander("补充缺失的指向信息"):
            confirmed = st.multiselect("我确认下列消息指向当前对象", list(unaddressed),
                                       format_func=lambda key: f"{unaddressed[key].sender_id} · {unaddressed[key].text[:70]}",
                                       key=f"confirmed-targets-{subject!r}")
        messages = [message.model_copy(update={"target_user_ids": (subject[1],)}) if message.key in confirmed else message
                    for message in messages]
    return subject, messages


def case_page(config, store: HistoryStore) -> None:
    with st.sidebar:
        st.subheader("待研判材料")
        try:
            messages = _load_user_messages()
        except (OSError, ValueError) as error:
            st.error(str(error))
            return
        if not messages:
            return
        subject, messages = _subject_input(messages)
        analyze = st.button("研判本次事件", type="primary", width="stretch")
        with st.expander("分析设置"):
            trained = st.radio("分析方式", ("规则演示", "模型辅助分析"),
                               help="分类模型使用虚构样例训练（synthetic_training_demo），仅用于演示。") != "规则演示"
            use_qwen = False
            if trained:
                use_qwen = st.checkbox("使用 AI 生成回复",
                                       help="开启后，分析所需的对话会发送至配置的服务。")
    signature = sha256(encode_record((messages, subject, trained, use_qwen)).encode()).hexdigest()
    if analyze:
        try:
            with st.spinner("正在分析聊天记录并整理支持建议…"):
                runtime = load_runtime(config, requested="trained" if trained else "demo", use_qwen=use_qwen)
                report = store.analyze(messages, config, runtime=runtime, affected_subject=subject,
                                       allowed_media_dir=media_directory())
                st.session_state["analysis_json"] = render_report(report)
                st.session_state["analysis_signature"] = signature
        except (OSError, ValueError, sqlite3.Error) as error:
            st.session_state.pop("analysis_json", None)
            st.error(f"分析未完成。{error}")
            return
    if st.session_state.get("analysis_signature") == signature and st.session_state.get("analysis_json"):
        report = json.loads(st.session_state["analysis_json"])
        if use_qwen and report["model_status"]["qwen_2_5_7b_instruct"] != "endpoint_available":
            st.info("AI 回复暂不可用。")
        elif trained and report["model_status"]["trained_mask_head"] != "available":
            st.info("辅助分析暂不可用。")
        events = report["events"]
        if events:
            index = len(events) - 1
            if len(events) > 1:
                index = st.selectbox("切换本次识别出的事件", range(len(events)), index=index,
                                     format_func=lambda i: _record_time(events[i]["event"]["start_time"]))
            render_case(events[index])
        else:
            st.info("暂未识别出该学生相关的欺凌事件。请核对受影响学生及消息指向，或补充相关对话。")
    else:
        with st.container(key="case_ready", gap=None):
            st.header("待研判", anchor=False)
            st.write("在侧栏选择聊天记录与受影响学生，然后点击「研判本次事件」。")
    with st.expander("查看聊天原文"):
        st.dataframe(_message_rows(messages), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="网络欺凌事件研判与跟进", layout="wide")
    st.html(ROOT / "assets" / "workspace.css")
    with st.container(key="workspace_header", gap=None):
        st.caption("学校心理咨询中心")
        st.title("网络欺凌事件研判与跟进", anchor=False)
    config = load_settings(ROOT / "config.yaml")
    store = HistoryStore(Path(os.environ.get("PATENT_HISTORY_DB", str(ROOT / "state/history.sqlite3"))))
    with st.sidebar, st.container(key="workspace_navigation"):
        navigation = st.radio("页面", (CASE_PAGE, HISTORY_PAGE))
    if navigation == HISTORY_PAGE:
        history_page(store)
    else:
        case_page(config, store)


if __name__ == "__main__":
    main()
