"""Interactive S1–S7 analysis and local history for the patent prototype."""

from collections import Counter
from datetime import datetime
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

USER_PAGE = "用户交互"
HISTORY_PAGE = "历史记录"
SOURCE_NAMES = {
    "qwen_2_5_7b_instruct": "AI 生成",
    "safe_template": "预设支持回复",
}
USER_ACTIONS = {
    "共情回应": ("找一个你信任的人", "给对方发一句：「群里有些话让我不舒服，你能听我说一会儿吗？」"),
    "温和询问": ("从一条消息说起", "挑一条你最想说的消息，告诉对方是谁说的，以及哪句话让你在意。"),
    "渐进式提问": ("记下发生的事", "先写下时间、在哪个群，以及让你不舒服的一句话。想不起来的部分可以先空着。"),
    "鼓励表达": ("说清你希望得到的帮助", "告诉对方你希望他先听你说，还是陪你去找老师。"),
    "证据保存": ("保存相关聊天记录", "截下带发送者、时间和前后消息的画面，保留在自己的设备上。"),
    "屏蔽或举报": ("减少对方继续打扰", "保存记录后，在聊天页屏蔽对方，或打开这条消息的举报菜单。"),
    "可信成年人求助引导": ("请成年人帮忙处理", "把记录给信任的家长、老师或辅导员看，告诉对方你希望得到什么帮助。"),
    "及时求助": ("现在就找成年人陪同", "先到安全的地方，告诉就近的老师、家长或其他可信任的成年人：「我现在担心自己的安全，请来帮我。」"),
}


def _record_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(DISPLAY_ZONE).isoformat(timespec="seconds")


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


def render_user_support(row: dict) -> None:
    risk = row["S6"]
    if risk["human_review_required"]:
        st.warning("本地记录已标记为需要人工复核。若担心安全，请尽快向可信任的成年人求助。外部通知未发送。")
    if risk["empathy_reply"]:
        with st.container(border=True):
            st.subheader("给你的回应")
            st.write(risk["empathy_reply"])
            st.caption(SOURCE_NAMES[risk["support_source"]])
        actions = ("及时求助",) if risk["human_review_required"] else (
            dict.fromkeys(step["action"] for step in risk["support_strategy"]["steps"])
        )
        st.subheader("下一步怎么做")
        for index, action in enumerate(actions, 1):
            title, instruction = USER_ACTIONS[action]
            st.markdown(f"**{index}. {title}**")
            st.write(instruction)
    elif row["S4"]["status"] == "masking_model_unavailable":
        st.info("当前模型不可用。可检查模型设置，或明确选择规则演示后重新分析。")
    else:
        st.info("没有找到足够相关的后续回应。你可以补充事件后的对话，再试一次。")


def history_page(store: HistoryStore) -> None:
    st.subheader("历史记录")
    records = store.list_events()
    if not records:
        st.info("暂无历史记录。")
        return
    people = sorted({(item["platform"], item["victim_id"]) for item in records})
    labels = _subject_labels(people)
    subject = st.selectbox("受影响对象", [None, *people],
                           format_func=lambda value: "全部" if value is None else labels[value])
    if subject:
        records = [item for item in records if (item["platform"], item["victim_id"]) == subject]
    st.dataframe([{"时间（UTC+08:00）": _record_time(row["started_utc"]), "受影响对象": row["victim_id"],
                   "事件摘要": row["summary"]} for row in records], hide_index=True)
    by_id = {item["event_id"]: item for item in records}
    event_id = st.selectbox("历史事件回看", list(by_id), format_func=lambda value: by_id[value]["summary"])
    record = store.load_event(event_id)
    if record is not None:
        render_user_support(record)


def _load_user_messages() -> list | None:
    source = st.radio("消息来源", ("虚构测试数据", "粘贴 / 编辑", "导入 JSON / CSV"), horizontal=True)
    if source == "虚构测试数据":
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
    subject = st.selectbox("当前受影响对象", subjects, format_func=_subject_labels(subjects).__getitem__)
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


def user_page(config, store: HistoryStore) -> None:
    st.subheader("从聊天记录开始")
    try:
        messages = _load_user_messages()
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    if not messages:
        return
    subject, messages = _subject_input(messages)
    st.dataframe(_message_rows(messages), hide_index=True)
    with st.expander("分析设置"):
        trained = st.radio("分析方式", ("规则演示", "模型辅助分析"), horizontal=True,
                           help="辅助分析模型使用虚构样例训练，仅用于演示。") != "规则演示"
        use_qwen = False
        if trained:
            use_qwen = st.checkbox("使用 AI 生成回复",
                                   help="开启后，分析所需的对话会发送至配置的服务。")
    signature = sha256(encode_record((messages, subject, trained, use_qwen)).encode()).hexdigest()
    if st.button("开始智能分析", type="primary"):
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
    if st.session_state.get("analysis_signature") != signature or not st.session_state.get("analysis_json"):
        return
    report = json.loads(st.session_state["analysis_json"])
    if use_qwen and report["model_status"]["qwen_2_5_7b_instruct"] != "endpoint_available":
        st.info("AI 回复暂不可用。")
    elif trained and report["model_status"]["trained_mask_head"] != "available":
        st.info("辅助分析暂不可用。")
    if not report["events"]:
        st.info("未生成该对象的事件记录。请核对受影响对象及消息指向，或补充相关对话。")
    else:
        events = report["events"]
        index = st.selectbox("查看事件的支持结果", range(len(events)), index=len(events) - 1,
                             format_func=lambda i: _record_time(events[i]["event"]["start_time"]))
        render_user_support(events[index])


def main() -> None:
    st.set_page_config(page_title="网络欺凌事件与回应分析", layout="wide")
    st.title("网络欺凌事件与回应分析")
    config = load_settings(ROOT / "config.yaml")
    store = HistoryStore(Path(os.environ.get("PATENT_HISTORY_DB", str(ROOT / "state/history.sqlite3"))))
    navigation = st.sidebar.radio("页面", (USER_PAGE, HISTORY_PAGE))
    if navigation == HISTORY_PAGE:
        history_page(store)
    else:
        user_page(config, store)


if __name__ == "__main__":
    main()
