"""S6 support templates and validation for the optional two-call Qwen path."""

from dataclasses import dataclass, replace
import re

from .llm_adapter import LLMUnavailable, QwenAdapter, SOURCE
from .risk_support import LEVELS, RiskResult

TEMPLATES = (
    "看到你遇到了这样的事。你愿意说说哪一段最让你在意吗？可以按自己的节奏来。",
    "谢谢你愿意回应。我们可以慢慢谈，你愿意先说说那件事发生时，接下来发生了什么吗？",
    "你可以先保存相关记录，并视情况屏蔽或向平台举报。也可以请家长、老师或其他可信任的成年人一起商量。",
    "请尽快找一位可信任的成年人当面陪伴和协助。如果眼前有危险，请先到安全的地方并联系当地紧急服务。当前记录已标记为需要人工复核。",
)
BLOCKED_SUPPORT = re.compile(
    r"(你(一定|肯定)(有|是)|你(已经|患上)(抑郁|焦虑)|[是都]你的错|活该|以牙还牙|报复对方|"
    r"伤害自己|自杀|别告诉|不要告诉|替你联系|已经联系|已通知|已报警|已经举报|https?://|\d{7,})"
)


@dataclass(frozen=True)
class SupportPlan:
    goal: str
    approach: tuple[str, ...]
    steps: tuple[dict[str, str], ...]


def _allowed_text(value, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit and not BLOCKED_SUPPORT.search(value)


def validate_strategy(payload: dict, allowed: tuple[str, ...]) -> SupportPlan:
    if set(payload) != {"goal", "approach", "steps"} or not _allowed_text(payload.get("goal"), 180):
        raise LLMUnavailable("strategy_goal_or_fields_rejected")
    approach, steps = payload["approach"], payload["steps"]
    if not isinstance(approach, list) or not 1 <= len(approach) <= len(allowed) or any(action not in allowed for action in approach):
        raise LLMUnavailable("strategy_approach_outside_allowed_actions")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 4:
        raise LLMUnavailable("strategy_step_count_rejected")
    for step in steps:
        if (not isinstance(step, dict) or set(step) != {"action", "text"}
                or step["action"] not in approach or not _allowed_text(step["text"], 180)):
            raise LLMUnavailable("strategy_step_outside_allowed_actions")
    if "人工复核" in allowed and ("及时求助" not in approach or "人工复核" not in approach):
        raise LLMUnavailable("high_risk_strategy_needs_review_and_assistance")
    return SupportPlan(payload["goal"], tuple(approach), tuple(steps))


def generate_support(
    risk: RiskResult, qwen: QwenAdapter | None, context: dict, *, unavailable_reason: str = "qwen_not_configured"
) -> RiskResult:
    if risk.risk_level is None:
        return risk
    index = LEVELS.index(risk.risk_level)
    reason = unavailable_reason
    if qwen is not None:
        try:
            prompt = {**context, "risk_level": risk.risk_level, "allowed_actions": risk.allowed_strategies}
            plan = validate_strategy(qwen.draft_strategy(prompt), risk.allowed_strategies)
            reply = qwen.draft_empathy({
                **context, "risk_level": risk.risk_level,
                "strategy": {"goal": plan.goal, "approach": plan.approach, "steps": plan.steps},
            })
            if not _allowed_text(reply, 380) or (risk.human_review_required and "成年人" not in reply):
                raise LLMUnavailable("empathy_safety_validation_rejected")
            return replace(
                risk, support_source=SOURCE, support_status="available",
                support_strategy={"goal": plan.goal, "approach": plan.approach, "steps": plan.steps},
                empathy_reply=reply,
            )
        except LLMUnavailable as error:
            reason = str(error)
    return replace(
        risk, support_source="safe_template", support_status=reason, empathy_reply=TEMPLATES[index],
        support_strategy={
            "goal": risk.risk_level, "approach": risk.allowed_strategies,
            "steps": tuple({"action": action, "text": action} for action in risk.allowed_strategies),
        },
    )
