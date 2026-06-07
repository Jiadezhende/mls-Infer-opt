"""grad — analyze 的编排入口：汇总态势 → 判停 / 给方向 → 产下一个 Policy，并每轮记一条事件。

每轮循环的「大脑」。控制权全程在确定性代码里：先看硬上限判停，再问 LLM 要方向（单次调用 +
确定性解析，见 prompt.py），LLM 不可用 / 失败 / 产垃圾就退回 rule-based 阶梯（heuristic.py）。

不变量纪律：
- **never-throw**：任何异常都翻成一条 error 事件 + 返回 None（= 停，best 已是安全产物）。
- **无发布权 / 不判定 stop_reason**：analyze 只产 Policy 或 None，并 **每轮** emit 一条
  ``source="analyze"`` 事件（见 [[analyze-record-via-events]]）；停因放进事件
  ``data["stop_reason"]``，由 loop 读后落到 ``LoopState.stop_reason`` 收尾——analyze **绝不**自写。
- 下一个 Policy 由 ``generate.policy.merge(best_policy, axes_delta=…, …, rationale=…)`` 构造，
  knob 只进 Policy.knobs，绝不碰 model_config（merge/aggregate 焊死）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..generate.policy import aggregate, default_policy, from_json, merge
from ..state.candidate import candidate_policy_path
from ..state.loop import AgentEvent, EventLevel, LoopState
from ..state.policy import Policy
from .heuristic import Decision, hard_stop_reason, heuristic_decision
from .prompt import build_analyze_prompt, parse_decision
from .situation import Situation, build_situation

__all__ = ["LLMClient", "analyze"]


class LLMClient(Protocol):
    """analyze 需要的最小 LLM 接口；与 generate 一致：优先 run_agent，兼容旧 generate(prompt)。

    ``available`` 为假或调用抛错 / 返回空时，analyze 退回 rule-based（不抛异常）。
    """

    available: bool

    def generate(self, prompt: str) -> str | None: ...


def analyze(state: LoopState, *, llm: LLMClient | None = None) -> Policy | None:
    """看反馈定方向，产下一个 Policy（带 rationale）或 None（停）。每轮 emit 一条 analyze 事件。

    never-throw：内部任何异常都记一条 error 事件并返回 None（停，best 兜底）。
    """
    try:
        sit = build_situation(state)
    except Exception as e:  # 连态势都建不起来：保守停机，记错。
        _emit(state, "判停：analyze 建态势异常", f"analyze crashed building situation: {e}",
              level="error", data={"decision": "stop", "stop_reason": "analyze_error"})
        return None

    # 1. 硬上限判停（确定性，优先于方向）。
    hard = hard_stop_reason(sit)
    if hard is not None:
        _emit(state, f"判停：{hard}", "", level="info",
              data={"decision": "stop", "stop_reason": hard, **_situation_data(sit)})
        return None

    # 2. 方向：LLM 优先，rule-based 兜底。
    best_policy = _load_best_policy(state)
    used_llm = False
    decision: Decision | None = None
    if llm is not None and getattr(llm, "available", False):
        decision = _ask_llm(state, sit, best_policy, llm)
        used_llm = decision is not None
    if decision is None:
        decision = heuristic_decision(sit, best_policy)

    # 3. analyze（或 LLM）主动判停。
    if decision.action == "stop":
        reason = decision.stop_reason or "analyze_decided_stop"
        _emit(state, f"判停：{reason}", decision.bottleneck, level="info",
              data={"decision": "stop", "stop_reason": reason, "used_llm": used_llm,
                    **_situation_data(sit)})
        return None

    # 4. 从 best 出发叠 delta → 合法的下一个 Policy。
    try:
        agg = merge(
            best_policy,
            axes_delta=decision.axes_delta,
            knobs_delta=decision.knobs_delta,
            kind="optimization",
            round=state.round + 1,
            parent_id=state.best_id,
            rationale=decision.rationale,
        )
    except Exception as e:  # merge 是纯逻辑、理论不抛；兜一层守 never-throw。
        _emit(state, "判停：analyze 构造 Policy 异常", f"analyze crashed building policy: {e}",
              level="error", data={"decision": "stop", "stop_reason": "analyze_error"})
        return None

    _emit(
        state,
        f"继续：{decision.bottleneck or '局部搜索下一步'}",
        decision.rationale,
        level="info",
        data={
            "decision": "continue",
            "used_llm": used_llm,
            "axes_delta": decision.axes_delta,
            "knobs_delta": decision.knobs_delta,
            "bottleneck": decision.bottleneck,
            "fixes": agg.fixes,
            "next_strategy_tags": _nondefault_tags(agg.policy),
            **_situation_data(sit),
        },
    )
    return agg.policy


# === 内部 =============================================================
def _ask_llm(
    state: LoopState, sit: Situation, best_policy: Policy, llm: LLMClient
) -> Decision | None:
    """问 LLM 要方向；never-throw（LLM 基建已承诺不抛，这里再兜一层）→ 失败返回 None。"""
    try:
        prompt = build_analyze_prompt(sit, best_policy)
        text = _call_llm(llm, prompt)
    except Exception:
        return None
    return parse_decision(text)


def _call_llm(llm: LLMClient, prompt: str) -> str | None:
    """优先新 run_agent（取 .text），兼容旧 generate(prompt)。"""
    runner = getattr(llm, "run_agent", None)
    if callable(runner):
        result = runner(prompt)
        if not getattr(result, "ok", False):
            return None
        text = getattr(result, "text", None)
        return text if isinstance(text, str) else None
    return llm.generate(prompt)


def _load_best_policy(state: LoopState) -> Policy:
    """取 best 候选的完整 Policy 作局部搜索的起点。

    真相在候选目录的 policy.json（含 knobs）；读盘失败 / 无 best 时优雅降级：先按 best 的
    strategy_tags 聚合还原，再退到全默认 baseline（不变量 #2：永远有可用起点）。
    """
    best = state.best_candidate()
    if best is not None:
        path = candidate_policy_path(state.task_context.run_dir, best.id)
        try:
            return from_json(Path(path).read_text(encoding="utf-8"))
        except Exception:
            pass
        # 降级：从 strategy_tags（axis:value）聚合还原（丢 knobs，但 axes 够 merge）。
        raw_axes: dict[str, str] = {}
        for tag in best.strategy_tags:
            key, sep, value = tag.partition(":")
            if sep:
                raw_axes[key] = value
        if raw_axes:
            return aggregate(raw_axes, kind="optimization", parent_id=best.id).policy
    return default_policy(round=state.round)


def _nondefault_tags(policy: Policy) -> list[str]:
    """下一个 Policy 的非默认轴摘要（记进事件，便于 report 还原「这轮为何这么走」）。"""
    from ..generate.policy import strategy_tags

    return strategy_tags(policy)


def _situation_data(sit: Situation) -> dict[str, Any]:
    """事件里随手带的态势数字（结构化，report 直接消费）。"""
    return {
        "round": sit.round,
        "best_id": sit.best_id,
        "best_score": sit.best_score,
        "stale_rounds": sit.stale_rounds,
        "n_candidates": sit.n_candidates,
        "n_rejected": sit.n_rejected,
    }


def _emit(
    state: LoopState,
    message: str,
    detail: str,
    *,
    level: EventLevel = "info",
    data: dict[str, Any] | None = None,
) -> None:
    """append 一条 analyze 事件。**每轮必发**（判停轮 / 无收益轮也发）。绝不写 stop_reason。"""
    payload = dict(data or {})
    if detail:
        payload.setdefault("detail", detail)
    state.add_event(
        AgentEvent(source="analyze", phase="grad", message=message, level=level, data=payload)
    )
