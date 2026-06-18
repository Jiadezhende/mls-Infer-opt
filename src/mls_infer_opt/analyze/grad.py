"""grad — analyze 的编排入口：汇总态势 → 给方向（或无方向）→ 产下一个 Policy，并每轮记一条事件。

每轮循环的「大脑」=求梯度：只算「往哪走」。问 LLM 要方向（单次调用 + 确定性解析，见 prompt.py），
LLM 不可用 / 失败 / 产垃圾就退回 rule-based 阶梯（heuristic.py）。

不变量纪律：
- **只算方向、不判停**：analyze 返回 ``Policy``（迈出的一步）或 ``NoMove(reason)``（gradient≈0，
  迈不出）。**硬上限判停（预算 / 轮数 / 连续无提升）不在这里**——那是总控的循环准则（见 loop）。
- **never-throw**：内部任何异常都翻成一条 error 事件 + 返回 ``NoMove("analyze_error")``。
- **无发布权 / 不写 stop_reason**：每轮 emit 一条 ``source="analyze"`` 事件作记录（见
  [[analyze-record-via-events]]）；是否因 NoMove 终止、停因落 ``LoopState.stop_reason``，由总控定。
- 下一个 Policy 由 ``searchspace.policy.merge(best_policy, axes_delta=…, …, rationale=…)`` 构造，
  knob 只进 Policy.knobs，绝不碰 model_config（merge/aggregate 焊死）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..llm.errors import LLMError  # 仅 errors（零依赖），避免经 llm/__init__ 拉入 present 导入环
from ..searchspace.policy import aggregate, default_policy, from_json, merge
from ..state.candidate import candidate_policy_path
from ..state.loop import LoopState, emit
from ..state.policy import NoMove, Policy
from .heuristic import Decision, heuristic_decision
from .prompt import build_analyze_prompt, parse_decision
from .situation import Situation, build_situation

__all__ = ["LLMClient", "analyze"]


class LLMClient(Protocol):
    """analyze 需要的最小 LLM 接口；与 generate 一致，统一走 run_agent。

    ``available`` 为假或调用抛错 / 返回空时，analyze 退回 rule-based（不抛异常）。
    """

    available: bool

    def run_agent(
        self, prompt: str, tools: list[Any] | None = ..., **kwargs: Any
    ) -> Any: ...


def analyze(state: LoopState, *, llm: LLMClient | None = None) -> Policy | NoMove:
    """只算方向：产下一个 Policy，或 NoMove(reason)（迈不出步）。每轮 emit 一条 analyze 事件。

    硬上限判停不在这里——是总控的循环准则。never-throw：内部异常 → 记错 + NoMove("analyze_error")。
    """
    try:
        sit = build_situation(state)
    except Exception as e:  # 连态势都建不起来：无方向可算，记错交总控。
        emit(state, source="analyze", phase="grad", message="无方向：analyze 建态势异常",
             level="error",
             data={"detail": f"analyze crashed building situation: {e}",
                   "decision": "stop", "stop_reason": "analyze_error"})
        return NoMove("analyze_error")

    # 1. 方向：LLM 优先，rule-based 兜底。
    best_policy = _load_best_policy(state)
    used_llm = False
    decision: Decision | None = None
    if llm is not None and getattr(llm, "available", False):
        decision = _ask_llm(sit, best_policy, llm)
        used_llm = decision is not None
    if decision is None:
        decision = heuristic_decision(sit, best_policy)

    # 2. 无方向（LLM 判定到位 / 阶梯走完）→ NoMove，交总控裁决是否终止。
    if decision.action == "stop":
        reason = decision.stop_reason or "no_direction"
        emit(state, source="analyze", phase="grad", message=f"无方向：{reason}",
             data={"detail": decision.bottleneck, "decision": "stop", "stop_reason": reason,
                   "used_llm": used_llm, **_situation_data(sit)})
        return NoMove(reason)

    # 3. 从 best 出发叠 delta → 合法的下一个 Policy。
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
        emit(state, source="analyze", phase="grad", message="无方向：analyze 构造 Policy 异常",
             level="error",
             data={"detail": f"analyze crashed building policy: {e}",
                   "decision": "stop", "stop_reason": "analyze_error"})
        return NoMove("analyze_error")

    emit(
        state,
        source="analyze",
        phase="grad",
        message=f"继续：{decision.bottleneck or '局部搜索下一步'}",
        data={
            "detail": decision.rationale,
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
def _ask_llm(sit: Situation, best_policy: Policy, llm: LLMClient) -> Decision | None:
    """问 LLM 要方向。C2（LLMError）穿透交总控；其它意外 → None（回 rule-based）。

    内容层失败（run_agent ok=False / 解析不出 Decision）由 _call_llm/parse_decision 返回 None，
    属 C1 邻域、回退规则；只有传输/基建失败（run_agent raise LLMCallError）才向上穿透。
    """
    try:
        prompt = build_analyze_prompt(sit, best_policy)
        text = _call_llm(llm, prompt)
    except LLMError:
        raise  # C2：基建失败，穿透到总控的循环边界
    except Exception:
        return None  # 其它非预期 → 当作没要到方向，回 rule-based
    return parse_decision(text)


def _call_llm(llm: LLMClient, prompt: str) -> str | None:
    """走 run_agent 取 .text；无 run_agent 或 ok=False → None（analyze 退回 rule-based）。"""
    runner = getattr(llm, "run_agent", None)
    if not callable(runner):
        return None
    result = runner(prompt)
    if not getattr(result, "ok", False):
        return None
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else None


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
    from ..searchspace.policy import strategy_tags

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


