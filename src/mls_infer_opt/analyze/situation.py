"""situation — 把 LoopState 的对象图汇总成一份**当前态势**（analyze 的输入视图）。

``Situation`` 是 ephemeral 计算视图（类比 generate 的 ``AggregateResult``）：只在 analyze 内部
传递、**绝不进 state / 不序列化**。它把诊断/判停/构 prompt 都要看的派生量算一遍，避免在
grad 与 prompt 两处各自重算 LoopState。

纯逻辑、只读：所有数字都从 LoopState 对象图派生（best/candidates/budget/limits），不落盘、
不改 state（见 [[contracts-are-object-graphs]] / [[analyze-record-via-events]]）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..searchspace.space import AXIS_BY_KEY
from ..state.candidate import Candidate
from ..state.eval import BenchResult, ValidationError
from ..state.loop import LoopState, candidate_status

__all__ = ["Situation", "build_situation"]

# 收集最近失败 / 历史时只看尾部若干条，prompt 不被灌爆。
_RECENT_FAILURES = 4


@dataclass
class RoundEntry:
    """单个候选在历史里的一行摘要（按轮聚合给 prompt 看）。"""

    round: int
    candidate_id: str
    status: str  # candidate_status: proposed/rejected/gated/measured/promoted
    score: float | None
    strategy_tags: list[str] = field(default_factory=list)


@dataclass
class Situation:
    """当前态势：analyze 据此定位瓶颈 / 判停 / 给方向。ephemeral，不进 state。"""

    round: int
    # —— best（永远存在，bootstrap 后；防御性允许 None）——
    best_id: str | None
    best_score: float
    best_axes: dict[str, str]
    best_strategy_tags: list[str]
    best_bench: BenchResult | None  # 对象引用，非副本
    applied_axes: dict[str, str]  # best 里取了非默认选项的轴（greedy ladder 据此跳过）
    # —— 进度 / 失败 ——
    stale_rounds: int
    n_candidates: int
    n_rejected: int
    recent_failures: list[ValidationError]  # 近期 rejected 候选的 gate 错误（诊断正确性边界）
    history: list[RoundEntry]
    # —— 预算（对照 limits 判停）——
    elapsed_s: float
    time_budget_s: int
    max_rounds: int
    max_stale_rounds: int


def _applied_axes(axes: dict[str, str]) -> dict[str, str]:
    """best 里取了非默认选项的轴（默认 = baseline 行为，不算「已应用」）。"""
    out: dict[str, str] = {}
    for key, value in axes.items():
        spec = AXIS_BY_KEY.get(key)
        if spec is not None and value != spec.default:
            out[key] = value
    return out


def build_situation(state: LoopState) -> Situation:
    """从 LoopState 对象图派生当前态势。纯函数、只读，不改 state。"""
    best = state.best_candidate()
    best_axes = _best_axes(state, best)

    history: list[RoundEntry] = []
    recent_failures: list[ValidationError] = []
    n_rejected = 0
    # 按轮、再按 id 稳定排序，保证 prompt / 测试可复现。
    for cid, cand in sorted(state.candidates.items(), key=lambda kv: (kv[1].round, kv[0])):
        status = candidate_status(state, cid)
        if status == "rejected":
            n_rejected += 1
        score = cand.bench.score if cand.bench is not None else None
        history.append(
            RoundEntry(
                round=cand.round,
                candidate_id=cid,
                status=status,
                score=score,
                strategy_tags=list(cand.strategy_tags),
            )
        )

    # 近期失败：取最后产生的几个 rejected 候选的结构化错误。
    for cand in sorted(
        (c for c in state.candidates.values() if candidate_status(state, c.id) == "rejected"),
        key=lambda c: (c.round, c.id),
    )[-_RECENT_FAILURES:]:
        if cand.gate is not None:
            recent_failures.extend(cand.gate.errors)

    limits = state.task_context.limits
    return Situation(
        round=state.round,
        best_id=state.best_id,
        best_score=state.best_score,
        best_axes=best_axes,
        best_strategy_tags=list(best.strategy_tags) if best is not None else [],
        best_bench=best.bench if best is not None else None,
        applied_axes=_applied_axes(best_axes),
        stale_rounds=state.stale_rounds,
        n_candidates=len(state.candidates),
        n_rejected=n_rejected,
        recent_failures=recent_failures,
        history=history,
        elapsed_s=state.budget.elapsed_s,
        time_budget_s=limits.time_budget_s,
        max_rounds=limits.max_rounds,
        max_stale_rounds=limits.max_stale_rounds,
    )


def _best_axes(state: LoopState, best: Candidate | None) -> dict[str, str]:
    """best 候选的完整 axes。

    Candidate 只存 strategy_tags（非默认轴的 ``axis:value``）；完整 axes 在 policy.json。这里
    只需「哪些轴取了什么」给 ladder 跳过用，从 strategy_tags 还原即可（默认轴留给 baseline_axes
    在 grad 里补全），无需读盘。
    """
    from ..searchspace.space import baseline_axes

    axes = baseline_axes()
    if best is None:
        return axes
    for tag in best.strategy_tags:
        key, sep, value = tag.partition(":")
        if sep and key in AXIS_BY_KEY:
            axes[key] = value
    return axes
