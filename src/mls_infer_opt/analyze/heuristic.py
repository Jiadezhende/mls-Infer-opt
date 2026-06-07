"""heuristic — 确定性决策：判停（hard stop）+ rule-based 方向（greedy ladder）。

LLM 是「可选增益」：不可用 / 失败时 analyze 退化到这里，靠优化主线先验做局部搜索，**不抛异常、
仍给出合法方向或明确停因**。也提供 ``Decision`` 这个 analyze 内部的共享决策类型（LLM 解析与
rule-based 都产出它，由 grad 统一消费）。

判停优先于方向：先看硬上限（预算 / 轮数 / 连续无提升）是否已到，再谈往哪走。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..generate.space import AXIS_BY_KEY
from ..state.policy import Policy
from .situation import Situation

__all__ = ["Action", "Decision", "hard_stop_reason", "MOVES", "heuristic_decision"]

Action = Literal["continue", "stop"]


@dataclass
class Decision:
    """analyze 单轮决策（LLM 解析 / rule-based 共用）。grad 据此 merge 出 Policy 或判停。

    ``action="continue"`` 时看 ``axes_delta``/``knobs_delta``/``rationale``/``bottleneck``；
    ``action="stop"`` 时看 ``stop_reason``。两者都可带 ``bottleneck``（记进事件）。
    """

    action: Action = "continue"
    axes_delta: dict[str, str] = field(default_factory=dict)
    knobs_delta: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    bottleneck: str = ""
    stop_reason: str = ""


# === 判停（确定性硬门）===============================================
def hard_stop_reason(sit: Situation) -> str | None:
    """到硬上限即返回停因字符串，否则 None。每条仅在对应 limit > 0（已配置）时生效。

    成本/达标类软停（收益不足等）交给 stale_rounds 与 LLM 判断；这里只管不可逾越的硬墙。
    """
    if sit.time_budget_s > 0 and sit.elapsed_s >= sit.time_budget_s:
        return "time_budget_exhausted"
    if sit.max_rounds > 0 and sit.round >= sit.max_rounds:
        return "max_rounds_reached"
    if sit.max_stale_rounds > 0 and sit.stale_rounds >= sit.max_stale_rounds:
        return "max_stale_rounds_reached"
    return None


# === rule-based 方向（贪心阶梯）======================================
@dataclass(frozen=True)
class _Move:
    """阶梯上的一步：把某条轴从 baseline 抬到下一个目标选项。"""

    axis: str
    option: str
    why: str


# 优化主线先验的贪心阶梯，**前置依赖在前、低风险(🟢)优先**：
# - kv_cache 收益最大（baseline 每步 decode 重算整段，O(n²)）；
# - decode_batch=batched 依赖 kv_cache、gqa=enable_gqa 依赖 attention=sdpa、
#   qkv/mlp 融合依赖 weight_layout=fused —— 故前置轴必须排在依赖轴之前（否则 compat 会把
#   依赖轴退默认、这一步等于空跑）；
# - 数值敏感(🔴)的 attention/compute_dtype 尽量靠后，先吃结构等价的稳收益。
MOVES: tuple[_Move, ...] = (
    _Move("kv_cache", "incremental", "decode 每步重算整段是最大瓶颈，先上增量 KV 缓存"),
    _Move("rope", "precomputed_table", "RoPE 预算 cos/sin 表按位置切片，省重复三角运算"),
    _Move("prefill_batch", "padded_batch", "prefill 跨 request padding 成批，提升预填吞吐"),
    _Move("decode_batch", "batched", "合批解码一次 forward 摊薄 kernel 启动（依赖 KV 缓存）"),
    _Move("attention", "sdpa", "naive matmul 注意力换 SDPA，省显存带宽与中间物化"),
    _Move("gqa", "enable_gqa", "GQA 不物化扩展 KV，省显存（依赖 SDPA）"),
    _Move("weight_layout", "fused", "加载期把 qkv/gate-up 权重 cat、转置，挪出热路径"),
    _Move("qkv_fusion", "fused_qkv", "q/k/v 投影融合成单次 matmul（依赖 fused 布局）"),
    _Move("mlp_fusion", "fused_gate_up", "gate/up 投影融合（依赖 fused 布局）"),
    _Move("compute_dtype", "bf16", "全局计算精度降到 bf16 提吞吐（数值敏感，最后尝试）"),
)


def _is_default(axes: dict[str, str], axis: str) -> bool:
    """该轴当前是否仍是 baseline 默认（== 尚未应用任何优化）。"""
    spec = AXIS_BY_KEY.get(axis)
    return spec is not None and axes.get(axis, spec.default) == spec.default


def heuristic_decision(sit: Situation, best_policy: Policy) -> Decision:
    """从 best 出发挑阶梯上第一条「尚未应用」的轴作为下一步；阶梯走完即建议停。

    determinstic、不读 LLM：相同 best → 相同下一步，保证可复现与可测。
    """
    for move in MOVES:
        if _is_default(best_policy.axes, move.axis):
            rationale = (
                f"[rule-based] 瓶颈：{move.why}。"
                f"从 best（已应用 {sorted(sit.applied_axes) or '无'}）出发，"
                f"沿优化主线把 {move.axis} 抬到 {move.option}。"
            )
            return Decision(
                action="continue",
                axes_delta={move.axis: move.option},
                rationale=rationale,
                bottleneck=move.why,
            )
    return Decision(
        action="stop",
        stop_reason="no_obvious_direction",
        bottleneck="贪心阶梯已走完，best 已应用主线全部优化轴",
    )
