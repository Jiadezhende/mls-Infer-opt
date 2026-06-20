"""gradient — Gradient 数据结构：analyze 的一步方向（共享契约，analyze 产 / generate 消费）。

每轮循环的「大脑」=求梯度（见 analyze/grad.py）：analyze 看态势，产出 ``Gradient``（迈出的一步）
或 ``NoMove``（gradient≈0，迈不出）。Gradient 是 analyze↔generate 之间交换的稳定结构，类型落在
state 层（最底层契约），analyze 与 generate 都能直接引用、互不横向 import。

刻意**不是**搜索空间里的恒合法定点：``suggest_axes`` 只是相对 best 的**松建议**（已过 searchspace
词表闸 sanitize、丢未知/非法），generate 的 agent 在搜索维度界内自由探索、不被它强制约束（外层
full gate 才是唯一权威）。实际采用了哪些轴由 agent 回报、落在 candidate.strategy_tags（见
generate.codegen），不由本结构决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidate import CandidateKind

__all__ = ["Gradient", "NoMove"]


@dataclass(frozen=True)
class NoMove:
    """analyze 的「无方向」结果（gradient≈0）：搜索空间走到头 / LLM 判定到位 / analyze 内部出错。

    analyze 的返回是 ``Gradient | NoMove``——``Gradient`` 是迈出的一步，``NoMove`` 表示这一步迈不出，
    并带上 ``reason`` 交总控。analyze **不判停、不写 stop_reason**：要不要因此终止、停因落到
    ``LoopState.stop_reason``，由总控（loop）裁决（停止是训练循环的准则，不是 gradient 的活）。
    """

    reason: str


@dataclass
class Gradient:
    """analyze 给 generate 的一步方向：往哪走（松建议）+ 为什么（rationale）+ 血缘。

    ``suggest_axes`` 是相对 best 的松建议（axis→option，已过词表闸），``knobs`` 是配套参数建议；
    二者都**不恒合法、不定点**——generate 的 agent 看完整搜索维度自由探索，建议只是优先方向。
    ``rationale``/``bottleneck`` 装瓶颈/思路/注意点的自然语言，渲进生成 prompt。其余为血缘字段。
    """

    suggest_axes: dict[str, str] = field(default_factory=dict)
    knobs: dict[str, Any] = field(default_factory=dict)
    kind: CandidateKind = "optimization"
    round: int = 0
    parent_id: str | None = None  # 仅 baseline 为 None
    bottleneck: str = ""
    rationale: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
