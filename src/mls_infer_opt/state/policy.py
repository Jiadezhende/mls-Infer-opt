"""policy — Policy 数据结构：搜索空间里的一个点（共享契约，analyze 产 / generate 消费）。

Policy 是 analyze（产出下一个点）与 generate（渲成 engine.py）之间交换的稳定结构，因此类型
落在 state 层——让 analyze 不必 import generate 即可产 Policy。

这里**只放纯 dataclass**：键齐全、恒合法的 axes/knobs + 血缘/审计字段。聚合/消解/序列化等
依赖搜索空间（generate.space / compat）的逻辑仍在 generate.policy（aggregate/merge/to_json…）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidate import CandidateKind

__all__ = ["Policy"]


@dataclass
class Policy:
    """搜索空间里的一个点：完整决定一份 engine。落盘为 candidate_policy_path 的 policy.json。

    ``axes`` 键齐全且已消解冲突（恒合法）；``knobs`` 只含被激活轴的参数。
    其余为血缘/审计字段，不影响渲染语义。
    """

    axes: dict[str, str]
    knobs: dict[str, Any] = field(default_factory=dict)
    kind: CandidateKind = "baseline"
    round: int = 0
    parent_id: str | None = None  # 仅 baseline 为 None
    notes: str = ""
    # analyze 给 generate 的上下文（瓶颈/方向/注意点）；属审计字段、不影响渲染语义
    # （engine 仍由 axes/knobs 唯一决定），仅供 prompt 注入。
    rationale: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
