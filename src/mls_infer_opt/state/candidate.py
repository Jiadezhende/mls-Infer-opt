"""candidate — generate 产出的 engine 候选（相当于训练的一次 train step 产物）。

候选是内存对象图里的一个节点：诞生时只有「不可变产生事实」（id/kind/parent…），评测结果
（gate/bench）随生命周期后填、**直接长在候选身上**——不走外键、不另起并行表。
「现在处于哪一步」由 gate/bench/best 的存在性派生（见 loop.candidate_status），不存字段。

落盘约定：每个候选有独立工作目录 ``runs/{run_id}/candidates/{cand_id}/``，其中
- ``engine.py``    候选源码（完整自包含纯 PyTorch）
- ``policy.json``  采用的 policy
源码与 policy 都落盘、不进 struct——否则全量候选会常驻内存并灌爆 report JSON。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from .common import utcnow_iso
from .eval import BenchResult, GateResult

__all__ = [
    "CandidateKind",
    "make_candidate_id",
    "candidate_dir",
    "candidate_engine_path",
    "candidate_policy_path",
    "Candidate",
]

# 用 Literal 让 mypy 卡住非法值。
# 生成走「agent 自带工具自闭环」：每次 generate 调用产 1 个候选（agent 内部已用 quick gate 自检
# 收敛，那些修复不单列候选）。repair 仅指**外层 full gate 没过、loop 再喊 agent 调**的那次重试。
CandidateKind = Literal["baseline", "optimization", "repair"]


def make_candidate_id(round_index: int, code: str) -> str:
    """稳定可追踪的候选 id：``r{round}-{sha1(code)[:8]}``。

    决定性（只依赖 round + 代码内容），便于内容去重与审计；同一份代码在同一轮恒等同 id，
    可 O(1) 判「这段代码是否已评测过」从而跳过重复评测。generate 在写盘前用它定 id。
    """
    digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:8]
    return f"r{round_index}-{digest}"


def candidate_dir(run_dir: str, candidate_id: str) -> str:
    """候选工作目录：``runs/{run_id}/candidates/{cand_id}``。run_dir 来自 TaskContext.run_dir。"""
    return f"{run_dir}/candidates/{candidate_id}"


def candidate_engine_path(run_dir: str, candidate_id: str) -> str:
    """候选源码落点；evaluate 从这里 import 来跑。"""
    return f"{candidate_dir(run_dir, candidate_id)}/engine.py"


def candidate_policy_path(run_dir: str, candidate_id: str) -> str:
    """候选采用的 policy 落点（审计 / report 用）。"""
    return f"{candidate_dir(run_dir, candidate_id)}/policy.json"


@dataclass
class Candidate:
    """一份 engine 候选的内存节点（源码在 candidate_engine_path，不进 struct）。

    诞生即定的不可变事实：它是什么 kind、第几轮、从哪个 parent 改来。
    评测结果 gate/bench 随生命周期后填，直接挂在本对象上（对象图，非外键）。
    """

    id: str  # 内容指纹 + 落盘目录名 + 序列化/日志句柄（非主键——内存里直接持引用，不靠它 join）
    kind: CandidateKind
    round: int = 0
    parent_id: str | None = None  # 仅 kind=="baseline" 为 None，其余必填；字符串以便序列化
    # 轻量策略摘要，给 analyze / report 用；完整 policy 在 policy.json。
    strategy_tags: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    # 评测结果随生命周期后填（对象图，非外键 / 非并行表）。
    # gate：外层权威 full gate（agent 内部 quick 自检 ephemeral，不进此处）。
    gate: GateResult | None = None
    bench: BenchResult | None = None  # 仅当 gate.passed 后才填（用 attach_bench 守护）
    extra: dict[str, Any] = field(default_factory=dict)

    def attach_bench(self, bench: BenchResult) -> None:
        """挂性能结果。守护设计点 #3：未过正确性门的候选不该有性能结果。"""
        if self.gate is None or not self.gate.passed:
            raise ValueError(
                f"candidate {self.id} has no passing gate; "
                "performance is only meaningful after correctness passes"
            )
        self.bench = bench
