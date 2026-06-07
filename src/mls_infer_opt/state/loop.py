"""loop — 驱动循环这层的状态与决策载体。

聚合 best/history/budget 的主状态（LoopState）、实时预算（BudgetUsage）、事件流（AgentEvent）。

analyze 每轮的「下一步」不再单列结构：它直接产出下一个 Policy（带 rationale，见 generate.policy），
判停时返回 None、停因落到 LoopState.stop_reason。

LoopState 是整个 run 唯一的主状态实例，贯穿全程；候选构成内存对象图，gate/bench 直接挂在
各 Candidate 上（不再有 gate_results/bench_results 并行 dict）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .candidate import Candidate
from .common import utcnow_iso
from .context import TaskContext

__all__ = [
    "EventLevel",
    "BudgetUsage",
    "AgentEvent",
    "LoopState",
    "candidate_status",
]

EventLevel = Literal["info", "warning", "error"]


@dataclass
class BudgetUsage:
    """实时预算消耗（对照 TaskContext.limits 判停）。"""

    started_at: str = field(default_factory=utcnow_iso)
    elapsed_s: float = 0.0
    llm_calls: int = 0
    eval_runs: int = 0  # correctness/benchmark 实际跑的次数
    tokens_in: int | None = None
    tokens_out: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """append-only 结构化事件。report 直接消费，调试可重放。"""

    source: str  # 产生事件的模块：loop | generate | evaluate | analyze
    phase: str
    message: str
    level: EventLevel = "info"
    candidate_id: str | None = None
    ts: str = field(default_factory=utcnow_iso)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopState:
    """Orchestrator(loop) 持有的主状态——整个 run 唯一实例，贯穿全程。

    candidates 是内容寻址的候选表（id = code 哈希，用于去重 / 落盘 / 序列化），gate/bench 直接挂在
    各 Candidate 上，不另起并行表。不变量：best_id 一旦设置即指向一个 gate.passed 的候选，且永不
    退化为更差或 None——用 set_best() 而非裸赋值来维持。
    """

    task_context: TaskContext = field(default_factory=TaskContext)
    round: int = 0
    candidates: dict[str, Candidate] = field(default_factory=dict)
    best_id: str | None = None
    best_score: float = float("-inf")
    stale_rounds: int = 0  # 连续无提升轮数，analyze 判停输入
    budget: BudgetUsage = field(default_factory=BudgetUsage)
    events: list[AgentEvent] = field(default_factory=list)
    stop_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # —— 轻量不变量守护（非业务决策；谁更优/要不要停在 loop/analyze 决定）——

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def set_best(self, candidate: Candidate, score: float) -> None:
        """把候选登记为 best。守护要点 #2：只接受已过门候选。

        是否「更优」由 loop 在调用前判断；本方法只保证不变量（过门 + 不为 None）。
        """
        if candidate.gate is None or not candidate.gate.passed:
            raise ValueError(
                f"refuse to set best to {candidate.id}: must pass correctness gate first"
            )
        self.best_id = candidate.id
        self.best_score = score

    def best_candidate(self) -> Candidate | None:
        return self.candidates.get(self.best_id) if self.best_id else None

    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)


def candidate_status(state: LoopState, candidate_id: str) -> str:
    """从 gate/bench/best 派生候选当前处于哪一步——单一真相源，Candidate 不存此字段。

    取值：``proposed``（未评测）→ ``rejected``（未过门）/ ``gated``（过门未测速）→
    ``measured``（已测速）→ ``promoted``（当前 best）。
    """
    cand = state.candidates.get(candidate_id)
    if cand is None or cand.gate is None:
        return "proposed"
    if not cand.gate.passed:
        return "rejected"
    if cand.bench is None:
        return "gated"
    if state.best_id == candidate_id:
        return "promoted"
    return "measured"
