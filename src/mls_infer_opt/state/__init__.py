"""state — 共享数据契约（数据层）。

定义 agent 内部、以及 agent ↔ 产物之间交换的稳定数据结构。所有业务模块（loop / generate /
evaluate / analyze）只通过这些结构通信，不直接共享内部对象。

设计要点（详见 ../ARCHITECTURE.md 与提交记录）：
1. state 只活在阶段 A，engine.py（阶段 B）**不依赖**它；序列化只为审计 / report / 续跑。
   因此这里只放轻量元数据 + 指标数字——engine 源码 / policy 落盘到候选目录
   （runs/{id}/candidates/{id}/engine.py、policy.json），**不进 struct**，避免全量候选常驻内存
   / 灌爆 report JSON。torch 张量 / 权重 / engine 实例 / logits / LLM client 一律**不进 state**。
2. 「永远有可用 best」靠结构焊死：best_id 在 bootstrap 后不为 None；set_best 只接受已过门候选。
3. 正确性与性能**物理分离**：GateResult（硬布尔门）与 BenchResult（性能）是两个结构，随生命
   周期后填、直接挂在 Candidate 上（candidate.gate / candidate.bench）；只有 gate.passed 的
   候选才会有 bench（Candidate.attach_bench 守护）。不走 candidate_id 外键、不另起并行表。
4. 失败信息结构化、可程序化读取（ValidationError 的 stage/case/max_abs_err/shape/traceback），
   供 generate.repair 与 analyze 消费，不是给人看的字符串。
5. 候选是可追溯的树（id/parent_id/kind）+ append-only 事件流（AgentEvent）；候选不存
   生命周期状态，proposed/gated/measured/promoted 由 gate/bench/best 派生（candidate_status）。
6. 评测昂贵：同一候选不重复评测（candidate.gate 已填即跳过）；reference logits 由 evaluate 跨候选
   复用（oracle 缓存）。候选 id 只是运行内序号、不内容寻址、不做去重。budget 区分静态上限
   （TaskContext.limits）与实时消耗（LoopState.budget）。
7. 向后兼容：可扩展结构带 extra:dict 逃生舱；读取方只依赖已声明字段，写入方只加不改。
8. model_config 的结构字段只读存在于 TaskContext，**绝不**作为可调 knob（knob 只属 Policy.knobs）。

本模块不含业务逻辑——纯结构 + 轻量构造 / 不变量校验 / 序列化。比较「谁更优」「要不要停」等
决策在 loop / analyze，不在这里。

文件划分（按数据流，粗粒度）：
- common.py    跨结构小工具（utcnow_iso / to_dict）
- context.py   只读会话上下文（Paths / Environment / Limits / TaskContext）
- candidate.py generate 产物（Candidate + id 工具）
- eval.py      evaluate 反馈（ValidationError / GateResult / BenchResult）
- policy.py    搜索空间里的一个点（Policy）——analyze↔generate 共享契约
- loop.py      驱动循环层（BudgetUsage / AgentEvent / LoopState）
"""

from __future__ import annotations

from .candidate import (
    Candidate,
    CandidateKind,
    candidate_dir,
    candidate_engine_path,
    candidate_policy_path,
    make_candidate_id,
)
from .common import to_dict, utcnow_iso
from .context import Environment, Limits, Paths, TaskContext
from .eval import BenchResult, EvalMode, GateResult, GateStage, ValidationError
from .loop import AgentEvent, BudgetUsage, EventLevel, LoopState, candidate_status
from .policy import NoMove, Policy

__all__ = [
    # candidate
    "CandidateKind",
    "make_candidate_id",
    "candidate_dir",
    "candidate_engine_path",
    "candidate_policy_path",
    "Candidate",
    # context
    "Paths",
    "Environment",
    "Limits",
    "TaskContext",
    # eval
    "GateStage",
    "EvalMode",
    "ValidationError",
    "GateResult",
    "BenchResult",
    # policy
    "Policy",
    "NoMove",
    # loop
    "EventLevel",
    "BudgetUsage",
    "AgentEvent",
    "LoopState",
    "candidate_status",
    # common
    "utcnow_iso",
    "to_dict",
]
