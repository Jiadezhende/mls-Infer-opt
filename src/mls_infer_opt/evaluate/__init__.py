"""evaluate — 评测候选，产出反馈信号（相当于训练的 eval）。

唯一可靠的反馈来源。对一个候选给出两层信号，决定它能否留下、好不好：

1. 正确性（硬约束 / gate）：syntax → api 契约 → 对官方 reference model 比 logits
   allclose(atol=1e-2, rtol=1e-2)，覆盖 single/multi prefill+decode、插入新请求、
   remove 后继续 decode。不过则该候选作废（性能分为 0）。ground truth 复用
   inference-core-ref/evaluator/{reference_model,test_correctness}.py。
2. 性能（分数）：prefill/decode/mixed 三类吞吐 + 峰值显存。计时只覆盖 prefill/decode/remove，
   不含 create_engine/权重加载。复用 inference-core-ref/evaluator/benchmark_throughput.py 口径。

输出要既能给 loop 做 keep-best 比较（归一化 score），又能给 analyze 定位问题
（结构化失败原因：stage/case/max_abs_err/shape/traceback；分项吞吐与显存）。

约定：
- 决定性、可复现、不调 LLM。
- 有次数预算并复用同一候选已有结果，避免重复跑昂贵评测；分 quick（循环内）/ full（发布前）。
- 只有通过正确性的候选才进入性能评测。

产出：填 candidate.gate / candidate.bench。依赖：state + vendored 参考实现（assets/）。

实现结构（子进程隔离，父进程零 torch）：
- 父进程（本文件 + runner + protocol，纯 stdlib）：编排 / spawn / 收口，**绝不抛异常给 loop**。
- 子进程（worker + gate + bench + oracle + cases，torch）：所有评测与风险代码，坏候选只死子进程。
"""

from __future__ import annotations

from ..state.candidate import Candidate, candidate_engine_path
from ..state.context import TaskContext
from ..state.eval import BenchResult, EvalMode, GateResult, ValidationError
from .protocol import JobSpec, bench_from_dict, gate_from_dict
from .runner import run_job

__all__ = ["evaluate", "run_gate", "run_bench", "quick_gate"]

# 正确性用固定 seed，保证 oracle expected 与候选输入的确定性可复现。
_DEFAULT_SEED = 7


def _oracle_cache_path(ctx: TaskContext, mode: EvalMode) -> str:
    """reference logits 缓存落点：本次 run 内、按 mode 区分，跨候选复用。"""
    return f"{ctx.run_dir}/oracle_cache_{mode}.pt"


def _build_spec(engine_path: str, ctx: TaskContext, mode: EvalMode, task: str) -> JobSpec:
    return JobSpec(
        engine_path=engine_path,
        weight_dir=ctx.weight_dir,
        model_config=ctx.model_config,
        device=ctx.device,
        mode=mode,
        task=task,  # type: ignore[arg-type]
        seed=_DEFAULT_SEED,
        oracle_cache_path=_oracle_cache_path(ctx, mode),
    )


def _runtime_gate(message: str) -> GateResult:
    """父进程编排自身异常时的兜底 gate（永远有结构化结果）。"""
    return GateResult(errors=[ValidationError(stage="runtime", message=message)])


def run_gate(
    engine_path: str, ctx: TaskContext, mode: EvalMode = "full", *, timeout_s: float | None = None
) -> GateResult:
    """只跑正确性门，返回 GateResult。永不抛——失败都落 GateResult.errors。"""
    try:
        result = run_job(_build_spec(engine_path, ctx, mode, "gate"), timeout_s)
        gate = result.get("gate")
        return gate_from_dict(gate) if gate else _runtime_gate("worker returned no gate")
    except Exception as e:  # 编排层最后一兜，绝不漏给 loop
        return _runtime_gate(f"evaluate.run_gate crashed: {e}")


def run_bench(
    engine_path: str, ctx: TaskContext, mode: EvalMode = "full", *, timeout_s: float | None = None
) -> BenchResult:
    """只跑性能 benchmark，返回 BenchResult（调用方需自行保证候选已过 gate）。永不抛。"""
    try:
        result = run_job(_build_spec(engine_path, ctx, mode, "bench"), timeout_s)
        bench = result.get("bench")
        return bench_from_dict(bench) if bench else BenchResult(mode=mode)
    except Exception as e:
        return BenchResult(mode=mode, warnings=[f"evaluate.run_bench crashed: {e}"])


def quick_gate(
    engine_path: str, ctx: TaskContext, *, timeout_s: float | None = None
) -> GateResult:
    """quick 正确性门——供 generate 的 agent 包成内层自检工具（方案1，ephemeral，不进 state）。"""
    return run_gate(engine_path, ctx, "quick", timeout_s=timeout_s)


def evaluate(
    candidate: Candidate,
    ctx: TaskContext,
    mode: EvalMode = "full",
    *,
    timeout_s: float | None = None,
) -> Candidate:
    """评测候选：gate → 过了才 bench → 挂到 candidate（对象图，非外键）→ 返回同一 candidate。

    幂等（candidate.gate 已存在则直接返回，对应「评测昂贵、复用结果」）。全程 never-throw：
    编排层任何异常都翻成 failed runtime gate，绝不让 loop 崩（不变量 #3）。
    """
    if candidate.gate is not None:
        return candidate

    try:
        engine_path = candidate_engine_path(ctx.run_dir, candidate.id)
        result = run_job(_build_spec(engine_path, ctx, mode, "both"), timeout_s)

        gate_d = result.get("gate")
        candidate.gate = (
            gate_from_dict(gate_d) if gate_d else _runtime_gate("worker returned no gate")
        )

        bench_d = result.get("bench")
        if candidate.gate.passed and bench_d:
            candidate.attach_bench(bench_from_dict(bench_d))  # 守护：仅 gate.passed 可挂
    except Exception as e:
        candidate.gate = _runtime_gate(f"evaluate crashed: {e}")

    return candidate
