"""trainer — loop 的可执行状态机：bootstrap → analyze → generate → evaluate → keep-best → finalize。

loop 是外层确定性控制器：它不问 LLM 做业务判断、不信候选自证，也不产 engine 代码。它只把
generate / evaluate / analyze 接起来，维护唯一的 ``LoopState``，并在 finalize 阶段发布当前
已验证 best。

第一版刻意保持小而稳：
- 依赖通过 ``LoopHooks`` 注入，测试可用 fake，不必真的跑 torch / LLM。
- 所有阶段 never-throw：异常转成 ``AgentEvent`` 和 stop_reason。
- 只有 gate.passed 的候选能进入 keep-best；只有当前 best 能被发布。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .. import present
from ..analyze import analyze as default_analyze
from ..evaluate import EvaluatorInfraError
from ..evaluate import evaluate as default_evaluate
from ..generate import bootstrap as default_bootstrap
from ..generate import propose as default_propose
from ..generate import repair as default_repair
from ..llm.errors import LLMError  # 仅 errors（零依赖），避免经 llm/__init__ 拉入 present 导入环
from ..state.candidate import Candidate, candidate_engine_path
from ..state.context import Environment, Limits, Paths, TaskContext
from ..state.eval import EvalMode, ValidationError, normalized_speedup_score
from ..state.loop import AgentEvent, EventLevel, LoopState, emit
from ..state.policy import NoMove, Policy

__all__ = [
    "AnalyzeFn",
    "BootstrapFn",
    "EvaluateFn",
    "LoopConfig",
    "LoopHooks",
    "ProposeFn",
    "RepairFn",
    "build_task_context",
    "finalize",
    "hard_stop_reason",
    "keep_best",
    "run_loop",
]


class BootstrapFn(Protocol):
    def __call__(self, ctx: TaskContext) -> Candidate: ...


class AnalyzeFn(Protocol):
    def __call__(self, state: LoopState, *, llm: Any | None = None) -> Policy | NoMove: ...


class ProposeFn(Protocol):
    def __call__(
        self, ctx: TaskContext, policy: Policy, parent_code: str, *, llm: Any | None
    ) -> Candidate | None: ...


class RepairFn(Protocol):
    def __call__(
        self,
        ctx: TaskContext,
        policy: Policy,
        parent_code: str,
        errors: list[ValidationError],
        *,
        llm: Any | None,
    ) -> Candidate | None: ...


class EvaluateFn(Protocol):
    def __call__(
        self,
        candidate: Candidate,
        ctx: TaskContext,
        mode: EvalMode = "full",
        *,
        timeout_s: float | None = None,
    ) -> Candidate: ...


@dataclass
class LoopHooks:
    """loop 依赖的业务函数集合。默认接真实模块；测试可替换成 fake。"""

    bootstrap: BootstrapFn = field(default=default_bootstrap)
    analyze: AnalyzeFn = field(default=default_analyze)
    propose: ProposeFn = field(default=default_propose)
    repair: RepairFn = field(default=default_repair)
    evaluate: EvaluateFn = field(default=default_evaluate)


@dataclass
class LoopConfig:
    """loop 自身的执行选项；业务预算仍放在 ``TaskContext.limits``。"""

    eval_timeout_s: float | None = None
    final_eval_timeout_s: float | None = None
    publish_artifacts: bool = True
    # 当 limits.max_rounds 未配置时的外层保险，防止 LLM/generate 持续无收益导致无限循环。
    safety_max_rounds: int = 32


def build_task_context(
    *,
    target_dir: str = "target",
    runs_dir: str = "runs",
    output_dir: str = "workspace",
    run_id: str | None = None,
    device: str = "cuda",
    limits: Limits | None = None,
    environment: Environment | None = None,
    extra: dict[str, Any] | None = None,
) -> TaskContext:
    """按 Phase3 目录约定构造 TaskContext；model_config 存在则读取，不存在则留空。

    这是轻量 INIT：不 import torch、不探测 GPU，避免 loop 层引入重依赖。后续 agent 装配层可在
    调用前填充更完整的 environment / limits。
    """

    rid = run_id or f"run-{int(time.time())}"
    paths = Paths(target_dir=target_dir, runs_dir=runs_dir, output_dir=output_dir)
    model_config: dict[str, Any] = {}
    model_config_path = Path(paths.target_dir) / "model_config.json"
    try:
        if model_config_path.exists():
            raw = json.loads(model_config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                model_config = raw
    except (OSError, ValueError):
        model_config = {}
    return TaskContext(
        model_config=model_config,
        device=device,
        run_id=rid,
        paths=paths,
        limits=limits or Limits(),
        environment=environment or Environment(),
        extra=extra or {},
    )


def run_loop(
    ctx: TaskContext,
    *,
    llm: Any | None = None,
    hooks: LoopHooks | None = None,
    config: LoopConfig | None = None,
) -> LoopState:
    """运行一次调优循环并 finalize，返回完整 LoopState。

    失败不抛给调用方：bootstrap / analyze / generate / evaluate / publish 任一阶段异常都会变成
    ``AgentEvent``，并尽量发布已验证 best。
    """

    hooks = hooks or LoopHooks()
    config = config or LoopConfig()
    state = LoopState(task_context=ctx)
    started = time.monotonic()
    _ensure_dirs(ctx)
    _install_observer(state)  # 每条事件实时(a)流 stderr、(b)增量 append results.log
    emit(state, "loop 启动", "init", data={"run_id": ctx.run_id})
    _emit_llm_status(state, llm)

    try:
        baseline = hooks.bootstrap(ctx)
        baseline = _register_candidate(state, baseline)
        emit(state, f"bootstrap 候选：{baseline.id}", "bootstrap", candidate_id=baseline.id)
        _evaluate_candidate(state, baseline, hooks.evaluate, config.eval_timeout_s)
        if keep_best(state, baseline):
            # bootstrap 提升后 best_score 恰为 baseline，冻结作 speedup 锚点（后续会被更优覆盖）。
            state.baseline_score = state.best_score
            emit(
                state,
                f"bootstrap 成为 best：{baseline.id}",
                "keep_best",
                candidate_id=baseline.id,
            )
            present.emit(present.fmt_banner(state))
        else:
            _stop(state, "bootstrap_failed", "bootstrap 未产生可发布候选", level="error")
    except EvaluatorInfraError as e:
        _stop(state, "evaluator_infra_failure", f"bootstrap 评测器基建失败：{e}", level="error")
    except Exception as e:
        _stop(state, "bootstrap_error", f"bootstrap crashed: {e}", level="error")

    while state.best_id and not state.stop_reason:
        _tick_budget(state, started)
        # 停止准则归总控：先看硬上限（预算 / 轮数 / 连续无提升），再看未配 max_rounds 的安全保险。
        reason = hard_stop_reason(state)
        if reason is not None:
            _stop(state, reason, "硬上限触发")
            break
        if _safety_stop(state, config):
            break

        present.emit("  · 分析中…")  # 瞬态进度：仅 stderr，不落 results.log
        try:
            result = hooks.analyze(state, llm=llm)
        except LLMError as e:  # C2 基建失败穿透到总控边界：记响亮 + 仍 finalize 发布 best-so-far。
            _stop(state, "llm_infra_failure", f"analyze LLM 调用失败：{e}", level="error")
            break
        except Exception as e:  # 其它非预期（analyze 承诺 never-throw）：当「无方向」处理。
            emit(state, f"analyze 异常：{e}", "grad", level="error")
            result = NoMove("analyze_crashed")

        # analyze 只算方向：NoMove 表示迈不出步，由总控（这里）裁决终止并写 stop_reason。
        if isinstance(result, NoMove):
            _stop(state, result.reason, "analyze 无方向")
            break

        policy = result
        try:
            improved = _run_policy_round(state, policy, llm, hooks, config)
        except LLMError as e:  # generate 的 C2 同样穿透到此：C2 停 + 仍发布。
            _stop(state, "llm_infra_failure", f"generate LLM 调用失败：{e}", level="error")
            break
        except EvaluatorInfraError as e:  # evaluate 的 C2 穿透到此：C2 停 + 仍发布。
            _stop(state, "evaluator_infra_failure", f"评测器基建失败：{e}", level="error")
            break
        state.round = max(state.round, policy.round)
        if not improved:
            state.stale_rounds += 1
        # 每轮落盘：不止 best 提升时刷 output3，使被 kill 时 rounds[] 含所有已完成轮（叙事抗-kill
        # 与 results.log 事件级 append 对齐）。提升轮已在 keep_best 内刷过，这里覆盖一遍代价可忽略。
        _publish_summary(state)

    _tick_budget(state, started)
    finalize(state, hooks=hooks, config=config)
    return state


def keep_best(state: LoopState, candidate: Candidate) -> bool:
    """若候选已过 gate 且分数严格更高，则提升为 best。返回是否提升。"""

    if candidate.gate is None or not candidate.gate.passed:
        emit(
            state,
            f"候选未过 gate，不参与 best：{candidate.id}",
            "keep_best",
            candidate_id=candidate.id,
            data={"passed": False},
        )
        return False

    score = candidate.bench.score if candidate.bench is not None else 0.0
    if score <= state.best_score:
        emit(
            state,
            f"候选未提升：{candidate.id}",
            "keep_best",
            candidate_id=candidate.id,
            data={
                "score": score,
                "best_score": state.best_score,
                "score_line": present.fmt_score_line(candidate.bench, state.baseline_score),
            },
        )
        return False

    old_best = state.best_id
    state.set_best(candidate, score)
    state.stale_rounds = 0

    # 增量发布：每刷新一次 best，立刻把这份已过 gate 的 engine.py 拷到发布点。
    # 候选源码早已落在 runs/.../candidates/{id}/engine.py，发布只是一次 copyfile。
    # 这样进程在任意时刻被外部 kill（评测墙），盘上始终是最新 best，不必等末尾 finalize；
    # finalize 仍是权威终发（含 final gate 复核），此处只做随时可用的安全网。
    ctx = state.task_context
    published = _copy_engine(
        candidate_engine_path(ctx.run_dir, candidate.id), ctx.engine_publish_path
    )
    emit(
        state,
        f"提升 best：{candidate.id}",
        "keep_best",
        candidate_id=candidate.id,
        data={
            "old_best_id": old_best,
            "best_score": score,
            "published": published,
            "score_line": present.fmt_score_line(candidate.bench, state.baseline_score),
        },
    )
    if not published:
        emit(
            state,
            f"增量发布失败（finalize 兜底）：{ctx.engine_publish_path}",
            "keep_best",
            level="warning",
            candidate_id=candidate.id,
        )
    _publish_summary(state)  # output3.json 随 best 同步刷新（与 engine.py 同时发布）
    return True


def finalize(
    state: LoopState,
    *,
    hooks: LoopHooks | None = None,
    config: LoopConfig | None = None,
) -> LoopState:
    """发布当前 best 到 ``ctx.engine_publish_path``，并写 output3.json / report.json / results.log。

    发布以 correctness gate 为硬门。没有 best 或 best 未过门时只写 artifact 说明，不发布 engine。
    """

    hooks = hooks or LoopHooks()
    config = config or LoopConfig()
    ctx = state.task_context
    _ensure_dirs(ctx)

    best = state.best_candidate()
    if best is None:
        if not state.stop_reason:
            state.stop_reason = "no_publishable_candidate"
        emit(state, "finalize 无 best 可发布", "finalize", level="error")
        _write_artifacts(state, enabled=config.publish_artifacts)
        present.emit(present.fmt_acceptance(state))
        return state

    if best.gate is None:
        try:
            _evaluate_candidate(state, best, hooks.evaluate, config.final_eval_timeout_s)
        except EvaluatorInfraError as e:
            # 终发复核遇评测器基建失败：记 C2、不发布未复核 best（增量发布的上一个 best 仍在盘上）。
            if not state.stop_reason:
                state.stop_reason = "evaluator_infra_failure"
            emit(state, f"finalize 评测器基建失败：{e}", "finalize", level="error",
                 candidate_id=best.id)

    if best.gate is not None and best.gate.passed:
        src = candidate_engine_path(ctx.run_dir, best.id)
        workspace_ok = _copy_engine(src, ctx.engine_publish_path)
        archive_path = str(_final_dir(ctx) / "engine.py")
        archive_ok = _copy_engine(src, archive_path)

        if workspace_ok:
            if not state.stop_reason:
                state.stop_reason = "completed"
            emit(
                state,
                f"发布 best：{best.id}",
                "finalize",
                candidate_id=best.id,
                data={
                    "engine_path": ctx.engine_publish_path,
                    "archived_engine_path": archive_path if archive_ok else "",
                },
            )
            if not archive_ok:
                emit(
                    state,
                    f"run final engine 留档失败：{archive_path}",
                    "finalize",
                    level="warning",
                    candidate_id=best.id,
                )
        else:
            state.stop_reason = "publish_failed"
            emit(
                state,
                f"发布失败：{ctx.engine_publish_path}",
                "finalize",
                level="error",
                candidate_id=best.id,
                data={"archived_engine_path": archive_path if archive_ok else ""},
            )
    else:
        if not state.stop_reason:
            state.stop_reason = "best_failed_final_gate"
        emit(
            state,
            f"best 未过最终 gate：{best.id}",
            "finalize",
            level="error",
            candidate_id=best.id,
        )

    _write_artifacts(state, enabled=config.publish_artifacts)
    present.emit(present.fmt_acceptance(state))
    return state


def _run_policy_round(
    state: LoopState,
    policy: Policy,
    llm: Any | None,
    hooks: LoopHooks,
    config: LoopConfig,
) -> bool:
    ctx = state.task_context
    parent_code = _read_best_code(state)
    if parent_code is None:
        _stop(state, "missing_best_code", "best 源码缺失，无法继续生成", level="error")
        return False

    present.emit("  · 生成候选中…")  # 瞬态进度：仅 stderr，不落 results.log
    try:
        candidate = hooks.propose(ctx, policy, parent_code, llm=llm)
    except LLMError:
        raise  # C2 穿透到 run_loop 边界
    except Exception as e:
        emit(state, f"generate.propose 异常：{e}", "generate", level="error")
        candidate = None

    if candidate is None:
        reason = _llm_failure_reason(llm)
        emit(
            state,
            "本轮未产出候选" + (f"（{reason}）" if reason else ""),
            "generate",
            data={"policy_round": policy.round, "parent_id": state.best_id, "reason": reason},
        )
        return False

    candidate = _register_candidate(state, candidate)
    emit(state, f"生成候选：{candidate.id}", "generate", candidate_id=candidate.id)
    _evaluate_candidate(state, candidate, hooks.evaluate, config.eval_timeout_s)
    if keep_best(state, candidate):
        return True
    if candidate.gate is None or candidate.gate.passed:
        return False
    return _run_repairs(state, policy, candidate, llm, hooks, config)


def _run_repairs(
    state: LoopState,
    policy: Policy,
    failed: Candidate,
    llm: Any | None,
    hooks: LoopHooks,
    config: LoopConfig,
) -> bool:
    ctx = state.task_context
    retries = max(0, ctx.limits.max_repair_retries)
    cur = failed
    for attempt in range(1, retries + 1):
        parent_code = _read_candidate_code(ctx, cur.id)
        if parent_code is None:
            emit(state, f"repair 跳过：候选源码缺失 {cur.id}", "repair", level="warning")
            return False
        errors = cur.gate.errors if cur.gate is not None else []
        present.emit(f"  · 修复中（第 {attempt} 次）…")  # 瞬态进度：仅 stderr，不落 results.log
        try:
            repaired = hooks.repair(ctx, policy, parent_code, errors, llm=llm)
        except LLMError:
            raise  # C2 穿透到 run_loop 边界
        except Exception as e:
            emit(state, f"generate.repair 异常：{e}", "repair", level="error")
            repaired = None
        if repaired is None:
            emit(
                state,
                "repair 未产出候选",
                "repair",
                data={"attempt": attempt, "failed_candidate_id": cur.id},
            )
            continue
        repaired = _register_candidate(state, repaired)
        emit(
            state,
            f"repair 候选：{repaired.id}",
            "repair",
            candidate_id=repaired.id,
            data={"attempt": attempt, "failed_candidate_id": cur.id},
        )
        _evaluate_candidate(state, repaired, hooks.evaluate, config.eval_timeout_s)
        if keep_best(state, repaired):
            return True
        if repaired.gate is not None and repaired.gate.passed:
            return False
        cur = repaired
    return False


def _register_candidate(state: LoopState, candidate: Candidate) -> Candidate:
    # id 是运行内单调序号（c{seq}），同一 run 内不会重复——直接登记，无需去重。
    state.add_candidate(candidate)
    return candidate


def _normalize_score(state: LoopState, candidate: Candidate) -> None:
    """用 baseline 的 per-列 tps 作参照，把候选 bench.score 归一化为加速比（口径见 state.eval）。

    参照系 = bootstrap baseline 候选的 bench（自校准到真实评测硬件）：baseline 自身 → score 1.0；
    后续候选 score≈×baseline，让 keep-best 严格比较与 speedup 展示都有诚实语义。无 baseline
    （bootstrap 评测当下尚未冻结）或缺 bench 时安全跳过，保留 worker 临时自评。是所有 score 消费者
    （keep_best / fmt_score_line / analyze）之前的唯一咽喉点。
    """
    bench = candidate.bench
    if bench is None:
        return
    ref = state.baseline_candidate()
    ref_bench = ref.bench if ref is not None else None
    if ref_bench is None:
        return
    bench.score = normalized_speedup_score(bench, ref_bench)
    bench.loss = -bench.score


def _evaluate_candidate(
    state: LoopState,
    candidate: Candidate,
    evaluate_fn: EvaluateFn,
    timeout_s: float | None,
) -> Candidate:
    already_evaluated = candidate.gate is not None
    present.emit(f"  · 评测中 {candidate.id}…")  # 瞬态进度：仅 stderr，不落 results.log
    try:
        evaluated = evaluate_fn(candidate, state.task_context, "full", timeout_s=timeout_s)
        if not already_evaluated:
            state.budget.eval_runs += 1
        if evaluated is not candidate:
            candidate = evaluated
            state.candidates[candidate.id] = candidate
    except EvaluatorInfraError:
        raise  # C2 评测器基建失败：穿透到总控边界（bootstrap / 循环 / finalize 各自接住）
    except Exception as e:
        emit(state, f"evaluate 异常：{e}", "evaluate", level="error", candidate_id=candidate.id)
        return candidate

    _normalize_score(state, candidate)
    passed = bool(candidate.gate and candidate.gate.passed)
    data: dict[str, Any] = {
        "passed": passed,
        "score": candidate.bench.score if candidate.bench is not None else None,
    }
    if candidate.bench is not None:
        # 把「带单位 + ×baseline」的分数行算好塞进 event.data——stream_event 只拿得到 event，
        # 这是它显示 speedup 的唯一来源；results.log 的数据块也复用同一份 score_line。
        data["score_line"] = present.fmt_score_line(candidate.bench, state.baseline_score)
    elif not passed and candidate.gate is not None:
        # 失败候选：把「卡在哪阶段 / 错因 / 多少 case 不过」塞进 event.data，让验收者看到为何不过，
        # 而非只见「评测失败」。错因本就挂在 candidate.gate 上，这里只是搬进事件供渲染。
        if candidate.gate.errors:
            err = candidate.gate.errors[0]
            data["gate_stage"] = err.stage
            data["gate_error"] = err.message
        cs = candidate.gate.case_summary or {}
        if "total" in cs:
            data["cases"] = f"{cs.get('passed', '?')}/{cs.get('total', '?')} 通过"
    emit(
        state,
        f"评测{'通过' if passed else '失败'}：{candidate.id}",
        "evaluate",
        candidate_id=candidate.id,
        data=data,
    )
    return candidate


def _read_best_code(state: LoopState) -> str | None:
    if state.best_id is None:
        return None
    return _read_candidate_code(state.task_context, state.best_id)


def _read_candidate_code(ctx: TaskContext, candidate_id: str) -> str | None:
    try:
        return Path(candidate_engine_path(ctx.run_dir, candidate_id)).read_text(encoding="utf-8")
    except OSError:
        return None


def hard_stop_reason(state: LoopState) -> str | None:
    """硬上限判停（总控的循环准则）：到上限即返回停因，否则 None。每条仅在对应 limit > 0 时生效。

    确定性、只读 limits + 实时量（budget.elapsed_s / round / stale_rounds）。从 analyze 上移到这里
    （停止是训练循环的准则、不是 gradient 的活）；达标 / 收益不足等软停由 analyze 的 NoMove 表达。
    """
    limits = state.task_context.limits
    if limits.time_budget_s > 0 and state.budget.elapsed_s >= limits.time_budget_s:
        return "time_budget_exhausted"
    if limits.max_rounds > 0 and state.round >= limits.max_rounds:
        return "max_rounds_reached"
    if limits.max_stale_rounds > 0 and state.stale_rounds >= limits.max_stale_rounds:
        return "max_stale_rounds_reached"
    return None


def _safety_stop(state: LoopState, config: LoopConfig) -> bool:
    if state.task_context.limits.max_rounds > 0:
        return False
    if config.safety_max_rounds <= 0:
        return False
    if state.round < config.safety_max_rounds:
        return False
    _stop(state, "safety_max_rounds_reached", "未配置 max_rounds，触发 loop 外层保险")
    return True


def _stop(state: LoopState, reason: str, message: str, *, level: EventLevel = "info") -> None:
    state.stop_reason = reason
    emit(
        state,
        f"停止：{reason}",
        "stop",
        level=level,
        data={"stop_reason": reason, "detail": message},
    )


def _tick_budget(state: LoopState, started: float) -> None:
    state.budget.elapsed_s = max(0.0, time.monotonic() - started)


def _ensure_dirs(ctx: TaskContext) -> None:
    for path in (ctx.run_dir, ctx.paths.output_dir, str(_final_dir(ctx))):
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def _final_dir(ctx: TaskContext) -> Path:
    """本次 run 的最终产物留档目录。workspace 对外发布，runs/final 供审计复盘。"""
    return Path(ctx.run_final_dir)


def _copy_engine(src: str, dst: str) -> bool:
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    except OSError:
        return False
    return True


def _install_observer(state: LoopState) -> None:
    """装事件观察者：每条事件实时 (a) 流 stderr、(b) 增量 append 到 runs/.../final/results.log。

    增量写让 results.log 在被 Ctrl-C / SIGTERM / 评测墙中途 kill 时也已落到最新一条，不必等
    finalize（与 engine.py 增量发布同一抗 kill 哲学）。results.log 是逐轮事件日志、只留档本次 run
    的独立目录（runs/{run_id}/final），不进对外 workspace（对外交付只有 engine.py + output3.json）。
    先清空 final/ 里上一个 run 的残留，再逐条 append；写盘失败则降级为仅 stderr。
    """
    final_log = _final_dir(state.task_context) / "results.log"
    log_path: Path | None = final_log
    try:
        final_log.write_text("", encoding="utf-8")  # 清空上一个 run 的残留
    except OSError:
        log_path = None

    def _sink(event: AgentEvent) -> None:
        present.stream_event(event)
        if log_path is not None:
            present.append_event(log_path, event)

    state.on_event(_sink)


def _publish_summary(state: LoopState) -> None:
    """把 output3.json 同步写到发布根（随 best 刷新即更新，与 engine.py 同时发布，抗中途 kill）。

    best 在 keep_best 里增量发布 engine.py，这里跟着把摘要也写出去，使任意中断点 output3.json
    都反映「当前最优」。finalize 仍会用完整状态覆盖一遍作权威终发。never-throw。
    """
    try:
        out = Path(state.task_context.paths.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "output3.json").write_text(
            present.json_dump(present.summary(state)), encoding="utf-8"
        )
    except OSError:
        pass


def _write_artifacts(state: LoopState, *, enabled: bool) -> None:
    """终发落盘：output3.json（含 result/rounds 推理）对外发布到 workspace + 留档 runs/final；
    results.log 只留档 runs/final；运行结束的「任务结果记录」写 runs/{run_id}/report.json。

    report3（摘要 + 完整 LoopState 快照）是开发报告、由人手写，运行时**不产**（不变量见模块约定）。
    """
    if not enabled:
        return
    ctx = state.task_context
    workspace = Path(ctx.paths.output_dir)
    archive = _final_dir(ctx)
    try:
        summary_json = present.json_dump(present.summary(state))
        # workspace 对外只交付 engine.py + output3.json；runs/final 同步留一份作审计复盘。
        for out_dir in (workspace, archive):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "output3.json").write_text(summary_json, encoding="utf-8")
        # results.log 是逐轮事件日志：只留 runs/{run_id}/final，不进对外 workspace。
        (archive / "results.log").write_text(present.render_events(state), encoding="utf-8")
        # 运行结束的任务结果记录，落本次 run 的独立目录根（runs/{run_id}）。
        (Path(ctx.run_dir) / "report.json").write_text(summary_json, encoding="utf-8")
    except OSError as e:
        emit(state, f"artifact 写入失败：{e}", "finalize", level="error")


def _emit_llm_status(state: LoopState, llm: Any | None) -> None:
    """开局记录 LLM 可用性——不可用时把原因落进 results.log，杜绝失败被静默吞掉。

    LLM 是 analyze 唯一方向源：不可用即首轮 NoMove、只发布 baseline。这条事件回答「为什么这条
    run 没动起来」：缺 key / 没装 SDK / disabled 等真因在 client 构造时已算进
    ``unavailable_reason``，这里把它显式留痕，而非事后靠时序猜。
    """
    fp = _llm_fingerprint(llm)
    available = bool(llm is not None and getattr(llm, "available", False))
    if available:
        emit(
            state,
            f"LLM 可用：key={fp['api_key']} base_url={fp['base_url']} model={fp['model']}",
            "llm",
            data={"available": True, **fp},
        )
        return
    reason = getattr(llm, "unavailable_reason", None) if llm is not None else None
    reason = reason or "llm client 未装配（None）"
    emit(
        state,
        f"LLM 不可用，analyze 将首轮 NoMove、只发布 baseline：{reason}"
        f"（key={fp['api_key']} base_url={fp['base_url']} model={fp['model']}）",
        "llm",
        level="warning",
        data={"available": False, "reason": reason, **fp},
    )


def _llm_fingerprint(llm: Any | None) -> dict[str, Any]:
    """运行时实际加载的 LLM 凭证指纹（脱敏）——回答「到底用了哪把 key / 哪个端点」。

    key 只留头 8 + 尾 4 + 长度，绝不落全量；base_url / model 原样留痕。专治「评测环境加载了
    与本地不同的 .env / 被注入了别的 OPENAI_*」这类只在 submit 复现、本地看不出的环境问题。
    """
    cfg = getattr(llm, "config", None)
    key = getattr(cfg, "api_key", None) or ""
    masked = f"{key[:8]}…{key[-4:]}(len={len(key)})" if key else "(empty)"
    return {
        "api_key": masked,
        "base_url": getattr(cfg, "base_url", None),
        "model": getattr(cfg, "model", None),
    }


def _llm_failure_reason(llm: Any | None) -> str | None:
    """提取本轮 generate 没出候选的 LLM 侧原因：不可用 → reason；可用但调用失败 → last_error。"""
    if llm is None:
        return "llm 未装配"
    if not getattr(llm, "available", False):
        return getattr(llm, "unavailable_reason", None) or "llm 不可用"
    err = getattr(llm, "last_error", None)
    if isinstance(err, dict):
        kind = err.get("kind", "?")
        msg = str(err.get("message", ""))[:200]
        return f"llm 调用失败: {kind}: {msg}"
    return None


