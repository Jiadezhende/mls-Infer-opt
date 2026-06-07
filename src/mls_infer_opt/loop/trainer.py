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
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..analyze import analyze as default_analyze
from ..evaluate import evaluate as default_evaluate
from ..generate import bootstrap as default_bootstrap
from ..generate import propose as default_propose
from ..generate import repair as default_repair
from ..state.candidate import Candidate, candidate_engine_path
from ..state.common import to_dict, utcnow_iso
from ..state.context import Environment, Limits, Paths, TaskContext
from ..state.eval import EvalMode, ValidationError
from ..state.loop import AgentEvent, EventLevel, LoopState
from ..state.policy import Policy

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
    "keep_best",
    "run_loop",
]


class BootstrapFn(Protocol):
    def __call__(self, ctx: TaskContext) -> Candidate: ...


class AnalyzeFn(Protocol):
    def __call__(self, state: LoopState, *, llm: Any | None = None) -> Policy | None: ...


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
    _emit(state, "loop 启动", "init", data={"run_id": ctx.run_id})

    try:
        baseline = hooks.bootstrap(ctx)
        baseline = _register_candidate(state, baseline)
        _emit(state, f"bootstrap 候选：{baseline.id}", "bootstrap", candidate_id=baseline.id)
        _evaluate_candidate(state, baseline, hooks.evaluate, config.eval_timeout_s)
        if keep_best(state, baseline):
            _emit(
                state,
                f"bootstrap 成为 best：{baseline.id}",
                "keep_best",
                candidate_id=baseline.id,
            )
        else:
            _stop(state, "bootstrap_failed", "bootstrap 未产生可发布候选", level="error")
    except Exception as e:
        _stop(state, "bootstrap_error", f"bootstrap crashed: {e}", level="error")

    while state.best_id and not state.stop_reason:
        _tick_budget(state, started)
        if _safety_stop(state, config):
            break

        try:
            policy = hooks.analyze(state, llm=llm)
        except Exception as e:
            _emit(state, f"analyze 异常：{e}", "grad", level="error")
            policy = None

        if policy is None:
            reason = _last_analyze_stop_reason(state) or "analyze_stopped"
            _stop(state, reason, "analyze 返回停止")
            break

        improved = _run_policy_round(state, policy, llm, hooks, config)
        state.round = max(state.round, policy.round)
        if not improved:
            state.stale_rounds += 1

    _tick_budget(state, started)
    finalize(state, hooks=hooks, config=config)
    return state


def keep_best(state: LoopState, candidate: Candidate) -> bool:
    """若候选已过 gate 且分数严格更高，则提升为 best。返回是否提升。"""

    if candidate.gate is None or not candidate.gate.passed:
        _emit(
            state,
            f"候选未过 gate，不参与 best：{candidate.id}",
            "keep_best",
            candidate_id=candidate.id,
            data={"passed": False},
        )
        return False

    score = candidate.bench.score if candidate.bench is not None else 0.0
    if score <= state.best_score:
        _emit(
            state,
            f"候选未提升：{candidate.id}",
            "keep_best",
            candidate_id=candidate.id,
            data={"score": score, "best_score": state.best_score},
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
    _emit(
        state,
        f"提升 best：{candidate.id}",
        "keep_best",
        candidate_id=candidate.id,
        data={"old_best_id": old_best, "best_score": score, "published": published},
    )
    if not published:
        _emit(
            state,
            f"增量发布失败（finalize 兜底）：{ctx.engine_publish_path}",
            "keep_best",
            level="warning",
            candidate_id=candidate.id,
        )
    return True


def finalize(
    state: LoopState,
    *,
    hooks: LoopHooks | None = None,
    config: LoopConfig | None = None,
) -> LoopState:
    """发布当前 best 到 ``ctx.engine_publish_path``，并写 output3/report3/results.log。

    发布仍以 correctness gate 为硬门。没有 best 或 best 未过门时只写 artifact 说明，不发布 engine。
    """

    hooks = hooks or LoopHooks()
    config = config or LoopConfig()
    ctx = state.task_context
    _ensure_dirs(ctx)

    best = state.best_candidate()
    if best is None:
        if not state.stop_reason:
            state.stop_reason = "no_publishable_candidate"
        _emit(state, "finalize 无 best 可发布", "finalize", level="error")
        _write_artifacts(state, enabled=config.publish_artifacts)
        return state

    if best.gate is None:
        _evaluate_candidate(state, best, hooks.evaluate, config.final_eval_timeout_s)

    if best.gate is not None and best.gate.passed:
        src = candidate_engine_path(ctx.run_dir, best.id)
        workspace_ok = _copy_engine(src, ctx.engine_publish_path)
        archive_path = str(_final_dir(ctx) / "engine.py")
        archive_ok = _copy_engine(src, archive_path)

        if workspace_ok:
            if not state.stop_reason:
                state.stop_reason = "completed"
            _emit(
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
                _emit(
                    state,
                    f"run final engine 留档失败：{archive_path}",
                    "finalize",
                    level="warning",
                    candidate_id=best.id,
                )
        else:
            state.stop_reason = "publish_failed"
            _emit(
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
        _emit(
            state,
            f"best 未过最终 gate：{best.id}",
            "finalize",
            level="error",
            candidate_id=best.id,
        )

    _write_artifacts(state, enabled=config.publish_artifacts)
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

    try:
        candidate = hooks.propose(ctx, policy, parent_code, llm=llm)
    except Exception as e:
        _emit(state, f"generate.propose 异常：{e}", "generate", level="error")
        candidate = None

    if candidate is None:
        _emit(
            state,
            "本轮未产出候选",
            "generate",
            data={"policy_round": policy.round, "parent_id": state.best_id},
        )
        return False

    candidate = _register_candidate(state, candidate)
    _emit(state, f"生成候选：{candidate.id}", "generate", candidate_id=candidate.id)
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
            _emit(state, f"repair 跳过：候选源码缺失 {cur.id}", "repair", level="warning")
            return False
        errors = cur.gate.errors if cur.gate is not None else []
        try:
            repaired = hooks.repair(ctx, policy, parent_code, errors, llm=llm)
        except Exception as e:
            _emit(state, f"generate.repair 异常：{e}", "repair", level="error")
            repaired = None
        if repaired is None:
            _emit(
                state,
                "repair 未产出候选",
                "repair",
                data={"attempt": attempt, "failed_candidate_id": cur.id},
            )
            continue
        repaired = _register_candidate(state, repaired)
        _emit(
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
    existing = state.candidates.get(candidate.id)
    if existing is not None:
        _emit(
            state,
            f"候选重复，复用已有结果：{candidate.id}",
            "dedupe",
            candidate_id=candidate.id,
        )
        return existing
    state.add_candidate(candidate)
    return candidate


def _evaluate_candidate(
    state: LoopState,
    candidate: Candidate,
    evaluate_fn: EvaluateFn,
    timeout_s: float | None,
) -> Candidate:
    already_evaluated = candidate.gate is not None
    try:
        evaluated = evaluate_fn(candidate, state.task_context, "full", timeout_s=timeout_s)
        if not already_evaluated:
            state.budget.eval_runs += 1
        if evaluated is not candidate:
            candidate = evaluated
            state.candidates[candidate.id] = candidate
    except Exception as e:
        _emit(state, f"evaluate 异常：{e}", "evaluate", level="error", candidate_id=candidate.id)
        return candidate

    passed = bool(candidate.gate and candidate.gate.passed)
    _emit(
        state,
        f"评测{'通过' if passed else '失败'}：{candidate.id}",
        "evaluate",
        candidate_id=candidate.id,
        data={
            "passed": passed,
            "score": candidate.bench.score if candidate.bench is not None else None,
        },
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


def _last_analyze_stop_reason(state: LoopState) -> str | None:
    for event in reversed(state.events):
        if event.source == "analyze" and event.data.get("decision") == "stop":
            reason = event.data.get("stop_reason")
            return reason if isinstance(reason, str) and reason else None
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
    _emit(
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
    return Path(ctx.run_dir) / "final"


def _copy_engine(src: str, dst: str) -> bool:
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    except OSError:
        return False
    return True


def _summary(state: LoopState) -> dict[str, Any]:
    best = state.best_candidate()
    return {
        "run_id": state.task_context.run_id,
        "stop_reason": state.stop_reason,
        "best_id": state.best_id,
        "best_score": state.best_score if math.isfinite(state.best_score) else None,
        "best_strategy_tags": list(best.strategy_tags) if best is not None else [],
        "n_candidates": len(state.candidates),
        "round": state.round,
        "stale_rounds": state.stale_rounds,
        "elapsed_s": state.budget.elapsed_s,
        "eval_runs": state.budget.eval_runs,
        "engine_path": state.task_context.engine_publish_path if best is not None else "",
        "archived_engine_path": str(_final_dir(state.task_context) / "engine.py")
        if best is not None
        else "",
        "run_final_dir": str(_final_dir(state.task_context)),
    }


def _write_artifacts(state: LoopState, *, enabled: bool) -> None:
    if not enabled:
        return
    output_dirs = (Path(state.task_context.paths.output_dir), _final_dir(state.task_context))
    try:
        summary = _summary(state)
        report = {"summary": summary, "state": to_dict(state)}
        events = _render_events(state)
        for out_dir in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "output3.json").write_text(_json_dump(summary), encoding="utf-8")
            (out_dir / "report3.json").write_text(_json_dump(report), encoding="utf-8")
            (out_dir / "results.log").write_text(events, encoding="utf-8")
    except OSError as e:
        _emit(state, f"artifact 写入失败：{e}", "finalize", level="error")


def _json_dump(payload: Any) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _render_events(state: LoopState) -> str:
    lines = []
    for event in state.events:
        cid = f" candidate={event.candidate_id}" if event.candidate_id else ""
        lines.append(
            f"[{event.ts}] {event.level} {event.source}.{event.phase}:{cid} {event.message}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _emit(
    state: LoopState,
    message: str,
    phase: str,
    *,
    level: EventLevel = "info",
    candidate_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    payload = dict(data or {})
    payload.setdefault("round", state.round)
    state.add_event(
        AgentEvent(
            source="loop",
            phase=phase,
            message=message,
            level=level,
            candidate_id=candidate_id,
            ts=utcnow_iso(),
            data=payload,
        )
    )
