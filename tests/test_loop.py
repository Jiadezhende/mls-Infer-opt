"""loop 测试：用 fake hooks 验证 trainer 状态机，不跑真实 torch/evaluate/LLM。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mls_infer_opt.evaluate import EvaluatorInfraError
from mls_infer_opt.llm import LLMCallError
from mls_infer_opt.loop import LoopConfig, LoopHooks, hard_stop_reason, keep_best, run_loop
from mls_infer_opt.searchspace.dims import strategy_tags
from mls_infer_opt.state.candidate import (
    Candidate,
    candidate_engine_path,
    make_candidate_id,
)
from mls_infer_opt.state.context import Limits, Paths, TaskContext
from mls_infer_opt.state.eval import BenchResult, EvalMode, GateResult, ValidationError
from mls_infer_opt.state.gradient import Gradient, NoMove
from mls_infer_opt.state.loop import AgentEvent, LoopState


def _ctx(tmp_path: Path, *, limits: Limits | None = None) -> TaskContext:
    return TaskContext(
        model_config={"num_hidden_layers": 1},
        device="cpu",
        run_id="t0",
        paths=Paths(
            target_dir=str(tmp_path / "target"),
            runs_dir=str(tmp_path / "runs"),
            output_dir=str(tmp_path / "workspace"),
        ),
        limits=limits or Limits(),
    )


def _persist_fake(
    ctx: TaskContext,
    gradient: Gradient,
    code: str,
    *,
    passed: bool = True,
    score: float = 1.0,
) -> Candidate:
    """镜像 generate._persist：落 engine.py、按 Gradient 血缘产 Candidate。

    tags 取自 gradient.suggest_axes（fake 无 agent 回报，用建议近似）；不写 applied.json。
    """
    base = Path(ctx.run_dir) / "candidates"
    seq = sum(1 for p in base.iterdir() if p.is_dir()) if base.exists() else 0
    cid = make_candidate_id(seq)
    engine_path = Path(candidate_engine_path(ctx.run_dir, cid))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_text(code, encoding="utf-8")
    return Candidate(
        id=cid,
        kind=gradient.kind,
        round=gradient.round,
        parent_id=gradient.parent_id,
        strategy_tags=strategy_tags(gradient.suggest_axes),
        extra={"fake_passed": passed, "fake_score": score},
    )


def _evaluate_fake(
    candidate: Candidate,
    ctx: TaskContext,
    mode: EvalMode = "full",
    *,
    timeout_s: float | None = None,
) -> Candidate:
    _ = (ctx, timeout_s)
    if candidate.gate is not None:
        return candidate
    passed = bool(candidate.extra.get("fake_passed", True))
    candidate.gate = GateResult(
        syntax_ok=passed,
        api_ok=passed,
        correctness_ok=passed,
        passed=passed,
        errors=[] if passed else [ValidationError(stage="correctness", message="fake failure")],
    )
    if passed:
        score = float(candidate.extra.get("fake_score", 0.0))
        # 真实 full bench 五个计量列都有值；这里全置为 score，使父进程归一化（对 baseline 各列
        # ratio）后 geomean 仍是 score/baseline_score，baseline(score=1.0) 下恰还原为 score，
        # 保持断言语义（_normalize_score 等权平铺 prefill/decode/mixed 的整体与 decode 两列）。
        candidate.attach_bench(
            BenchResult(
                mode=mode,
                score=score,
                decode_tps=score,
                decode_overall_tps=score,
                mixed_tps=score,
                mixed_decode_tps=score,
                prefill_tps=score,
            )
        )
    return candidate


def _stop_analyze(reason: str = "done"):
    def analyze(state: LoopState, *, llm: Any | None = None) -> Gradient | NoMove:
        _ = (state, llm)
        return NoMove(reason)  # analyze 只产 NoMove；停因由总控读取并写 stop_reason

    return analyze


def _scripted_analyze(
    steps: list[Callable[[LoopState], Gradient] | None],
    *,
    stop_reason: str = "done",
):
    def analyze(state: LoopState, *, llm: Any | None = None) -> Gradient | NoMove:
        _ = llm
        if not steps:
            return _stop_analyze(stop_reason)(state)
        step = steps.pop(0)
        if step is None:
            return _stop_analyze(stop_reason)(state)
        return step(state)

    return analyze


def _kv_grad(state: LoopState) -> Gradient:
    gradient = Gradient(
        suggest_axes={"kv_cache": "incremental"},
        kind="optimization",
        round=state.round + 1,
        parent_id=state.best_id,
        rationale="try kv cache",
    )
    # 镜像生产 analyze（grad.py）：每产一个 gradient 都发一条 source="analyze" 的 continue 事件，
    # 它是 _reasoning_trace 的逐轮锚（无此事件则该轮不进 rounds[]）。
    state.add_event(
        AgentEvent(
            source="analyze",
            phase="grad",
            message="continue",
            data={
                "decision": "continue",
                "used_llm": False,
                "bottleneck": "kv",
                "detail": "try kv cache",
                "suggest_axes": dict(gradient.suggest_axes),
                "knobs": {},
            },
        )
    )
    return gradient


# === hard_stop_reason（总控硬上限判停，从 analyze 上移） =============
def test_hard_stop_inactive_when_limits_zero(tmp_path: Path):
    state = LoopState(task_context=_ctx(tmp_path), round=99, stale_rounds=99)  # 默认 limits 全 0
    assert hard_stop_reason(state) is None


def test_hard_stop_each_limit(tmp_path: Path):
    rounds = LoopState(task_context=_ctx(tmp_path, limits=Limits(max_rounds=5)), round=5)
    assert hard_stop_reason(rounds) == "max_rounds_reached"

    stale = LoopState(
        task_context=_ctx(tmp_path, limits=Limits(max_stale_rounds=3)), stale_rounds=3
    )
    assert hard_stop_reason(stale) == "max_stale_rounds_reached"

    timed = LoopState(task_context=_ctx(tmp_path, limits=Limits(time_budget_s=10)))
    timed.budget.elapsed_s = 11.0
    assert hard_stop_reason(timed) == "time_budget_exhausted"


def test_hard_stop_under_limit_returns_none(tmp_path: Path):
    state = LoopState(task_context=_ctx(tmp_path, limits=Limits(max_rounds=5)), round=2)
    assert hard_stop_reason(state) is None


def test_run_loop_hard_stops_on_max_stale_rounds(tmp_path: Path):
    """非提升轮把 stale_rounds 推到上限 → 总控在下一轮顶部硬停（不再依赖 analyze 判停）。"""
    ctx = _ctx(tmp_path, limits=Limits(max_stale_rounds=1))

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "slower engine", score=0.5)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, _kv_grad, _kv_grad]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=10))

    assert state.stop_reason == "max_stale_rounds_reached"
    assert state.best_score == 1.0  # baseline 始终是 best


def test_run_loop_c2_stops_on_llm_infra_failure(tmp_path: Path):
    """LLM 调用失败（C2）穿透到总控边界 → stop_reason=llm_infra_failure，但仍发布 best-so-far。"""
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def boom_analyze(state: LoopState, *, llm: Any | None = None) -> Gradient | NoMove:
        _ = (state, llm)
        raise LLMCallError("api down")

    hooks = LoopHooks(bootstrap=bootstrap, analyze=boom_analyze, evaluate=_evaluate_fake)
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.stop_reason == "llm_infra_failure"
    # must-publish 不破：C2 中止后 finalize 仍把已验证 baseline 发布出去。
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"


def test_run_loop_evaluator_c2_stops_and_keeps_published_best(tmp_path: Path):
    """循环中评测器基建失败（C2）穿透总控边界 → evaluator_infra_failure，已发布 best 仍在盘上。"""
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "optimized engine", score=2.0)

    calls = {"n": 0}

    def flaky_eval(
        candidate: Candidate, ctx: TaskContext, mode: EvalMode = "full",
        *, timeout_s: float | None = None,
    ) -> Candidate:
        calls["n"] += 1
        if calls["n"] == 1:
            return _evaluate_fake(candidate, ctx, mode, timeout_s=timeout_s)  # baseline 评测正常
        raise EvaluatorInfraError("worker exited with code 1")  # 优化轮评测器进程级死亡

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, _kv_grad]),
        propose=propose,
        evaluate=flaky_eval,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.stop_reason == "evaluator_infra_failure"
    # baseline 早已增量发布；C2 中止后仍是盘上 best（未退化）。
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"


def test_run_loop_bootstraps_and_publishes_when_analyze_stops(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_stop_analyze("done"),
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.stop_reason == "done"
    assert state.best_score == 1.0
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"
    final_dir = Path(ctx.run_dir) / "final"
    workspace = Path(ctx.paths.output_dir)
    assert (final_dir / "engine.py").read_text(encoding="utf-8") == "baseline engine"
    assert (final_dir / "output3.json").exists()
    assert (final_dir / "results.log").exists()
    # 运行结束的任务结果记录落本次 run 独立目录根；report3（开发报告）运行时不产。
    assert (Path(ctx.run_dir) / "report.json").exists()
    assert not (final_dir / "report3.json").exists()
    assert not (workspace / "report3.json").exists()
    # workspace 对外只交付 engine.py + output3.json；results.log 是日志、只留 runs/。
    assert not (workspace / "results.log").exists()
    output = json.loads((workspace / "output3.json").read_text(encoding="utf-8"))
    assert output["best_id"] == state.best_id
    assert output["stop_reason"] == "done"
    assert output["run_final_dir"] == str(final_dir)
    assert output["archived_engine_path"] == str(final_dir / "engine.py")
    assert "rounds" in output and "result" in output
    report = json.loads((Path(ctx.run_dir) / "report.json").read_text(encoding="utf-8"))
    assert report["best_id"] == state.best_id and report["stop_reason"] == "done"


def test_run_loop_promotes_strictly_better_candidate(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "optimized engine", score=2.0)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, None]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.best_score == 2.0
    assert state.round == 1
    assert state.stale_rounds == 0
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "optimized engine"

    # 逐轮叙事每条自带分项（哪个轴动了）+ 血缘（从谁 fork）+ 增量。
    output = json.loads((Path(ctx.paths.output_dir) / "output3.json").read_text(encoding="utf-8"))
    r0 = output["rounds"][0]
    assert r0["candidate_id"] == state.best_id
    bd = r0["score_breakdown"]
    assert bd["decode_tps"] == 2.0 and bd["prefill_tps"] == 2.0 and bd["mixed_decode_tps"] == 2.0
    # 从 bootstrap baseline fork（贪心爬山留痕）。
    assert r0["parent_id"] == make_candidate_id(0)
    assert r0["delta"] is None  # 首轮无前序，delta 为 None
    # round 与含候选的轮数自洽（≥ state.round），不再比实际少一拍。
    assert output["round"] == 1 and output["round"] >= state.round


def test_run_loop_keeps_best_when_candidate_is_not_better(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "slower engine", score=0.5)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, None]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.best_score == 1.0
    assert state.stale_rounds == 1
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"


def test_output3_records_non_improving_rounds_every_round(tmp_path: Path):
    """每轮落盘：候选不及 best 也不刷新发布点，但该轮仍进 output3.rounds（叙事抗-kill）。"""

    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "slower engine", score=0.5)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, _kv_grad, None]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    # best 始终是 bootstrap baseline，两轮都没提升。
    assert state.best_score == 1.0
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"

    output = json.loads((Path(ctx.paths.output_dir) / "output3.json").read_text(encoding="utf-8"))
    executed = [r for r in output["rounds"] if r.get("candidate_id")]
    # 两个非提升轮都被记录，不因 best 未刷新而丢叙事。
    assert len(executed) >= 2
    # round 与含候选的轮数自洽。
    assert output["round"] == len(executed) == state.round
    # 非提升轮也带分项 + 血缘 + 增量。
    assert executed[0]["score_breakdown"]["decode_tps"] == 0.5
    assert executed[0]["parent_id"] == make_candidate_id(0)
    assert executed[1]["delta"] == 0.0  # 两轮同分，增量 0


def test_run_loop_repairs_failed_candidate_and_promotes_repair(tmp_path: Path):
    ctx = _ctx(tmp_path, limits=Limits(max_repair_retries=1))

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, Gradient(kind="baseline"), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, gradient: Gradient, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, gradient, "broken engine", passed=False)

    def repair(
        ctx: TaskContext,
        gradient: Gradient,
        parent_code: str,
        errors: list[ValidationError],
        *,
        llm: Any | None,
    ) -> Candidate | None:
        _ = (parent_code, errors, llm)
        return _persist_fake(ctx, gradient, "repaired engine", score=2.0)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_grad, None]),
        propose=propose,
        repair=repair,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.best_score == 2.0
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "repaired engine"
    assert any(e.phase == "repair" for e in state.events)


def test_keep_best_publishes_engine_immediately(tmp_path: Path):
    """每刷新一次 best 就当场发布——不依赖末尾 finalize，保证被中途 kill 时盘上已是最新 best。"""

    ctx = _ctx(tmp_path)
    candidate = _evaluate_fake(
        _persist_fake(ctx, Gradient(kind="baseline"), "first best engine", score=1.0), ctx
    )
    state = LoopState(task_context=ctx)

    # 关键：只调 keep_best，绝不调 finalize / run_loop。
    promoted = keep_best(state, candidate)

    assert promoted is True
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "first best engine"
    promote_evt = next(e for e in state.events if e.message.startswith("提升 best"))
    assert promote_evt.data.get("published") is True


def test_keep_best_republishes_on_strict_improvement(tmp_path: Path):
    """后续更优 best 覆盖发布点；未提升的候选不动发布点。"""

    ctx = _ctx(tmp_path)
    state = LoopState(task_context=ctx)

    first = _evaluate_fake(
        _persist_fake(ctx, Gradient(kind="baseline"), "engine v1", score=1.0), ctx
    )
    keep_best(state, first)
    worse = _evaluate_fake(
        _persist_fake(ctx, Gradient(round=1), "engine worse", score=0.5), ctx
    )
    assert keep_best(state, worse) is False
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "engine v1"

    better = _evaluate_fake(
        _persist_fake(ctx, Gradient(round=2), "engine v2", score=2.0), ctx
    )
    assert keep_best(state, better) is True
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "engine v2"
