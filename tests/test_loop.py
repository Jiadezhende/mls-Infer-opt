"""loop 测试：用 fake hooks 验证 trainer 状态机，不跑真实 torch/evaluate/LLM。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mls_infer_opt.loop import LoopConfig, LoopHooks, keep_best, run_loop
from mls_infer_opt.searchspace.policy import aggregate, default_policy, strategy_tags, to_json
from mls_infer_opt.state.candidate import (
    Candidate,
    candidate_engine_path,
    candidate_policy_path,
    make_candidate_id,
)
from mls_infer_opt.state.context import Limits, Paths, TaskContext
from mls_infer_opt.state.eval import BenchResult, EvalMode, GateResult, ValidationError
from mls_infer_opt.state.loop import AgentEvent, LoopState
from mls_infer_opt.state.policy import Policy


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
    policy: Policy,
    code: str,
    *,
    passed: bool = True,
    score: float = 1.0,
) -> Candidate:
    cid = make_candidate_id(policy.round, code)
    engine_path = Path(candidate_engine_path(ctx.run_dir, cid))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_text(code, encoding="utf-8")
    Path(candidate_policy_path(ctx.run_dir, cid)).write_text(to_json(policy), encoding="utf-8")
    return Candidate(
        id=cid,
        kind=policy.kind,
        round=policy.round,
        parent_id=policy.parent_id,
        strategy_tags=strategy_tags(policy),
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
        # 真实 full bench 三类 tps 都有值；这里都置为 score，使父进程归一化（对 baseline 各类
        # ratio）得 score/baseline_score，baseline(score=1.0) 下恰还原为 score，保持断言语义。
        candidate.attach_bench(
            BenchResult(
                mode=mode,
                score=score,
                decode_tps=score,
                mixed_decode_tps=score,
                prefill_tps=score,
            )
        )
    return candidate


def _stop_analyze(reason: str = "done"):
    def analyze(state: LoopState, *, llm: Any | None = None) -> Policy | None:
        _ = llm
        state.add_event(
            AgentEvent(
                source="analyze",
                phase="grad",
                message=f"stop {reason}",
                data={"decision": "stop", "stop_reason": reason},
            )
        )
        return None

    return analyze


def _scripted_analyze(
    steps: list[Callable[[LoopState], Policy] | None],
    *,
    stop_reason: str = "done",
):
    def analyze(state: LoopState, *, llm: Any | None = None) -> Policy | None:
        _ = llm
        if not steps:
            return _stop_analyze(stop_reason)(state)
        step = steps.pop(0)
        if step is None:
            return _stop_analyze(stop_reason)(state)
        return step(state)

    return analyze


def _kv_policy(state: LoopState) -> Policy:
    return aggregate(
        {"kv_cache": "incremental"},
        kind="optimization",
        round=state.round + 1,
        parent_id=state.best_id,
        rationale="try kv cache",
    ).policy


def test_run_loop_bootstraps_and_publishes_when_analyze_stops(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, default_policy(), "baseline engine", score=1.0)

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
    assert (final_dir / "engine.py").read_text(encoding="utf-8") == "baseline engine"
    assert (final_dir / "output3.json").exists()
    assert (final_dir / "report3.json").exists()
    assert (final_dir / "results.log").exists()
    output = json.loads((Path(ctx.paths.output_dir) / "output3.json").read_text(encoding="utf-8"))
    assert output["best_id"] == state.best_id
    assert output["stop_reason"] == "done"
    assert output["run_final_dir"] == str(final_dir)
    assert output["archived_engine_path"] == str(final_dir / "engine.py")


def test_run_loop_promotes_strictly_better_candidate(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, default_policy(), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, policy: Policy, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, policy, "optimized engine", score=2.0)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_policy, None]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.best_score == 2.0
    assert state.round == 1
    assert state.stale_rounds == 0
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "optimized engine"


def test_run_loop_keeps_best_when_candidate_is_not_better(tmp_path: Path):
    ctx = _ctx(tmp_path)

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, default_policy(), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, policy: Policy, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, policy, "slower engine", score=0.5)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_policy, None]),
        propose=propose,
        evaluate=_evaluate_fake,
    )
    state = run_loop(ctx, hooks=hooks, config=LoopConfig(safety_max_rounds=4))

    assert state.best_score == 1.0
    assert state.stale_rounds == 1
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "baseline engine"


def test_run_loop_repairs_failed_candidate_and_promotes_repair(tmp_path: Path):
    ctx = _ctx(tmp_path, limits=Limits(max_repair_retries=1))

    def bootstrap(ctx: TaskContext) -> Candidate:
        return _persist_fake(ctx, default_policy(), "baseline engine", score=1.0)

    def propose(
        ctx: TaskContext, policy: Policy, parent_code: str, *, llm: Any | None
    ) -> Candidate | None:
        _ = (parent_code, llm)
        return _persist_fake(ctx, policy, "broken engine", passed=False)

    def repair(
        ctx: TaskContext,
        policy: Policy,
        parent_code: str,
        errors: list[ValidationError],
        *,
        llm: Any | None,
    ) -> Candidate | None:
        _ = (parent_code, errors, llm)
        return _persist_fake(ctx, policy, "repaired engine", score=2.0)

    hooks = LoopHooks(
        bootstrap=bootstrap,
        analyze=_scripted_analyze([_kv_policy, None]),
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
        _persist_fake(ctx, default_policy(), "first best engine", score=1.0), ctx
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

    first = _evaluate_fake(_persist_fake(ctx, default_policy(), "engine v1", score=1.0), ctx)
    keep_best(state, first)
    worse = _evaluate_fake(
        _persist_fake(ctx, aggregate({}, round=1).policy, "engine worse", score=0.5), ctx
    )
    assert keep_best(state, worse) is False
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "engine v1"

    better = _evaluate_fake(
        _persist_fake(ctx, aggregate({}, round=2).policy, "engine v2", score=2.0), ctx
    )
    assert keep_best(state, better) is True
    assert Path(ctx.engine_publish_path).read_text(encoding="utf-8") == "engine v2"
