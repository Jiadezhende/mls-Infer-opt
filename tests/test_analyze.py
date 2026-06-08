"""analyze 测试：态势汇总 / 判停 / rule-based 阶梯 / 解析（纯逻辑）+ 编排兜底（fake LLM）。

全部纯逻辑、不需 torch。LLM 路径用 llm.FakeAgentClient 驱动（对齐 test_generate.py 约定）。
"""

from __future__ import annotations

from pathlib import Path

from mls_infer_opt.analyze import (
    MOVES,
    analyze,
    build_analyze_prompt,
    build_situation,
    hard_stop_reason,
    heuristic_decision,
    parse_decision,
)
from mls_infer_opt.llm import FakeAgentClient
from mls_infer_opt.searchspace.policy import aggregate, default_policy, to_json
from mls_infer_opt.state.candidate import Candidate, candidate_policy_path
from mls_infer_opt.state.context import Limits, Paths, TaskContext
from mls_infer_opt.state.eval import BenchResult, GateResult, ValidationError
from mls_infer_opt.state.loop import LoopState


# === 夹具 =============================================================
def _passing_gate() -> GateResult:
    return GateResult(syntax_ok=True, api_ok=True, correctness_ok=True, passed=True)


def make_state(
    tmp_path,
    *,
    best_axes: dict[str, str] | None = None,
    best_score: float = 1.0,
    round: int = 1,
    stale_rounds: int = 0,
    limits: Limits | None = None,
    write_policy: bool = True,
) -> LoopState:
    """构一个带「已过门 + 已测速」best 的 LoopState，并把 best 的 policy.json 落盘。"""
    paths = Paths(
        target_dir=str(tmp_path / "target"),
        runs_dir=str(tmp_path / "runs"),
        output_dir=str(tmp_path / "out"),
    )
    ctx = TaskContext(
        model_config={"num_hidden_layers": 2},
        device="cpu",
        run_id="t0",
        paths=paths,
        limits=limits or Limits(),
    )
    state = LoopState(task_context=ctx, round=round, stale_rounds=stale_rounds)

    policy = aggregate(best_axes or {}, kind="baseline").policy
    best = Candidate(
        id="r0-base",
        kind="baseline",
        round=0,
        parent_id=None,
        strategy_tags=[f"{k}:{v}" for k, v in (best_axes or {}).items()],
        gate=_passing_gate(),
    )
    best.attach_bench(BenchResult(score=best_score, prefill_tps=100.0, decode_tps=50.0))
    state.add_candidate(best)
    state.set_best(best, best_score)

    if write_policy:
        path = Path(candidate_policy_path(ctx.run_dir, best.id))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_json(policy), encoding="utf-8")
    return state


def _add_rejected(state: LoopState, cid: str, err: ValidationError, round: int = 1) -> None:
    cand = Candidate(id=cid, kind="optimization", round=round, parent_id="r0-base")
    cand.gate = GateResult(syntax_ok=True, api_ok=True, correctness_ok=False, passed=False,
                           errors=[err])
    state.add_candidate(cand)


def _json_block(payload: str) -> str:
    return f"前言\n```json\n{payload}\n```\n后记"


# === 1. build_situation（纯逻辑） ====================================
def test_build_situation_derives_best_and_failures(tmp_path):
    state = make_state(tmp_path, best_axes={"attention": "sdpa"}, best_score=2.5)
    _add_rejected(
        state, "r1-bad",
        ValidationError(stage="correctness", message="logits mismatch", case="multi_decode",
                        max_abs_err=0.5),
    )

    sit = build_situation(state)
    assert sit.best_id == "r0-base"
    assert sit.best_score == 2.5
    assert sit.best_axes["attention"] == "sdpa"
    assert sit.applied_axes == {"attention": "sdpa"}  # 只列非默认轴
    assert sit.best_bench is not None and sit.best_bench.prefill_tps == 100.0
    assert sit.n_candidates == 2 and sit.n_rejected == 1
    assert any("logits mismatch" in e.message for e in sit.recent_failures)


# === 2. hard_stop_reason（纯逻辑） ===================================
def test_hard_stop_inactive_when_limits_zero(tmp_path):
    state = make_state(tmp_path, round=99, stale_rounds=99)  # 默认 Limits 全 0 → 不生效
    assert hard_stop_reason(build_situation(state)) is None


def test_hard_stop_each_limit(tmp_path):
    rounds = make_state(tmp_path, round=5, limits=Limits(max_rounds=5))
    assert hard_stop_reason(build_situation(rounds)) == "max_rounds_reached"

    stale = make_state(tmp_path, stale_rounds=3, limits=Limits(max_stale_rounds=3))
    assert hard_stop_reason(build_situation(stale)) == "max_stale_rounds_reached"

    timed = make_state(tmp_path, limits=Limits(time_budget_s=10))
    timed.budget.elapsed_s = 11.0
    assert hard_stop_reason(build_situation(timed)) == "time_budget_exhausted"


def test_hard_stop_under_limit_returns_none(tmp_path):
    state = make_state(tmp_path, round=2, limits=Limits(max_rounds=5))
    assert hard_stop_reason(build_situation(state)) is None


# === 3. heuristic_decision（贪心阶梯，纯逻辑） =======================
def test_heuristic_picks_kv_cache_first_from_baseline(tmp_path):
    sit = build_situation(make_state(tmp_path))
    decision = heuristic_decision(sit, default_policy())
    assert decision.action == "continue"
    assert decision.axes_delta == {"kv_cache": "incremental"}
    assert "rule-based" in decision.rationale


def test_heuristic_skips_already_applied_axis(tmp_path):
    best = aggregate({"kv_cache": "incremental"}, kind="baseline").policy
    sit = build_situation(make_state(tmp_path, best_axes={"kv_cache": "incremental"}))
    decision = heuristic_decision(sit, best)
    # kv_cache 已应用 → 跳到阶梯下一条 rope
    assert decision.axes_delta == {"rope": "precomputed_table"}


def test_heuristic_stops_when_ladder_exhausted(tmp_path):
    all_moves = {m.axis: m.option for m in MOVES}
    best = aggregate(all_moves, kind="baseline").policy
    sit = build_situation(make_state(tmp_path, best_axes=all_moves))
    decision = heuristic_decision(sit, best)
    assert decision.action == "stop"
    assert decision.stop_reason == "no_obvious_direction"


# === 4. parse_decision（防御式，纯逻辑） ============================
def test_parse_decision_valid_json_block():
    text = _json_block(
        '{"action":"continue","axes_delta":{"kv_cache":"incremental"},'
        '"knobs_delta":{"kv_capacity_init":512},"rationale":"上 KV cache","bottleneck":"decode 慢"}'
    )
    d = parse_decision(text)
    assert d is not None and d.action == "continue"
    assert d.axes_delta == {"kv_cache": "incremental"}
    assert d.knobs_delta == {"kv_capacity_init": 512}
    assert d.bottleneck == "decode 慢"


def test_parse_decision_stop():
    d = parse_decision(_json_block('{"action":"stop","stop_reason":"diminishing_returns"}'))
    assert d is not None and d.action == "stop" and d.stop_reason == "diminishing_returns"


def test_parse_decision_garbage_returns_none():
    assert parse_decision("no json here at all") is None
    assert parse_decision("```json\nnot valid json :\n```") is None
    assert parse_decision(None) is None


# === 5. analyze 编排（fake LLM 兜底） ===============================
def _analyze_events(state: LoopState):
    return [e for e in state.events if e.source == "analyze"]


def test_analyze_uses_llm_direction(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient([
        _json_block(
            '{"action":"continue","axes_delta":{"kv_cache":"incremental"},'
            '"rationale":"decode 是瓶颈，先上增量 KV","bottleneck":"decode O(n²)"}'
        )
    ])
    policy = analyze(state, llm=llm)

    assert policy is not None
    assert policy.axes["kv_cache"] == "incremental"
    assert policy.kind == "optimization"
    assert policy.parent_id == "r0-base"
    assert policy.round == state.round + 1
    assert "decode 是瓶颈" in policy.rationale
    events = _analyze_events(state)
    assert len(events) == 1
    assert events[0].data["decision"] == "continue"
    assert events[0].data["used_llm"] is True


def test_analyze_merges_delta_onto_best_policy(tmp_path):
    # best 已应用 kv_cache（落在 policy.json）；LLM 只给 rope delta → 结果应同时含两者。
    state = make_state(tmp_path, best_axes={"kv_cache": "incremental"})
    llm = FakeAgentClient([
        _json_block('{"action":"continue","axes_delta":{"rope":"precomputed_table"},'
                    '"rationale":"预算 RoPE 表"}')
    ])
    policy = analyze(state, llm=llm)
    assert policy is not None
    assert policy.axes["kv_cache"] == "incremental"  # 从 best 继承
    assert policy.axes["rope"] == "precomputed_table"  # delta 叠加


def test_analyze_falls_back_to_heuristic_on_garbage(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient(["completely unparseable response"])
    policy = analyze(state, llm=llm)
    assert policy is not None
    assert policy.axes["kv_cache"] == "incremental"  # 阶梯首步
    events = _analyze_events(state)
    assert len(events) == 1 and events[0].data["used_llm"] is False


def test_analyze_falls_back_on_llm_exception(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient([RuntimeError("boom")])
    policy = analyze(state, llm=llm)
    assert policy is not None and policy.axes["kv_cache"] == "incremental"
    assert len(_analyze_events(state)) == 1


def test_analyze_no_llm_uses_heuristic(tmp_path):
    state = make_state(tmp_path)
    policy = analyze(state, llm=None)
    assert policy is not None and policy.axes["kv_cache"] == "incremental"
    assert len(_analyze_events(state)) == 1


def test_analyze_unavailable_llm_uses_heuristic(tmp_path):
    state = make_state(tmp_path)
    down = FakeAgentClient([], available=False)
    policy = analyze(state, llm=down)
    assert policy is not None and policy.axes["kv_cache"] == "incremental"
    assert not down.prompts  # 不可用 → 根本没问 LLM


def test_analyze_hard_stop_returns_none_and_records_event(tmp_path):
    state = make_state(tmp_path, stale_rounds=3, limits=Limits(max_stale_rounds=3))
    policy = analyze(state, llm=None)
    assert policy is None
    events = _analyze_events(state)
    assert len(events) == 1
    assert events[0].data["decision"] == "stop"
    assert events[0].data["stop_reason"] == "max_stale_rounds_reached"
    # analyze 绝不自己写 stop_reason，交给 loop
    assert state.stop_reason == ""


def test_analyze_llm_stop_decision_returns_none(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient([_json_block('{"action":"stop","stop_reason":"target_reached"}')])
    policy = analyze(state, llm=llm)
    assert policy is None
    events = _analyze_events(state)
    assert len(events) == 1 and events[0].data["stop_reason"] == "target_reached"


def test_analyze_degrades_without_best_policy_file(tmp_path):
    # policy.json 缺失 → _load_best_policy 走 strategy_tags 还原 → 仍能产 Policy。
    state = make_state(tmp_path, best_axes={"kv_cache": "incremental"}, write_policy=False)
    policy = analyze(state, llm=None)
    assert policy is not None
    assert policy.axes["kv_cache"] == "incremental"  # 从 strategy_tags 还原


# === 6. build_analyze_prompt（纯逻辑 smoke） =========================
def test_build_prompt_contains_space_and_situation(tmp_path):
    state = make_state(tmp_path, best_axes={"attention": "sdpa"})
    sit = build_situation(state)
    best = aggregate({"attention": "sdpa"}, kind="baseline").policy
    text = build_analyze_prompt(sit, best)
    assert "kv_cache" in text  # 搜索空间菜单在场
    assert "attention=sdpa" in text  # 当前 best 非默认轴
    assert "json" in text  # 输出格式约定
