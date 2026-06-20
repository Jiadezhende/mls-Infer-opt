"""analyze 测试：态势汇总 / 解析（纯逻辑）+ 编排（fake LLM）。

全部纯逻辑、不需 torch。LLM 是唯一方向源：不可用 → NoMove("llm_unavailable")；内容失败重试一次
仍败 → NoMove("llm_content_failure")；C2 穿透。analyze 产 Gradient（松建议，不 merge 定点）。
LLM 路径用 llm.FakeAgentClient 驱动（对齐 test_generate.py 约定）。
"""

from __future__ import annotations

import pytest

from mls_infer_opt.analyze import (
    analyze,
    build_analyze_prompt,
    build_situation,
    parse_gradient,
)
from mls_infer_opt.llm import FakeAgentClient, LLMCallError, LLMError
from mls_infer_opt.state.candidate import Candidate
from mls_infer_opt.state.context import Limits, Paths, TaskContext
from mls_infer_opt.state.eval import BenchResult, GateResult, ValidationError
from mls_infer_opt.state.gradient import Gradient, NoMove
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
) -> LoopState:
    """构一个带「已过门 + 已测速」best 的 LoopState。

    best 的「已应用轴」靠 honest strategy_tags 表达（analyze 不再读 policy.json，situation 从
    strategy_tags 还原 best_axes）。
    """
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


# === 2. 硬上限判停已上移总控（见 test_loop.test_hard_stop_*）；analyze 不再判停 ====

# === 3. parse_gradient（防御式，纯逻辑） ============================
def test_parse_gradient_valid_continue_block():
    text = _json_block(
        '{"action":"continue","suggest_axes":{"kv_cache":"incremental"},'
        '"knobs":{"kv_capacity_init":512},"rationale":"上 KV cache","bottleneck":"decode 慢"}'
    )
    g = parse_gradient(text)
    assert isinstance(g, Gradient)
    assert g.suggest_axes == {"kv_cache": "incremental"}
    assert g.knobs == {"kv_capacity_init": 512}
    assert g.bottleneck == "decode 慢"


def test_parse_gradient_sanitizes_invalid_axes():
    # 未知轴 / 非法选项被词表闸丢弃；合法的留下。
    text = _json_block(
        '{"action":"continue","suggest_axes":{"kv_cache":"incremental","bogus":"x",'
        '"attention":"not_an_option"}}'
    )
    g = parse_gradient(text)
    assert isinstance(g, Gradient)
    assert g.suggest_axes == {"kv_cache": "incremental"}


def test_parse_gradient_stop_returns_nomove():
    g = parse_gradient(_json_block('{"action":"stop","stop_reason":"diminishing_returns"}'))
    assert isinstance(g, NoMove) and g.reason == "diminishing_returns"


def test_parse_gradient_garbage_returns_none():
    assert parse_gradient("no json here at all") is None
    assert parse_gradient("```json\nnot valid json :\n```") is None
    assert parse_gradient(None) is None


# === 5. analyze 编排（fake LLM 兜底） ===============================
def _analyze_events(state: LoopState):
    return [e for e in state.events if e.source == "analyze"]


def test_analyze_uses_llm_direction(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient([
        _json_block(
            '{"action":"continue","suggest_axes":{"kv_cache":"incremental"},'
            '"rationale":"decode 是瓶颈，先上增量 KV","bottleneck":"decode O(n²)"}'
        )
    ])
    g = analyze(state, llm=llm)

    assert isinstance(g, Gradient)
    assert g.suggest_axes["kv_cache"] == "incremental"
    assert g.kind == "optimization"
    assert g.parent_id == "r0-base"
    assert g.round == state.round + 1
    assert "decode 是瓶颈" in g.rationale
    events = _analyze_events(state)
    assert len(events) == 1
    assert events[0].data["decision"] == "continue"
    assert events[0].data["used_llm"] is True
    assert events[0].data["suggest_axes"] == {"kv_cache": "incremental"}


def test_analyze_does_not_merge_with_best(tmp_path):
    # best 已应用 kv_cache（strategy_tags）；LLM 只给 rope → gradient 只含 rope（不再 merge 定点），
    # best 的 kv_cache 仍出现在诊断 prompt（供 LLM 参考），但不混进 suggest_axes。
    state = make_state(tmp_path, best_axes={"kv_cache": "incremental"})
    llm = FakeAgentClient([
        _json_block('{"action":"continue","suggest_axes":{"rope":"precomputed_table"},'
                    '"rationale":"预算 RoPE 表"}')
    ])
    g = analyze(state, llm=llm)
    assert isinstance(g, Gradient)
    assert g.suggest_axes == {"rope": "precomputed_table"}  # 只 LLM 的建议、不 merge best
    assert "kv_cache" not in g.suggest_axes
    assert "kv_cache=incremental" in llm.prompts[0]  # best 应用轴进了诊断 prompt


def test_analyze_empty_suggestion_still_proposes(tmp_path):
    # 建议全被词表闸滤空 + action=continue → 仍返回 Gradient（交 propose 自由探索），不误判 NoMove。
    state = make_state(tmp_path)
    llm = FakeAgentClient([
        _json_block('{"action":"continue","suggest_axes":{"bogus":"x"},"rationale":"放手探索"}')
    ])
    g = analyze(state, llm=llm)
    assert isinstance(g, Gradient)
    assert g.suggest_axes == {}
    assert g.kind == "optimization" and g.round == state.round + 1


def test_analyze_content_failure_retries_once_then_nomove(tmp_path):
    # C1 内容失败：解析不出 → 重试一次（共调 2 次）仍败 → NoMove("llm_content_failure")。
    state = make_state(tmp_path)
    llm = FakeAgentClient(["garbage one", "garbage two"])
    result = analyze(state, llm=llm)
    assert isinstance(result, NoMove)
    assert result.reason == "llm_content_failure"
    assert len(llm.prompts) == 2  # 初试 + 重试一次
    events = _analyze_events(state)
    assert len(events) == 1 and events[0].data["used_llm"] is True
    assert events[0].data["stop_reason"] == "llm_content_failure"


def test_analyze_retries_once_then_recovers(tmp_path):
    # 第一次 garbage、第二次有效：重试救回，返回 Gradient（验证确实再问一次且能恢复）。
    state = make_state(tmp_path)
    llm = FakeAgentClient([
        "unparseable",
        _json_block('{"action":"continue","suggest_axes":{"kv_cache":"incremental"},'
                    '"rationale":"重试后给出方向"}'),
    ])
    g = analyze(state, llm=llm)
    assert isinstance(g, Gradient)
    assert g.suggest_axes["kv_cache"] == "incremental"
    assert len(llm.prompts) == 2


def test_analyze_unexpected_non_llm_error_is_content_failure(tmp_path):
    # 非 LLMError 的意外异常当内容失败：进重试，仍败 → NoMove("llm_content_failure")。
    state = make_state(tmp_path)
    llm = FakeAgentClient([RuntimeError("boom")])  # 第二次 responses 空 → ok=False
    result = analyze(state, llm=llm)
    assert isinstance(result, NoMove) and result.reason == "llm_content_failure"
    assert len(llm.prompts) == 2


def test_analyze_propagates_c2_llm_infra_error(tmp_path):
    # C2 传输失败：analyze 不吞、不重试、穿透交总控。
    state = make_state(tmp_path)
    llm = FakeAgentClient([LLMCallError("network down")])
    with pytest.raises(LLMError):
        analyze(state, llm=llm)


def test_analyze_no_llm_returns_unavailable_nomove(tmp_path):
    state = make_state(tmp_path)
    result = analyze(state, llm=None)
    assert isinstance(result, NoMove) and result.reason == "llm_unavailable"
    events = _analyze_events(state)
    assert len(events) == 1 and events[0].data["used_llm"] is False


def test_analyze_unavailable_llm_returns_unavailable_nomove(tmp_path):
    state = make_state(tmp_path)
    down = FakeAgentClient([], available=False)
    result = analyze(state, llm=down)
    assert isinstance(result, NoMove) and result.reason == "llm_unavailable"
    assert not down.prompts  # 不可用 → 根本没问 LLM


def test_analyze_ignores_hard_limits_now_in_loop(tmp_path):
    # 硬上限判停已上移总控：analyze 即便 stale 到上限也只算方向（不再判停），返回 Gradient。
    state = make_state(tmp_path, stale_rounds=3, limits=Limits(max_stale_rounds=3))
    llm = FakeAgentClient([
        _json_block('{"action":"continue","suggest_axes":{"kv_cache":"incremental"},'
                    '"rationale":"仍给方向"}')
    ])
    result = analyze(state, llm=llm)
    assert isinstance(result, Gradient)
    assert result.suggest_axes["kv_cache"] == "incremental"


def test_analyze_llm_stop_decision_returns_nomove(tmp_path):
    state = make_state(tmp_path)
    llm = FakeAgentClient([_json_block('{"action":"stop","stop_reason":"target_reached"}')])
    result = analyze(state, llm=llm)
    assert isinstance(result, NoMove)
    assert result.reason == "target_reached"
    # analyze 绝不自己写 stop_reason，交给总控
    assert state.stop_reason == ""
    events = _analyze_events(state)
    assert len(events) == 1 and events[0].data["stop_reason"] == "target_reached"


# === 6. build_analyze_prompt（纯逻辑 smoke） =========================
def test_build_prompt_contains_dims_and_situation(tmp_path):
    state = make_state(tmp_path, best_axes={"attention": "sdpa"})
    sit = build_situation(state)
    text = build_analyze_prompt(sit)
    assert "kv_cache" in text  # 搜索维度菜单在场
    assert "attention=sdpa" in text  # 当前 best 非默认轴
    assert "json" in text  # 输出格式约定
