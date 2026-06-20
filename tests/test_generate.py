"""generate 测试：prompt 渲染 / guards（纯逻辑）+ bootstrap 真跑（torch）+ 编排兜底（mock LLM）。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mls_infer_opt.generate import (
    bootstrap,
    build_prompt,
    check_self_contained,
    propose,
    repair,
)
from mls_infer_opt.generate.codegen import _BASELINE_PATH, _MAX_SELF_CHECK_ROUNDS
from mls_infer_opt.llm import FakeAgentClient, LLMConfig, LLMError, OpenAIAgentClient
from mls_infer_opt.state.candidate import candidate_dir, candidate_engine_path
from mls_infer_opt.state.context import Paths, TaskContext
from mls_infer_opt.state.eval import ValidationError
from mls_infer_opt.state.gradient import Gradient

SMALL_CONFIG = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_size": 64,
    "intermediate_size": 128,
    "vocab_size": 100,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "torch_dtype": "float32",
}

BASELINE_CODE = _BASELINE_PATH.read_text(encoding="utf-8")


def make_ctx(tmp_path) -> TaskContext:
    target = tmp_path / "target"
    (target / "weights").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    return TaskContext(
        model_config=SMALL_CONFIG,
        device="cpu",
        run_id="t0",
        paths=Paths(
            target_dir=str(target),
            runs_dir=str(tmp_path / "runs"),
            output_dir=str(tmp_path / "out"),
        ),
    )


def _write_toy_weights(ctx: TaskContext) -> None:
    """合成随机权重落 ``weight_dir/model.pt``（key 布局对齐 target/generate_toy_weights.py）。

    供 quick_gate 自检路径用：oracle 与候选共用同一份权重，baseline 应过 allclose。
    """
    import os

    import torch

    c = SMALL_CONFIG
    h, heads, kv, hd, inter, vocab, layers = (
        c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"],
        c["head_dim"], c["intermediate_size"], c["vocab_size"], c["num_hidden_layers"],
    )
    q_out, kv_out = heads * hd, kv * hd
    sd = {"embed_tokens.weight": torch.randn(vocab, h), "norm.weight": torch.ones(h),
          "lm_head.weight": torch.randn(vocab, h)}
    for i in range(layers):
        p = f"layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(h)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(q_out, h)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_out, h)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_out, h)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, q_out)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, h)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, h)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(h, inter)
    os.makedirs(ctx.weight_dir, exist_ok=True)
    torch.save(sd, os.path.join(ctx.weight_dir, "model.pt"))


def _load_engine_module(path: str):
    spec = importlib.util.spec_from_file_location("cand_engine_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# === 1. prompt 渲染（纯逻辑） =========================================
def test_build_prompt_renders_suggestions_menu_and_knobs():
    gradient = Gradient(
        suggest_axes={"kv_cache": "incremental", "attention": "sdpa"},
        knobs={"kv_capacity_init": 256},
        kind="optimization",
        round=1,
        parent_id="r0-aaaa",
    )
    text = build_prompt(gradient, make_ctx_stub(), mode="propose", parent_code="# parent")

    # analyze 松建议渲成清单
    assert "kv_cache = incremental" in text
    assert "attention = sdpa" in text
    assert "kv_capacity_init=256" in text  # 配套 knob 建议值
    # 完整搜索维度（界）+ 依赖规则在场，供 agent 自由探索
    assert "可探索的搜索维度" in text and "torch_compile" in text
    assert "轴间依赖" in text
    # 契约稳定知识在场
    assert "create_engine" in text and "allclose" in text


def test_build_prompt_renders_rationale_in_propose():
    gradient = Gradient(
        suggest_axes={"attention": "sdpa"},
        kind="optimization",
        round=1,
        parent_id="r0-base",
        rationale="decode 是瓶颈：合批后注意力仍是 O(n²)，优先上 SDPA。",
    )
    text = build_prompt(gradient, make_ctx_stub(), mode="propose", parent_code="# parent")
    assert "analyze 方向提示" in text
    assert "decode 是瓶颈" in text


def test_build_prompt_empty_suggestions_says_free_explore():
    text = build_prompt(Gradient(), make_ctx_stub(), mode="propose", parent_code="# parent")
    assert "无具体建议" in text  # 空建议 → 明确放手在维度界内自由探索


def test_build_prompt_repair_includes_validation_error():
    err = ValidationError(
        stage="correctness",
        message="logits mismatch on decode",
        case="multi_decode",
        max_abs_err=0.5,
        expected_shape=[2, 100],
        actual_shape=[2, 100],
    )
    text = build_prompt(
        Gradient(kind="repair"), make_ctx_stub(), mode="repair", parent_code="# bad", errors=[err]
    )
    assert "logits mismatch on decode" in text
    assert "multi_decode" in text
    assert "待修复" in text


def make_ctx_stub() -> TaskContext:
    return TaskContext(model_config=SMALL_CONFIG, device="cpu")


# === 2. 自包含 guards（纯逻辑） ======================================
def test_guards_accept_baseline():
    assert check_self_contained(BASELINE_CODE) == []


def test_guards_reject_agent_import():
    bad = "import mls_infer_opt\n" + BASELINE_CODE
    problems = check_self_contained(bad)
    assert any("forbidden import" in p for p in problems)


def test_guards_reject_missing_api():
    # 改名 prefill 方法 → API 契约不满足
    stripped = BASELINE_CODE.replace(
        "    def prefill(self, request_ids, input_ids):",
        "    def _disabled(self, request_ids, input_ids):",
    )
    problems = check_self_contained(stripped)
    assert any("prefill" in p for p in problems)


def test_guards_reject_syntax_error():
    problems = check_self_contained("def create_engine(:\n  pass")
    assert problems and "syntax error" in problems[0]


# === 3. bootstrap 真能跑（需 torch） =================================
def test_bootstrap_engine_runs(tmp_path):
    torch = pytest.importorskip("torch")
    ctx = make_ctx(tmp_path)

    cand = bootstrap(ctx)
    assert cand.kind == "baseline"
    assert cand.parent_id is None

    _write_toy_weights(ctx)
    vocab = SMALL_CONFIG["vocab_size"]

    mod = _load_engine_module(candidate_engine_path(ctx.run_dir, cand.id))
    engine = mod.create_engine(SMALL_CONFIG, ctx.weight_dir, "cpu")

    out = engine.prefill([0, 1], [torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    assert tuple(out.shape) == (2, vocab)
    out2 = engine.decode([0, 1], torch.tensor([6, 7]))
    assert tuple(out2.shape) == (2, vocab)
    engine.remove([0])
    out3 = engine.decode([1], torch.tensor([8]))
    assert tuple(out3.shape) == (1, vocab)


# === 4. propose/repair 编排兜底（模型只回文本、没调工具 → 抽码 + 确定性复核的回退路径） ===
def _opt_grad(**kw) -> Gradient:
    base = dict(suggest_axes={"attention": "sdpa"}, kind="optimization", round=1,
                parent_id="r0-base")
    base.update(kw)
    return Gradient(**base)


def test_propose_persists_candidate_on_good_llm(tmp_path):
    ctx = make_ctx(tmp_path)
    llm = FakeAgentClient([f"好的：\n```python\n{BASELINE_CODE}\n```\n完成"])

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert cand.kind == "optimization"
    assert cand.parent_id == "r0-base"
    # honest tags：回退文本路径无 agent 采用回报 → 空 tags（不再照 gradient 的 sdpa 撒谎）。
    assert cand.strategy_tags == []
    # 真落盘
    import os
    assert os.path.exists(candidate_engine_path(ctx.run_dir, cand.id))


def test_propose_returns_none_on_garbage(tmp_path):
    ctx = make_ctx(tmp_path)
    llm = FakeAgentClient(["no code here at all"])
    assert propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=llm) is None


def test_propose_returns_none_on_llm_exception(tmp_path):
    ctx = make_ctx(tmp_path)
    llm = FakeAgentClient([RuntimeError("boom")])
    assert propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=llm) is None


def test_propose_returns_none_when_unavailable(tmp_path):
    ctx = make_ctx(tmp_path)
    down = FakeAgentClient([], available=False)
    assert propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=down) is None
    assert propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=None) is None


def test_repair_persists_candidate(tmp_path):
    ctx = make_ctx(tmp_path)
    gradient = Gradient(kind="repair", round=2, parent_id="r1-bbbb")
    err = ValidationError(stage="syntax", message="unexpected indent")
    llm = FakeAgentClient([f"```python\n{BASELINE_CODE}\n```"])
    cand = repair(ctx, gradient, parent_code=BASELINE_CODE, errors=[err], llm=llm)
    assert cand is not None and cand.kind == "repair"


def test_propose_accepts_agent_client_interface(tmp_path):
    ctx = make_ctx(tmp_path)
    llm = FakeAgentClient([f"```python\n{BASELINE_CODE}\n```"])

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert cand.kind == "optimization"
    assert llm.prompts and "attention = sdpa" in llm.prompts[0]  # 松建议渲进 prompt


# === 5. Gradient 契约在 state =========================================
def test_gradient_dataclass_lives_in_state():
    from mls_infer_opt.state import Gradient as StateGradient

    # 契约真身在 state（analyze 不必 import generate 即可产 Gradient）。
    assert StateGradient is Gradient
    assert StateGradient.__module__ == "mls_infer_opt.state.gradient"


# === 6. agent 工具自检自闭环（真 OpenAIAgentClient + 脚本化 Responses；需 torch + 权重）====
class _FakeResponses:
    """脚本化 Responses.create：按序吐 output（dict）或抛 Exception（测 never-throw）。"""

    def __init__(self, outputs) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeOpenAI:
    def __init__(self, outputs) -> None:
        self.responses = _FakeResponses(outputs)


def _check_engine_call(
    code: str,
    call_id: str = "c1",
    *,
    applied_axes: dict | None = None,
    applied_knobs: dict | None = None,
) -> dict:
    """一轮模型请求调 check_engine(code, applied_axes?, applied_knobs?) 的 Responses item。"""
    args: dict = {"code": code}
    if applied_axes is not None:
        args["applied_axes"] = applied_axes
    if applied_knobs is not None:
        args["applied_knobs"] = applied_knobs
    return {
        "output": [
            {
                "type": "function_call",
                "name": "check_engine",
                "arguments": json.dumps(args),
                "call_id": call_id,
            }
        ]
    }


def _final_message(text: str = "done") -> dict:
    """一轮模型给最终答复、无更多工具调用 → agent 收束。"""
    return {"output_text": text, "output": []}


def _agent_client(outputs) -> tuple[OpenAIAgentClient, _FakeOpenAI]:
    fake = _FakeOpenAI(outputs)
    return OpenAIAgentClient(LLMConfig(api_key="x"), client=fake), fake


def test_propose_agent_loop_degrades_without_weights(tmp_path):
    """无 model.pt → check_engine 静态过即放行（quick 跳过）→ 出候选；落盘码即 agent 提交码。"""
    ctx = make_ctx(tmp_path)  # 不写权重
    llm, fake = _agent_client([_check_engine_call(BASELINE_CODE), _final_message()])

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None and cand.kind == "optimization"
    persisted = Path(candidate_engine_path(ctx.run_dir, cand.id)).read_text(encoding="utf-8")
    assert persisted == BASELINE_CODE  # 持久化「过 check_engine 的码」
    # 模型确实拿到了 check_engine 工具
    assert "check_engine" in [t["name"] for t in fake.responses.calls[0]["tools"]]


def _read_applied(ctx: TaskContext, cand_id: str) -> dict:
    return json.loads(
        Path(f"{candidate_dir(ctx.run_dir, cand_id)}/applied.json").read_text(encoding="utf-8")
    )


def test_propose_agent_report_drives_honest_tags(tmp_path):
    """agent 过门时回报 applied_axes/knobs → strategy_tags 取自回报（过词表闸），与 policy 无关。"""
    ctx = make_ctx(tmp_path)  # 无权重 → 静态过即捕获回报
    # gradient 建议 attention=sdpa，但 agent 自述实际只动了 kv_cache（且夹带一个非法轴 bogus）。
    llm, _ = _agent_client(
        [
            _check_engine_call(
                BASELINE_CODE,
                applied_axes={"kv_cache": "incremental", "bogus": "x"},
                applied_knobs={"kv_capacity_init": 128},
            ),
            _final_message(),
        ]
    )

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    # honest：tags 来自 agent 回报、过 sanitize（bogus 被丢），不再来自 gradient 的 attention:sdpa。
    assert cand.strategy_tags == ["kv_cache:incremental"]
    assert "attention:sdpa" not in cand.strategy_tags
    # applied.json 留档实际采用（axes 已过词表闸，knobs 原样）；不再写 policy.json（定点已删）。
    applied = _read_applied(ctx, cand.id)
    assert applied["applied_axes"] == {"kv_cache": "incremental"}
    assert applied["applied_knobs"] == {"kv_capacity_init": 128}
    assert not Path(f"{candidate_dir(ctx.run_dir, cand.id)}/policy.json").exists()


def test_propose_agent_no_report_yields_empty_tags(tmp_path):
    """agent 过门但没回报 applied_axes → 空 tags（诚实「未报」），不照 gradient 撒谎。"""
    ctx = make_ctx(tmp_path)
    llm, _ = _agent_client([_check_engine_call(BASELINE_CODE), _final_message()])

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert cand.strategy_tags == []
    assert _read_applied(ctx, cand.id)["applied_axes"] == {}


def test_propose_agent_loop_passes_quick_gate_with_weights(tmp_path):
    """有权重 → check_engine 真跑 quick_gate 子进程 → baseline 过门 → 出候选。"""
    pytest.importorskip("torch")
    ctx = make_ctx(tmp_path)
    _write_toy_weights(ctx)
    llm, _ = _agent_client([_check_engine_call(BASELINE_CODE), _final_message()])

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None and cand.kind == "optimization"


def test_propose_agent_loop_retries_then_fixes_on_error(tmp_path):
    """首次提交静态不过 → check_engine 回 errors → agent 改对 → 出候选；落盘是修好的码。"""
    pytest.importorskip("torch")
    ctx = make_ctx(tmp_path)
    _write_toy_weights(ctx)
    bad = "import mls_infer_opt\n" + BASELINE_CODE  # forbidden import → 静态不过、不写 captured
    llm, fake = _agent_client(
        [_check_engine_call(bad, "c1"), _check_engine_call(BASELINE_CODE, "c2"), _final_message()]
    )

    cand = propose(ctx, _opt_grad(), parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    persisted = Path(candidate_engine_path(ctx.run_dir, cand.id)).read_text(encoding="utf-8")
    assert persisted == BASELINE_CODE and "import mls_infer_opt" not in persisted  # captured-wins
    assert len(fake.responses.calls) == 3  # 两次 check_engine + 终轮 message


def test_propose_propagates_c2_on_create_exception(tmp_path):
    """create 抛错 = C2 传输失败 → run_agent raise LLMCallError → propose 穿透（不吞）。"""
    ctx = make_ctx(tmp_path)
    llm, _ = _agent_client([RuntimeError("net")])

    with pytest.raises(LLMError):
        propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=llm)


def test_propose_agent_loop_returns_none_when_never_passes(tmp_path):
    """恒提交静态不过的码 → 触顶 max_tool_rounds、ok=False → None（这轮无收益）。"""
    ctx = make_ctx(tmp_path)
    bad = "import mls_infer_opt\nthis is not valid python ："
    # max_tool_rounds=_MAX_SELF_CHECK_ROUNDS → 客户端最多 rounds+1 次 create
    outputs = [_check_engine_call(bad, f"c{i}") for i in range(_MAX_SELF_CHECK_ROUNDS + 1)]
    llm, fake = _agent_client(outputs)

    assert propose(ctx, Gradient(), parent_code=BASELINE_CODE, llm=llm) is None
    assert len(fake.responses.calls) == _MAX_SELF_CHECK_ROUNDS + 1  # 触顶自检预算
