"""generate 测试：prompt 渲染 / guards（纯逻辑）+ bootstrap 真跑（torch）+ 编排兜底（mock LLM）。"""

from __future__ import annotations

import importlib.util

import pytest

from mls_infer_opt.generate import (
    bootstrap,
    build_prompt,
    check_self_contained,
    propose,
    repair,
)
from mls_infer_opt.generate.codegen import _BASELINE_PATH
from mls_infer_opt.llm import FakeAgentClient
from mls_infer_opt.searchspace import aggregate, default_policy, merge
from mls_infer_opt.state.candidate import candidate_engine_path
from mls_infer_opt.state.context import Paths, TaskContext
from mls_infer_opt.state.eval import ValidationError

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


class FakeLLM:
    """最小 mock：available + generate(prompt)。text 为 Exception 时抛错（测兜底）。"""

    def __init__(self, text, available: bool = True) -> None:
        self.text = text
        self.available = available

    def generate(self, prompt: str):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _load_engine_module(path: str):
    spec = importlib.util.spec_from_file_location("cand_engine_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# === 1. prompt 渲染（纯逻辑） =========================================
def test_build_prompt_renders_nondefault_axes_and_knobs():
    policy = aggregate(
        {"kv_cache": "incremental", "attention": "sdpa"},
        {"kv_capacity_init": 256},
        kind="optimization",
        round=1,
        parent_id="r0-aaaa",
    ).policy
    text = build_prompt(policy, make_ctx_stub(), mode="propose", parent_code="# parent")

    assert "kv_cache = incremental" in text
    assert "attention = sdpa" in text
    assert "kv_capacity_init=256" in text  # 激活轴的 knob 值被注入
    # 默认轴不出现（norm 默认是 rmsnorm_fp32）
    assert "norm = rmsnorm_fp32" not in text
    # 契约稳定知识在场
    assert "create_engine" in text and "allclose" in text


def test_build_prompt_renders_policy_rationale_in_propose():
    policy = aggregate(
        {"attention": "sdpa"},
        kind="optimization",
        round=1,
        parent_id="r0-base",
        rationale="decode 是瓶颈：合批后注意力仍是 O(n²)，优先上 SDPA。",
    ).policy
    text = build_prompt(policy, make_ctx_stub(), mode="propose", parent_code="# parent")
    assert "analyze 方向提示" in text
    assert "decode 是瓶颈" in text


def test_merge_overlays_delta_and_carries_rationale():
    parent = aggregate({"attention": "sdpa"}, parent_id="r0-base").policy
    child = merge(
        parent,
        axes_delta={"kv_cache": "incremental"},
        kind="optimization",
        round=2,
        parent_id="r1-base",
        rationale="加 KV cache 去掉重算。",
    ).policy
    # parent 的轴被继承，delta 叠加，结果合法
    assert child.axes["attention"] == "sdpa"
    assert child.axes["kv_cache"] == "incremental"
    assert child.rationale == "加 KV cache 去掉重算。"
    assert child.round == 2 and child.parent_id == "r1-base"


def test_build_prompt_repair_includes_validation_error():
    policy = default_policy()
    err = ValidationError(
        stage="correctness",
        message="logits mismatch on decode",
        case="multi_decode",
        max_abs_err=0.5,
        expected_shape=[2, 100],
        actual_shape=[2, 100],
    )
    text = build_prompt(policy, make_ctx_stub(), mode="repair", parent_code="# bad", errors=[err])
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


# === 4. propose/repair 编排兜底（mock LLM） ===================
def test_propose_persists_candidate_on_good_llm(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = aggregate(
        {"attention": "sdpa"}, kind="optimization", round=1, parent_id="r0-base"
    ).policy
    llm = FakeLLM(f"好的：\n```python\n{BASELINE_CODE}\n```\n完成")

    cand = propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert cand.kind == "optimization"
    assert cand.parent_id == "r0-base"
    assert "attention:sdpa" in cand.strategy_tags
    # 真落盘
    import os
    assert os.path.exists(candidate_engine_path(ctx.run_dir, cand.id))


def test_propose_returns_none_on_garbage(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = default_policy()
    llm = FakeLLM("no code here at all")
    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm) is None


def test_propose_returns_none_on_llm_exception(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = default_policy()
    llm = FakeLLM(RuntimeError("boom"))
    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm) is None


def test_propose_returns_none_when_unavailable(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = default_policy()
    down = FakeLLM("x", available=False)
    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=down) is None
    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=None) is None


def test_repair_persists_candidate(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = aggregate({}, kind="repair", round=2, parent_id="r1-bbbb").policy
    err = ValidationError(stage="syntax", message="unexpected indent")
    llm = FakeLLM(f"```python\n{BASELINE_CODE}\n```")
    cand = repair(ctx, policy, parent_code=BASELINE_CODE, errors=[err], llm=llm)
    assert cand is not None and cand.kind == "repair"


def test_propose_accepts_agent_client_interface(tmp_path):
    ctx = make_ctx(tmp_path)
    policy = aggregate(
        {"attention": "sdpa"}, kind="optimization", round=1, parent_id="r0-base"
    ).policy
    llm = FakeAgentClient([f"```python\n{BASELINE_CODE}\n```"])

    cand = propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert cand.kind == "optimization"
    assert llm.prompts and "attention = sdpa" in llm.prompts[0]


# === 5. Policy 下沉 state =============================================
def test_policy_dataclass_lives_in_state():
    from mls_infer_opt.searchspace import Policy as SearchspacePolicy
    from mls_infer_opt.state import Policy as StatePolicy

    # searchspace 仅 re-export，类型真身在 state（analyze 不必 import generate 即可产 Policy）。
    assert SearchspacePolicy is StatePolicy
    assert StatePolicy.__module__ == "mls_infer_opt.state.policy"


# === 6. 自检重试自闭环（generate 驱动；需 torch + 权重） ==============
def test_propose_degrades_without_weights_emits_immediately(tmp_path):
    """无 model.pt → 无法 quick 自检 → 静态过即出候选，不重试、不起子进程。"""
    ctx = make_ctx(tmp_path)  # make_ctx 不写权重
    policy = default_policy()
    llm = FakeAgentClient([f"```python\n{BASELINE_CODE}\n```"])

    cand = propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None
    assert len(llm.prompts) == 1  # 一次出码、直接放行


def test_propose_passes_quick_gate_with_weights(tmp_path):
    """有权重 → 真跑 quick_gate（子进程）→ baseline 过门 → 出候选。"""
    pytest.importorskip("torch")
    ctx = make_ctx(tmp_path)
    _write_toy_weights(ctx)
    policy = aggregate({"attention": "sdpa"}, kind="optimization", parent_id="r0-base").policy
    llm = FakeAgentClient([f"```python\n{BASELINE_CODE}\n```"])

    cand = propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None and cand.kind == "optimization"
    assert len(llm.prompts) == 1  # 首次即过 quick


def test_propose_self_check_retries_on_static_failure(tmp_path):
    """首轮静态不过 → 回灌结构化错误、转 repair 提示 → 次轮修对 → 出候选。"""
    pytest.importorskip("torch")
    ctx = make_ctx(tmp_path)
    _write_toy_weights(ctx)
    policy = aggregate({"attention": "sdpa"}, kind="optimization", parent_id="r0-base").policy
    bad = f"```python\nimport mls_infer_opt\n{BASELINE_CODE}\n```"  # forbidden import → 静态不过
    good = f"```python\n{BASELINE_CODE}\n```"
    llm = FakeAgentClient([bad, good])

    cand = propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm)
    assert cand is not None and cand.kind == "optimization"
    assert len(llm.prompts) == 2
    # 次轮 prompt 含首轮错误且切到 repair 模式（修自己的码）
    assert "forbidden import" in llm.prompts[1]
    assert "待修复" in llm.prompts[1]


def test_propose_never_throws_on_llm_exception_each_round(tmp_path):
    """每轮 LLM 调用都抛 → 守卫成 None（不漏给 loop），并在预算内逐轮重试。"""
    ctx = make_ctx(tmp_path)
    policy = default_policy()
    llm = FakeAgentClient([RuntimeError("net")] * 4)

    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm) is None
    assert len(llm.prompts) == 4  # 异常逐轮跳过、耗满自检预算，never-throw


def test_propose_returns_none_when_self_check_never_passes(tmp_path):
    """持续产静态不过的码 → 耗尽自检预算 → None（这轮无收益）。"""
    ctx = make_ctx(tmp_path)
    policy = default_policy()
    bad = "import mls_infer_opt\nthis is not valid python ："
    llm = FakeAgentClient([bad] * 10)

    assert propose(ctx, policy, parent_code=BASELINE_CODE, llm=llm) is None
    assert len(llm.prompts) == 4  # _MAX_SELF_CHECK_ROUNDS
