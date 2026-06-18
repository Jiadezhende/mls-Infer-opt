"""evaluate 测试：serde（纯逻辑）+ 子进程隔离评测（gate 通过/失败、api/syntax、超时/崩溃、
bench、never-throw）。

隔离类用例真起子进程跑 worker，较慢但正是要验证的核心——坏候选只死子进程、父进程拿结构化结果。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mls_infer_opt.evaluate.protocol import (
    JobSpec,
    bench_from_dict,
    gate_from_dict,
    job_from_json,
    job_to_json,
)
from mls_infer_opt.state.candidate import Candidate, candidate_engine_path
from mls_infer_opt.state.common import to_dict
from mls_infer_opt.state.context import Paths, TaskContext
from mls_infer_opt.state.eval import BenchResult, GateResult, ValidationError

# evaluate 只依赖 state——用测试自带的「已知正确引擎」当 golden，不借 generate 的 baseline，
# 避免 evaluate 测试耦合 generate（golden_engine.py 是冻结 stage-B API 的纯 torch 物料）。
_GOLDEN = Path(__file__).parent / "assets" / "golden_engine.py"
BASELINE_CODE = _GOLDEN.read_text(encoding="utf-8")

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


def _make_ctx(tmp_path) -> TaskContext:
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


def _write_weights(ctx: TaskContext) -> None:
    import torch

    torch.manual_seed(1234)
    c = SMALL_CONFIG
    h, heads, kv, hd, inter, vocab, layers = (
        c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"],
        c["head_dim"], c["intermediate_size"], c["vocab_size"], c["num_hidden_layers"],
    )
    q_out, kv_out = heads * hd, kv * hd
    sd = {
        "embed_tokens.weight": torch.randn(vocab, h) * 0.02,
        "norm.weight": torch.ones(h),
        "lm_head.weight": torch.randn(vocab, h) * 0.02,
    }
    for i in range(layers):
        p = f"layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(h)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(h)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(q_out, h) * 0.02
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_out, h) * 0.02
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_out, h) * 0.02
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, q_out) * 0.02
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, h) * 0.02
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, h) * 0.02
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(h, inter) * 0.02
    torch.save(sd, f"{ctx.weight_dir}/model.pt")


def _setup_candidate(ctx: TaskContext, code: str, cid: str = "r0-test") -> Candidate:
    """把候选源码落到候选目录，合成权重，返回 Candidate 元数据。"""
    engine_path = candidate_engine_path(ctx.run_dir, cid)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "w", encoding="utf-8") as f:
        f.write(code)
    _write_weights(ctx)
    return Candidate(id=cid, kind="baseline")


# === 0. run_job C1/C2 分流 + 重试（monkeypatch 子进程，无 torch） ======
def _spec() -> JobSpec:
    return JobSpec(
        engine_path="/x/engine.py", weight_dir="/w", model_config={"vocab_size": 100},
        device="cpu", mode="quick", task="both", seed=11, oracle_cache_path="/c.pt",
    )


def _proc(returncode: int, stdout: str, stderr: str = ""):
    import types

    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_job_trusts_worker_verdict_without_retry(monkeypatch):
    from mls_infer_opt.evaluate import runner

    calls: list[int] = []

    def fake_run(*a, **k):
        calls.append(1)
        return _proc(0, json.dumps({"gate": {"passed": False}, "bench": None}))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    out = runner.run_job(_spec())
    assert out["gate"]["passed"] is False
    assert len(calls) == 1  # 有裁决（C1/通过）→ 不重试


def test_run_job_retries_once_then_raises_c2(monkeypatch):
    from mls_infer_opt.evaluate import EvaluatorInfraError, runner

    calls: list[int] = []

    def fake_run(*a, **k):
        calls.append(1)
        return _proc(1, "", "segfault")  # 非零退出且无 JSON = C2

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(EvaluatorInfraError):
        runner.run_job(_spec())
    assert len(calls) == 2  # 重试一次


def test_run_job_retry_recovers_on_second_attempt(monkeypatch):
    from mls_infer_opt.evaluate import runner

    seq = [_proc(1, "", "boom"), _proc(0, json.dumps({"gate": {"passed": True}, "bench": None}))]
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: seq.pop(0))
    out = runner.run_job(_spec())
    assert out["gate"]["passed"] is True  # 第二次成功 → 不抛


def test_run_job_timeout_is_c1_no_retry(monkeypatch):
    import subprocess as _sp

    from mls_infer_opt.evaluate import runner

    calls: list[int] = []

    def fake_run(*a, **k):
        calls.append(1)
        raise _sp.TimeoutExpired(cmd="worker", timeout=1)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    out = runner.run_job(_spec(), timeout_s=1)  # 超时 = C1：返回结构化失败、不抛、不重试
    assert out["gate"]["errors"][0]["stage"] == "runtime"
    assert "tim" in out["gate"]["errors"][0]["message"].lower()
    assert len(calls) == 1


# === 1. serde（纯逻辑，无 torch / 无子进程） ===========================
def test_jobspec_json_roundtrip():
    spec = JobSpec(
        engine_path="/x/engine.py", weight_dir="/w", model_config={"vocab_size": 100},
        device="cpu", mode="quick", task="gate", seed=11, oracle_cache_path="/c.pt",
    )
    back = job_from_json(job_to_json(spec))
    assert back == spec


def test_gate_from_dict_roundtrip():
    gate = GateResult(
        syntax_ok=True, api_ok=True, correctness_ok=False, passed=False,
        errors=[ValidationError(stage="correctness", message="m", case="c", max_abs_err=0.5)],
        case_summary={"single_prefill": True}, duration_s=1.2,
    )
    back = gate_from_dict(to_dict(gate))
    assert back == gate


def test_bench_from_dict_roundtrip():
    bench = BenchResult(mode="full", decode_tps=12.0, score=7.0, loss=-7.0, raw={"a": 1})
    back = bench_from_dict(to_dict(bench))
    assert back == bench


def test_score_monotonic_in_decode_tps():
    bench = pytest.importorskip("mls_infer_opt.evaluate.bench")
    assert bench._score(0, 100, 0) > bench._score(0, 10, 0)
    assert bench._score(10, 0, 0) > bench._score(0, 0, 0)


# === 2. 子进程隔离评测（需 torch；起 worker 子进程） ===================
@pytest.fixture(scope="module")
def _torch():
    return pytest.importorskip("torch")


def test_evaluate_baseline_passes_and_benches(tmp_path, _torch):
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, BASELINE_CODE)

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=120)
    assert out.gate is not None
    assert out.gate.passed, out.gate.errors
    assert out.gate.syntax_ok and out.gate.api_ok and out.gate.correctness_ok
    # 过门候选才有 bench
    assert out.bench is not None
    assert out.bench.score > 0
    assert out.bench.decode_tps > 0
    assert out.bench.peak_memory_mb == 0.0  # CPU 无显存计量


def test_evaluate_idempotent_skips_when_gate_present(tmp_path, _torch):
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, BASELINE_CODE)
    cand.gate = GateResult(syntax_ok=True, api_ok=True, correctness_ok=True, passed=True)

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=5)  # 应直接返回、不起子进程
    assert out.bench is None  # 没重新评测


def test_gate_fails_on_wrong_logits(tmp_path, _torch):
    # 把最后一层 logits 平移 +1000 → 必然 allclose 不过
    broken = BASELINE_CODE.replace(
        'return F.linear(x, self.w["lm_head.weight"])',
        'return F.linear(x, self.w["lm_head.weight"]) + 1000.0',
    )
    assert broken != BASELINE_CODE
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, broken)

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=120)
    assert out.gate is not None and not out.gate.passed
    assert out.gate.syntax_ok and out.gate.api_ok and not out.gate.correctness_ok
    err = out.gate.errors[0]
    assert err.stage == "correctness" and err.max_abs_err is not None
    assert out.bench is None  # 未过门不测性能


def test_gate_fails_api_on_wrong_shape(tmp_path, _torch):
    bad = (
        "import torch\n"
        "def create_engine(model_config, weight_dir, device='cpu'):\n"
        "    return Engine()\n"
        "class Engine:\n"
        "    def prefill(self, request_ids, input_ids):\n"
        "        return torch.zeros(1, 7)\n"  # 错 vocab 维
        "    def decode(self, request_ids, token_ids):\n"
        "        return torch.zeros(1, 7)\n"
        "    def remove(self, request_ids):\n"
        "        pass\n"
    )
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, bad)

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=60)
    assert out.gate is not None and not out.gate.passed
    assert out.gate.syntax_ok and not out.gate.api_ok
    assert out.gate.errors[0].stage == "api"


def test_gate_fails_syntax_on_garbage(tmp_path, _torch):
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, "this is not @@@ python")

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=60)
    assert out.gate is not None and not out.gate.passed
    assert not out.gate.syntax_ok
    assert out.gate.errors[0].stage == "syntax"


def test_isolation_timeout_does_not_kill_parent(tmp_path, _torch):
    slow = (
        "import time, torch\n"
        "def create_engine(model_config, weight_dir, device='cpu'):\n"
        "    time.sleep(60)\n"
        "    return None\n"
        "class Engine:\n"
        "    def prefill(self, r, i):\n        return torch.zeros(1, 1)\n"
        "    def decode(self, r, t):\n        return torch.zeros(1, 1)\n"
        "    def remove(self, r):\n        pass\n"
    )
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, slow)

    from mls_infer_opt.evaluate import evaluate

    out = evaluate(cand, ctx, "full", timeout_s=8)  # 父进程仍存活、拿结构化失败
    assert out.gate is not None and not out.gate.passed
    assert out.gate.errors[0].stage == "runtime"
    assert "tim" in out.gate.errors[0].message.lower()


def test_isolation_hard_crash_escalates_c2(tmp_path, _torch):
    # import 期硬退出 = 进程级死亡、worker 没产出裁决 = C2 → 重试一次仍死 → evaluate 穿透抛 C2。
    crash = "import os\nos._exit(1)\n"
    ctx = _make_ctx(tmp_path)
    cand = _setup_candidate(ctx, crash)

    from mls_infer_opt.evaluate import EvaluatorInfraError, evaluate

    with pytest.raises(EvaluatorInfraError):
        evaluate(cand, ctx, "full", timeout_s=60)


def test_quick_gate_passes_on_baseline(tmp_path, _torch):
    ctx = _make_ctx(tmp_path)
    _setup_candidate(ctx, BASELINE_CODE)

    from mls_infer_opt.evaluate import quick_gate

    gate = quick_gate(candidate_engine_path(ctx.run_dir, "r0-test"), ctx, timeout_s=120)
    assert gate.passed, gate.errors
