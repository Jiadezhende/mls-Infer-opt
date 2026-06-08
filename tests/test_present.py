"""present 纯函数测试：score 格式化 / 数据块渲染 / 验收块。不跑 torch/LLM。"""

from __future__ import annotations

from mls_infer_opt import present
from mls_infer_opt.state.eval import BenchResult
from mls_infer_opt.state.loop import AgentEvent


def _bench(**kw) -> BenchResult:
    base = dict(decode_tps=312.0, prefill_tps=1840.0, mixed_decode_tps=280.0, score=301.0)
    base.update(kw)
    return BenchResult(mode="full", **base)


def test_score_line_with_baseline():
    line = present.fmt_score_line(_bench(), 210.0)
    assert "decode 312 / prefill 1840 / mixed 280 tok/s → score 301.00" in line
    assert "(1.43× baseline)" in line


def test_score_line_shows_peak_memory_when_set():
    # 显存 >0（GPU）才附 · NNNMB 护栏段；CPU(mem=0) 不污染行
    assert " · 587MB → score" in present.fmt_score_line(_bench(peak_memory_mb=587.0), 210.0)
    assert "MB" not in present.fmt_score_line(_bench(), 210.0)


def test_score_line_missing_baseline_marks_baseline():
    # baseline 锚点缺失（-inf）或为零都退化成 (baseline)，不报错、不给倍率
    assert present.fmt_score_line(_bench(), float("-inf")).endswith("(baseline)")
    assert present.fmt_score_line(_bench(), 0.0).endswith("(baseline)")


def test_score_line_anchor_false_drops_suffix():
    line = present.fmt_score_line(_bench(), None, anchor=False)
    assert "baseline" not in line and "×" not in line


def test_score_line_no_bench():
    assert present.fmt_score_line(None, 210.0) == "(无 bench)"


def test_speedup_guards():
    assert present.speedup(301.0, 210.0) is not None
    assert present.speedup(301.0, 0.0) is None
    assert present.speedup(301.0, float("-inf")) is None
    assert present.speedup(float("nan"), 210.0) is None


def test_format_data_block_skips_empty_and_renders_whitelist():
    ev = AgentEvent(
        source="analyze",
        phase="grad",
        message="继续",
        data={
            "bottleneck": "decode_throughput",
            "next_strategy_tags": ["attention:paged_kv", "kvcache:fp8"],
            "detail": "理由全文",
            "fixes": "",  # 空值应被跳过
            "axes_delta": {},  # 空 dict 应被跳过
            "round": 2,  # round 不渲染（表头已隐含）
            "best_score": 210.0,
        },
    )
    block = present.format_data_block(ev)
    text = "\n".join(block)
    assert "bottleneck: decode_throughput" in text
    assert "strategy  : attention:paged_kv, kvcache:fp8" in text
    assert "rationale : 理由全文" in text
    assert "fixes" not in text and "axes_delta" not in text
    assert "situation : best_score=210" in text
    # round 不应单独成行
    assert not any(line.strip().startswith("round") for line in block)


def test_format_data_block_truncates_rationale():
    long = "x" * 800
    ev = AgentEvent(source="analyze", phase="grad", message="m", data={"detail": long})
    block = present.format_data_block(ev)
    assert block and block[0].endswith("…")
    assert len(block[0]) < 600  # 截断到 ~500 + 标签宽度


def test_format_data_block_empty_data():
    ev = AgentEvent(source="loop", phase="init", message="启动", data={})
    assert present.format_data_block(ev) == []
