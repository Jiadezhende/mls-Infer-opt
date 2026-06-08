"""present — 把运行状态变成人类输出：实时 stderr 进度 + results.log 文本。

合并了「写出原语」（emit/stream_enabled）与「格式化」（score 行 / 事件数据块 / banner / 验收块）。
纯叶子模块：不反向依赖包内任何东西，故 loop 与 llm 都可向下 import。约定：走 stderr（不碰
worker↔parent 的 stdout 结果管道）、全部 never-throw、``MLS_LOG_STREAM=0`` 关停。
score 口径见 evaluate/bench.py（越大越好）；speedup = score / baseline(bootstrap 候选分数)。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

__all__ = [
    "stream_enabled",
    "emit",
    "score_breakdown",
    "speedup",
    "fmt_score_line",
    "format_data_block",
    "render_event",
    "append_event",
    "fmt_banner",
    "result_verdict",
    "fmt_acceptance",
]

_W = 60  # banner / 验收块横线宽度


def stream_enabled() -> bool:
    """是否把进度实时写 stderr。默认开；``MLS_LOG_STREAM=0/false/no`` 关停。每次读 env 便于测试。"""
    return os.environ.get("MLS_LOG_STREAM", "1").strip().lower() not in ("0", "false", "no", "")


def emit(text: str) -> None:
    """写一行进度到 stderr：门控 + never-throw + flush（逐行可见、不被缓冲攒住）。"""
    if not stream_enabled():
        return
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 — 可观测性绝不拖垮主流程
        pass


def _finite(x: Any) -> float | None:
    """转有限 float；None / 非数 / inf / nan → None。"""
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _num(x: Any) -> str:
    """整数化展示；非有限用 ``—`` 占位（如未设的 baseline_score=-inf）。"""
    f = _finite(x)
    return f"{f:.0f}" if f is not None else "—"


def score_breakdown(bench: Any) -> dict[str, float]:
    """BenchResult 的 headline 分量（tok/s + 合成 score）；bench=None → {}。"""
    if bench is None:
        return {}
    return {
        "decode_tps": _finite(getattr(bench, "decode_tps", None)) or 0.0,
        "prefill_tps": _finite(getattr(bench, "prefill_tps", None)) or 0.0,
        "mixed_decode_tps": _finite(getattr(bench, "mixed_decode_tps", None)) or 0.0,
        "score": _finite(getattr(bench, "score", None)) or 0.0,
    }


def speedup(score: Any, baseline: Any) -> float | None:
    """score / baseline；baseline 非有限/≤0 或 score 非有限 → None。"""
    s, b = _finite(score), _finite(baseline)
    return s / b if (s is not None and b is not None and b > 0) else None


def fmt_score_line(bench: Any, baseline_score: Any, *, anchor: bool = True) -> str:
    """``decode 312 / prefill 1840 / mixed 280 tok/s → score 301 (1.43× baseline)``。

    bench=None → ``(无 bench)``。anchor=False 时只给分量+score、不附倍率（验收块自带 vs baseline）。
    """
    if bench is None:
        return "(无 bench)"
    bd = score_breakdown(bench)
    base = (
        f"decode {bd['decode_tps']:.0f} / prefill {bd['prefill_tps']:.0f} / "
        f"mixed {bd['mixed_decode_tps']:.0f} tok/s → score {bd['score']:.0f}"
    )
    if not anchor:
        return base
    sp = speedup(bd["score"], baseline_score)
    return f"{base} ({sp:.2f}× baseline)" if sp is not None else f"{base} (baseline)"


# results.log 事件数据块：(event.data key, 展示标签)。只渲染白名单，不裸 dump。
_BLOCK_KEYS: tuple[tuple[str, str], ...] = (
    ("score_line", "score"),
    ("gate_stage", "fail_stage"),
    ("gate_error", "fail_reason"),
    ("cases", "cases"),
    ("bottleneck", "bottleneck"),
    ("axes_delta", "axes_delta"),
    ("knobs_delta", "knobs_delta"),
    ("next_strategy_tags", "strategy"),
    ("detail", "rationale"),
    ("fixes", "fixes"),
    ("stop_reason", "stop_reason"),
    ("used_llm", "used_llm"),
)
_SITUATION_KEYS = ("best_id", "best_score", "stale_rounds", "n_candidates", "n_rejected")
_LABEL_W = 10
_TRUNC = 500  # rationale / 错因截断宽度（全文见 results.log / output3.json rounds[]）


def _fmt_value(key: str, value: Any) -> str:
    if isinstance(value, dict):
        body = ", ".join(f"{k}={v}" for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        body = ", ".join(str(v) for v in value)
    else:
        body = str(value)
    if key == "gate_error" and body.strip():  # 错因可能多行：只取首行
        body = body.strip().splitlines()[0]
    if key in ("detail", "gate_error") and len(body) > _TRUNC:
        body = body[:_TRUNC] + "…"
    return body


def format_data_block(event: Any) -> list[str]:
    """把 event.data 里已采集、results.log 旧版丢弃的信号渲染成缩进行（表头行由调用方保留）。"""
    try:
        data = getattr(event, "data", None) or {}
        lines = [
            f"    {label:<{_LABEL_W}}: {_fmt_value(key, data[key])}"
            for key, label in _BLOCK_KEYS
            if data.get(key) not in (None, "", [], {})
        ]
        parts = [
            f"{k}={_num(data[k]) if k == 'best_score' else data[k]}"
            for k in _SITUATION_KEYS
            if k in data
        ]
        if parts:
            lines.append(f"    {'situation':<{_LABEL_W}}: " + " ".join(parts))
        return lines
    except Exception:  # noqa: BLE001
        return []


def render_event(event: Any) -> str:
    """单条事件 → results.log 文本：表头行（可 grep）+ 缩进数据块。不含尾换行。"""
    cid = f" candidate={event.candidate_id}" if getattr(event, "candidate_id", None) else ""
    head = f"[{event.ts}] {event.level} {event.source}.{event.phase}:{cid} {event.message}"
    return "\n".join([head, *format_data_block(event)])


def append_event(path: Any, event: Any) -> None:
    """把一条事件实时 append 到 results.log（每次 open/close，落盘即抗中途 kill）。never-throw。"""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(render_event(event) + "\n")
    except OSError:
        pass


def fmt_banner(state: Any) -> str:
    """开头横幅：run / 模型 / 设备 / 预算 / baseline 分数。"""
    try:
        ctx = state.task_context
        lim, env = ctx.limits, ctx.environment
        dev = ctx.device or "?"
        if getattr(env, "gpu_name", None):
            mem = _finite(getattr(env, "total_memory_mb", None))
            dev += f" ({env.gpu_name}" + (f", {mem:.0f}MB)" if mem is not None else ")")
        mc = ctx.model_config or {}
        model = (
            " ".join(
                s
                for s in (
                    f"{mc['num_hidden_layers']}L" if mc.get("num_hidden_layers") else "",
                    f"hidden={mc['hidden_size']}" if mc.get("hidden_size") else "",
                    f"heads={mc['num_attention_heads']}" if mc.get("num_attention_heads") else "",
                    f"vocab={mc['vocab_size']}" if mc.get("vocab_size") else "",
                )
                if s
            )
            or "?"
        )
        rounds = str(lim.max_rounds) if lim.max_rounds and lim.max_rounds > 0 else "∞"
        base = state.baseline_candidate()
        base_line = fmt_score_line(base.bench, None) if base is not None else "(无 baseline)"
        return "\n".join(
            [
                "═" * _W,
                "  MLS Infer-Opt",
                f"  run        : {ctx.run_id}",
                f"  model      : {model}",
                f"  device     : {dev}",
                f"  budget     : time {lim.time_budget_s}s | rounds {rounds} "
                f"| stale {lim.max_stale_rounds}",
                f"  baseline   : {base_line}",
                "═" * _W,
            ]
        )
    except Exception:  # noqa: BLE001
        return ""


def _correctness(gate: Any) -> str:
    """correctness 行：过→PASSED+三阶段勾；不过→卡在哪阶段 + 首行错因。"""
    if gate is None:
        return "correctness: 未评测"
    if getattr(gate, "passed", False):
        flags = (
            f"syntax {'✓' if gate.syntax_ok else '✗'} "
            f"api {'✓' if gate.api_ok else '✗'} "
            f"correctness {'✓' if gate.correctness_ok else '✗'}"
        )
        return f"correctness: PASSED ({flags})"
    errors = getattr(gate, "errors", None) or []
    stage = errors[0].stage if errors else "?"
    msg = errors[0].message.strip().splitlines()[0] if errors and errors[0].message.strip() else ""
    return f"correctness: FAILED at {stage}" + (f" — {msg}" if msg else "")


def result_verdict(state: Any) -> str:
    """任务结果判定：best 未过门 → failed；过门且 speedup>1 → accepted；否则 published_baseline。

    fmt_acceptance 的标题、output3.json 与 runs/report.json 的 ``result`` 字段共用同一口径。
    never-throw：异常 → ``failed``（最保守）。
    """
    try:
        best = state.best_candidate()
        if best is None or best.gate is None or not getattr(best.gate, "passed", False):
            return "failed"
        sp = speedup(state.best_score, state.baseline_score)
        return "accepted" if sp is not None and sp > 1.0 else "published_baseline"
    except Exception:  # noqa: BLE001
        return "failed"


def fmt_acceptance(state: Any) -> str:
    """结尾验收块——验收者的头条。按结果切标题，失败时给原因。"""
    try:
        ctx = state.task_context
        best = state.best_candidate()
        baseline = state.baseline_score
        sp = speedup(state.best_score, baseline) if best is not None else None
        verdict = result_verdict(state)
        if verdict == "failed":
            title = "Result: FAILED (best did not pass final gate)"
        elif verdict == "accepted":
            title = f"Result: ACCEPTED (beat baseline {sp:.2f}×)"
        else:
            title = "Result: PUBLISHED BASELINE (no improvement found)"

        lines = ["═" * _W, f"  {title}"]
        if best is not None:
            tags = ", ".join(best.strategy_tags) if best.strategy_tags else "(none)"
            lines.append(f"  best       : {best.id}   strategy: {tags}")
            lines.append(f"  score      : {fmt_score_line(best.bench, None, anchor=False)}")
            if sp is not None and _finite(baseline) is not None:
                lines.append(
                    f"  vs baseline: {_num(baseline)} → {_num(state.best_score)} "
                    f"({(sp - 1.0) * 100.0:+.0f}%, {sp:.2f}×)"
                )
            lines.append(f"  {_correctness(best.gate)}")
        b = state.budget
        lines.append(
            f"  budget     : {state.round} rounds | {b.eval_runs} evals | "
            f"{b.elapsed_s:.0f}s / {ctx.limits.time_budget_s}s"
        )
        lines.append(
            f"  published  : {ctx.engine_publish_path if best is not None else '(none)'}"
            f"   stop_reason: {state.stop_reason or '?'}"
        )
        lines.append("═" * _W)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""
