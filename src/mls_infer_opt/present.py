"""present — 把运行状态变成人类输出：实时 stderr 进度 + results.log 文本。

合并了「写出原语」（emit/stream_enabled）与「格式化」（score 行 / 事件数据块 / banner / 验收块）。
纯叶子模块：不反向依赖包内任何东西，故 loop 与 llm 都可向下 import。约定：走 stderr（不碰
worker↔parent 的 stdout 结果管道）、全部 never-throw、``MLS_LOG_STREAM=0`` 关停。
score 口径见 evaluate/bench.py（越大越好）；speedup = score / baseline(bootstrap 候选分数)。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
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
    "summary",
    "reasoning_trace",
    "render_events",
    "json_dump",
    "stream_event",
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
    """BenchResult 的 headline 分量（grader 两列 tok/s + 显存护栏 + score）；bench=None → {}。"""
    if bench is None:
        return {}
    return {
        "decode_tps": _finite(getattr(bench, "decode_tps", None)) or 0.0,
        "prefill_tps": _finite(getattr(bench, "prefill_tps", None)) or 0.0,
        "mixed_decode_tps": _finite(getattr(bench, "mixed_decode_tps", None)) or 0.0,
        # —— grader 同样逐 case 报「整体 tokens/s」：留进 breakdown 供 rounds[] 复盘 ——
        "decode_overall_tps": _finite(getattr(bench, "decode_overall_tps", None)) or 0.0,
        "mixed_tps": _finite(getattr(bench, "mixed_tps", None)) or 0.0,
        "peak_memory_mb": _finite(getattr(bench, "peak_memory_mb", None)) or 0.0,
        "score": _finite(getattr(bench, "score", None)) or 0.0,
    }


def speedup(score: Any, baseline: Any) -> float | None:
    """score / baseline；baseline 非有限/≤0 或 score 非有限 → None。"""
    s, b = _finite(score), _finite(baseline)
    return s / b if (s is not None and b is not None and b > 0) else None


def fmt_score_line(bench: Any, baseline_score: Any, *, anchor: bool = True) -> str:
    """``decode 312 / prefill 1840 / mixed 280 tok/s · 587MB → score 1.43 (1.43× baseline)``。

    显示 decode 列三类 tps（最具诊断性）+ 峰值显存护栏（>0 才附，CPU 下省略）；score 保两位小数，
    便于读出归一化后 ~1–3 倍的轮间增量（裸 .0f 会把 2.04 截成 2）。整体 tps 等完整分项见
    output3.json 的 rounds[].score_breakdown。bench=None → ``(无 bench)``；anchor=False 时不附倍率。
    """
    if bench is None:
        return "(无 bench)"
    bd = score_breakdown(bench)
    base = (
        f"decode {bd['decode_tps']:.0f} / prefill {bd['prefill_tps']:.0f} / "
        f"mixed {bd['mixed_decode_tps']:.0f} tok/s"
    )
    mem = bd.get("peak_memory_mb", 0.0)
    if mem > 0:
        base += f" · {mem:.0f}MB"
    base += f" → score {bd['score']:.2f}"
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
    ("suggest_axes", "strategy"),
    ("knobs", "knobs"),
    ("detail", "rationale"),
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


# === 运行结果摘要 / 逐轮叙事 / 序列化（state → dict|text，loop 负责何时何处落盘） ===
def summary(state: Any) -> dict[str, Any]:
    """把 LoopState 摊成机读摘要 dict，嵌进 output3.json / runs/report.json。

    含 result 判定、speedup、best 分项与逐轮推理叙事（rounds[]）。纯派生、不落盘——
    何时/写到哪里由 loop（_publish_summary / _write_artifacts）决定。
    """
    best = state.best_candidate()
    gate = best.gate if best is not None else None
    gate_fail_stage = ""
    if gate is not None and not gate.passed and gate.errors:
        gate_fail_stage = gate.errors[0].stage
    rounds = reasoning_trace(state)
    # round 取「已执行轮数（含候选的 trace 条目）」与 state.round 的较大者：让中途快照不再比实际少
    # 一拍（state.round 在 _run_policy_round 返回后才 +1，而落盘发生在其内部）。
    executed_rounds = sum(1 for r in rounds if r.get("candidate_id"))
    final_dir = Path(state.task_context.run_final_dir)
    return {
        "run_id": state.task_context.run_id,
        "stop_reason": state.stop_reason,
        "best_id": state.best_id,
        "best_score": state.best_score if math.isfinite(state.best_score) else None,
        "best_strategy_tags": list(best.strategy_tags) if best is not None else [],
        "n_candidates": len(state.candidates),
        "round": max(state.round, executed_rounds),
        "stale_rounds": state.stale_rounds,
        "elapsed_s": state.budget.elapsed_s,
        "eval_runs": state.budget.eval_runs,
        "engine_path": state.task_context.engine_publish_path if best is not None else "",
        "archived_engine_path": str(final_dir / "engine.py") if best is not None else "",
        "run_final_dir": str(final_dir),
        # —— 让验收者一眼读懂「比 baseline 快多少 / 正确性」的增量字段 ——
        "result": result_verdict(state),
        "baseline_score": state.baseline_score if math.isfinite(state.baseline_score) else None,
        "speedup": speedup(state.best_score, state.baseline_score),
        "score_breakdown": score_breakdown(best.bench) if best is not None else {},
        "correctness_passed": bool(gate and gate.passed),
        "gate_stage_on_fail": gate_fail_stage,
        # —— 逐轮推理叙事（诊断→策略→评测→结论），让 output3.* 自带推理而非只摘要 ——
        "rounds": rounds,
    }


def reasoning_trace(state: Any) -> list[dict[str, Any]]:
    """逐轮推理的紧凑机读数组，嵌进 output3.json / runs/report.json。

    以每条 analyze 事件为一轮的锚：取诊断（bottleneck）/方向（strategy_tags/knobs_delta）/理由
    （rationale=event.data["detail"]）/判定（continue|stop），再并入该 analyze 之后、下一条 analyze
    之前最后一次 evaluate 的 {candidate_id, passed, score} 作结论，并由 candidate_id 反查候选补上
    分项 score_breakdown（哪个轴动了）、parent_id（从谁 fork——留痕贪心爬山）、delta（相对上一轮的
    分数增量）。never-throw → []。
    """
    try:
        trace: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        prev_score: float | None = None
        for ev in state.events:
            data = ev.data or {}
            if ev.source == "analyze":
                cur = {
                    "step": len(trace) + 1,
                    "decision": data.get("decision"),
                    "bottleneck": data.get("bottleneck") or "",
                    "rationale": data.get("detail") or "",
                    # analyze 的松建议（方向）；实际采用 strategy_tags 在评测后从候选回填（honest）
                    "suggest_axes": dict(data.get("suggest_axes") or {}),
                    "knobs": dict(data.get("knobs") or {}),
                    "strategy_tags": [],
                    "used_llm": bool(data.get("used_llm", False)),
                }
                if data.get("stop_reason"):
                    cur["stop_reason"] = data["stop_reason"]
                trace.append(cur)
            elif ev.source == "loop" and ev.phase == "evaluate" and cur is not None:
                # 该轮 analyze 之后的评测结论：覆盖式取最后一次（含 repair 后的最终评测）。
                cur["candidate_id"] = ev.candidate_id
                cur["passed"] = bool(data.get("passed", False))
                score = data.get("score")
                if score is not None:
                    cur["score"] = score
                    cur["delta"] = (score - prev_score) if prev_score is not None else None
                    prev_score = score
                # 由 candidate_id 反查候选：补分项（哪个轴动了）+ 血缘（从谁 fork）+ 实际采用轴。
                cand = state.candidates.get(ev.candidate_id) if ev.candidate_id else None
                if cand is not None:
                    if cand.bench is not None:
                        cur["score_breakdown"] = score_breakdown(cand.bench)
                    cur["parent_id"] = cand.parent_id
                    # strategy_tags 来自 agent 实际回报，非 analyze 建议——诚实记录这轮真动了什么。
                    cur["strategy_tags"] = list(cand.strategy_tags)
        return trace
    except Exception:  # noqa: BLE001 — 叙事采集绝不拖垮 finalize 落盘
        return []


def render_events(state: Any) -> str:
    """results.log 全量渲染（finalize 用）。与增量 append 走同一 render_event，格式一致。"""
    if not state.events:
        return ""
    return "\n".join(render_event(e) for e in state.events) + "\n"


def json_dump(payload: Any) -> str:
    """稳定 JSON 序列化：非有限 float → null，tuple → list，键序固定。"""
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


# 无 score_line 的事件，仍按这几个高频键点出指标；其余完整落 results.log（runs/final 留档）。
_STREAM_KEYS = ("passed", "stop_reason", "attempt", "published")


def stream_event(event: Any) -> None:
    """事件观察者：把单条 AgentEvent 实时写 stderr，给评测终端逐轮反馈。

    由 ``run_loop`` 通过 ``state.on_event`` 装到 state 上，于是 loop / analyze 等**所有**来源的
    事件一产生就出现在终端，而不必等 finalize 落 results.log。格式化失败被 emit 吞掉，绝不影响主
    流程（事件本身早已入表）。

    extra 的取舍：评测/keep_best 事件优先显示带单位 + ×baseline 的 ``score_line``（取代裸 score）；
    analyze 事件追加 bottleneck + strategy，让诊断行自解释。于是 analyze→generate→evaluate→
    keep_best 四行天然连成「诊断→策略→评测→结论」叙事。
    """

    data = event.data
    rnd = data.get("round")
    prefix = f"[r{rnd}]" if rnd is not None else "[--]"
    cid = f" {event.candidate_id}" if event.candidate_id else ""

    extra = ""
    if data.get("score_line"):
        extra = f"  {data['score_line']}"
    elif event.source == "analyze":
        bits = []
        if data.get("bottleneck"):
            bits.append(f"bottleneck={data['bottleneck']}")
        if data.get("suggest_axes"):
            sg = ", ".join(f"{k}={v}" for k, v in data["suggest_axes"].items())
            bits.append(f"strategy={sg}")
        extra = f"  ({'; '.join(bits)})" if bits else ""
    else:
        bits = [f"{k}={data[k]}" for k in _STREAM_KEYS if data.get(k) is not None]
        extra = f"  ({', '.join(bits)})" if bits else ""

    where = f"{event.source}.{event.phase}"
    emit(f"{prefix} {event.level:<7} {where}:{cid} {event.message}{extra}")
