"""prompt — 把当前态势拼成喂 LLM 的诊断提示词 + 把回复解析成 Gradient（无副作用、无 LLM 调用）。

analyze 的 LLM 用法刻意对齐 generate（见 [[agentic-generate]]）：**单次调用 + 确定性解析**，
不依赖 function-calling / tool-loop。LLM 看「搜索维度菜单 + 依赖规则 + 当前 best + 历史/失败 +
预算」，回一个 ```json 决策块；解析失败就回 None，由 grad 重试一次、仍失败则 NoMove。

本模块只拼字符串 / 抽 json，并把回复解析成 ``Gradient``（迈出的一步）或 ``NoMove``（gradient≈0）；
真正的 LLM 调用、判停执行、发事件都在 grad.py。Gradient 是 analyze↔generate 的共享契约（state 层）。
``suggest_axes`` 是相对 best 的**松建议**、已过词表闸——不定点，generate 的 agent 在界内自由探索。
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..searchspace.compat import render_constraints
from ..searchspace.dims import grouped_axes, render_search_dims, sanitize_axes
from ..searchspace.space import GROUP_ORDER
from ..state.eval import BenchResult, ValidationError
from ..state.gradient import Gradient, NoMove
from .situation import Situation

__all__ = [
    "ANALYZE_CONTRACT",
    "build_analyze_prompt",
    "parse_gradient",
]


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


# 注入每个诊断 prompt 的稳定知识（措辞稳定，利于缓存 / 审计 diff）。
ANALYZE_CONTRACT = f"""\
你是一个推理引擎自动调优器的 **grad（梯度）**：看本轮评测反馈与历史，定位瓶颈，从当前最优
（best）出发做**局部搜索**，给出下一步该往哪些轴探索（松建议），或判断应当停止。

## 评分口径（grader 实际怎么打分）
grader 逐 case 报两列：**整体 tokens/s**=(prefill+decode)/elapsed 与 **decode tokens/s**=
decode/elapsed，外加峰值显存；**不公布合成权重**。含一个纯 prefill case（decode 列=0）。要点：
- prefill 不是边角料：它独占一个 case，且每个 case 的「整体 tokens/s」分母都含 prefill 时间 →
  prefill 提速会同时抬高所有 case 的整体列。
- decode 仍是大头：多个 case 的 decode 列单独计量；baseline 每步重算整段，decode 提速空间最大。
- 显存目前只观测、不直接扣分，但别盲目拿显存换速度。
故方向上**整体吞吐与 decode 吞吐都要顾**，不要只盯单一指标。

## 优化主线先验（搜索维度的方向）
baseline 每步 decode 重算整段（O(n²)）→ 增量 KV 缓存收益最大 → batched prefill / GQA / 合理
dtype / 显存复用 / SDPA /（视设备）torch.compile。数值敏感(🔴)轴可能顶破 allclose 容差，谨慎靠后。

## 可探索的搜索维度（轴 → 选项；建议只在这些轴上选选项，knob 配套给）
{render_search_dims()}

## 轴间依赖（尽量自洽；非法组合会被外层 full gate 拦下，不在此自动降级）
{render_constraints()}

## 输出格式（只输出一个 ```json 代码块，不要额外解释）
```json
{{
  "action": "continue" 或 "stop",
  "stop_reason": "若 stop，给简短英文 slug，如 target_reached / diminishing_returns",
  "suggest_axes": {{"轴名": "选项名"}},   // 相对 best 建议探索的轴；可空=放手让生成器定
  "knobs": {{"knob名": 值}},              // 可空；配套被建议轴的 knob
  "rationale": "给 generate 的方向提示：瓶颈/思路/注意点（自然语言，会渲进生成 prompt）",
  "bottleneck": "一句话当前最该解决的问题"
}}
```
只建议确有把握的少数轴；最终采用哪些由生成器在维度界内自定，无需强求一一落实。"""


def _fmt_bench(b: BenchResult | None) -> str:
    if b is None:
        return "（best 尚无 bench 数据）"
    # 对齐 grader 的两列口径：每类 case 同时给「整体 tokens/s」与「decode tokens/s」。
    line = (
        f"score={b.score:.4f} loss={b.loss:.4f} | "
        f"prefill={b.prefill_tps:.0f} | "
        f"decode={b.decode_tps:.0f}(整体{b.decode_overall_tps:.0f}) | "
        f"mixed_decode={b.mixed_decode_tps:.0f}(整体{b.mixed_tps:.0f}) | "
        f"peak_mem={b.peak_memory_mb:.0f}MB"
    )
    # 跨 seed 的 decode tps 离散度 = 抗不规则鲁棒性信号：min/max 拉得越开，说明对请求顺序越敏感，
    # 是优化该优先收敛的方向。仅 full bench 的 extra 里有；缺则不追加。
    per_seed = (b.extra or {}).get("per_seed", {})
    ds = [v for v in per_seed.get("decode_tps", []) if v > 0]
    if len(ds) >= 2:
        line += f" | decode_tps×{len(ds)}seed [min={min(ds):.1f} max={max(ds):.1f}]"
    return line


def _fmt_failure(e: ValidationError) -> str:
    bits = [f"- [{e.stage}] {e.message}"]
    if e.case:
        bits.append(f"case={e.case}")
    if e.max_abs_err is not None:
        bits.append(f"max_abs_err={e.max_abs_err}")
    return " ".join(bits)


def _fmt_best_axes(axes: dict[str, str]) -> str:
    """best 已应用的非默认轴（来自 honest strategy_tags 还原，见 situation.best_axes）。"""
    groups = grouped_axes(axes)
    items = [f"{axis}={opt}" for group in GROUP_ORDER for axis, opt in groups[group].items()]
    return ", ".join(items) if items else "（全 baseline 默认）"


def build_analyze_prompt(sit: Situation) -> str:
    """拼出完整诊断 prompt：契约 + 当前 best + 历史/失败 + 预算态势。"""
    parts: list[str] = [ANALYZE_CONTRACT, ""]

    parts.append(f"## 当前态势（第 {sit.round} 轮）")
    parts.append(f"- best 已应用轴：{_fmt_best_axes(sit.best_axes)}")
    parts.append(f"- best 性能：{_fmt_bench(sit.best_bench)}")
    parts.append(
        f"- 进度：候选 {sit.n_candidates} 个（拒 {sit.n_rejected}），"
        f"连续无提升 {sit.stale_rounds} 轮"
    )

    if sit.history:
        parts.append("## 历史（轮: 候选 状态 score 策略）")
        for h in sit.history[-8:]:
            score = f"{h.score:.4f}" if h.score is not None else "—"
            tags = ",".join(h.strategy_tags) or "baseline"
            parts.append(f"- r{h.round}: {h.candidate_id} {h.status} score={score} [{tags}]")

    if sit.recent_failures:
        parts.append("## 近期正确性失败（定位数值边界，避免重蹈）")
        parts.extend(_fmt_failure(e) for e in sit.recent_failures)

    parts.append("## 预算")
    parts.append(
        f"- 用时 {sit.elapsed_s:.0f}s / 上限 {sit.time_budget_s or '∞'}s；"
        f"轮 {sit.round} / 上限 {sit.max_rounds or '∞'}；"
        f"无提升 {sit.stale_rounds} / patience {sit.max_stale_rounds or '∞'}"
    )

    parts.append("")
    parts.append("请输出下一步决策，只用一个 ```json 代码块。")
    return "\n".join(parts)


def parse_gradient(text: str | None) -> Gradient | NoMove | None:
    """从 LLM 回复抽 ```json 块 → ``Gradient``（continue）/ ``NoMove``（stop）/ ``None``（不合规）。

    防御式：先取围栏内 json，无围栏则整段试解析。``action=="stop"`` → NoMove(stop_reason)；否则
    Gradient——``suggest_axes`` 过词表闸 sanitize（丢未知/非法，可空），其余字段缺省走默认。
    None 表示内容失败，交 grad 重试 / 判 NoMove；round/parent_id/kind 由 grad 据态势补全。
    """
    if not text:
        return None
    m = _FENCE.search(text)
    raw = (m.group(1) if m else text).strip()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    if data.get("action") == "stop":
        return NoMove(str(data.get("stop_reason") or "no_direction"))

    suggest = data.get("suggest_axes")
    suggest_axes = (
        sanitize_axes({str(k): str(v) for k, v in suggest.items()})
        if isinstance(suggest, dict)
        else {}
    )
    knobs_raw = data.get("knobs")
    knobs: dict[str, Any] = dict(knobs_raw) if isinstance(knobs_raw, dict) else {}
    return Gradient(
        suggest_axes=suggest_axes,
        knobs=knobs,
        rationale=str(data.get("rationale", "")),
        bottleneck=str(data.get("bottleneck", "")),
    )
