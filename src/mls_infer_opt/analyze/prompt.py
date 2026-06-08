"""prompt — 把当前态势拼成喂 LLM 的诊断提示词 + 把回复解析成 Decision（无副作用、无 LLM 调用）。

analyze 的 LLM 用法刻意对齐 generate（见 [[agentic-generate]]）：**单次调用 + 确定性解析**，
不依赖 function-calling / tool-loop。LLM 看「搜索空间菜单 + 当前 best + 历史/失败 + 预算」，
回一个 ```json 决策块；解析失败就回 None，由 grad 退回 rule-based。

本模块只拼字符串 / 抽 json；真正的 LLM 调用、判停执行、merge 出 Policy、发事件都在 grad.py。
"""

from __future__ import annotations

import json
import re

from ..generate.policy import grouped_axes
from ..generate.space import AXES, GROUP_ORDER
from ..state.eval import BenchResult, ValidationError
from ..state.policy import Policy
from .heuristic import Action, Decision
from .situation import Situation

__all__ = ["ANALYZE_CONTRACT", "render_search_space", "build_analyze_prompt", "parse_decision"]

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def render_search_space() -> str:
    """把搜索空间渲成菜单：按组件分层列轴/选项/敏感度/knob，供 LLM 选下一步。"""
    lines: list[str] = []
    by_group: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for ax in AXES:
        risk = "🔴数值敏感" if ax.sensitive else "🟢结构等价"
        opts = " | ".join(ax.options)
        knob_list = ", ".join(f"{k.key}(默认{k.default})" for k in ax.knobs)
        knobs = f"；knobs：{knob_list}" if ax.knobs else ""
        by_group[ax.group].append(f"  - {ax.key} [{risk}]：{opts}。{ax.summary}{knobs}")
    for group in GROUP_ORDER:
        lines.append(f"### {group}")
        lines.extend(by_group[group])
    return "\n".join(lines)


# 注入每个诊断 prompt 的稳定知识（措辞稳定，利于缓存 / 审计 diff）。
ANALYZE_CONTRACT = f"""\
你是一个推理引擎自动调优器的 **grad（梯度）**：看本轮评测反馈与历史，定位瓶颈，从当前最优
（best）出发做**局部搜索**，给出下一步该动哪条/哪几条轴，或判断应当停止。

## 优化主线先验（搜索空间的方向）
baseline 每步 decode 重算整段（O(n²)）→ 增量 KV 缓存收益最大 → batched prefill / GQA / 合理
dtype / 显存复用 / SDPA /（视设备）torch.compile。前置依赖：合批解码需 KV 缓存、enable_gqa 需
SDPA、qkv/mlp 融合需 weight_layout=fused。数值敏感(🔴)轴可能顶破 allclose 容差，谨慎、靠后用。

## 可动的搜索空间（轴 → 选项；只能在这些轴上选选项，knob 只属 Policy.knobs，绝不碰模型结构）
{render_search_space()}

## 输出格式（只输出一个 ```json 代码块，不要额外解释）
```json
{{
  "action": "continue" 或 "stop",
  "stop_reason": "若 stop，给简短英文 slug，如 target_reached / diminishing_returns",
  "axes_delta": {{"轴名": "选项名"}},      // 相对 best 要改动的轴（continue 时）
  "knobs_delta": {{"knob名": 值}},         // 可空；只对被激活轴的 knob 生效
  "rationale": "给 generate 的方向提示：瓶颈/思路/注意点（自然语言，会渲进生成 prompt）",
  "bottleneck": "一句话当前最该解决的问题"
}}
```
只改动确有把握的少数轴；冲突/非法组合会被自动降级，但应尽量自洽。"""


def _fmt_bench(b: BenchResult | None) -> str:
    if b is None:
        return "（best 尚无 bench 数据）"
    line = (
        f"score={b.score:.4f} loss={b.loss:.4f} | prefill_tps={b.prefill_tps:.1f} "
        f"decode_tps={b.decode_tps:.1f} mixed_tps={b.mixed_tps:.1f} "
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


def _fmt_best_axes(policy: Policy) -> str:
    groups = grouped_axes(policy)
    items = [f"{axis}={opt}" for group in GROUP_ORDER for axis, opt in groups[group].items()]
    return ", ".join(items) if items else "（全 baseline 默认）"


def build_analyze_prompt(sit: Situation, best_policy: Policy) -> str:
    """拼出完整诊断 prompt：契约 + 当前 best + 历史/失败 + 预算态势。"""
    parts: list[str] = [ANALYZE_CONTRACT, ""]

    parts.append(f"## 当前态势（第 {sit.round} 轮）")
    parts.append(f"- best 已应用轴：{_fmt_best_axes(best_policy)}")
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


def parse_decision(text: str | None) -> Decision | None:
    """从 LLM 回复抽取 ```json 决策块并解析成 Decision；任何不合规都返回 None（退回 rule-based）。

    防御式：先取围栏内 json，无围栏则整段试解析；字段缺省走 Decision 默认，类型不对则丢弃该字段。
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

    action: Action = "stop" if data.get("action") == "stop" else "continue"
    axes_delta = data.get("axes_delta")
    knobs_delta = data.get("knobs_delta")
    return Decision(
        action=action,
        axes_delta={str(k): str(v) for k, v in axes_delta.items()}
        if isinstance(axes_delta, dict)
        else {},
        knobs_delta=dict(knobs_delta) if isinstance(knobs_delta, dict) else {},
        rationale=str(data.get("rationale", "")),
        bottleneck=str(data.get("bottleneck", "")),
        stop_reason=str(data.get("stop_reason", "")),
    )
