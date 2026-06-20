"""compat — 轴间依赖约束（声明 + 渲成 prompt 规则文本）。

搜索维度是若干正交轴，但轴之间有真实依赖（合批解码需要 KV 缓存、enable_gqa 需要 SDPA…）。

不再做「消解成恒合法定点」：generate 的 agent 在搜索维度界内自由写码，外层 full gate 是唯一权威，
而这些硬依赖恰是 full gate 的**子集**（跑不起来类被 syntax/runtime 抓、语义冲突类被 allclose 抓）。
故本模块只保留约束的**声明**，并把它们渲成 prompt 规则文本（``render_constraints``）注入给 analyze
与 generate 的 LLM，让其尽量自洽；越界的非法组合交 full gate 拦，不再在 Python 里反向降级。

本模块纯逻辑、零依赖（除 space）。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Requires", "CONSTRAINTS", "render_constraints"]


@dataclass(frozen=True)
class Requires:
    """依赖约束：``axes[when_axis] ∈ when_options`` 时要求 ``axes[need_axis] ∈ need_options``。

    仅作声明 + 渲成 prompt 规则；不在 Python 里执行降级（交 full gate 拦非法组合）。
    """

    name: str
    when_axis: str
    when_options: tuple[str, ...]
    need_axis: str
    need_options: tuple[str, ...]
    summary: str = ""


# === 约束表 ===========================================================
# 仅覆盖会导致「跑不起来 / 语义错误 / 无意义」的硬依赖；纯性能取舍不在此（交给 evaluate）。
_CACHE_PRESENT = ("incremental", "static_prealloc", "paged")
_SDPA = ("sdpa", "sdpa_causal")

CONSTRAINTS: tuple[Requires, ...] = (
    Requires(
        name="batched_decode_needs_cache",
        when_axis="decode_batch",
        when_options=("batched",),
        need_axis="kv_cache",
        need_options=_CACHE_PRESENT,
        summary="合批解码要求有 KV 缓存，否则无法跨 request 对齐重算。",
    ),
    Requires(
        name="enable_gqa_needs_sdpa",
        when_axis="gqa",
        when_options=("enable_gqa",),
        need_axis="attention",
        need_options=_SDPA,
        summary="enable_gqa 依赖 SDPA 的 GQA 支持。",
    ),
    Requires(
        name="fused_qkv_needs_layout",
        when_axis="qkv_fusion",
        when_options=("fused_qkv",),
        need_axis="weight_layout",
        need_options=("fused",),
        summary="融合 QKV 需加载期把权重 cat 在一起（weight_layout=fused）。",
    ),
    Requires(
        name="fused_mlp_needs_layout",
        when_axis="mlp_fusion",
        when_options=("fused_gate_up",),
        need_axis="weight_layout",
        need_options=("fused",),
        summary="融合 gate/up 需加载期 cat 权重（weight_layout=fused）。",
    ),
    Requires(
        name="cache_dtype_needs_cache",
        when_axis="cache_dtype",
        when_options=("fp16", "bf16"),
        need_axis="kv_cache",
        need_options=_CACHE_PRESENT,
        summary="cache_dtype 仅在存在 KV 缓存时才有意义。",
    ),
    Requires(
        name="autocast_conflicts_fp32",
        when_axis="autocast",
        when_options=("on",),
        need_axis="compute_dtype",
        need_options=("config_default", "bf16", "fp16"),
        summary="autocast 与 compute_dtype=fp32 语义冲突。",
    ),
)


def render_constraints() -> str:
    """把依赖约束渲成 prompt 规则文本，注入 analyze / generate 的 LLM（让其组合尽量自洽）。

    形如「- 取 X∈{…} 时需 Y∈{…}：<summary>」。非法组合不在此降级，交 full gate 拦。
    """
    lines: list[str] = []
    for c in CONSTRAINTS:
        when = " | ".join(c.when_options)
        need = " | ".join(c.need_options)
        lines.append(f"- 取 {c.when_axis}∈{{{when}}} 时需 {c.need_axis}∈{{{need}}}：{c.summary}")
    return "\n".join(lines)
