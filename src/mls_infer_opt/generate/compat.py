"""compat — 策略冲突校验：把非法/无意义的轴组合降级成合法组合。

搜索空间是若干正交轴，但轴之间有真实依赖（合批解码需要 KV 缓存、enable_gqa 需要 SDPA…）。
LLM / analyze 可能给出冲突组合，渲染前必须先消解，否则会产出跑不起来或语义错误的 engine。

降级原则：**依赖方让步**。约束形如「轴 X 取某些选项 → 要求轴 Y 取某些选项」，违反时把
更激进的依赖轴 X 退回它的 baseline 默认（而非反向去开启 Y），因为退回默认恒等价 baseline、
绝不会引入新风险，且结果对同一输入是决定性的（保证候选 id 可复现）。

本模块纯逻辑、零依赖（除 space）。输入需是已归一化、键齐全的 axes（见 policy.normalize）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .space import AXIS_BY_KEY

__all__ = ["Requires", "Violation", "CONSTRAINTS", "validate", "resolve"]


@dataclass(frozen=True)
class Requires:
    """依赖约束：``axes[when_axis] ∈ when_options`` 时要求 ``axes[need_axis] ∈ need_options``。

    违反时降级动作固定为：把 ``when_axis`` 退回其 baseline 默认（依赖方让步）。
    """

    name: str
    when_axis: str
    when_options: tuple[str, ...]
    need_axis: str
    need_options: tuple[str, ...]
    summary: str = ""

    def is_violated(self, axes: Mapping[str, str]) -> bool:
        return axes.get(self.when_axis) in self.when_options and (
            axes.get(self.need_axis) not in self.need_options
        )


@dataclass
class Violation:
    """一次冲突命中（供 report / repair 诊断）。"""

    constraint: str
    summary: str
    when_axis: str
    when_value: str
    need_axis: str
    fix: str  # 将把 when_axis 退回的默认值


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


def validate(axes: Mapping[str, str]) -> list[Violation]:
    """只检测、不修改，返回全部冲突。axes 需已归一化、键齐全。"""
    violations: list[Violation] = []
    for c in CONSTRAINTS:
        if c.is_violated(axes):
            violations.append(
                Violation(
                    constraint=c.name,
                    summary=c.summary,
                    when_axis=c.when_axis,
                    when_value=axes[c.when_axis],
                    need_axis=c.need_axis,
                    fix=AXIS_BY_KEY[c.when_axis].default,
                )
            )
    return violations


def resolve(axes: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """消解全部冲突，返回 (合法 axes, 降级说明列表)。

    依赖方让步：违反约束的 when_axis 退回 baseline 默认。退回默认不会触发新冲突
    （约束只在 when_axis 取非默认时命中），故单遍即达不动点；仍迭代到稳定以防新增约束破坏该性质。
    """
    resolved = dict(axes)
    notes: list[str] = []
    for _ in range(len(CONSTRAINTS) + 1):
        hits = validate(resolved)
        if not hits:
            break
        for v in hits:
            resolved[v.when_axis] = v.fix
            notes.append(
                f"{v.constraint}: {v.when_axis}={v.when_value!r} → {v.fix!r}（{v.summary}）"
            )
    return resolved, notes
