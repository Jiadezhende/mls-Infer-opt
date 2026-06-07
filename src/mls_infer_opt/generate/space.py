"""space — decoder-only 推理引擎的搜索空间（按组件分层声明）。

这里是**唯一的搜索空间真相源**：把「一次 forward 数据流过的层次」拆成若干正交的轴
（axis），每条轴是一组从 baseline → 激进的可选实现（option）。约定：

- **baseline-first**：每条轴 ``options[0]`` 都等价于现有 workspace/engine.py。
  因此「全默认 axes」渲染出来就是 baseline 本身——bootstrap 不需要特例。
- **轴名/选项名即契约**：它们同时是 (1) policy.axes 的键值、(2) 片段库 registry 的 key、
  (3) Candidate.strategy_tags 的来源、(4) analyze 下发策略时引用的标识。
- **数值敏感性** (``sensitive``)：🔴 改了会动 logits、可能顶破 allclose 容差；
  🟢 正确实现即与 baseline 数学等价（风险是 bug 不是数值）。repair 优先回退 🔴 轴。
- **knobs**：挂在轴上的数值/布尔参数，仅当该轴被选为非默认选项时才生效。

本模块是纯声明、零依赖（不 import torch）：只描述空间，不做渲染、不做校验。
冲突校验见 compat.py，归一/聚合见 policy.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "GroupKey",
    "KnobSpec",
    "AxisSpec",
    "AXES",
    "AXIS_BY_KEY",
    "GROUP_ORDER",
    "axis_spec",
    "baseline_axes",
    "axis_keys",
]

# 组件分层，按数据流先后排序（也决定 strategy_tags / 展示的稳定顺序）。
GroupKey = Literal["cache", "batching", "operator", "precision", "weight_layout"]

GROUP_ORDER: tuple[GroupKey, ...] = (
    "cache",
    "batching",
    "operator",
    "precision",
    "weight_layout",
)


@dataclass(frozen=True)
class KnobSpec:
    """挂在某条轴上的可调参数；仅当宿主轴取非默认选项时纳入 policy.knobs。"""

    key: str
    default: Any
    summary: str = ""


@dataclass(frozen=True)
class AxisSpec:
    """一条搜索轴：一组从 baseline 到激进的互斥实现选项。"""

    key: str
    group: GroupKey
    options: tuple[str, ...]  # 有序，options[0] 必为 baseline 等价实现
    summary: str = ""
    sensitive: bool = False  # 🔴 数值敏感 / 🟢 结构等价
    knobs: tuple[KnobSpec, ...] = ()

    @property
    def default(self) -> str:
        """baseline-first：默认即第一个选项。"""
        return self.options[0]


# === 搜索空间定义 =====================================================
# 顺序即声明顺序，归一化时据此构造稳定有序的 axes（保证 policy.json / 候选 id 可复现）。
AXES: tuple[AxisSpec, ...] = (
    # --- 1. 状态/缓存层：收益最大。baseline 每步 decode 重算整段，是 O(n²)。 ---
    AxisSpec(
        key="kv_cache",
        group="cache",
        options=("recompute_full", "incremental", "static_prealloc", "paged"),
        summary="KV 缓存策略；incremental 起 decode 只算新 token，需按 past_len 偏移 rope/mask。",
        sensitive=False,
        knobs=(
            KnobSpec("kv_capacity_init", 256, "预分配初始长度"),
            KnobSpec("kv_capacity_growth_factor", 2.0, "扩容倍数"),
        ),
    ),
    AxisSpec(
        key="cache_dtype",
        group="cache",
        options=("same_as_compute", "fp16", "bf16"),
        summary="KV 缓存存储精度；低精度省显存但动数值。仅在有缓存时有意义。",
        sensitive=True,
    ),
    # --- 2. 批处理/调度层 ---
    AxisSpec(
        key="prefill_batch",
        group="batching",
        options=("per_request_loop", "padded_batch"),
        summary="prefill 是否跨 request padding 成批；padded 需 batch mask。",
        sensitive=False,
    ),
    AxisSpec(
        key="decode_batch",
        group="batching",
        options=("per_request_loop", "batched"),
        summary="decode 是否跨 request 合批一次 forward；强依赖 KV 缓存。",
        sensitive=False,
        knobs=(KnobSpec("min_batch_for_batched_decode", 2, "触发合批的最小并发请求数"),),
    ),
    # --- 3. 算子实现层（per-layer） ---
    AxisSpec(
        key="attention",
        group="operator",
        options=("naive_matmul", "sdpa", "sdpa_causal"),
        summary="注意力实现；baseline 在 fp32 上算 matmul+softmax；SDPA 低精度 reduction 顺序变。",
        sensitive=True,
    ),
    AxisSpec(
        key="attn_upcast",
        group="operator",
        options=("fp32", "keep_dtype"),
        summary="注意力是否上溢到 fp32 再算；关掉更快但更易漂移，单列方便 repair 回退。",
        sensitive=True,
    ),
    AxisSpec(
        key="gqa",
        group="operator",
        options=("repeat_interleave", "enable_gqa"),
        summary="GQA 处理；enable_gqa 不物化扩展 KV，依赖 SDPA + torch 版本。",
        sensitive=False,
    ),
    AxisSpec(
        key="rope",
        group="operator",
        options=("recompute_each", "precomputed_table"),
        summary="RoPE 是否预算 cos/sin 表按位置切片；decode 需按 past_len 取偏移。",
        sensitive=False,
    ),
    AxisSpec(
        key="norm",
        group="operator",
        options=("rmsnorm_fp32", "keep_dtype"),
        summary="RMSNorm 是否 fp32 上溢。",
        sensitive=True,
    ),
    AxisSpec(
        key="qkv_fusion",
        group="operator",
        options=("separate", "fused_qkv"),
        summary="q/k/v 投影是否融合成单次 matmul；需加载期 cat 权重（配 weight_layout=fused）。",
        sensitive=False,
    ),
    AxisSpec(
        key="mlp_fusion",
        group="operator",
        options=("separate", "fused_gate_up"),
        summary="gate/up 投影是否融合；需加载期 cat 权重（配 weight_layout=fused）。",
        sensitive=False,
    ),
    # --- 4. 全局精度/编译层 ---
    AxisSpec(
        key="compute_dtype",
        group="precision",
        options=("config_default", "fp32", "bf16", "fp16"),
        summary="计算精度；漂移幅度取决于 reference 用什么 dtype。",
        sensitive=True,
    ),
    AxisSpec(
        key="torch_compile",
        group="precision",
        options=("off", "default", "reduce_overhead", "max_autotune"),
        summary="torch.compile 模式；变长 prefill 必须 dynamic 否则反复重编译。",
        sensitive=False,
        knobs=(KnobSpec("torch_compile_dynamic", True, "变长 shape 走 dynamic 编译"),),
    ),
    AxisSpec(
        key="autocast",
        group="precision",
        options=("off", "on"),
        summary="是否启用 autocast；与 compute_dtype=fp32 语义冲突。",
        sensitive=True,
    ),
    # --- 5. 权重布局层（一次性，在 create_engine，不计入 hotpath 计时） ---
    AxisSpec(
        key="weight_layout",
        group="weight_layout",
        options=("as_is", "pretranspose", "fused"),
        summary="权重加载布局；fused 把 qkv/gate-up 的 cat、转置挪到加载期。",
        sensitive=False,
    ),
    AxisSpec(
        key="contiguous",
        group="weight_layout",
        options=("off", "on"),
        summary="加载期是否 .contiguous() 权重。",
        sensitive=False,
    ),
)

# 轴名 → spec 的查表（保持 AXES 顺序）。
AXIS_BY_KEY: dict[str, AxisSpec] = {ax.key: ax for ax in AXES}


def axis_spec(key: str) -> AxisSpec:
    """按轴名取 spec；未知轴抛 KeyError（调用方应先归一化过滤）。"""
    return AXIS_BY_KEY[key]


def axis_keys() -> tuple[str, ...]:
    """全部轴名，声明顺序。"""
    return tuple(AXIS_BY_KEY)


def baseline_axes() -> dict[str, str]:
    """全默认 axes（== baseline）。归一化与 bootstrap 的起点。"""
    return {ax.key: ax.default for ax in AXES}
