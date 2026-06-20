"""searchspace — analyze 与 generate 共享的搜索空间领域层（依赖只向下到 state）。

把「优化引擎的搜索维度 + 围着它的工具」抽成独立一层，介于 state（纯 dataclass 契约）与
generate/analyze（消费方）之间：

- ``space``   搜索维度声明：按组件分层的轴/选项/knobs（唯一真相源，纯声明、零依赖）。
- ``compat``  轴间依赖约束声明 + 渲成 prompt 规则（不再消解成定点，交 full gate 拦非法组合）。
- ``dims``    维度工具：sanitize_axes 词表闸 / strategy_tags / grouped_axes / render_search_dims。

analyze 与 generate 都向下 import 本层、把搜索维度渲进各自 prompt，agent 在维度界内自由探索；
彼此不再横向 import——搜索维度不属于任一阶段的业务，而是它们共享的领域模型。
"""

from __future__ import annotations

from .compat import CONSTRAINTS, Requires, render_constraints
from .dims import (
    grouped_axes,
    render_search_dims,
    sanitize_axes,
    strategy_tags,
)
from .space import (
    AXES,
    AXIS_BY_KEY,
    GROUP_ORDER,
    AxisSpec,
    GroupKey,
    KnobSpec,
    axis_keys,
    axis_spec,
    baseline_axes,
)

__all__ = [
    # space
    "AXES",
    "AXIS_BY_KEY",
    "GROUP_ORDER",
    "GroupKey",
    "AxisSpec",
    "KnobSpec",
    "baseline_axes",
    "axis_spec",
    "axis_keys",
    # compat
    "CONSTRAINTS",
    "Requires",
    "render_constraints",
    # dims
    "sanitize_axes",
    "strategy_tags",
    "grouped_axes",
    "render_search_dims",
]
