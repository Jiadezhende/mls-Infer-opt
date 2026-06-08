"""searchspace — analyze 与 generate 共享的搜索空间领域层（依赖只向下到 state）。

把「优化引擎的搜索空间 + 如何在其中导航/落点」抽成独立一层，介于 state（纯 dataclass 契约）与
generate/analyze（消费方）之间：

- ``space``   搜索空间声明：按组件分层的轴/选项/knobs（唯一真相源，纯声明、零依赖）。
- ``compat``  轴间冲突校验与消解（依赖方让步，决定性）。
- ``policy``  Policy 聚合：normalize → resolve → fill knobs → 合法可复现 Policy + strategy_tags。

generate 据此把 Policy 渲成 engine.py；analyze 据此从 best 叠 ``axes_delta`` 构造下一个合法 Policy。
两者都向下 import 本层，彼此不再横向 import——搜索空间不属于任一阶段的业务，而是它们共享的领域模型。
"""

from __future__ import annotations

from .compat import CONSTRAINTS, Requires, Violation, resolve, validate
from .policy import (
    AggregateResult,
    Policy,
    aggregate,
    default_policy,
    from_json,
    grouped_axes,
    merge,
    strategy_tags,
    to_json,
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
    "Violation",
    "validate",
    "resolve",
    # policy
    "Policy",
    "AggregateResult",
    "aggregate",
    "default_policy",
    "merge",
    "strategy_tags",
    "grouped_axes",
    "to_json",
    "from_json",
]
