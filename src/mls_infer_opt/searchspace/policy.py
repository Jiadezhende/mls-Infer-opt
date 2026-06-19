"""policy — Policy 聚合：把零散的轴选择/knobs 归并成一份合法、可复现的 Policy。

Policy 数据结构本身（dataclass）已下沉到 [state.policy]（analyze↔generate 的共享契约）；本模块
re-export 它，并提供依赖搜索空间（space/compat）的聚合/序列化逻辑：先确定 policy（本模块），
再渲成 prompt 让 LLM 产 engine.py（见 prompt.py / codegen.py）。

聚合（aggregate）是核心入口，按固定管线把任意来源的 axes/knobs 收敛成规范 Policy：

    raw axes/knobs
      → normalize  填默认、丢未知轴/非法选项（baseline 兜底）
      → resolve    消解轴间冲突（compat，依赖方让步）
      → fill knobs 仅为「被激活的非默认轴」纳入 knob 默认，再覆盖用户值
      → Policy + strategy_tags（只列非默认轴，作为轻量摘要）

三种触发场景都走这条管线：bootstrap = 空输入聚合；propose = parent.axes 叠 delta 后聚合；
repair = parent.axes 把出错轴退默认后聚合。决定性：相同输入 → 相同 Policy → 相同候选 id。

纯逻辑、零依赖（除 space/compat 与 state 的类型别名）；不 import torch、不落盘。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..state.candidate import CandidateKind
from ..state.policy import Policy
from .compat import resolve as resolve_conflicts
from .space import AXIS_BY_KEY, GROUP_ORDER, GroupKey, baseline_axes

__all__ = [
    "Policy",
    "AggregateResult",
    "aggregate",
    "default_policy",
    "merge",
    "sanitize_axes",
    "strategy_tags",
    "grouped_axes",
    "to_json",
    "from_json",
]


def sanitize_axes(raw: Mapping[str, str]) -> dict[str, str]:
    """把任意来源的 axes 过滤到「已知轴 + 合法选项」，丢未知/非法，**不填默认**。

    与 ``_normalize_axes`` 的区别：那是「铺满全轴的恒合法定点」（聚合用）；这里只保留确有取值的合法
    轴，是 agent 回报实际采用轴 / analyze 松建议进入 prompt 前的词表闸——既诚实（只留真值）又干净
    （越界即丢）。按 AXES 声明顺序输出，保证 strategy_tags / 序列化可复现。
    """
    out: dict[str, str] = {}
    for key in AXIS_BY_KEY:  # 按声明顺序遍历，过滤 raw
        if key not in raw:
            continue
        value = raw[key]
        if value in AXIS_BY_KEY[key].options:
            out[key] = value
    return out


@dataclass
class AggregateResult:
    """聚合产物：规范 Policy + 这次为消解冲突/丢非法值做的降级说明（供 report/事件）。"""

    policy: Policy
    fixes: list[str] = field(default_factory=list)


def _normalize_axes(raw: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """以 baseline 为底铺满全部轴，只接受已知轴的合法选项；非法值丢弃并记说明。

    按 AXES 声明顺序构造，保证序列化与候选 id 可复现。
    """
    axes = baseline_axes()
    notes: list[str] = []
    for key, value in raw.items():
        spec = AXIS_BY_KEY.get(key)
        if spec is None:
            notes.append(f"drop unknown axis {key!r}")
            continue
        if value not in spec.options:
            notes.append(f"drop invalid {key}={value!r} → {spec.default!r}")
            continue
        axes[key] = value
    return axes, notes


def _fill_knobs(axes: Mapping[str, str], raw: Mapping[str, Any]) -> dict[str, Any]:
    """仅为「取了非默认选项」的轴纳入其 knob：先填默认，再覆盖用户提供值。

    未被激活轴的 knob 一律丢弃（无意义），保证 knobs 与 axes 一致、policy 规范化。
    """
    knobs: dict[str, Any] = {}
    for key, value in axes.items():
        spec = AXIS_BY_KEY[key]
        if value == spec.default:
            continue
        for k in spec.knobs:
            knobs[k.key] = raw.get(k.key, k.default)
    return knobs


def aggregate(
    axes: Mapping[str, str],
    knobs: Mapping[str, Any] | None = None,
    *,
    kind: CandidateKind = "optimization",
    round: int = 0,
    parent_id: str | None = None,
    notes: str = "",
    rationale: str = "",
) -> AggregateResult:
    """核心入口：任意来源的 axes/knobs → 一份合法、可复现的 Policy。见模块文档的四步管线。"""
    norm_axes, drop_notes = _normalize_axes(axes)
    legal_axes, conflict_notes = resolve_conflicts(norm_axes)
    final_knobs = _fill_knobs(legal_axes, knobs or {})
    fixes = drop_notes + conflict_notes
    policy = Policy(
        axes=legal_axes,
        knobs=final_knobs,
        kind=kind,
        round=round,
        parent_id=parent_id,
        notes=notes,
        rationale=rationale,
    )
    return AggregateResult(policy=policy, fixes=fixes)


def default_policy(*, round: int = 0) -> Policy:
    """全默认 Policy（== baseline）；bootstrap 的产物，也是永久兜底。"""
    return aggregate({}, kind="baseline", round=round).policy


def merge(
    parent: Policy,
    *,
    axes_delta: Mapping[str, str] | None = None,
    knobs_delta: Mapping[str, Any] | None = None,
    kind: CandidateKind = "optimization",
    round: int = 0,
    parent_id: str | None = None,
    notes: str = "",
    rationale: str = "",
) -> AggregateResult:
    """在 parent 之上叠加 delta 再聚合：propose（叠策略）/ repair（把出错轴退默认）共用。

    analyze 据此从 best.policy 出发做局部搜索构造下一个点：选好要动的轴/knob 作 delta，
    merge 得到合法的完整 Policy，并附上给 generate 的 ``rationale``。
    """
    axes = {**parent.axes, **(axes_delta or {})}
    knobs = {**parent.knobs, **(knobs_delta or {})}
    return aggregate(
        axes,
        knobs,
        kind=kind,
        round=round,
        parent_id=parent_id,
        notes=notes,
        rationale=rationale,
    )


def strategy_tags(policy: Policy) -> list[str]:
    """轻量策略摘要：只列「非默认」轴，形如 ``axis:option``，按组件顺序。

    供 Candidate.strategy_tags / report 用；也是 analyze 反向引用策略点的稳定标识。
    """
    by_group = grouped_axes(policy)
    tags: list[str] = []
    for group in GROUP_ORDER:
        for key, value in by_group[group].items():
            tags.append(f"{key}:{value}")
    return tags


def grouped_axes(policy: Policy) -> dict[GroupKey, dict[str, str]]:
    """按组件分组的「非默认」轴（默认轴省略）；用于分层展示 / 渲染派发。"""
    out: dict[GroupKey, dict[str, str]] = {g: {} for g in GROUP_ORDER}
    for key in policy.axes:
        spec = AXIS_BY_KEY[key]
        value = policy.axes[key]
        if value != spec.default:
            out[spec.group][key] = value
    return out


def to_json(policy: Policy) -> str:
    """规范化序列化（sort_keys 保证字节级可复现，利于缓存/审计 diff）。"""
    payload = {
        "axes": policy.axes,
        "knobs": policy.knobs,
        "kind": policy.kind,
        "round": policy.round,
        "parent_id": policy.parent_id,
        "notes": policy.notes,
        "rationale": policy.rationale,
        "extra": policy.extra,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)


def from_json(text: str) -> Policy:
    """从 policy.json 还原；缺字段走 dataclass 默认，未知键进 extra 不丢失。"""
    data = json.loads(text)
    known = {"axes", "knobs", "kind", "round", "parent_id", "notes", "rationale", "extra"}
    extra = dict(data.get("extra", {}))
    for k, v in data.items():
        if k not in known:
            extra[k] = v
    return Policy(
        axes=dict(data.get("axes", baseline_axes())),
        knobs=dict(data.get("knobs", {})),
        kind=data.get("kind", "baseline"),
        round=int(data.get("round", 0)),
        parent_id=data.get("parent_id"),
        notes=data.get("notes", ""),
        rationale=data.get("rationale", ""),
        extra=extra,
    )
