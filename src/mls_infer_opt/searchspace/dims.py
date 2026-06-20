"""dims — 搜索维度的轻量工具：词表闸 / 标签 / 分组 / 渲成 prompt 菜单。

搜索维度（axis/option/knob）的唯一声明在 [space]；本模块只提供围着它转的纯函数：

- ``sanitize_axes``  把任意来源的 axes 过滤到「已知轴 + 合法选项」（词表闸，不填默认）。
- ``strategy_tags``  axes dict → 非默认轴的 ``axis:option`` 标签（Candidate.strategy_tags 用）。
- ``grouped_axes``   axes dict → 按组件分组的非默认轴（展示/渲染用）。
- ``render_search_dims`` 把完整搜索维度渲成 prompt 菜单（analyze 与 generate 共用，消重）。

不再有「聚合成恒合法定点 / 序列化 Policy」那套——generate 的 agent 在维度界内自由探索、外层
full gate 唯一权威；这里只描述维度、过词表、做展示，不定点、不落盘、零 torch。
"""

from __future__ import annotations

from collections.abc import Mapping

from .space import AXES, AXIS_BY_KEY, GROUP_ORDER, GroupKey

__all__ = [
    "sanitize_axes",
    "strategy_tags",
    "grouped_axes",
    "render_search_dims",
]


def sanitize_axes(raw: Mapping[str, str]) -> dict[str, str]:
    """把任意来源的 axes 过滤到「已知轴 + 合法选项」，丢未知/非法，**不填默认**。

    只保留确有取值的合法轴，是 agent 回报实际采用轴 / analyze 松建议进入 prompt 前的词表闸——
    既诚实（只留真值）又干净（越界即丢）。按 AXES 声明顺序输出，保证 tags / 序列化可复现。
    """
    out: dict[str, str] = {}
    for key in AXIS_BY_KEY:  # 按声明顺序遍历，过滤 raw
        if key not in raw:
            continue
        value = raw[key]
        if value in AXIS_BY_KEY[key].options:
            out[key] = value
    return out


def grouped_axes(axes: Mapping[str, str]) -> dict[GroupKey, dict[str, str]]:
    """按组件分组的「非默认」轴（默认轴省略）；用于分层展示 / 渲染派发。

    入参是已过词表闸或本就合法的 axes dict；未知轴静默跳过。
    """
    out: dict[GroupKey, dict[str, str]] = {g: {} for g in GROUP_ORDER}
    for key, value in axes.items():
        spec = AXIS_BY_KEY.get(key)
        if spec is not None and value != spec.default:
            out[spec.group][key] = value
    return out


def strategy_tags(axes: Mapping[str, str]) -> list[str]:
    """轻量策略摘要：只列「非默认」轴，形如 ``axis:option``，按组件顺序。

    供 Candidate.strategy_tags / report 用；也是 analyze 在历史里引用某轮策略的稳定标识。
    """
    by_group = grouped_axes(axes)
    tags: list[str] = []
    for group in GROUP_ORDER:
        for key, value in by_group[group].items():
            tags.append(f"{key}:{value}")
    return tags


def render_search_dims() -> str:
    """把完整搜索维度渲成 prompt 菜单：按组件分层列轴/选项/敏感度/knob，供 LLM 在界内选/探索。"""
    by_group: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for ax in AXES:
        risk = "🔴数值敏感" if ax.sensitive else "🟢结构等价"
        opts = " | ".join(ax.options)
        knob_list = ", ".join(f"{k.key}(默认{k.default})" for k in ax.knobs)
        knobs = f"；knobs：{knob_list}" if ax.knobs else ""
        by_group[ax.group].append(f"  - {ax.key} [{risk}]：{opts}。{ax.summary}{knobs}")
    lines: list[str] = []
    for group in GROUP_ORDER:
        lines.append(f"### {group}")
        lines.extend(by_group[group])
    return "\n".join(lines)
