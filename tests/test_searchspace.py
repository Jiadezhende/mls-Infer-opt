"""searchspace 测试：词表闸 sanitize_axes（PR-A）。"""

from __future__ import annotations

from mls_infer_opt.searchspace import sanitize_axes
from mls_infer_opt.searchspace.space import AXES


def test_sanitize_keeps_known_legal_axes():
    out = sanitize_axes({"kv_cache": "incremental", "attention": "sdpa"})
    assert out == {"kv_cache": "incremental", "attention": "sdpa"}


def test_sanitize_drops_unknown_axis():
    assert sanitize_axes({"bogus_axis": "whatever"}) == {}


def test_sanitize_drops_invalid_option():
    # kv_cache 是已知轴，但 "turbo" 不是它的合法选项 → 丢弃。
    assert sanitize_axes({"kv_cache": "turbo"}) == {}


def test_sanitize_does_not_fill_defaults():
    # 与 normalize 不同：不铺满全轴、不填 baseline，只留传入的合法项。
    out = sanitize_axes({"attention": "sdpa"})
    assert out == {"attention": "sdpa"}
    assert len(out) == 1


def test_sanitize_preserves_declaration_order():
    # 乱序传入 → 按 AXES 声明顺序输出（保证 tags / 序列化可复现）。
    declared = [ax.key for ax in AXES]
    raw = {"attention": "sdpa", "kv_cache": "incremental"}
    keys = list(sanitize_axes(raw).keys())
    assert keys == sorted(keys, key=declared.index)
