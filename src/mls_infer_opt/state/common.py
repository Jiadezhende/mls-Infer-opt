"""state 内部共用的小工具——跨结构复用，不属于任何单个结构。"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

__all__ = ["utcnow_iso", "to_dict"]


def utcnow_iso() -> str:
    """统一时间戳（UTC ISO8601）。state 里所有 ts 用它。"""
    return datetime.now(timezone.utc).isoformat()


def to_dict(obj: Any) -> Any:
    """递归转 JSON-friendly dict（evaluate worker 跨进程传 gate/bench 用）。

    dataclass 实例 → dict（含嵌套）；其余原样。结果可直接 json.dumps。
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj
