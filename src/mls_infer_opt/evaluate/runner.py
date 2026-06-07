"""runner — 父进程侧子进程驱动（纯 stdlib，零 torch，绝不抛异常）。

把 ``JobSpec`` 喂给 ``python -m mls_infer_opt.evaluate.worker`` 子进程，拿回结果 JSON dict。
隔离的价值全在这层兜底：超时 → 杀子进程；非零退出 / 崩溃 / stdout 非法 → 取 stderr 尾部，
一律翻成 ``{"gate": <failed runtime GateResult>, "bench": None}``，让上层永远拿到结构化结果。

父进程刻意不 import torch：编排层健壮、导入快；torch 只活在子进程。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..state.common import to_dict
from ..state.eval import GateResult, GateStage, ValidationError
from .protocol import JobSpec, job_to_json

__all__ = ["run_job"]

_WORKER_MODULE = "mls_infer_opt.evaluate.worker"
# src 布局根：.../src/mls_infer_opt/evaluate/runner.py → parents[2] == .../src
_SRC_ROOT = str(Path(__file__).resolve().parents[2])


def _failure(stage: GateStage, message: str, traceback_tail: str | None = None) -> dict[str, Any]:
    """构造结构化失败结果（gate 不过、无 bench）。"""
    gate = GateResult(
        errors=[ValidationError(stage=stage, message=message, traceback_tail=traceback_tail)]
    )
    return {"gate": to_dict(gate), "bench": None}


def _child_env() -> dict[str, str]:
    """注入 PYTHONPATH 保证 src 布局下子进程能 import 本包，不依赖 pip install。"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_ROOT + (os.pathsep + existing if existing else "")
    return env


def run_job(spec: JobSpec, timeout_s: float | None = None) -> dict[str, Any]:
    """跑一次隔离评测，返回 ``{"gate": dict|None, "bench": dict|None}``。永不抛。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", _WORKER_MODULE],
            input=job_to_json(spec),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_child_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure("runtime", f"evaluation timed out after {timeout_s}s (process killed)")
    except Exception as e:  # spawn 本身失败（极少见）也兜住
        return _failure("runtime", f"failed to spawn worker: {e}")

    if proc.returncode != 0:
        return _failure(
            "runtime",
            f"worker exited with code {proc.returncode}",
            traceback_tail=(proc.stderr or "")[-1600:],
        )

    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        tail = (proc.stdout or "")[-400:] + "\n--- stderr ---\n" + (proc.stderr or "")[-800:]
        return _failure("runtime", "worker produced no valid JSON result", traceback_tail=tail)

    if not isinstance(result, dict) or "gate" not in result:
        return _failure("runtime", "worker result missing 'gate'")
    return result
