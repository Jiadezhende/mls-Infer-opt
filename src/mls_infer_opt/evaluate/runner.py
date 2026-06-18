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

__all__ = ["run_job", "EvaluatorInfraError"]


class EvaluatorInfraError(RuntimeError):
    """评测器基建失败（C2）：子进程起不来 / 进程级死亡（非超时）/ 输出非法。

    与候选域失败（C1）严格区分——C1 是 worker 产出的结构化裁决（gate fail）或候选超时（太慢），
    照常拒候选、继续；C2 是 worker **根本没产出裁决**（spawn 失败 / 非零退出且无 JSON / 输出非法）。
    run_job 对 C2 **重试一次**，仍失败才抛本异常，交总控在循环边界接住、记 C2 并仍发布 best-so-far。

    前提：worker 已把候选自身异常 catch 成结构化 C1 裁决（见 worker.main），故落到 C2 分支的
    只剩真·进程级死亡（段错误 / OOM kill / import 期 os._exit 等）。
    """

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
    """跑隔离评测，返回 ``{"gate": dict|None, "bench": dict|None}``。

    失败分两类（见 EvaluatorInfraError）：
    - **C1**：worker 产出结构化裁决（gate fail）→ 照常返回；候选超时（太慢/挂死）→ 返回 runtime
      failed gate，**不重试**（重试只会再烧一个 timeout 且大概率再超）。
    - **C2**：worker 没产出裁决（spawn 失败 / 非零退出且无 JSON / 输出非法）→ **重试一次**，仍失败
      则抛 EvaluatorInfraError（交总控记 C2 + 仍发布 best-so-far）。
    """
    last_msg = "evaluator infra failure"
    last_tail: str | None = None
    for _attempt in (1, 2):  # C2 重试一次；C1（超时 / 有裁决）在循环内直接 return
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
            # 超时 = 候选太慢/挂死 = C1：返回结构化失败，不重试、不升级。
            return _failure("runtime", f"evaluation timed out after {timeout_s}s (process killed)")
        except Exception as e:  # spawn 本身失败（极少见）= C2，重试。
            last_msg, last_tail = f"failed to spawn worker: {e}", None
            continue

        if proc.returncode != 0:  # 非零退出 = 进程级死亡、无裁决 = C2，重试。
            last_msg = f"worker exited with code {proc.returncode}"
            last_tail = (proc.stderr or "")[-1600:]
            continue

        try:
            result = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):  # 输出非法 = C2，重试。
            last_msg = "worker produced no valid JSON result"
            last_tail = (
                (proc.stdout or "")[-400:] + "\n--- stderr ---\n" + (proc.stderr or "")[-800:]
            )
            continue

        if not isinstance(result, dict) or "gate" not in result:  # 缺裁决 = C2，重试。
            last_msg, last_tail = "worker result missing 'gate'", None
            continue

        return result  # worker 产出了结构化裁决（C1 / 通过）——信它

    raise EvaluatorInfraError(last_msg + (f"\n{last_tail}" if last_tail else ""))
