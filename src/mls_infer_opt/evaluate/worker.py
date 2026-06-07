"""worker — evaluate 子进程入口（``python -m mls_infer_opt.evaluate.worker``）。

父进程把 ``JobSpec`` JSON 喂到 stdin，本进程跑 torch 评测，把 ``{"gate":…, "bench":…}`` JSON
吐回 stdout。**所有 torch / 风险代码只在这里跑**——坏候选崩溃/超时只死本进程，父进程兜回结构化失败。

stdout 纪律：评测期间把 ``sys.stdout`` 改指向 stderr，避免候选自己的 ``print`` 污染结果通道；
只有最后那行结果 JSON 写真正的 stdout。双层兜底：worker 自身异常也翻成 failed gate 再吐。
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

import torch

from ..state.common import to_dict
from ..state.eval import GateResult, ValidationError
from .protocol import JobSpec, job_from_json


def _run(spec: JobSpec) -> dict[str, Any]:
    from .bench import run_bench
    from .gate import run_gate

    gate = None
    bench = None

    if spec.task in ("gate", "both"):
        gate = run_gate(spec)

    if spec.task == "bench":
        bench = run_bench(spec)
    elif spec.task == "both" and gate is not None and gate.passed:
        bench = run_bench(spec)

    return {"gate": to_dict(gate), "bench": to_dict(bench)}


def main() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # 评测期间隔离候选的 print
    try:
        spec = job_from_json(sys.stdin.read())
        torch.manual_seed(spec.seed)
        output = _run(spec)
    except Exception as e:  # worker 自身异常 → 翻成 failed gate，绝不让父进程拿到空 stdout
        gate = GateResult(
            errors=[
                ValidationError(
                    stage="runtime",
                    message=f"worker crashed: {e}",
                    traceback_tail=traceback.format_exc()[-1600:],
                )
            ]
        )
        output = {"gate": to_dict(gate), "bench": None}
    finally:
        sys.stdout = real_stdout

    json.dump(output, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
