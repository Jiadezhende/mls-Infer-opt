"""protocol — evaluate 父子进程之间的 IPC 契约（纯 stdlib，父子共享）。

子进程 worker 跑 torch、吐 ``dataclasses.asdict`` 形式的结果 JSON；父进程只 import 本模块
重建回 ``GateResult`` / ``BenchResult``。**张量永不过进程边界**——所有结果都是 primitive。

设计点：
- ``JobSpec`` 是父→子唯一入参（一段 JSON）；``WorkerOutput`` 是子→父唯一出参（一段 JSON）。
- 重建 from_dict 放这里、不放 state——这是 evaluate 内部的 IPC 关切，state 保持纯结构。
- 容错重建：worker 万一漏字段也不炸，缺啥用 dataclass 默认值兜（评测器宁可降级不崩，不变量 #3）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ..state.eval import BenchResult, EvalMode, GateResult, ValidationError

__all__ = [
    "JobTask",
    "JobSpec",
    "job_to_json",
    "job_from_json",
    "validation_error_from_dict",
    "gate_from_dict",
    "bench_from_dict",
]

# 一次 worker 调用要做什么：只 gate / 只 bench / 两者（gate 过了才接着 bench）。
JobTask = Literal["gate", "bench", "both"]


@dataclass
class JobSpec:
    """父→子的完整作业描述。全是 JSON-friendly 标量 / dict，无张量、无回调。

    ``oracle_cache_path`` 为 reference logits 的落盘缓存（worker 侧 oracle 读写）；同一
    ``(config, weights, case-set, seed)`` 跨候选恒定，算一次复用。
    """

    engine_path: str
    weight_dir: str
    model_config: dict[str, Any]
    device: str = "cpu"
    mode: EvalMode = "full"
    task: JobTask = "both"
    seed: int = 7
    oracle_cache_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def job_to_json(spec: JobSpec) -> str:
    """JobSpec → JSON 字符串（喂 worker stdin）。确定性：sort_keys，保中文。"""
    payload = {
        "engine_path": spec.engine_path,
        "weight_dir": spec.weight_dir,
        "model_config": spec.model_config,
        "device": spec.device,
        "mode": spec.mode,
        "task": spec.task,
        "seed": spec.seed,
        "oracle_cache_path": spec.oracle_cache_path,
        "extra": spec.extra,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def job_from_json(text: str) -> JobSpec:
    """JSON 字符串 → JobSpec（worker 侧解析）。缺字段用默认值兜。"""
    d = json.loads(text)
    return JobSpec(
        engine_path=d["engine_path"],
        weight_dir=d["weight_dir"],
        model_config=d.get("model_config", {}),
        device=d.get("device", "cpu"),
        mode=d.get("mode", "full"),
        task=d.get("task", "both"),
        seed=int(d.get("seed", 7)),
        oracle_cache_path=d.get("oracle_cache_path"),
        extra=d.get("extra", {}),
    )


# === 子→父：把 worker 吐的 dict 重建回 state 结构 =========================
def validation_error_from_dict(d: dict[str, Any]) -> ValidationError:
    """重建单个 ValidationError；未知字段忽略，已声明字段缺省用默认值。"""
    return ValidationError(
        stage=d.get("stage", "runtime"),
        message=d.get("message", ""),
        case=d.get("case"),
        max_abs_err=d.get("max_abs_err"),
        max_rel_err=d.get("max_rel_err"),
        expected_shape=d.get("expected_shape"),
        actual_shape=d.get("actual_shape"),
        traceback_tail=d.get("traceback_tail"),
        extra=d.get("extra", {}),
    )


def gate_from_dict(d: dict[str, Any]) -> GateResult:
    """重建 GateResult（含嵌套 errors）。passed 直接信 worker 给的布尔。"""
    return GateResult(
        syntax_ok=bool(d.get("syntax_ok", False)),
        api_ok=bool(d.get("api_ok", False)),
        correctness_ok=bool(d.get("correctness_ok", False)),
        passed=bool(d.get("passed", False)),
        errors=[validation_error_from_dict(e) for e in d.get("errors", [])],
        case_summary=d.get("case_summary", {}),
        duration_s=float(d.get("duration_s", 0.0)),
        extra=d.get("extra", {}),
    )


def bench_from_dict(d: dict[str, Any]) -> BenchResult:
    """重建 BenchResult。"""
    return BenchResult(
        mode=d.get("mode", "quick"),
        prefill_tps=float(d.get("prefill_tps", 0.0)),
        decode_tps=float(d.get("decode_tps", 0.0)),
        mixed_tps=float(d.get("mixed_tps", 0.0)),
        mixed_decode_tps=float(d.get("mixed_decode_tps", 0.0)),
        peak_memory_mb=float(d.get("peak_memory_mb", 0.0)),
        score=float(d.get("score", 0.0)),
        loss=float(d.get("loss", 0.0)),
        raw=d.get("raw", {}),
        duration_s=float(d.get("duration_s", 0.0)),
        warnings=list(d.get("warnings", [])),
        extra=d.get("extra", {}),
    )
