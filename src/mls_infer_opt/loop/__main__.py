"""阶段 A 进程入口：``python -m mls_infer_opt.loop``（run.sh 调用）。

这一层只做 trainer 做不到的两件事，**不含任何业务逻辑**：

1. 装配——探测真实 environment/device、按 env 建真实 LLM client，组 ``LoopHooks``；
   ``run_loop`` 故意不做这些重活（见 trainer.build_task_context 的轻量 INIT 注释）。
2. 进程外壳——``run_loop`` 只返回 ``LoopState``、不控制退出码，其 never-throw 也兜不住「它自己
   崩」。这里用最外层 try/finally 保证两条对外契约即便 run_loop 整个崩溃也成立：
   **始终 exit 0**，且 **workspace/engine.py 必在盘上**（一切失败时补 pristine baseline 兜底）。

兜底链（与 ARCHITECTURE 一致）：最优正确候选 → 已验证 baseline → 原始 baseline。前两者由
loop.finalize 发布；最后一条「原始 baseline」是这里的职责——run_loop 没能跑到发布时的最后保险。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from ..generate import baseline_engine_source
from ..state.context import Environment, Limits
from .trainer import LoopConfig, LoopHooks, build_task_context, run_loop


def _probe_environment() -> tuple[Environment, str]:
    """客观采集 torch / GPU 快照并选 device；torch 缺失或探测失败都降级为 CPU，绝不抛。"""

    env = Environment(python_version=platform.python_version())
    device = "cpu"
    try:
        import torch

        env.torch_version = getattr(torch, "__version__", "") or ""
        if torch.cuda.is_available():
            device = "cuda"
            env.cuda_version = getattr(torch.version, "cuda", None)
            env.gpu_count = torch.cuda.device_count()
            env.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env.total_memory_mb = float(props.total_memory) / (1024 * 1024)
    except Exception:  # noqa: BLE001 — 探测是尽力而为；任何失败都退回 CPU 继续跑
        pass
    return env, device


def _build_llm() -> object | None:
    """按 env 建真实 LLM client；不可用时把原因打到 stderr 后**仍返回该 client**。

    返回不可用 client（其 ``available=False``）而非 None：consumers（analyze/generate）本就按
    ``.available`` 决策、行为不变，但 loop 能据此把 ``unavailable_reason`` 落进 results.log——
    杜绝「缺 key / 没装 SDK」这类降级被静默吞掉、事后只能靠时序猜。仅装配阶段自身抛异常时返回 None。
    """

    try:
        from ..llm import OpenAIAgentClient

        client = OpenAIAgentClient()
    except Exception as e:  # noqa: BLE001 — LLM 是可选增益，绝不能让装配阶段拖垮进程
        print(f"[loop.__main__] LLM 装配异常，退回规则兜底: {e}", file=sys.stderr)
        return None
    if not client.available:
        print(
            f"[loop.__main__] LLM 不可用，退回规则兜底: {client.unavailable_reason}",
            file=sys.stderr,
        )
    return client


def _ensure_fallback_engine(publish_path: str) -> None:
    """原始兜底：发布点缺 engine.py 时补一份 pristine baseline。best-effort，绝不抛。

    只在文件不存在/为空时写——已由 finalize 发布的已验证 best 不会被覆盖。
    """

    try:
        dst = Path(publish_path)
        if dst.exists() and dst.stat().st_size > 0:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(baseline_engine_source(), encoding="utf-8")
    except Exception:  # noqa: BLE001 — 已尽力；连兜底都写不进时不再有可做的事
        pass


def _time_budget_s() -> int:
    """从 env 读时间硬墙（秒）；缺省/非法都退回 0 = 不限。

    评测有外部墙（约 30min）时，run.sh 把它设成略小值（如 1680=28min），loop 会在 elapsed
    到点后于下一轮开始前优雅停 + finalize 发布 best，避免被外部 kill 导致只剩兜底 baseline。
    """

    raw = os.environ.get("MLS_TIME_BUDGET_S", "0")
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    """装配并跑一次调优循环。无论发生什么都保证 engine.py 落盘并返回 0。"""

    environment, device = _probe_environment()
    ctx = build_task_context(
        target_dir=os.environ.get("MLS_TARGET_DIR", "target"),
        runs_dir=os.environ.get("MLS_RUNS_DIR", "runs"),
        output_dir=os.environ.get("MLS_OUTPUT_DIR", "workspace"),
        device=device,
        environment=environment,
        limits=Limits(time_budget_s=_time_budget_s()),
    )
    try:
        run_loop(ctx, llm=_build_llm(), hooks=LoopHooks(), config=LoopConfig())
    except Exception as e:  # noqa: BLE001 — run_loop 之外的最后一道防线
        print(f"[loop.__main__] run_loop crashed, falling back to baseline: {e}", file=sys.stderr)
    finally:
        _ensure_fallback_engine(ctx.engine_publish_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
