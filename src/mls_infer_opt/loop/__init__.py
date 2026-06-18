"""loop — 顶层调优循环（相当于训练框架的 Trainer）。

把整件事看成一次「训练」：generate=train（产出 engine），evaluate=eval（正确性+吞吐
反馈），analyze=grad（定位问题、给搜索空间里的下一步方向）。loop 是驱动三者不断循环的
那段控制代码。

一次会话的主线：

    INIT          建立 TaskContext（读 model_config/weights/device、探测能力、预算约束）
    BOOTSTRAP     generate 产保守初始 engine → evaluate 验证 → 成为 best
    LOOP(每轮):
        analyze   看 best/history/反馈，定位瓶颈、判停、给下一步方向（grad）
        generate  按方向产新候选；过不了正确性则按报错修复（train）
        evaluate  正确性 gate + 吞吐 benchmark（eval）
        keep-best 与 best 比较，严格更优且已验证才替换（checkpoint 选优）
    FINALIZE      对 best 做完整验证 → 发布 workspace/engine.py → 写 output3.json（含 rounds[]
                  推理）+ runs/{run_id}/report.json（任务结果记录）；report3 是人手写开发报告、不产

loop 自己持有的职责（不外包给业务模块的「训练器」逻辑）：
- INIT/context、keep-best 选优、停止判定的执行、finalize 与 report 落盘。
- **唯一发布出口**：候选先落临时区，只有过正确性的 best 才由此处写入 workspace/engine.py。
- 兜底：任何时刻有一个已验证可用的 best；generate/analyze 失败只是「这轮没收益」。

不变量：未过 correctness 不发布；产物永不退化或缺失；generate/analyze 无直接发布权；
agent 始终 exit 0。

依赖：generate / evaluate / analyze / state。
"""

from __future__ import annotations

from .trainer import (
    AnalyzeFn,
    BootstrapFn,
    EvaluateFn,
    LoopConfig,
    LoopHooks,
    ProposeFn,
    RepairFn,
    build_task_context,
    finalize,
    hard_stop_reason,
    keep_best,
    run_loop,
)

__all__ = [
    "AnalyzeFn",
    "BootstrapFn",
    "EvaluateFn",
    "LoopConfig",
    "LoopHooks",
    "ProposeFn",
    "RepairFn",
    "build_task_context",
    "finalize",
    "hard_stop_reason",
    "keep_best",
    "run_loop",
]
