"""generate — 产出 engine 候选（相当于训练的 train step）。

一切「生成一份 engine 代码」的逻辑都在这里。三种触发场景，本质同一件事，只是条件不同：

- bootstrap：产一个保守、语义正确的初始 engine，不依赖 LLM、随时可得，既是搜索起点也是永久兜底。
  源码即 generate/assets/baseline_engine.py（pristine baseline 副本）。
- propose：按 analyze 产出的 Gradient（松建议 + rationale：瓶颈/方向/注意点）产新候选。
- repair：外层 full gate 没过时，拿结构化报错让 agent 调整（agent 内部的自修复不单列候选）。

生成方式（方案1 · agent 自带工具自闭环）：给 agent 提供 Read/Edit/Write（候选暂存区）+ quick
正确性 gate 作为工具，agent 边写边自检、连续迭代收敛，**每次调用产 1 个候选**。

共同约定：
- 产物是完整 engine.py：自包含纯 PyTorch、零硬编码、全部从 model_config 动态构建，
  不 import agent 包 / 不依赖网络。
- quick 自检是 agent 内层用的、ephemeral 不进 state；挂到 candidate.gate 的**只有外层 full
  gate**。**本模块只产候选、没有发布权**，正确性由外层 evaluate 权威保证、绝不自证（不变量 #5）。
- LLM 不可用 / 失败 / 产垃圾都只返回空，由 loop 走回退；bootstrap 不依赖 LLM、永久兜底。
- 预算分层：max_repair_retries=外层重试；agent 内部 tool-call 上限归 agent/llm 配置。

产出：Candidate（kind ∈ baseline|optimization|repair，带 parent_id/lineage）。
依赖：llm、state。bootstrap/propose/repair 的内部拆分与签名 TBD。

搜索维度（space / compat 约束 / dims 工具）已抽到独立的 ``searchspace`` 领域层，generate 与 analyze
都向下依赖它、彼此不再横向 import。本包只保留「产 engine 代码」自身的两件事：

- prompt  Gradient + 完整搜索维度 + 生成规范 → LLM prompt（纯文本拼接）
- codegen bootstrap/propose/repair：驱动 agent 工具自闭环（写 + quick 自检）→ 自包含校验 →
  落盘，返回 Candidate|None（实现待从 one-shot 迁移到 tool-loop）
"""

from __future__ import annotations

from .codegen import (
    LLMClient,
    baseline_engine_source,
    bootstrap,
    check_self_contained,
    propose,
    repair,
)
from .prompt import ENGINE_CONTRACT, build_prompt, render_suggestions

__all__ = [
    # prompt
    "ENGINE_CONTRACT",
    "build_prompt",
    "render_suggestions",
    # codegen
    "LLMClient",
    "baseline_engine_source",
    "bootstrap",
    "propose",
    "repair",
    "check_self_contained",
]
