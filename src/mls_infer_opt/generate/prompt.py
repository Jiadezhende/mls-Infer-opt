"""prompt — 把 Policy + 生成规范拼成喂给 LLM 的提示词（纯文本，无副作用、无 LLM 调用）。

初版的 policy→engine 不做确定性 render，而是把「生成规范 + 选定策略」注入 prompt 让 LLM 直接
产出完整 engine.py。本模块只负责拼 prompt：

- ``ENGINE_CONTRACT``：注入每个 prompt 的稳定知识（API 契约 / 自包含规则 / 正确性目标 / 输出格式）。
- ``build_prompt``：把 Policy 的非默认轴渲成中文策略指令（复用 space 的 AxisSpec.summary），
  叠上 model_config、参考代码、analyze 方向（policy.rationale，propose）或结构化报错（repair）。

LLM 调用、产物抽取、自包含校验、落盘都在 codegen.py。
"""

from __future__ import annotations

import json
from typing import Literal

from ..state.context import TaskContext
from ..state.eval import ValidationError
from .policy import Policy, grouped_axes
from .space import AXIS_BY_KEY, GROUP_ORDER

__all__ = ["ENGINE_CONTRACT", "build_prompt", "render_policy_instructions"]

PromptMode = Literal["propose", "repair"]

# 注入每个 prompt 的稳定知识。措辞稳定，便于缓存与审计 diff。
ENGINE_CONTRACT = """\
你是一个 LLM 推理引擎代码生成器。请产出一份**自包含的纯 PyTorch** `engine.py`。

## API 契约（评测器据此调用，签名/语义不可改）
- `create_engine(model_config: dict, weight_dir: str, device: str = "cuda") -> Engine`
- `Engine.prefill(request_ids, input_ids) -> Tensor`
  为每个新请求预填，返回末位 logits，形状 `[batch, vocab_size]`。
- `Engine.decode(request_ids, token_ids) -> Tensor`
  为每个请求追加一个 token 续算，返回 logits `[batch, vocab_size]`。
- `Engine.remove(request_ids) -> None`：结束请求、释放其状态。

## 自包含硬规则（违反任何一条都会被判废）
- 只允许 `import torch`、`import torch.nn.functional as F`、`import math`、`import os`；
  **禁止** import 本 agent 包（`mls_infer_opt`）、禁止任何网络/HTTP/LLM 调用、禁止读 `.env`。
- **零结构硬编码**：层数/隐藏维/头数/head_dim/vocab/rope_theta 等一律从 `model_config` 动态读取。
- 权重从 `os.path.join(weight_dir, "model.pt")` 用 `torch.load` 加载，key 命名固定：
  `embed_tokens.weight`、`layers.{i}.self_attn.{q,k,v,o}_proj.weight`、
  `layers.{i}.mlp.{gate,up,down}_proj.weight`、`layers.{i}.{input,post_attention}_layernorm.weight`、
  `norm.weight`、`lm_head.weight`。
- dtype/device 自适应：无 CUDA 退回 CPU+float32。

## 正确性目标
- 对官方 reference model 的 logits 满足 `allclose(atol=1e-2, rtol=1e-2)`，
  覆盖单/多请求 prefill+decode、运行中插入新请求、remove 后继续 decode。优化不得破坏数值正确性。

## 输出格式
- **只输出一个 ```python 代码块**，包含完整可直接运行的单文件 engine.py，不要额外解释。
"""


def render_policy_instructions(policy: Policy) -> str:
    """把 Policy 的**非默认轴**渲成中文策略清单（默认轴 = baseline 行为，不提）。"""
    groups = grouped_axes(policy)
    lines: list[str] = []
    for group in GROUP_ORDER:
        for axis, option in groups[group].items():
            spec = AXIS_BY_KEY[axis]
            knobs = [f"{k.key}={policy.knobs[k.key]}" for k in spec.knobs if k.key in policy.knobs]
            suffix = f"（参数：{', '.join(knobs)}）" if knobs else ""
            lines.append(f"- {axis} = {option}：{spec.summary}{suffix}")
    if not lines:
        return "- 无（保持保守 baseline 实现即可）"
    return "\n".join(lines)


def _format_error(e: ValidationError) -> str:
    bits = [f"- [{e.stage}] {e.message}"]
    if e.case:
        bits.append(f"  case={e.case}")
    if e.expected_shape is not None or e.actual_shape is not None:
        bits.append(f"  expected_shape={e.expected_shape} actual_shape={e.actual_shape}")
    if e.max_abs_err is not None or e.max_rel_err is not None:
        bits.append(f"  max_abs_err={e.max_abs_err} max_rel_err={e.max_rel_err}")
    if e.traceback_tail:
        bits.append(f"  traceback:\n{e.traceback_tail}")
    return "\n".join(bits)


def build_prompt(
    policy: Policy,
    ctx: TaskContext,
    *,
    mode: PromptMode,
    parent_code: str | None = None,
    errors: list[ValidationError] | None = None,
) -> str:
    """拼出完整 prompt。

    propose：parent_code 作参考基线 + policy.rationale 给方向；
    repair：parent_code 作待修代码 + errors 定向。
    """
    parts: list[str] = [ENGINE_CONTRACT, ""]

    parts.append("## 模型配置 model_config")
    parts.append("```json")
    parts.append(json.dumps(ctx.model_config, ensure_ascii=False, indent=2, sort_keys=True))
    parts.append("```")

    if parent_code:
        if mode == "repair":
            parts.append("## 待修复的 engine.py（正确性未过，请在此基础上定向修复）")
        else:
            parts.append("## 参考基线 engine.py（在此基础上按下方策略改写）")
        parts.append("```python")
        parts.append(parent_code)
        parts.append("```")

    parts.append("## 本次要采用的优化策略")
    parts.append(render_policy_instructions(policy))

    if mode == "propose" and policy.rationale:
        parts.append("## analyze 方向提示")
        parts.append(policy.rationale)

    if mode == "repair" and errors:
        parts.append("## 上次正确性失败，请定向修复")
        parts.extend(_format_error(e) for e in errors)

    parts.append("")
    parts.append("请输出完整的 engine.py，只用一个 ```python 代码块。")
    return "\n".join(parts)
