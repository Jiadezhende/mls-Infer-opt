"""prompt — 把 Gradient + 生成规范拼成喂给 LLM 的提示词（纯文本，无副作用、无 LLM 调用）。

policy→engine 不做确定性 render：把「生成规范 + 完整搜索维度（界） + 依赖规则 + analyze 的松建议」
注入 prompt，让 agent 在维度界内自由探索、产出完整 engine.py。本模块只负责拼 prompt：

- ``ENGINE_CONTRACT``：注入每个 prompt 的稳定知识（API 契约 / 自包含规则 / 正确性目标 / 输出格式）。
- ``build_prompt``：叠上 model_config、参考代码、完整搜索维度 + 依赖规则、analyze 方向
  （Gradient.suggest_axes 松建议 + rationale，propose）或结构化报错（repair）。

LLM 调用、产物抽取、自包含校验、回报落盘都在 codegen.py。
"""

from __future__ import annotations

import json
from typing import Literal

from ..searchspace.compat import render_constraints
from ..searchspace.dims import render_search_dims
from ..searchspace.space import AXIS_BY_KEY
from ..state.context import TaskContext
from ..state.eval import ValidationError
from ..state.gradient import Gradient

__all__ = ["ENGINE_CONTRACT", "build_prompt", "render_suggestions"]

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


def render_suggestions(gradient: Gradient) -> str:
    """把 analyze 的松建议渲成中文清单：建议优先探索的轴 + 配套 knob。空则明确放手。"""
    lines: list[str] = []
    for axis, option in gradient.suggest_axes.items():
        spec = AXIS_BY_KEY.get(axis)
        summary = spec.summary if spec is not None else ""
        lines.append(f"- {axis} = {option}：{summary}")
    if gradient.knobs:
        kv = ", ".join(f"{k}={v}" for k, v in gradient.knobs.items())
        lines.append(f"- 配套参数建议：{kv}")
    if not lines:
        return "- 无具体建议：请在上方搜索维度界内自行判断该动哪些轴。"
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
    gradient: Gradient,
    ctx: TaskContext,
    *,
    mode: PromptMode,
    parent_code: str | None = None,
    errors: list[ValidationError] | None = None,
) -> str:
    """拼出完整 prompt。

    propose：parent_code 作参考基线 + 完整搜索维度（界）+ 依赖规则 + analyze 松建议/rationale；
    repair：parent_code 作待修代码 + errors 定向（维度/规则仍给作上下文）。
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
            parts.append("## 参考基线 engine.py（在此基础上探索优化）")
        parts.append("```python")
        parts.append(parent_code)
        parts.append("```")

    # 完整搜索维度 = 探索边界；依赖规则供自洽（非法组合交 full gate 拦）。
    parts.append("## 可探索的搜索维度（在这些轴/选项内自由组合）")
    parts.append(render_search_dims())
    parts.append("## 轴间依赖（尽量自洽）")
    parts.append(render_constraints())

    if mode == "propose":
        parts.append("## analyze 的方向建议（松提示，最终采用由你定）")
        parts.append(render_suggestions(gradient))
        if gradient.rationale:
            parts.append("## analyze 方向提示")
            parts.append(gradient.rationale)

    if mode == "repair" and errors:
        parts.append("## 上次正确性失败，请定向修复")
        parts.extend(_format_error(e) for e in errors)

    parts.append("")
    parts.append("请输出完整的 engine.py，只用一个 ```python 代码块。")
    return "\n".join(parts)
