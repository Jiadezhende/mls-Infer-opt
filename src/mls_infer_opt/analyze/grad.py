"""grad — analyze 的编排入口：汇总态势 → 给方向（或无方向）→ 产 Gradient，并每轮记一条事件。

每轮循环的「大脑」=求梯度：只算「往哪走」。**LLM 是唯一方向源**（单次调用 + 确定性解析，见
prompt.py）：

- LLM 不可用 / 未配置 → 首轮即 ``NoMove("llm_unavailable")``，交总控停、发布 baseline。
- LLM 内容失败（``ok=False`` / 解析不出，C1 邻域）→ **重试一次**，仍失败则 NoMove。
- LLM 调用基建失败（``LLMError``，C2）→ 穿透交总控（不在此降级）。

不变量纪律：
- **只算方向、不判停**：analyze 返回 ``Gradient``（迈出的一步）或 ``NoMove(reason)``（gradient≈0，
  迈不出）。**硬上限判停（预算 / 轮数 / 连续无提升）不在这里**——那是总控的循环准则（见 loop）。
- **never-throw（除 C2）**：内部非 C2 异常都翻成一条 error 事件 + 返回 ``NoMove("analyze_error")``。
- **无发布权 / 不写 stop_reason**：每轮 emit 一条 ``source="analyze"`` 事件作记录（见
  [[analyze-record-via-events]]）；是否因 NoMove 终止、停因落 ``LoopState.stop_reason``，由总控定。
- **Gradient 是松建议、不定点**：``suggest_axes`` 已过词表闸（parse_gradient 里 sanitize）；空建议
  **照常 propose**（生成器在维度界内自由探索），不当作 NoMove。实际采用哪些轴由 generate 的 agent
  回报、落 candidate.strategy_tags（PR-A）——analyze 不再 merge 定点、不读 policy.json。
"""

from __future__ import annotations

from typing import Any, Protocol

from ..llm.errors import LLMError  # 仅 errors（零依赖），避免经 llm/__init__ 拉入 present 导入环
from ..state.gradient import Gradient, NoMove
from ..state.loop import LoopState, emit
from .prompt import build_analyze_prompt, parse_gradient
from .situation import Situation, build_situation

__all__ = ["LLMClient", "analyze"]


class LLMClient(Protocol):
    """analyze 需要的最小 LLM 接口；与 generate 一致，统一走 run_agent。

    ``available`` 为假 → analyze 首轮 NoMove；调用抛 LLMError(C2) 穿透；返回空 / 解析不出（C1）
    重试一次后 NoMove。
    """

    available: bool

    def run_agent(
        self, prompt: str, tools: list[Any] | None = ..., **kwargs: Any
    ) -> Any: ...


def analyze(state: LoopState, *, llm: LLMClient | None = None) -> Gradient | NoMove:
    """只算方向：产 Gradient，或 NoMove(reason)（迈不出步）。每轮 emit 一条 analyze 事件。

    LLM 是唯一方向源：不可用 → NoMove("llm_unavailable")；内容失败重试一次仍败 →
    NoMove("llm_content_failure")；C2 穿透。硬上限判停不在这里——是总控的循环准则。
    never-throw（除 C2）：内部异常 → 记错 + NoMove("analyze_error")。
    """
    try:
        sit = build_situation(state)
    except Exception as e:  # 连态势都建不起来：无方向可算，记错交总控。
        emit(state, source="analyze", phase="grad", message="无方向：analyze 建态势异常",
             level="error",
             data={"detail": f"analyze crashed building situation: {e}",
                   "decision": "stop", "stop_reason": "analyze_error"})
        return NoMove("analyze_error")

    # 1. LLM 是唯一方向源：不可用即首轮无方向，交总控停、发布 baseline。
    if llm is None or not getattr(llm, "available", False):
        emit(state, source="analyze", phase="grad", message="无方向：LLM 不可用",
             data={"detail": "LLM 未配置或不可用，无方向可算", "decision": "stop",
                   "stop_reason": "llm_unavailable", "used_llm": False, **_situation_data(sit)})
        return NoMove("llm_unavailable")

    # 2. 问 LLM 要方向（内容失败重试一次；C2 穿透）。
    result = _ask_llm(sit, llm)
    if result is None:
        emit(state, source="analyze", phase="grad", message="无方向：LLM 内容失败",
             data={"detail": "LLM 回复解析不出有效方向（重试一次后仍失败）", "decision": "stop",
                   "stop_reason": "llm_content_failure", "used_llm": True, **_situation_data(sit)})
        return NoMove("llm_content_failure")

    # 3. LLM 判定到位 → NoMove，交总控裁决是否终止。
    if isinstance(result, NoMove):
        emit(state, source="analyze", phase="grad", message=f"无方向：{result.reason}",
             data={"decision": "stop", "stop_reason": result.reason,
                   "used_llm": True, **_situation_data(sit)})
        return result

    # 4. 得到一步方向：据态势补全血缘（round/parent_id/kind）。空建议照常 propose（生成器自由探索）
    gradient = result
    gradient.round = state.round + 1
    gradient.parent_id = state.best_id
    gradient.kind = "optimization"

    emit(
        state,
        source="analyze",
        phase="grad",
        message=f"继续：{gradient.bottleneck or '局部搜索下一步'}",
        data={
            "detail": gradient.rationale,
            "decision": "continue",
            "used_llm": True,
            "suggest_axes": gradient.suggest_axes,
            "knobs": gradient.knobs,
            "bottleneck": gradient.bottleneck,
            **_situation_data(sit),
        },
    )
    return gradient


# === 内部 =============================================================
def _ask_llm(sit: Situation, llm: LLMClient) -> Gradient | NoMove | None:
    """问 LLM 要方向，**内容失败重试一次**。返回 Gradient / NoMove(stop) / None（仍要不到）。

    内容层失败（run_agent ok=False / 解析不出，C1 邻域）→ 再问一次同样的 prompt；传输/基建失败
    （run_agent raise LLMError，C2）→ 向上穿透，不重试、不降级。其它非预期异常当内容失败处理。
    stop 决策（NoMove）是有效结果、立即返回不重试。
    """
    prompt = build_analyze_prompt(sit)
    for _ in range(2):  # 1 次初试 + 1 次重试
        try:
            text = _call_llm(llm, prompt)
        except LLMError:
            raise  # C2：基建失败，穿透到总控的循环边界
        except Exception:
            text = None  # 其它非预期 → 当内容失败，进重试
        result = parse_gradient(text)
        if result is not None:
            return result
    return None


def _call_llm(llm: LLMClient, prompt: str) -> str | None:
    """走 run_agent 取 .text；无 run_agent 或 ok=False → None（C1 内容失败，由 _ask_llm 重试）。"""
    runner = getattr(llm, "run_agent", None)
    if not callable(runner):
        return None
    result = runner(prompt)
    if not getattr(result, "ok", False):
        return None
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else None


def _situation_data(sit: Situation) -> dict[str, Any]:
    """事件里随手带的态势数字（结构化，report 直接消费）。"""
    return {
        "round": sit.round,
        "best_id": sit.best_id,
        "best_score": sit.best_score,
        "stale_rounds": sit.stale_rounds,
        "n_candidates": sit.n_candidates,
        "n_rejected": sit.n_rejected,
    }
