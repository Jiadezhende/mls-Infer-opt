"""analyze — 定位问题 + 给搜索空间里的下一步方向（相当于训练的 grad）。

每轮循环的「大脑」：看 evaluate 的反馈和历史，告诉 loop 往哪走、还要不要走。

职责：
- 汇总反馈：best metrics、history、近期失败、近期收益、剩余预算 → 一份当前态势。
- 定位瓶颈：从分项吞吐/显存/失败原因判断当前最该解决什么（prefill 慢？decode 慢？显存爆？
  正确性边界？）。
- 给方向：从 best.policy 出发做局部搜索——选要动的轴/knob 作 delta，用
  ``searchspace.policy.merge(best.policy, axes_delta=, knobs_delta=, rationale=)`` 构造**下一个
  合法 Policy** 返回（knob 只属 Policy.knobs，绝不碰 model_config 的结构字段）。``rationale``
  装瓶颈/方向/注意点的自然语言，generate 直接渲进 prompt。
- 判停：预算耗尽 / 连续多轮无提升 / 失败率过高 / 达标 / 收益不足 / 轮数或时间上限
  —— 返回 None，停因经 AgentEvent 记录、由 loop 落到 LoopState.stop_reason 执行收尾。

优化主线知识（搜索空间的先验）：baseline 每步重跑整段 → KV cache 增量解码 → batched prefill、
GQA、合理 dtype、显存复用、SDPA、（视设备）torch.compile。

LLM 可选：不可用时退化为基于规则的方向选择，不抛异常。
产出：下一个 Policy（带 rationale）或 None（停机）。依赖：llm、searchspace、state（不碰 generate）。

内部分层（对齐 generate 的粒度）：
- situation  汇总 LoopState → 当前态势视图（ephemeral，不进 state）
- heuristic  确定性判停（hard stop）+ rule-based 贪心阶梯方向 + 共享 Decision 类型
- prompt     态势 → LLM 诊断 prompt；LLM 回复 → Decision（单次调用 + 确定性解析）
- grad       编排入口 analyze()：判停 / 给方向 / merge 出 Policy，并每轮 emit 一条 analyze 事件
"""

from __future__ import annotations

from .grad import LLMClient, analyze
from .heuristic import MOVES, Decision, hard_stop_reason, heuristic_decision
from .prompt import ANALYZE_CONTRACT, build_analyze_prompt, parse_decision
from .situation import Situation, build_situation

__all__ = [
    # grad（主入口）
    "analyze",
    "LLMClient",
    # situation
    "Situation",
    "build_situation",
    # heuristic
    "Decision",
    "hard_stop_reason",
    "MOVES",
    "heuristic_decision",
    # prompt
    "ANALYZE_CONTRACT",
    "build_analyze_prompt",
    "parse_decision",
]
