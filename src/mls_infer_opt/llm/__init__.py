"""llm — LLM 客户端基建。

角色：generate 与 analyze 共用的唯一 LLM 接入点。封装 provider、API 模式、tool loop
与健壮性。LLM 只在这两个业务模块里被调用，决定性的 loop/evaluate 不碰 LLM。

职责与经验约定（来自 MIGRATION.md 第 6 节，参考 sdk_reference/llm.py）：
- OpenAI Responses API 主路径：封装模型调用、function tools、tool loop 与响应提取。
  provider/base_url/model 可由 env 强制覆盖。
- 配置走 env，自动加载 .env 但不覆盖已 export 的 shell 变量。
- 健壮性：缺 key / 没装 SDK **不抛异常**——暴露 ``available`` 标志（构造时一次裁定、运行中不翻），
  不可用时调用返回 ok=False。**但传输/基建调用失败（C2）会 raise LLMCallError 穿透**——绝不静默
  降级，由总控在循环边界接住（见 PIPELINE_SPEC §3）。内容层失败（没给答复 / 工具循环没收敛）
  仍返回 ok=False，属 C1 邻域、上层可回退。
- 响应解析对各 API 形态防御式提取；从文本稳健抽取代码块并先 compile() 语法检查再用。
- tool loop 有最大轮数上限；handler 抛错包成结构化错误回灌模型，不中断循环。

定位：LLM 是「可选增益」，不是硬依赖。本模块不含业务策略，只提供干净的调用原语。
"""

from __future__ import annotations

from .config import LLMConfig
from .errors import LLMCallError, LLMError, LLMUnavailableError, ToolExecutionError
from .fake import FakeAgentClient
from .openai_client import AgentResult, OpenAIAgentClient, ToolCallRecord
from .tooling import ToolExecutor, ToolRegistry, ToolResult, ToolSpec, to_openai_tools

__all__ = [
    "LLMConfig",
    "LLMError",
    "LLMUnavailableError",
    "LLMCallError",
    "ToolExecutionError",
    "FakeAgentClient",
    "AgentResult",
    "OpenAIAgentClient",
    "ToolCallRecord",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "to_openai_tools",
]
