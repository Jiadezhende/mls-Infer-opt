"""llm — LLM 客户端基建。

角色：generate 与 analyze 共用的唯一 LLM 接入点。封装 provider、API 模式、tool loop
与健壮性。LLM 只在这两个业务模块里被调用，决定性的 loop/evaluate 不碰 LLM。

职责与经验约定（来自 MIGRATION.md 第 6 节，参考 sdk_reference/llm.py）：
- 多 provider：Chat Completions（OpenAI-compatible，如 DeepSeek）+ Responses API（原生 OpenAI）。
  provider/base_url/模式可探测 + 可 env 强制覆盖。工具主路径只在 Chat Completions 上实现。
- 配置走 env，自动加载 .env 但不覆盖已 export 的 shell 变量。
- 健壮性（硬要求）：缺 key / 没装 SDK **不抛异常**——暴露 ``available`` 标志，不可用时调用返回
  None，由上层走回退。网络调用内置有限次指数退避，最终失败返回 None。
- 响应解析对各 API 形态防御式提取；从文本稳健抽取代码块并先 compile() 语法检查再用。
- tool loop 有最大轮数上限；handler 抛错包成结构化错误回灌模型，不中断循环。

定位：LLM 是「可选增益」，不是硬依赖。本模块不含业务策略，只提供干净的调用原语。
接口 TBD。
"""
