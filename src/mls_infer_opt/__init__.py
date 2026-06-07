"""推理框架自动调优 agent（阶段 A）。

整个 agent 是一个训练循环：generate(train) → evaluate(eval) → analyze(grad)，由
loop 驱动不断迭代 engine。导览见 ``src/mls_infer_opt/ARCHITECTURE.md``。

业务模块：loop(trainer) / generate(产 engine) / evaluate(评测) / analyze(方向)。
基建：llm（LLM 客户端）/ state（共享数据契约）。
"""

__version__ = "0.1.0"
