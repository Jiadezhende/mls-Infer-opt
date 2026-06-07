"""state — 共享数据契约（数据层）。

角色：定义 agent 内部、以及 agent ↔ 产物之间交换的稳定数据结构。所有子模块只通过这些结构
通信，不直接共享内部对象。

预期承载的核心对象（字段后续收敛，先对齐 specs/00_shared_contracts.md）：
- TaskContext                   —— loop INIT 产出
- Candidate                     —— 候选 engine 代码 + lineage/status（baseline|optimization|repair），generate 产出
- EvalResult                    —— 正确性结果 + metrics + 结构化诊断，evaluate 产出
- OptimizationPlan              —— analyze 产出，generate 消费（含 stop 建议）
- LoopState                     —— loop 主状态（candidates/history/budget/best/round/stop）
- AgentEvent                    —— 结构化事件流，report 消费

版本兼容不变量：每个结构保留未知字段；读取方只依赖已声明字段，写入方可加字段不改既有语义。
本模块不含业务逻辑，纯结构 + 轻量构造/校验。接口 TBD。
"""
