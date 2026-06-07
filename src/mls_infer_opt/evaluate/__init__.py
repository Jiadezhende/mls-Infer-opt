"""evaluate — 评测候选，产出反馈信号（相当于训练的 eval）。

唯一可靠的反馈来源。对一个候选给出两层信号，决定它能否留下、好不好：

1. 正确性（硬约束 / gate）：syntax → api 契约 → 对官方 reference model 比 logits
   allclose(atol=1e-2, rtol=1e-2)，覆盖 single/multi prefill+decode、插入新请求、
   remove 后继续 decode。不过则该候选作废（性能分为 0）。ground truth 复用
   inference-core-ref/evaluator/{reference_model,test_correctness}.py。
2. 性能（分数）：prefill/decode/mixed 三类吞吐 + 峰值显存。计时只覆盖 prefill/decode/remove，
   不含 create_engine/权重加载。复用 inference-core-ref/evaluator/benchmark_throughput.py 口径。

输出要既能给 loop 做 keep-best 比较（归一化 score），又能给 analyze 定位问题
（结构化失败原因：stage/case/max_abs_err/shape/traceback；分项吞吐与显存）。

约定：
- 决定性、可复现、不调 LLM。
- 有次数预算并复用同一候选已有结果，避免重复跑昂贵评测；分 quick（循环内）/ full（发布前）。
- 只有通过正确性的候选才进入性能评测。

产出：EvalResult（correctness + metrics + 结构化诊断）。依赖：state、外部 evaluator。接口 TBD。
"""
