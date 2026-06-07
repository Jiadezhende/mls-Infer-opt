"""analyze — 定位问题 + 给搜索空间里的下一步方向（相当于训练的 grad）。

每轮循环的「大脑」：看 evaluate 的反馈和历史，告诉 loop 往哪走、还要不要走。

职责：
- 汇总反馈：best metrics、history、近期失败、近期收益、剩余预算 → 一份当前态势。
- 定位瓶颈：从分项吞吐/显存/失败原因判断当前最该解决什么（prefill 慢？decode 慢？显存爆？
  正确性边界？）。
- 给方向：产出下一步策略 + runtime_knobs（**只允许运行时实现参数**，禁止模型结构字段
  num_hidden_layers/hidden_size/heads/head_dim/vocab/rope_theta…）+ 风险点 + 预期收益 +
  给 generate 的 prompt 提示。
- 判停：预算耗尽 / 连续多轮无提升 / 失败率过高 / 达标 / 收益不足 / 轮数或时间上限
  —— 给建议，由 loop 执行。

优化主线知识（搜索空间的先验）：baseline 每步重跑整段 → KV cache 增量解码 → batched prefill、
GQA、合理 dtype、显存复用、SDPA、（视设备）torch.compile。

LLM 可选：不可用时退化为基于规则的方向选择，不抛异常。
产出：当前态势 + OptimizationPlan（含 stop 建议）。依赖：llm、state。接口 TBD。
"""
