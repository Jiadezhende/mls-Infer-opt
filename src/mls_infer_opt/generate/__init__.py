"""generate — 产出 engine 候选（相当于训练的 train step）。

一切「生成一份 engine 代码」的逻辑都在这里。三种触发场景，本质同一件事，只是条件不同：

- bootstrap：产一个保守、语义正确的初始 engine（KV-cache 增量解码、能用 SDPA 就用、dtype 安全），
  不依赖 LLM、随时可得，既是搜索起点也是永久兜底。参考 inference-core-ref/baseline。
- propose：在 analyze 给的方向（瓶颈 / 策略 / runtime_knobs）下产新候选。
- repair：候选过不了正确性时，拿 evaluate 的结构化报错，产修复后的候选。

共同约定：
- 产物是完整 engine.py：自包含纯 PyTorch、零硬编码、全部从 model_config 动态构建，
  不 import agent 包 / 不依赖网络。
- 优先在稳定骨架上替换策略点，整文件生成作兜底。
- LLM 是可选增益：不可用 / 失败 / 产垃圾都只返回空，由 loop 走回退；
  **本模块只产候选，没有任何发布权**，正确性由 evaluate 保证、不自证。

产出：Candidate（kind ∈ baseline|optimization|repair，带 parent_id/lineage）。
依赖：llm、state。bootstrap/propose/repair 的内部拆分与签名 TBD。
"""
