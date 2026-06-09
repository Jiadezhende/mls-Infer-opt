# Phase 3 评测结果分析

## 提交信息

| 项 | 值 |
|---|---|
| output_file id | `0e16044dcaec8b4822c7612ed779d29d` |
| run id | `run-1780944148` |
| 结果判定 | `accepted`（正确性通过） |
| 运行时长 | 1593 s（28min 硬墙内 finalize） |
| 候选数 / 评测轮 | 11 / 11 |
| 停因 | `budget_exhausted_diminishing_returns` |
| 模型 | 公开示例：12L · hidden=768 · heads=4 · vocab=4096 |

## 真实 grader 逐 case 结果

grader 按三列口径报告（整体 tps / decode tps / 峰值显存），无合成权重：

| case | 整体 tok/s | decode tok/s | 峰值显存 (MB) |
|---|---|---|---|
| 1 | 40703 | 0（纯 prefill） | 851 |
| 2 | 4380 | 487 | 907 |
| 3 | 6099 | 273 | 875 |
| 4 | 12030 | 275 | 886 |

baseline 参照（pristine engine）：decode ~89–91 tok/s · 266 MB。

## 与 baseline 对比

- **decode 吞吐**：case 2/3/4 从 ~90 提到 273–487 tok/s，约 **3–5×**，是主要收益来源。case 1 为纯 prefill case，decode=0 属正常，其整体 tps 由 prefill 决定。
- **显存**：266 → ~850–900 MB，约 **3×**。静态预分配 KV + padded batch + torch.compile 的代价；绝对值仍 < 24GB 卡的 4%，不构成约束。
- **综合分**（内部等权 geomean，对齐 grader 三列）：**1.000 → 2.908**，即 **≈2.9× 加速**，正确性 PASSED。

## 优化轨迹

每轮 `used_llm: true`，由 analyze → generate（agent 自检自闭环）→ gate 驱动：

| 轮 | 关键策略增量 | score | delta |
|---|---|---|---|
| r1 | 增量 KV 缓存 | 0.965 | （较 baseline 略降） |
| r2 | + batched decode | 1.564 | +0.599 |
| r3 | + 静态预分配 KV + RoPE 预计算表 | 1.588 | +0.024 |
| r4 | + qkv/mlp 融合 + 融合权重布局 | 1.689 | +0.101 |
| r5 | **+ padded batch prefill** | 2.662 | **+0.973** |
| r6 | + SDPA + enable_gqa | 2.845 | +0.183 |
| r7 | + torch.compile reduce_overhead | 2.899 | +0.054 |
| r9 | torch.compile → max_autotune | **2.908** | +0.009 |
| r10 | fp16 KV 缓存（回退） | 2.801 | −0.108 |
| r11 | attention 保持 compute dtype | 2.850 | +0.050 |

best 落在 **r9-838f014d**（score 2.908）。策略组合：静态预分配 KV、padded prefill、batched decode、SDPA+GQA、RoPE 预计算、qkv/mlp 融合、融合权重布局、contiguous、torch.compile max_autotune。

两次主跳：**r2 的 batched decode**（+0.60）和 **r5 的 padded prefill**（+0.97）拿到结构性收益；r6 之后多为 compile 模式微调，单轮增量降到 0.05 上下，r10 的 fp16 KV 数值敏感项还回退了。停因 `diminishing_returns` 与该曲线一致——安全的结构性空间已基本耗尽。

## 保留点

1. **泛化性未知**：以上为公开示例模型的数。隐藏评测会替换模型规模、权重、batch/prompt 长度与 trace，绝对吞吐和加速比都可能变化。
2. **剩余空间偏险**：往下主要剩数值敏感项（bf16 compute、attention 精度放宽、cache dtype），收益不确定且需正确性兜底；r10 fp16 KV 回退即一例。
3. **显存口径**：当前以 ~3× baseline 显存换 3–5× decode 吞吐。若隐藏评测显存权重更敏感或模型更大导致 KV 预分配吃紧，该 tradeoff 需重估。

## 此前 submit 异常（已修复，附记）

修复前连续多次 submit 均 ~75s 收尾、只发 baseline。根因：评测容器运行时注入 `OPENAI_API_KEY`（课程 key）盖过仓库 `.env`，但未注入 base_url → 课程 key 打 xiaoai 端点 → 每轮 `responses.create` 返回 `401 Invalid token`、generate 全程产不出候选。本地无注入故未暴露。修复将凭证 precedence 收敛在 `config._merged_env`：`.env` 对凭证逐键权威、环境变量仅在 `.env` 未设时兜底。本次 run 即为修复后结果。
