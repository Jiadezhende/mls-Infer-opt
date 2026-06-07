"""eval — evaluate 产出的反馈信号：门控（正确性）+ 性能。

正确性与性能**物理分离**：GateResult 是硬布尔门，BenchResult 是性能。两者随生命周期后填、
直接挂在 Candidate 上（candidate.gate / candidate.bench），不走 candidate_id 外键；只有
gate.passed 的候选才会有 bench（不变量在 Candidate.attach_bench 焊死）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "GateStage",
    "EvalMode",
    "ValidationError",
    "GateResult",
    "BenchResult",
]

GateStage = Literal["syntax", "api", "correctness", "runtime"]
# quick：agent 自带工具在内层迭代自检用（小批 / 少 case，便宜、快；ephemeral，不进 state）。
# full ：外层 loop/evaluate 在 keep-best / 发布前跑的权威校验，是 candidate.gate 的唯一真相。
EvalMode = Literal["quick", "full"]


@dataclass
class ValidationError:
    """结构化失败原因——给 generate.repair / analyze 程序化消费，不是 human message。"""

    stage: GateStage
    message: str
    case: str | None = None
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    expected_shape: list[int] | None = None
    actual_shape: list[int] | None = None
    traceback_tail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """语法 / 接口 / 正确性硬门。passed == syntax_ok and api_ok and correctness_ok。

    correctness 判定：对官方 reference model 比 logits allclose(atol=1e-2, rtol=1e-2)，覆盖
    guide 6 类 case（single/multi prefill+decode、插入新请求、remove 后继续 decode），并在变化的
    batch / 长度 / 顺序上做泛化抽测；逐 case / 逐配置通过情况放 case_summary（dict，不另起嵌套
    结构）。不过则候选作废。

    生成采用「agent 自带工具自闭环」：generate 的 agent 用 quick gate 在内层边写边自检收敛，
    但那些自检是 ephemeral 的、不进 state。挂到 candidate.gate 的**只有外层 full gate**——loop
    在 keep-best / 发布前重跑、作唯一真相，绝不信 agent 自报（不变量 #5）。不存 candidate_id 回指。
    """

    syntax_ok: bool = False
    api_ok: bool = False
    correctness_ok: bool = False
    passed: bool = False
    errors: list[ValidationError] = field(default_factory=list)
    case_summary: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    """性能测量（严格对照 PHASE3 guide 口径）。**只有 gate.passed 的候选才会有 bench。**

    guide 定义（计时只覆盖 prefill/decode/remove，不含 create_engine / 权重加载）：
    - throughput      tokens/s = (prefill + decode tokens) / elapsed
    - decode tokens/s          = decode tokens / elapsed
    - 三类 benchmark：prefill（长 prompt 批量预填）/ decode（多请求连续解码）/ mixed（含 remove）
    字段对应：prefill_tps / decode_tps / mixed_tps / mixed_decode_tps + peak_memory_mb。

    泛化：隐藏评测会变 batch / prompt 长度 / decode 步数 / 请求顺序，故 bench 在多组配置上跑，
    headline 字段取这些配置上的代表 / 聚合值，逐配置明细放 raw（不另起嵌套结构）。
    score 给 loop keep-best 比较（归一化标量）；loss 给 analyze 诊断。挂在 candidate.bench 上。
    """

    mode: EvalMode = "quick"
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    mixed_tps: float = 0.0
    mixed_decode_tps: float = 0.0
    peak_memory_mb: float = 0.0
    score: float = 0.0
    loss: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
