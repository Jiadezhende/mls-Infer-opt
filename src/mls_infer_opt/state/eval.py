"""eval — evaluate 产出的反馈信号：门控（正确性）+ 性能。

正确性与性能**物理分离**：GateResult 是硬布尔门，BenchResult 是性能。两者随生命周期后填、
直接挂在 Candidate 上（candidate.gate / candidate.bench），不走 candidate_id 外键；只有
gate.passed 的候选才会有 bench（不变量在 Candidate.attach_bench 焊死）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "GateStage",
    "EvalMode",
    "ValidationError",
    "GateResult",
    "BenchResult",
    "geomean_score",
    "normalized_speedup_score",
]

# 放本模块（零 torch）以便 worker(bench) 与父进程(trainer) 共用同一份公式，不把 torch 拖进父进程。
_SCORE_EPS = 1e-9


def geomean_score(*ratios: float) -> float:
    """传入比值的**等权**几何平均（>0，越大越好）。scale-free：不被某类量级绑架。

    口径对齐真实 grader：它逐 case 报「整体 tokens/s」与「decode tokens/s」两列、外加显存，**不公布
    任何合成权重**。故这里不再内置 decode/mixed/prefill 权重，改由调用方把每个被计量的比值（各 case
    的 overall-tps ratio + decode-tps ratio，见 normalized_speedup_score）平铺进来，等权合成——
    overall:decode 的相对话语权由「平铺了几项」自然决定，而非我们拍的常数。

    父进程传"对 baseline 的 ratio"（baseline 自身→各 1.0→score 1.0）；worker 临时自评传原始 tps
    （参照 ref=1）。惩罚偏科：任一项塌到 0（EPS 下限）即把整体拉到近 0，符合"该项失败=灾难"。
    每个入参单调。无入参 → 0.0。
    """
    vals = [math.log(max(r, _SCORE_EPS)) for r in ratios]
    if not vals:
        return 0.0
    return math.exp(sum(vals) / len(vals))

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
    每类都按 grader 的两列计量：整体 tokens/s 与 decode tokens/s。字段对应——
      · prefill：prefill_tps（整体；该 case 无 decode）
      · decode ：decode_overall_tps（整体）/ decode_tps（decode 列）
      · mixed  ：mixed_tps（整体）/ mixed_decode_tps（decode 列）
    外加 peak_memory_mb（grader 每 case 都报；当前只计量+展示，不入 score，作护栏）。

    泛化：隐藏评测会变 batch / prompt 长度 / decode 步数 / 请求顺序，故 bench 在多组配置上跑，
    headline 字段取这些配置上的代表 / 聚合值，逐配置明细放 raw（不另起嵌套结构）。
    score 给 loop keep-best 比较（归一化标量）；loss 给 analyze 诊断。挂在 candidate.bench 上。
    """

    mode: EvalMode = "quick"
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    decode_overall_tps: float = 0.0
    mixed_tps: float = 0.0
    mixed_decode_tps: float = 0.0
    peak_memory_mb: float = 0.0
    score: float = 0.0
    loss: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def normalized_speedup_score(bench: BenchResult, ref_bench: BenchResult) -> float:
    """把 bench 的各计量列 tps 对 ref（baseline）求比值，平铺进等权几何平均加速比。

    口径对齐真实 grader：它逐 case 报「整体 tokens/s」与「decode tokens/s」两列、不公布合成权重。
    这里就把 grader 会量的每一列对 baseline 求比值、平铺进等权几何平均：
      · prefill case：整体 tps（无 decode 列）
      · decode  case：整体 tps + decode 列
      · mixed   case：整体 tps + decode 列
    overall:decode 的相对话语权由「平铺了几项」自然给出（overall 3 项 / decode 2 项），不再拍
    0.60/0.25/0.15 这种合成权重。peak_memory 不入分（grader 虽报，合成口径未知，先作护栏）。

    参照系 = bootstrap baseline 的 per-列 tps（自校准到真实评测硬件）：baseline 自身 ref 即自身
    → 各 ratio 1.0 → score 1.0；后续候选 score≈×baseline，让 keep-best 严格比较与 speedup 展示
    都有诚实语义。是 score 在所有消费者（keep_best / fmt_score_line / analyze）之前的唯一咽喉口径。
    """

    def _ratio(cand_tps: float, ref_tps: float) -> float:
        # baseline 该列无数据（case 失败/0，如纯 prefill 的 decode 列）→ ratio 中性 1.0，
        # 不让它把整体几何平均拖塌；候选在有 baseline 信号的列丢分则照实惩罚。
        if ref_tps <= _SCORE_EPS:
            return 1.0
        return cand_tps / ref_tps

    return geomean_score(
        _ratio(bench.prefill_tps, ref_bench.prefill_tps),            # prefill 整体
        _ratio(bench.decode_overall_tps, ref_bench.decode_overall_tps),  # decode 整体
        _ratio(bench.decode_tps, ref_bench.decode_tps),             # decode 列
        _ratio(bench.mixed_tps, ref_bench.mixed_tps),               # mixed 整体
        _ratio(bench.mixed_decode_tps, ref_bench.mixed_decode_tps),  # mixed decode 列
    )
