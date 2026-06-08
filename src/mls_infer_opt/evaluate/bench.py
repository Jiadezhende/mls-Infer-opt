"""bench — 性能评测：prefill/decode/mixed 三类吞吐 + 峰值显存（worker 侧，产 BenchResult）。

口径照搬外部 ``benchmark_throughput.py``：计时只覆盖 prefill/decode/remove（不含 create_engine /
权重加载）；每个 case 先 warmup 再 repeat 次、取**中位数**那次的测量；峰值显存只有 CUDA 有
（CPU 下为 0）。**只有 gate.passed 的候选才会调本模块**（守护在 Candidate.attach_bench）。

full 工况刻意**不规则**（ragged）：decode/mixed 每请求长度/停止步各异、跨多个 seed，逐 seed 取
**下半均值**聚合——隐藏评测会换 batch/长度/步数/顺序，等长 lockstep 流过拟合分组策略，ragged 流才
逼出 varlen 批处理/KV-cache 的真实鲁棒性。逐 (case,seed) 明细落 ``raw``，跨 seed 离散度落
``extra``。

口径对齐真实 grader：它逐 case 报「整体 tokens/s」+「decode tokens/s」两列 + 峰值显存，故每类
case 同时留**整体 tps**（prefill_tps / decode_overall_tps / mixed_tps）与 **decode tps**
（decode_tps / mixed_decode_tps）；显存 peak_memory_mb 只计量不入分（护栏）。

score：worker 侧只算**临时自评**（等权几何平均，参照 ref=1），仅供 loop 外直接 ``evaluate()``
用；loop 内的**权威 score 由父进程按 baseline per-列 tps 归一化后覆盖**（见
loop/trainer._normalize_score，把各 case 的 overall/decode 两列比值平铺进等权几何平均）。
几何平均 scale-free、惩罚偏科；loss = -score。
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..state.eval import BenchResult, EvalMode, geomean_score
from .protocol import JobSpec

__all__ = ["run_bench"]

# 跨 seed 跑不规则工况的随机种子；7 居首延续 oracle cache / 历史可比。
_FULL_SEEDS = (7, 101, 202)
# 单候选 bench 软墙（秒）：超时则跳过剩余 (case,seed)，保护 28min 共享硬墙。_agg 容忍少值。
_BENCH_WALL_S = 90.0


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _rand(shape: tuple[int, ...], vocab: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, vocab, shape, generator=gen, dtype=torch.long)


def _gen(seed: int) -> torch.Generator:
    """每 seed 独立 Generator——同 seed 恒产逐元素一致的事件流，跨 seed 互不相干。"""
    return torch.Generator().manual_seed(int(seed))


def _prefill_events(
    batch: int, prompt_len: int, vocab: int, gen: torch.Generator, device: str
) -> list[dict[str, Any]]:
    rids = list(range(batch))
    inputs = [_rand((prompt_len,), vocab, gen).to(device) for _ in range(batch)]
    return [
        {"op": "prefill", "request_ids": rids, "input_ids": inputs},
        {"op": "remove", "request_ids": rids},
    ]


def _decode_events(
    batch: int, prompt_len: int, steps: int, vocab: int, gen: torch.Generator, device: str
) -> list[dict[str, Any]]:
    rids = list(range(batch))
    inputs = [_rand((prompt_len,), vocab, gen).to(device) for _ in range(batch)]
    events: list[dict[str, Any]] = [{"op": "prefill", "request_ids": rids, "input_ids": inputs}]
    for _ in range(steps):
        tok = _rand((batch,), vocab, gen).to(device)
        events.append({"op": "decode", "request_ids": rids, "token_ids": tok})
    events.append({"op": "remove", "request_ids": rids})
    return events


def _mixed_events(vocab: int, gen: torch.Generator, device: str) -> list[dict[str, Any]]:
    """含 remove 的动态调度（照搬 benchmark_throughput.make_mixed_case 结构）。"""
    events: list[dict[str, Any]] = []
    active: set[int] = set()
    next_id = 0
    schedule = [
        ("prefill", 4, 64), ("decode", 4, None), ("decode", 4, None),
        ("prefill", 2, 128), ("decode", 6, None), ("remove", 2, None),
        ("prefill", 4, 32), ("decode", 8, None), ("decode", 8, None), ("remove", 8, None),
    ]
    for op, count, prompt_len in schedule:
        if op == "prefill":
            assert prompt_len is not None
            rids = list(range(next_id, next_id + count))
            next_id += count
            active.update(rids)
            inputs = [_rand((prompt_len,), vocab, gen).to(device) for _ in rids]
            events.append({"op": "prefill", "request_ids": rids, "input_ids": inputs})
        elif op == "decode":
            rids = sorted(active)[:count]
            tok = _rand((len(rids),), vocab, gen).to(device)
            events.append({"op": "decode", "request_ids": rids, "token_ids": tok})
        else:  # remove
            rids = sorted(active)[:count]
            for r in rids:
                active.remove(r)
            events.append({"op": "remove", "request_ids": rids})
    if active:
        events.append({"op": "remove", "request_ids": sorted(active)})
    return events


def _randint(lo: int, hi: int, gen: torch.Generator) -> int:
    """确定性整数 ∈ [lo, hi]（含端点），只依赖传入 Generator。"""
    if hi <= lo:
        return lo
    return lo + int(torch.randint(0, hi - lo + 1, (1,), generator=gen).item())


def _ragged_decode_events(
    batch: int,
    plen_lo: int,
    plen_hi: int,
    steps_lo: int,
    steps_hi: int,
    vocab: int,
    gen: torch.Generator,
    device: str,
) -> list[dict[str, Any]]:
    """不规则 decode：每请求 prompt 长度 ∈[plen_lo,plen_hi]、停止步 ∈[steps_lo,steps_hi]。

    请求**不同时长完成**（到点即 remove），长度**永不对齐** → 等长分组退化、逼出 varlen 批处理。
    """
    rids = list(range(batch))
    plens = [_randint(plen_lo, plen_hi, gen) for _ in rids]
    stops = [_randint(steps_lo, steps_hi, gen) for _ in rids]
    inputs = [_rand((plens[i],), vocab, gen).to(device) for i in rids]
    events: list[dict[str, Any]] = [{"op": "prefill", "request_ids": rids, "input_ids": inputs}]

    active = list(rids)
    step = 0
    max_steps = max(stops) if stops else 0
    while active and step < max_steps:
        step += 1
        tok = _rand((len(active),), vocab, gen).to(device)
        events.append({"op": "decode", "request_ids": list(active), "token_ids": tok})
        finished = [r for r in active if stops[r] <= step]
        if finished:
            events.append({"op": "remove", "request_ids": finished})
            active = [r for r in active if r not in finished]
    if active:
        events.append({"op": "remove", "request_ids": list(active)})
    return events


def _ragged_mixed_events(vocab: int, gen: torch.Generator, device: str) -> list[dict[str, Any]]:
    """不规则 mixed：到达数 / prompt 长度 / decode 突发 / remove 点都从 gen 抽。

    替代写死 schedule，跨 seed 产生不同到达/移除顺序 + ragged 活跃集（长度参差）。
    """
    events: list[dict[str, Any]] = []
    active: list[int] = []
    next_id = 0
    n_waves = _randint(3, 5, gen)
    for _ in range(n_waves):
        # 一波新请求到达（长度各异）。
        count = _randint(2, 6, gen)
        rids = list(range(next_id, next_id + count))
        next_id += count
        inputs = [_rand((_randint(16, 128, gen),), vocab, gen).to(device) for _ in rids]
        events.append({"op": "prefill", "request_ids": rids, "input_ids": inputs})
        active.extend(rids)
        # 若干步 decode（每步对当前活跃集，长度因到达先后而参差）。
        for _ in range(_randint(1, 4, gen)):
            if not active:
                break
            tok = _rand((len(active),), vocab, gen).to(device)
            events.append({"op": "decode", "request_ids": list(active), "token_ids": tok})
        # 随机移除一部分老请求。
        if active and _randint(0, 1, gen):
            drop = _randint(1, len(active), gen)
            removed = active[:drop]
            events.append({"op": "remove", "request_ids": removed})
            active = active[drop:]
    if active:
        events.append({"op": "remove", "request_ids": list(active)})
    return events


def _agg(values: list[float]) -> float:
    """跨 seed 聚合成一个数：下半均值（≈p25–50，最差倾向）。

    惩罚"某些 order 下崩溃"，又不被单个倒霉 seed 独霸；n<=2 退回最差值；无正值 → 0。
    """
    xs = sorted(v for v in values if v > 0.0)
    if not xs:
        return 0.0
    if len(xs) <= 2:
        return xs[0]
    half = xs[: max(1, len(xs) // 2)]
    return sum(half) / len(half)


def _timed_run(engine: Any, events: list[dict[str, Any]], device: str) -> dict[str, float]:
    """跑一遍事件流计时 + 计 token + 峰值显存。"""
    prefill_tokens = 0
    decode_tokens = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for ev in events:
            if ev["op"] == "prefill":
                engine.prefill(ev["request_ids"], ev["input_ids"])
                prefill_tokens += sum(int(x.numel()) for x in ev["input_ids"])
            elif ev["op"] == "decode":
                engine.decode(ev["request_ids"], ev["token_ids"])
                decode_tokens += int(ev["token_ids"].numel())
            elif ev["op"] == "remove":
                engine.remove(ev["request_ids"])
    _sync(device)
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.startswith("cuda") else 0.0
    total = prefill_tokens + decode_tokens
    return {
        "elapsed_s": elapsed_s,
        "prefill_tokens": float(prefill_tokens),
        "decode_tokens": float(decode_tokens),
        "tps": total / elapsed_s,
        "decode_tps": decode_tokens / elapsed_s if decode_tokens else 0.0,
        "peak_memory_mb": peak,
    }


def _measure_case(
    mod: Any, spec: JobSpec, events: list[dict[str, Any]], warmup: int, repeat: int
) -> dict[str, float]:
    """warmup + repeat 次、取中位 elapsed 的那次（每次新建引擎，计时不含 create_engine）。"""
    device = spec.device
    for _ in range(warmup):
        engine = mod.create_engine(spec.model_config, spec.weight_dir, device)
        _timed_run(engine, events, device)
    runs = []
    for _ in range(max(repeat, 1)):
        engine = mod.create_engine(spec.model_config, spec.weight_dir, device)
        runs.append(_timed_run(engine, events, device))
    runs.sort(key=lambda r: r["elapsed_s"])
    return runs[len(runs) // 2]


def _score(
    prefill_tps: float,
    decode_tps: float,
    mixed_decode_tps: float,
    decode_overall_tps: float = 0.0,
    mixed_tps: float = 0.0,
) -> float:
    """worker 侧临时自评（仅供 loop 外直接 evaluate）：各计量列原始 tps 的等权几何平均，参照 ref=1。

    与权威口径同形：把各 case 的整体/decode 两列平铺等权（0 值被 EPS 下限吞，自评粗排够用）。
    loop 内的权威 score 由父进程按 baseline per-列 tps 归一化后覆盖（见 trainer._normalize_score）。
    """
    return geomean_score(
        prefill_tps, decode_overall_tps, decode_tps, mixed_tps, mixed_decode_tps
    )


def run_bench(spec: JobSpec) -> BenchResult:
    """跑三类 case，聚合 headline + score/loss。永不抛——异常落 warnings、返回零分结果。"""
    start = time.perf_counter()
    mode: EvalMode = spec.mode
    warnings: list[str] = []

    try:
        from .gate import load_engine_module

        mod = load_engine_module(spec.engine_path)
    except Exception as e:  # bench 只在 gate 过后调，import 仍失败属异常，降级零分。
        return BenchResult(
            mode=mode, warnings=[f"load failed: {e}"], duration_s=time.perf_counter() - start
        )

    vocab = int(spec.model_config["vocab_size"])
    quick = mode == "quick"

    raw: dict[str, Any] = {}

    def _run(
        name: str, events: list[dict[str, Any]], warmup: int, repeat: int
    ) -> dict[str, float] | None:
        try:
            m = _measure_case(mod, spec, events, warmup, repeat)
            raw[name] = m
            return m
        except Exception as e:
            warnings.append(f"{name} case failed: {e}")
            return None

    # v3：等权几何平均，平铺 grader 两列（各 case 整体 tps + decode tps），不再内置 0.6/0.25/0.15。
    extra: dict[str, Any] = {"score_kind": "geomean_v3"}

    if quick:
        # quick：单形状单 seed，便宜（agent 自检用），不测不规则。
        gen = torch.Generator().manual_seed(spec.seed)
        pf = _run("prefill", _prefill_events(2, 32, vocab, gen, spec.device), 0, 1)
        dc = _run("decode", _decode_events(2, 16, 4, vocab, gen, spec.device), 0, 1)
        prefill_tps = pf["tps"] if pf else 0.0
        decode_tps = dc["decode_tps"] if dc else 0.0
        decode_overall_tps = dc["tps"] if dc else 0.0
        mixed_tps = 0.0
        mixed_decode_tps = 0.0
        peak = max((m["peak_memory_mb"] for m in (pf, dc) if m), default=0.0)
    else:
        # full：prefill 单 seed（并行、对不规则不敏感、最便宜）；decode/mixed 多 seed ragged。
        # repeat 3→2（多 seed 已提供方差平均）；逐 (case,seed) 前查软墙，超时跳过剩余。
        def _wall_hit() -> bool:
            if time.perf_counter() - start > _BENCH_WALL_S:
                warnings.append("bench wall hit, partial seeds")
                return True
            return False

        pf_events = _prefill_events(4, 128, vocab, _gen(_FULL_SEEDS[0]), spec.device)
        pf = _run("prefill", pf_events, 1, 2)

        decode_list: list[float] = []
        decode_overall_list: list[float] = []
        mixed_decode_list: list[float] = []
        mixed_tps_list: list[float] = []
        peaks: list[float] = [pf["peak_memory_mb"]] if pf else []

        for s in _FULL_SEEDS:
            if _wall_hit():
                break
            dc_s = _run(
                f"decode.s{s}",
                _ragged_decode_events(8, 16, 96, 6, 24, vocab, _gen(s), spec.device),
                1,
                2,
            )
            if dc_s:
                decode_list.append(dc_s["decode_tps"])
                decode_overall_list.append(dc_s["tps"])
                peaks.append(dc_s["peak_memory_mb"])
            if _wall_hit():
                break
            mx_s = _run(
                f"mixed.s{s}", _ragged_mixed_events(vocab, _gen(s), spec.device), 1, 2
            )
            if mx_s:
                mixed_decode_list.append(mx_s["decode_tps"])
                mixed_tps_list.append(mx_s["tps"])
                peaks.append(mx_s["peak_memory_mb"])

        prefill_tps = pf["tps"] if pf else 0.0
        decode_tps = _agg(decode_list)
        decode_overall_tps = _agg(decode_overall_list)
        mixed_decode_tps = _agg(mixed_decode_list)
        mixed_tps = _agg(mixed_tps_list)
        peak = max(peaks, default=0.0)
        extra["per_seed"] = {
            "decode_tps": decode_list,
            "mixed_decode_tps": mixed_decode_list,
        }

    score = _score(prefill_tps, decode_tps, mixed_decode_tps, decode_overall_tps, mixed_tps)
    return BenchResult(
        mode=mode,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        decode_overall_tps=decode_overall_tps,
        mixed_tps=mixed_tps,
        mixed_decode_tps=mixed_decode_tps,
        peak_memory_mb=peak,
        score=score,
        loss=-score,
        raw=raw,
        duration_s=time.perf_counter() - start,
        warnings=warnings,
        extra=extra,
    )
