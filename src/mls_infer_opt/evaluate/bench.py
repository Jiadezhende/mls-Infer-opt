"""bench — 性能评测：prefill/decode/mixed 三类吞吐 + 峰值显存（worker 侧，产 BenchResult）。

口径照搬外部 ``benchmark_throughput.py``：计时只覆盖 prefill/decode/remove（不含 create_engine /
权重加载）；每个 case 先 warmup 再 repeat 次、取**中位数**那次的测量；峰值显存只有 CUDA 有
（CPU 下为 0）。**只有 gate.passed 的候选才会调本模块**（守护在 Candidate.attach_bench）。

score：给 loop keep-best 的归一化标量，决定性、文档化（decode 吞吐为主、mixed/prefill 次之）；
loss：给 analyze 的诊断量（= -score，越小越好）。公式只保证单调可比，后续可调。
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..state.eval import BenchResult, EvalMode
from .protocol import JobSpec

__all__ = ["run_bench"]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _rand(shape: tuple[int, ...], vocab: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, vocab, shape, generator=gen, dtype=torch.long)


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


def _score(prefill_tps: float, decode_tps: float, mixed_decode_tps: float) -> float:
    """归一化标量（越大越好）。decode 吞吐主导，mixed/prefill 次之。单调可比即可，公式可调。"""
    return 0.6 * decode_tps + 0.25 * mixed_decode_tps + 0.15 * prefill_tps


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
    gen = torch.Generator().manual_seed(spec.seed)
    quick = mode == "quick"
    warmup = 0 if quick else 1
    repeat = 1 if quick else 3

    raw: dict[str, Any] = {}

    def _run(name: str, events: list[dict[str, Any]]) -> dict[str, float] | None:
        try:
            m = _measure_case(mod, spec, events, warmup, repeat)
            raw[name] = m
            return m
        except Exception as e:
            warnings.append(f"{name} case failed: {e}")
            return None

    if quick:
        pf = _run("prefill", _prefill_events(2, 32, vocab, gen, spec.device))
        dc = _run("decode", _decode_events(2, 16, 4, vocab, gen, spec.device))
        mx = None
    else:
        pf = _run("prefill", _prefill_events(4, 128, vocab, gen, spec.device))
        dc = _run("decode", _decode_events(8, 32, 16, vocab, gen, spec.device))
        mx = _run("mixed", _mixed_events(vocab, gen, spec.device))

    prefill_tps = pf["tps"] if pf else 0.0
    decode_tps = dc["decode_tps"] if dc else 0.0
    mixed_tps = mx["tps"] if mx else 0.0
    mixed_decode_tps = mx["decode_tps"] if mx else 0.0
    peak = max((m["peak_memory_mb"] for m in (pf, dc, mx) if m), default=0.0)

    score = _score(prefill_tps, decode_tps, mixed_decode_tps)
    return BenchResult(
        mode=mode,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        mixed_tps=mixed_tps,
        mixed_decode_tps=mixed_decode_tps,
        peak_memory_mb=peak,
        score=score,
        loss=-score,
        raw=raw,
        duration_s=time.perf_counter() - start,
        warnings=warnings,
    )
