"""oracle — 正确性比对的 ground truth：参考模型逐 checkpoint 的 expected logits（worker 侧）。

参考实现来自 vendored ``assets/reference_model.py``（独立、可信、纯 torch；**不能用候选起点
baseline 当参考，会循环自证**）。对固定 ``(config, weights, mode, seed)`` expected 恒定，故
算一次落盘缓存（``.pt``），后续候选 worker 直接 load——参考前向是评测里最贵的部分，跨候选复用。

oracle 与 gate 走同一份 ``cases.correctness_schedule``（同 seed → 同输入），保证 expected 与
候选被喂的输入逐元素对齐。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..state.eval import EvalMode
from .cases import correctness_schedule

__all__ = ["expected_logits"]

_REFERENCE_PATH = Path(__file__).parent / "assets" / "reference_model.py"
_CASESET_VERSION = "1"  # cases.correctness_schedule 变更时手动 bump，使旧缓存失效


def _load_reference_class() -> Any:
    """从 vendored asset 加载 ReferenceModel 类（importlib，避免它进 typed 包）。"""
    spec = importlib.util.spec_from_file_location("evaluate_reference_model", _REFERENCE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - asset 必存在
        raise RuntimeError(f"cannot load reference model from {_REFERENCE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ReferenceModel


def _cache_key(model_config: dict[str, Any], weight_dir: str, mode: EvalMode, seed: int) -> str:
    """缓存键：依赖 config 内容 + 权重文件 stat + mode + seed + caseset 版本。"""
    weight_path = os.path.join(weight_dir, "model.pt")
    try:
        st = os.stat(weight_path)
        weight_sig = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        weight_sig = "missing"
    payload = json.dumps(
        {
            "config": model_config,
            "weight": weight_sig,
            "mode": mode,
            "seed": seed,
            "caseset": _CASESET_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _compute(
    model_config: dict[str, Any], weight_dir: str, device: str, mode: EvalMode, seed: int
) -> dict[str, torch.Tensor]:
    """跑参考模型，沿确定性事件流记每个 checkpoint 的 expected 最后一位 logits。"""
    reference_cls = _load_reference_class()
    ref = reference_cls(model_config, weight_dir, device)

    gen = torch.Generator().manual_seed(seed)
    vocab_size = int(model_config["vocab_size"])
    ops = correctness_schedule(vocab_size, mode, gen)

    # 参考无 KV cache，每步喂整段累积序列重算（与 reference_model.forward 一致）。
    acc: dict[int, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for op in ops:
            if op.kind == "remove":
                for rid in op.request_ids:
                    acc.pop(rid, None)
                continue

            rows = []
            for rid, inp in zip(op.request_ids, op.inputs, strict=True):
                tokens = inp.reshape(-1)
                if op.kind == "prefill":
                    acc[rid] = tokens.clone()
                else:  # decode：单步 token 追加到累积序列
                    acc[rid] = torch.cat([acc[rid], tokens])
                logits = ref.forward(acc[rid].unsqueeze(0))  # [1, L, vocab]
                rows.append(logits[0, -1, :].detach().float().cpu())

            assert op.case is not None
            expected[op.case] = torch.stack(rows, dim=0)

    return expected


def expected_logits(
    model_config: dict[str, Any],
    weight_dir: str,
    device: str,
    mode: EvalMode,
    seed: int,
    cache_path: str | None,
) -> dict[str, torch.Tensor]:
    """取 expected logits：命中缓存即 load，否则算一次并落盘。缓存损坏/不匹配则重算。"""
    key = _cache_key(model_config, weight_dir, mode, seed)

    if cache_path and os.path.exists(cache_path):
        try:
            blob = torch.load(cache_path, map_location="cpu")
            if isinstance(blob, dict) and blob.get("key") == key:
                return {k: v for k, v in blob["expected"].items()}
        except Exception:  # 缓存损坏不致命，重算覆盖即可。
            pass

    expected = _compute(model_config, weight_dir, device, mode, seed)

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save({"key": key, "expected": expected}, cache_path)
        except OSError:  # 落盘失败只是丢了复用，不影响本次正确性。
            pass

    return expected
