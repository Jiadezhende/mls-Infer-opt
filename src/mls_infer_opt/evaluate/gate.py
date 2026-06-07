"""gate — 正确性硬门：syntax → api → correctness 三阶段（worker 侧，产 GateResult）。

静态早筛（语法 / import 白名单 / API arity）已由 ``generate.check_self_contained`` 在评测前做过；
这里做「真跑得起来 + 数值对不对」那一层：

- syntax：候选 ``engine.py`` 能否被 import（运行期 import/exec 错落这一档）。
- api   ：``create_engine`` 能实例化、``prefill`` 冒烟返回的 logits 形状/类型合规。
- correctness：沿 ``cases`` 确定性事件流喂候选，与 ``oracle`` 的 expected 比
  ``allclose(atol=1e-2, rtol=1e-2)``，fail-fast（与 stage B 评测器一致）。

任一档失败 → 结构化 ``ValidationError``（喂 generate.repair / analyze），``passed`` 立即为 False。
本模块不抛到 worker 之外的语义异常——所有失败都进 GateResult.errors。
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from typing import Any

import torch

from ..state.eval import GateResult, GateStage, ValidationError
from .cases import correctness_schedule
from .oracle import expected_logits
from .protocol import JobSpec

__all__ = ["run_gate", "load_engine_module"]

_ATOL = 1e-2
_RTOL = 1e-2


def load_engine_module(engine_path: str) -> Any:
    """importlib 从磁盘加载候选 engine 模块（独立模块名，避免与本包/其他候选冲突）。"""
    spec = importlib.util.spec_from_file_location("evaluate_candidate_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tb_tail(limit: int = 1600) -> str:
    """取 traceback 尾部（给 repair 看，不灌爆）。"""
    return traceback.format_exc()[-limit:]


def _stage_err(stage: GateStage, message: str) -> ValidationError:
    """带当前 traceback 尾部的结构化错（语法/接口/正确性崩溃共用）。"""
    return ValidationError(stage=stage, message=message, traceback_tail=_tb_tail())


def _to_float_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu()


def run_gate(spec: JobSpec) -> GateResult:
    """跑三阶段门控，返回 GateResult。永不抛——失败都落 errors。"""
    start = time.perf_counter()
    errors: list[ValidationError] = []
    case_summary: dict[str, Any] = {}

    config = spec.model_config
    device = spec.device

    # --- stage syntax：能否 import ---
    try:
        mod = load_engine_module(spec.engine_path)
    except Exception as e:
        errors.append(_stage_err("syntax", f"import failed: {e}"))
        return GateResult(
            syntax_ok=False, errors=errors, duration_s=time.perf_counter() - start
        )

    # --- stage api：实例化 + 冒烟 prefill 形状/类型 ---
    try:
        engine = mod.create_engine(config, spec.weight_dir, device)
        smoke_ids = torch.zeros(3, dtype=torch.long, device=device)
        smoke = engine.prefill([0], [smoke_ids])
        vocab = int(config["vocab_size"])
        if not torch.is_tensor(smoke):
            raise TypeError(f"prefill must return a tensor, got {type(smoke).__name__}")
        if tuple(smoke.shape) != (1, vocab):
            raise ValueError(
                f"prefill logits shape {tuple(smoke.shape)} != expected (1, {vocab})"
            )
        if not torch.isfinite(_to_float_cpu(smoke)).all():
            raise ValueError("prefill logits contain non-finite values")
    except Exception as e:
        errors.append(_stage_err("api", f"api contract failed: {e}"))
        return GateResult(
            syntax_ok=True, api_ok=False, errors=errors, duration_s=time.perf_counter() - start
        )

    # --- stage correctness：对照 oracle expected ---
    try:
        expected = expected_logits(
            config, spec.weight_dir, device, spec.mode, spec.seed, spec.oracle_cache_path
        )
        err = _run_correctness(mod, spec, expected, case_summary)
    except Exception as e:
        errors.append(_stage_err("correctness", f"correctness run crashed: {e}"))
        return GateResult(
            syntax_ok=True, api_ok=True, correctness_ok=False, errors=errors,
            case_summary=case_summary, duration_s=time.perf_counter() - start,
        )

    if err is not None:
        errors.append(err)
        return GateResult(
            syntax_ok=True, api_ok=True, correctness_ok=False, errors=errors,
            case_summary=case_summary, duration_s=time.perf_counter() - start,
        )

    return GateResult(
        syntax_ok=True, api_ok=True, correctness_ok=True, passed=True,
        case_summary=case_summary, duration_s=time.perf_counter() - start,
    )


def _run_correctness(
    mod: Any, spec: JobSpec, expected: dict[str, torch.Tensor], case_summary: dict[str, Any]
) -> ValidationError | None:
    """用全新引擎沿确定性事件流逐 case 比对；fail-fast 返回首个失败，否则 None。"""
    device = spec.device
    engine = mod.create_engine(spec.model_config, spec.weight_dir, device)

    gen = torch.Generator().manual_seed(spec.seed)
    vocab_size = int(spec.model_config["vocab_size"])
    ops = correctness_schedule(vocab_size, spec.mode, gen)

    with torch.no_grad():
        for op in ops:
            if op.kind == "remove":
                engine.remove(op.request_ids)
                continue

            if op.kind == "prefill":
                inputs = [t.to(device) for t in op.inputs]
                student = engine.prefill(op.request_ids, inputs)
            else:  # decode：每请求单步 token → [num_requests]
                token_ids = torch.cat([t.reshape(1) for t in op.inputs]).to(device)
                student = engine.decode(op.request_ids, token_ids)

            assert op.case is not None
            ref = expected[op.case]
            err = _compare(op.case, _to_float_cpu(student), ref)
            if err is not None:
                case_summary[op.case] = False
                return err
            case_summary[op.case] = True

    return None


def _compare(case: str, student: torch.Tensor, reference: torch.Tensor) -> ValidationError | None:
    """逐 case allclose；不过则带 max_abs/max_rel/shape 的结构化错。"""
    if student.shape != reference.shape:
        return ValidationError(
            stage="correctness",
            message=f"{case}: shape mismatch",
            case=case,
            expected_shape=list(reference.shape),
            actual_shape=list(student.shape),
        )
    if torch.allclose(student, reference, atol=_ATOL, rtol=_RTOL):
        return None

    diff = (student - reference).abs()
    max_abs = float(diff.max())
    ref_scale = reference.abs().clamp_min(1e-12)
    max_rel = float((diff / ref_scale).max())
    return ValidationError(
        stage="correctness",
        message=f"{case}: logits mismatch (max_abs={max_abs:.6g}, max_rel={max_rel:.6g})",
        case=case,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        expected_shape=list(reference.shape),
        actual_shape=list(student.shape),
    )
