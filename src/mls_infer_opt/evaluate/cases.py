"""cases — 正确性评测的确定性事件调度（worker 侧，oracle 与 gate 共享）。

oracle（参考）与 gate（候选）必须喂**完全相同**的输入序列，否则比对没意义。做法：两边都调
``correctness_schedule(...)`` 并传**同一个 seed 生成的 Generator**，得到逐字节一致的事件流；
oracle 沿事件流跑参考、记每个 checkpoint 的 expected logits，gate 沿同一事件流跑候选与之比对。

full 调度照搬外部 ``test_correctness.py`` 的 6 类 case（single/multi prefill+decode、remove、
insert_after_remove、decode_after_remove），再加少量泛化抽测（变 batch/长度）；quick 取便宜子集。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..state.eval import EvalMode

__all__ = ["Op", "correctness_schedule"]


@dataclass
class Op:
    """一个评测事件。``kind`` 为 remove 时 ``case`` 为 None（不产生可比对的 logits）。

    - prefill：``inputs`` 是每个请求的 prompt token 序列（list[Tensor[L]]）
    - decode ：``inputs`` 是单步 token（list[Tensor[1]]，每个请求一个）
    - remove ：``inputs`` 为空
    所有张量都在 CPU、dtype=long；喂引擎前由调用方按需 .to(device)。
    """

    kind: str  # "prefill" | "decode" | "remove"
    case: str | None
    request_ids: list[int]
    inputs: list[torch.Tensor]


def _rand_ids(length: int, vocab_size: int, gen: torch.Generator) -> torch.Tensor:
    """确定性随机 token 序列（CPU/long），只依赖传入的 Generator。"""
    return torch.randint(0, vocab_size, (length,), generator=gen, dtype=torch.long)


def correctness_schedule(
    vocab_size: int, mode: EvalMode, gen: torch.Generator
) -> list[Op]:
    """构造确定性事件流。同一 (vocab_size, mode, gen 种子) 恒返回逐元素一致的序列。"""
    ops: list[Op] = []

    # --- canonical：与 test_correctness.py 等价的核心序列 ---
    ids0 = _rand_ids(11, vocab_size, gen)
    ops.append(Op("prefill", "single_prefill", [0], [ids0]))

    tok0 = _rand_ids(1, vocab_size, gen)
    ops.append(Op("decode", "single_decode", [0], [tok0]))

    if mode == "quick":
        # quick：再补一个 multi_prefill 就收手（便宜、覆盖批量路径）。
        ids1 = _rand_ids(7, vocab_size, gen)
        ids2 = _rand_ids(13, vocab_size, gen)
        ops.append(Op("prefill", "multi_prefill", [1, 2], [ids1, ids2]))
        return ops

    # --- full：补齐 multi decode、remove、insert_after_remove、decode_after_remove ---
    ids1 = _rand_ids(7, vocab_size, gen)
    ids2 = _rand_ids(13, vocab_size, gen)
    ops.append(Op("prefill", "multi_prefill", [1, 2], [ids1, ids2]))

    toks = [_rand_ids(1, vocab_size, gen) for _ in range(3)]
    ops.append(Op("decode", "multi_decode", [0, 1, 2], toks))

    ops.append(Op("remove", None, [1], []))

    ids3 = _rand_ids(5, vocab_size, gen)
    ops.append(Op("prefill", "insert_after_remove", [3], [ids3]))

    toks2 = [_rand_ids(1, vocab_size, gen) for _ in range(3)]
    ops.append(Op("decode", "decode_after_remove", [0, 2, 3], toks2))

    ops.append(Op("remove", None, [0, 2, 3], []))

    # --- 泛化抽测：换 batch / 长度 / 顺序，验证不是只对固定 case 过拟合 ---
    gids = [4, 5, 6]
    glens = [9, 4, 17]
    ginputs = [_rand_ids(length, vocab_size, gen) for length in glens]
    ops.append(Op("prefill", "general_prefill", gids, ginputs))

    gtoks = [_rand_ids(1, vocab_size, gen) for _ in gids]
    ops.append(Op("decode", "general_decode", gids, gtoks))

    ops.append(Op("remove", None, gids, []))

    return ops
