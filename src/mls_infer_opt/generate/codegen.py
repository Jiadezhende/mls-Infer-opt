"""codegen — 产一份 engine 候选：自检重试自闭环（方案1，generate 驱动）。

三个入口本质同一件事，差别只在 prompt 与是否用 LLM：

- ``bootstrap``：不依赖 LLM，直接落 pristine baseline 源码——永久兜底、零风险、必产 Candidate。
- ``propose``：按 Policy（带 analyze 的 rationale）产新候选（LLM）。
- ``repair``：按结构化报错定向修复候选（LLM）。

propose/repair 不是「调一次」：generate 调 LLM 出 ``engine.py`` → 自己跑
``check_self_contained``（静态）+ ``evaluate.quick_gate``（子进程隔离、对 oracle 比 logits）→
不过就把结构化错误回灌、转 repair 提示让模型修自己的码，循环到过门或耗自检预算。控制权全程在
确定性代码里（不靠模型自觉调工具）。**自检 ephemeral、不进 state**；挂到 candidate.gate 的只有
外层 loop 跑的 full gate（权威，不变量 #5）。无权重无法自检时优雅降级为「静态过即出候选」。

不变量：本模块只产候选、无发布权；**LLM 不可用 / 失败 / 产垃圾 / 自检始终不过 → 返回 None**
（loop 当作「这轮没收益」走兜底），绝不抛异常给上层。成功则把 engine.py + policy.json 落到
候选工作目录并返回 [Candidate]，code 不进 struct（见 state.candidate）。
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Protocol

from ..evaluate import quick_gate
from ..state.candidate import (
    Candidate,
    candidate_engine_path,
    candidate_policy_path,
    make_candidate_id,
)
from ..state.context import TaskContext
from ..state.eval import GateResult, GateStage, ValidationError
from .policy import Policy, default_policy, strategy_tags, to_json
from .prompt import PromptMode, build_prompt

__all__ = [
    "LLMClient",
    "baseline_engine_source",
    "bootstrap",
    "propose",
    "repair",
    "check_self_contained",
]

_BASELINE_PATH = Path(__file__).parent / "assets" / "baseline_engine.py"

# 自检重试预算：generate 内部循环上限（与外层 loop 的 max_repair_retries 不同层）。
_MAX_SELF_CHECK_ROUNDS = 4

# 自包含 import 白名单：根模块必须在内；engine 应是纯 torch，余量给常见 stdlib。
_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "torch",
        "math",
        "os",
        "sys",
        "typing",
        "dataclasses",
        "collections",
        "itertools",
        "functools",
        "warnings",
        "json",
        "contextlib",
        "abc",
        "numpy",
    }
)
# 显式黑名单（即便误入白名单也拒）：本 agent 包绝不能被 engine import。
_FORBIDDEN_IMPORT_ROOTS = frozenset({"mls_infer_opt"})

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class LLMClient(Protocol):
    """generate 需要的最小 LLM 接口；真实实现来自 llm/（TBD），测试期可 mock。

    ``available`` 为假或调用返回 None/抛错时，本模块走兜底（返回 None）。新 LLM 基建可
    额外提供 ``run_agent(prompt, tools=...)``；本模块会优先使用它，但仍兼容旧的
    ``generate(prompt)`` mock。
    """

    available: bool

    def generate(self, prompt: str) -> str | None: ...


# === 公共入口 =========================================================
def baseline_engine_source() -> str:
    """pristine baseline 的源码字符串——原始兜底。

    bootstrap 把它落成首个候选；进程外壳（loop.__main__）在一切都失败时也用它直接补出
    ``workspace/engine.py``，保证「engine.py 必在盘上」这条对外契约即便 run_loop 自己崩溃也成立。
    """
    return _BASELINE_PATH.read_text(encoding="utf-8")


def bootstrap(ctx: TaskContext) -> Candidate:
    """不依赖 LLM 的保守初始候选——直接落 pristine baseline。永久兜底，必产。"""
    code = _BASELINE_PATH.read_text(encoding="utf-8")
    problems = check_self_contained(code)
    if problems:  # 自带基线不自包含属开发期 bug，应当场暴露（这是兜底基石）。
        raise RuntimeError(f"baseline asset failed self-containment: {problems}")
    return _persist(ctx, default_policy(), code)


def propose(
    ctx: TaskContext,
    policy: Policy,
    parent_code: str,
    *,
    llm: LLMClient | None,
) -> Candidate | None:
    """按 Policy（含 analyze 的 rationale）产新候选；任何失败返回 None。"""
    return _generate(ctx, policy, parent_code, mode="propose", llm=llm, errors=None)


def repair(
    ctx: TaskContext,
    policy: Policy,
    parent_code: str,
    errors: list[ValidationError],
    *,
    llm: LLMClient | None,
) -> Candidate | None:
    """按结构化报错定向修复候选；任何失败返回 None。"""
    return _generate(ctx, policy, parent_code, mode="repair", llm=llm, errors=errors)


# === 内部流程 =========================================================
def _generate(
    ctx: TaskContext,
    policy: Policy,
    parent_code: str,
    *,
    mode: str,
    llm: LLMClient | None,
    errors: list[ValidationError] | None,
    max_self_check_rounds: int = _MAX_SELF_CHECK_ROUNDS,
) -> Candidate | None:
    """自检重试自闭环：出码 → 静态/quick 自检 → 不过则带错回灌转 repair，循环到过门或耗预算。

    首轮按入参 mode/parent_code/errors；任一轮自检不过后，把模型自己产的码作 parent、转 repair
    提示、回灌结构化错误，让它修自己的代码。仅当过 quick（或无权重无法自检）才出候选。

    **永不抛**（不变量 #3/#5）：LLM 调用 / build_prompt / 暂存写盘 / quick_gate / 落盘 任一异常都
    收敛为「这轮无收益」（None），绝不漏给 loop。单次 LLM 失败只跳过本轮、在预算内重试。
    """
    if llm is None or not getattr(llm, "available", False):
        return None

    cur_mode: PromptMode = "repair" if mode == "repair" else "propose"
    cur_parent = parent_code
    cur_errors = errors

    try:
        for _ in range(max(1, max_self_check_rounds)):
            prompt = build_prompt(
                policy, ctx, mode=cur_mode, parent_code=cur_parent, errors=cur_errors
            )
            try:
                text = _call_llm(llm, prompt)
            except Exception:
                continue  # 瞬时失败（含超时）：跳过本轮、在预算内重试（max_retries=0 兜底）
            code = _extract_code(text) if text else None
            if code is None:
                continue  # 没产出代码块：再给一次机会（仍受预算上限约束）

            static = check_self_contained(code)
            if static:  # 静态不过：省掉子进程，直接回灌错误让模型修自己的码。
                cur_mode, cur_parent = "repair", code
                cur_errors = [_static_to_error(p) for p in static]
                continue

            gate = _quick_self_check(ctx, code)
            if gate is None or gate.passed:  # 过 quick（或无权重无法自检）→ 出候选。
                return _persist(ctx, policy, code)

            cur_mode, cur_parent, cur_errors = "repair", code, gate.errors  # correctness 错回灌
    except Exception:
        # build_prompt / 暂存 / 落盘 等任何意外 → 无收益，不漏给 loop（never-throw，不变量 #3/#5）。
        return None

    return None  # 耗尽自检预算仍未过 quick → 这轮无收益（不变量 #2 兜底）


def _quick_self_check(ctx: TaskContext, code: str) -> GateResult | None:
    """把候选码写暂存区，跑 quick 正确性门（子进程隔离、never-throw）。

    无权重（``weight_dir/model.pt`` 不存在）时无法比 logits，返回 None → 调用方按「静态过即出
    候选」降级，交外层 full gate 把关。
    """
    if not os.path.exists(os.path.join(ctx.weight_dir, "model.pt")):
        return None
    try:
        staging = os.path.join(ctx.run_dir, "_staging", "engine.py")
        os.makedirs(os.path.dirname(staging), exist_ok=True)
        Path(staging).write_text(code, encoding="utf-8")
    except OSError:
        return None  # 写暂存失败：无法自检，降级交外层 full gate 把关
    return quick_gate(staging, ctx)


def _static_to_error(problem: str) -> ValidationError:
    """把静态问题字符串包成 ValidationError（喂 build_prompt 的 repair 块）。"""
    stage: GateStage = "syntax" if "syntax" in problem else "api"
    return ValidationError(stage=stage, message=problem)


def _call_llm(llm: LLMClient, prompt: str) -> str | None:
    """Call either the new agent API or the legacy generate(prompt) API."""

    runner = getattr(llm, "run_agent", None)
    if callable(runner):
        result = runner(prompt)
        if not getattr(result, "ok", False):
            return None
        text = getattr(result, "text", None)
        return text if isinstance(text, str) else None
    return llm.generate(prompt)


def _extract_code(text: str) -> str | None:
    """从 LLM 回复抽取代码：优先取首个 ```python``` 代码块，无围栏则整段裸文本兜底。"""
    m = _FENCE.search(text)
    code = (m.group(1) if m else text).strip()
    return code or None


def _persist(ctx: TaskContext, policy: Policy, code: str) -> Candidate:
    """落盘候选工作目录（engine.py + policy.json），算 id，返回 Candidate 元数据。"""
    run_dir = ctx.run_dir
    cid = make_candidate_id(policy.round, code)
    engine_path = candidate_engine_path(run_dir, cid)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    Path(engine_path).write_text(code, encoding="utf-8")
    Path(candidate_policy_path(run_dir, cid)).write_text(to_json(policy), encoding="utf-8")
    return Candidate(
        id=cid,
        kind=policy.kind,
        round=policy.round,
        parent_id=policy.parent_id,
        strategy_tags=strategy_tags(policy),
    )


def check_self_contained(code: str) -> list[str]:
    """确定性自包含早筛（跑在 evaluate 之前）。返回问题列表，空即通过。

    查：语法、import 白名单（显式禁 mls_infer_opt / 相对 import）、API 契约
    （顶层 create_engine + class Engine 含 prefill/decode/remove）。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    problems: list[str] = []

    # --- imports ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    problems.append(f"forbidden import: {alias.name}")
                elif root not in _ALLOWED_IMPORT_ROOTS:
                    problems.append(f"non-allowlisted import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                problems.append("relative import not allowed in self-contained engine")
                continue
            root = (node.module or "").split(".", 1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                problems.append(f"forbidden import: {node.module}")
            elif root and root not in _ALLOWED_IMPORT_ROOTS:
                problems.append(f"non-allowlisted import: {node.module}")

    # --- API 契约 ---
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    ce = funcs.get("create_engine")
    if ce is None:
        problems.append("missing top-level create_engine()")
    elif len(ce.args.args) < 2:
        problems.append("create_engine must accept (model_config, weight_dir, device)")

    engine = classes.get("Engine")
    if engine is None:
        problems.append("missing class Engine")
    else:
        methods = {n.name: n for n in engine.body if isinstance(n, ast.FunctionDef)}
        for name, min_args in (("prefill", 3), ("decode", 3), ("remove", 2)):
            m = methods.get(name)
            if m is None:
                problems.append(f"Engine missing method {name}()")
            elif len(m.args.args) < min_args:
                problems.append(f"Engine.{name} has wrong arity")

    return problems
