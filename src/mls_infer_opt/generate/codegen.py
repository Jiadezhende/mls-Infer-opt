"""codegen — 产一份 engine 候选：自检重试自闭环（方案1，generate 驱动）。

三个入口本质同一件事，差别只在 prompt 与是否用 LLM：

- ``bootstrap``：不依赖 LLM，直接落 pristine baseline 源码——永久兜底、零风险、必产 Candidate。
- ``propose``：按 Policy（带 analyze 的 rationale）产新候选（LLM）。
- ``repair``：按结构化报错定向修复候选（LLM）。

propose/repair 不是「调一次」：把 ``check_engine`` 工具（包 ``check_self_contained`` 静态 +
``evaluate.quick_gate`` 子进程隔离对 oracle 比 logits）交给 agent，让它**边写边自检**——写完整
``engine.py`` → 调 ``check_engine(code=…)`` → 按返回的结构化 ``errors`` 修自己的码 → 再调，直到
``passed=true`` 或耗 tool-loop 预算（``run_agent`` 的 ``max_tool_rounds``）。自检循环在 agent 内部，
不再靠确定性外层 Python 驱动。client 必须提供 ``run_agent``；模型若始终没成功调用工具，则回退抽取
最终文本并做一次确定性静态/quick 复核。**自检 ephemeral、不进 state**；只持久化「最近一次过
check_engine 的码」，挂到 candidate.gate 的只有外层 loop 跑的 full gate（权威，不变量 #5）。
无权重无法自检时优雅降级为「静态过即出候选」。

不变量：本模块只产候选、无发布权；**LLM 不可用 / 失败 / 产垃圾 / 自检始终不过 → 返回 None**
（loop 当作「这轮没收益」走兜底），绝不抛异常给上层。成功则把 engine.py + policy.json 落到
候选工作目录并返回 [Candidate]，code 不进 struct（见 state.candidate）。
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..evaluate import quick_gate

# 直 import tooling/errors（仅依赖 stdlib）而非 ..llm：避免经 llm/__init__ 拉入
# openai_client→present 的导入环。
from ..llm.errors import LLMError
from ..llm.tooling import ToolResult, ToolSpec
from ..searchspace.policy import Policy, default_policy, strategy_tags, to_json
from ..state.candidate import (
    Candidate,
    candidate_engine_path,
    candidate_policy_path,
    make_candidate_id,
)
from ..state.context import TaskContext
from ..state.eval import GateResult, ValidationError
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

# 自检重试预算：agent tool-loop 上限（与外层 loop 的 max_repair_retries 不同层）。
_MAX_SELF_CHECK_ROUNDS = 4

# check_engine 工具的执行超时：必须 ≥ quick_gate 子进程预算 + 余量，否则 ToolExecutor 的
# ThreadPoolExecutor 会按 LLMConfig.timeout_s（默认 120s）提前 cancel、孤儿化 quick_gate 子进程。
_CHECK_ENGINE_TOOL_TIMEOUT_S = 600.0

# 交给 agent 的自检指令：让它边写边用 check_engine 自检，而非一次性出码。
_AGENT_SELF_CHECK_INSTRUCTIONS = (
    "你在生成一份自包含的 engine.py。流程：\n"
    "1. 写出完整的 engine.py 源码。\n"
    "2. 调用 check_engine(code=<完整源码>) 自检——它跑静态规则 + quick 正确性门，"
    "返回 {passed, errors}。\n"
    "3. 若 passed=false，按 errors 修正你自己的代码，再次调用 check_engine；"
    "重复直到 passed=true。\n"
    "4. 你最近一次让 check_engine 返回 passed=true 的入参代码，就是最终采用的 engine.py——"
    "务必让那次调用包含完整、可直接运行的单文件源码。\n"
    "不要 import mls_infer_opt；遵守 prompt 里的 API 契约与自包含硬规则。"
)

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
    """generate 需要的最小 LLM 接口；真实实现来自 llm/，测试期可 mock。

    ``available`` 为假或 ``run_agent`` 返回 ok=False/抛错时，本模块走兜底（返回 None）。
    """

    available: bool

    def run_agent(
        self, prompt: str, tools: list[Any] | None = ..., **kwargs: Any
    ) -> Any: ...


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
    """产一份候选：把 check_engine 工具交给模型，让它边写边自检自闭环。

    agent 路径：把 check_engine 工具交给模型，让它写完整 engine.py → 自检 → 按 errors 修自己的码 →
    重调，直到过门（run_agent 内部 tool-loop，上限 max_self_check_rounds）。只持久化「最近一次过
    check_engine 的码」；模型从没成功调用工具时回退抽取最终文本并确定性复核。

    **永不抛**（不变量 #3/#5）：build_prompt / run_agent / 暂存写盘 / quick_gate / 落盘 任一异常都
    收敛为「这轮无收益」（None），绝不漏给 loop。仅当过 quick（或无权重无法自检）才出候选。
    """
    if llm is None or not getattr(llm, "available", False):
        return None
    if not callable(getattr(llm, "run_agent", None)):
        return None  # 唯一支持的路径是 agent 工具自检自闭环；无 run_agent 视作不可用。

    prompt_mode: PromptMode = "repair" if mode == "repair" else "propose"
    try:
        return _generate_agent_loop(
            ctx,
            policy,
            parent_code,
            mode=prompt_mode,
            llm=llm,
            errors=errors,
            max_self_check_rounds=max_self_check_rounds,
        )
    except LLMError:
        # C2 基建失败（run_agent 传输层抛错）：穿透交总控，绝不静默吞成「这轮没收益」。
        raise
    except Exception:
        # 其它意外（build_prompt / 暂存 / 落盘 等）→ 无收益，不漏给 loop（never-throw）。
        return None


def _generate_agent_loop(
    ctx: TaskContext,
    policy: Policy,
    parent_code: str,
    *,
    mode: PromptMode,
    llm: LLMClient,
    errors: list[ValidationError] | None,
    max_self_check_rounds: int,
) -> Candidate | None:
    """agent 持有 check_engine 工具、内部边写边自检；返回过门候选或 None。"""
    prompt = build_prompt(policy, ctx, mode=mode, parent_code=parent_code, errors=errors)
    captured: dict[str, str] = {}
    tool = _build_check_engine_tool(ctx, captured)
    result = llm.run_agent(  # type: ignore[attr-defined]
        prompt,
        tools=[tool],
        instructions=_AGENT_SELF_CHECK_INSTRUCTIONS,
        max_tool_rounds=max(1, max_self_check_rounds),
    )

    code = captured.get("code")
    if code is not None:  # agent 已让 check_engine 返回 passed → 该码即已验证
        return _persist(ctx, policy, code)

    # 回退：模型没成功调用工具（或末次没过）→ 抽最终文本，确定性复核后才出候选。
    # 绝不持久化未经 gate 的码。
    text = getattr(result, "text", None) if getattr(result, "ok", False) else None
    code = _extract_code(text) if text else None
    if code is None or check_self_contained(code):
        return None
    gate = _quick_self_check(ctx, code)
    if gate is None or gate.passed:
        return _persist(ctx, policy, code)
    return None


def _build_check_engine_tool(ctx: TaskContext, captured: dict[str, str]) -> ToolSpec:
    """造 agent 的内层自检工具：收完整 engine.py 源码 → 静态 + quick 门 → {passed, errors}。

    过门（或无权重无法自检的降级）时把该码写进 ``captured``，供调用方持久化；不过则只回错误、不写。
    自检 ephemeral：只写 run_dir/_staging、不进 LoopState。handler 防御式产 ToolResult。
    """

    def _handler(args: Mapping[str, Any]) -> ToolResult:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult.failure("invalid_arguments", "code 必须是非空源码字符串")

        static = check_self_contained(code)
        if static:
            return ToolResult.success({"passed": False, "errors": static})

        gate = _quick_self_check(ctx, code)
        if gate is None:  # 无权重 / 暂存写失败：无法 quick 自检，降级交外层 full gate 把关
            captured["code"] = code
            return ToolResult.success(
                {"passed": True, "errors": [], "note": "static-only (no weights)"}
            )
        if gate.passed:
            captured["code"] = code
            return ToolResult.success({"passed": True, "errors": []})
        return ToolResult.success(
            {"passed": False, "errors": [_error_to_payload(e) for e in gate.errors]}
        )

    return ToolSpec(
        name="check_engine",
        description=(
            "校验候选 engine.py：跑自包含静态规则 + quick 正确性门（对官方 reference 比 logits）。"
            "返回 {passed: bool, errors: [...]}；passed=false 时按 errors 修代码后重试。"
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=_handler,
        timeout_s=_CHECK_ENGINE_TOOL_TIMEOUT_S,
    )


def _error_to_payload(e: ValidationError) -> dict[str, Any]:
    """把 ValidationError 摊成 JSON 友好 dict 喂回模型修码（对齐 prompt._format_error 的字段）。"""
    payload: dict[str, Any] = {"stage": e.stage, "message": e.message}
    if e.case:
        payload["case"] = e.case
    if e.expected_shape is not None:
        payload["expected_shape"] = e.expected_shape
    if e.actual_shape is not None:
        payload["actual_shape"] = e.actual_shape
    if e.max_abs_err is not None:
        payload["max_abs_err"] = e.max_abs_err
    if e.max_rel_err is not None:
        payload["max_rel_err"] = e.max_rel_err
    if e.traceback_tail:
        payload["traceback_tail"] = e.traceback_tail
    return payload


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


def _extract_code(text: str) -> str | None:
    """从 LLM 回复抽取代码：优先取首个 ```python``` 代码块，无围栏则整段裸文本兜底。"""
    m = _FENCE.search(text)
    code = (m.group(1) if m else text).strip()
    return code or None


def _next_seq(run_dir: str) -> int:
    """本次 run 内下一个候选序号 = ``candidates/`` 下已有候选目录数。

    loop 单写者顺序执行（一次只生成一个候选），无竞态；候选目录只增不删，故序号单调。
    用文件系统而非 LoopState 计数，让 generate 保持与 LoopState 解耦（手上只有 ctx）。
    """
    base = Path(run_dir) / "candidates"
    if not base.exists():
        return 0
    return sum(1 for p in base.iterdir() if p.is_dir())


def _persist(ctx: TaskContext, policy: Policy, code: str) -> Candidate:
    """落盘候选工作目录（engine.py + policy.json），分配序号 id，返回 Candidate 元数据。"""
    run_dir = ctx.run_dir
    cid = make_candidate_id(_next_seq(run_dir))
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
