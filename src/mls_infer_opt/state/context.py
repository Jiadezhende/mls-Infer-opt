"""context — 一次调优会话的只读上下文（loop INIT 产出，全程只读）。

全是「任务开始时给定 / INIT 探测到的东西」：模型配置、设备、路径三根、判停上限、环境快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Paths", "Environment", "Limits", "TaskContext"]


def _major_version(version: str) -> int:
    """从 ``"2.4.1"`` 取主版本号；解析不出来返回 0（用于 has_sdpa 这类版本派生）。"""
    head = version.split(".", 1)[0].strip()
    return int(head) if head.isdigit() else 0


@dataclass
class Paths:
    """三个语义不同的根：target 只读、run_dir 可乱写、output_dir 受控发布。

    其余路径（weight_dir / model_config / engine.py 发布点）不存字段，由 TaskContext 的
    property 按约定算出来——这样 engine.py 不可能被写到约定外的地方，key 也不会拼错。
    """

    target_dir: str = ""   # 只读输入根：model_config.json + weights/，agent 绝不写
    runs_dir: str = ""      # runs/ 根；本次工作目录 = runs_dir/run_id
    output_dir: str = ""    # 发布根：engine.py / output3.* / report3.*


@dataclass
class Environment:
    """环境快照：客观采集，主要喂 LLM prompt + 落 report / 审计。

    只存「客观事实」（版本 / GPU / 显存）。「试了才知道」的能力（如 torch.compile 是否真能
    跑通）不进这里——那是 generate 的可尝试策略、由 evaluate 验证，存个可能撒谎的 bool 没意义。
    代码需要 branch 的硬开关用 @property 从客观事实派生，不单独存字段。
    """

    torch_version: str = ""
    cuda_version: str | None = None   # None 即无 cuda
    gpu_name: str | None = None
    gpu_count: int = 0
    total_memory_mb: float | None = None
    python_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        return self.cuda_version is not None

    @property
    def has_sdpa(self) -> bool:
        """torch≥2.0 起有 scaled_dot_product_attention。"""
        return _major_version(self.torch_version) >= 2


@dataclass
class Limits:
    """静态判停上限。每个都对照一个实时量（LoopState / BudgetUsage），由 loop/analyze 读。

    成本类上限（max_llm_calls / max_eval_runs / max_tokens）先不做一等字段——先跑出结果，
    需要按成本判停时再从 extra 提成正式字段。
    """

    time_budget_s: int = 0       # 硬墙；对照 BudgetUsage.elapsed_s
    max_rounds: int = 0          # 防空转；对照 LoopState.round
    max_stale_rounds: int = 0    # 早停 patience；对照 LoopState.stale_rounds（analyze 读）
    max_repair_retries: int = 0  # 单候选 repair 内层重试上限（generate 读）
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    """一次调优会话的只读上下文——全是任务开始时给定 / INIT 探测到的东西。

    model_config 含模型结构字段（num_hidden_layers / hidden_size / heads / head_dim /
    vocab_size / rope_theta …），**只读**、由 engine 动态读取，绝不进入搜索空间
    （搜索空间见 searchspace.space；可调 knob 只属 Policy.knobs）。
    """

    model_config: dict[str, Any] = field(default_factory=dict)
    device: str = ""                                    # create_engine 透传
    run_id: str = ""                                    # runs/{run_id}；report 标识本次运行
    paths: Paths = field(default_factory=Paths)
    limits: Limits = field(default_factory=Limits)
    environment: Environment = field(default_factory=Environment)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def weight_dir(self) -> str:
        """create_engine 透传用；= target_dir/weights。"""
        return f"{self.paths.target_dir}/weights"

    @property
    def model_config_path(self) -> str:
        return f"{self.paths.target_dir}/model_config.json"

    @property
    def run_dir(self) -> str:
        """本次运行的暂存 / 草稿目录 = runs_dir/run_id。"""
        return f"{self.paths.runs_dir}/{self.run_id}"

    @property
    def engine_publish_path(self) -> str:
        """唯一发布出口，固定约定；只由 loop finalize 写。"""
        return f"{self.paths.output_dir}/engine.py"
