# Agent for MLS — 系统设计报告（Phase 2）

23302010024 印伟辰

## 1. 项目目标

Phase-2 评测脚本只读一份文件 `./optimized_lora.cu`，用 `cpp_extension.load` 编译后调 `mod.forward(W, X, A, B)`，与 PyTorch 参考实现对照算 `speedup_geomean`。系统的任务是：在 30 分钟内自动搜索一个**既正确又更快**的 CUDA kernel，并实时把当前最佳版本写到 `./optimized_lora.cu`。

由此衍生出三条贯穿全局的硬约束：

* **算子无关**：评测后续可能换算子，系统对 LoRA 没有写死任何形状或公式。
* **LLM 不许量自己**：编译结果、speedup 等数值全部由确定性 Python 算出来，LLM 没有任何工具能短路。
* **断点可续**：进程随时被杀，重新指定 `run_id` 必须能根据磁盘工件接着跑。

## 2. 顶层架构

系统物理上分两层：

* **`mls_agent/`** —— 独立的通用 ReAct 框架。提供 `Agent` / `ReActLoop` / `Tool` / `ToolRegistry` / OpenAI 兼容 backend / CUDA `Executor`。不知道有"算子"这件事，可单独复用。
* **`operator_opt_pipe/`** —— 业务层（pipeline）。把 ReAct 能力组装成一条带状态机、工作区和确定性评测的流水线。

调用关系自上而下：

1. **入口**（`main.py`）— 加载算子契约、做 build 环境预检（nvcc / g++ / `load_inline` 烟测）、构造编排器。
2. **编排器**（`PipelineOrchestrator`）— 纯代码状态机；唯一拥有"同步 `optimized_lora.cu`"权限的组件。
3. **Stage 集合** — 5 个阶段串成一条流水线。
4. **Agent 角色** — 5 个 LLM 角色，全部跑在同一套 ReAct loop 上，差别只是 system prompt 和工具集。
5. **确定性资源**（`resources/` + `operators/`）— agent 不可见、编排器直接调用：生成输入、测 PyTorch 基线、编译候选、跑多形状 benchmark。

## 3. mls_agent 框架层

* **ReAct 循环**：一次 `Agent.run()` 走完整套 Thought → Validate → Act → Observe → Apply → Decide。退出有四种原因：工具主动终止、连续多次不调工具、超过迭代上限、LLM 异常。业务层只接受"主动终止"为成功，其它一律视作 stage 失败。
* **Tool 抽象**：每个工具是一个类，暴露名称、JSON Schema 参数、`run()` 方法和统一的 `ToolResponse`。**终止只能通过 `ToolResponse.terminate=True`**，没有异常控流。
* **内置工具组**：
  * `read_skill / list_skills` —— 读 `skills/*.md` 给 LLM 提供经验材料；
  * `record_measurement / flag_event` —— LLM 主动写审计；
  * `terminate` —— 通用结束工具；
  * `make_profile_tools(executor)` —— CUDA 工具集（编译探针、ncu、nsys、torch 测时、环境探查）。
* **CUDA Executor**：唯一真正起子进程做编译/分析的代码，负责工具链探测、`cpp_extension.load_inline` 烟测、ncu/nsys 跑批与每候选独立工作区。

## 4. operator_opt_pipe 业务层

### 4.1 算子契约

两件套，都定义在 `operator_opt_pipe/operators/<name>.py`：

* **`OperatorContract`**（frozen dataclass）—— 描述输入/输出张量、参考公式、`forward` 参数顺序、形状范围、容差。
* **`OperatorOps`**（ABC）—— 可执行行为：`make_inputs / reference / forward_call`。

**新增一个算子 = 写一个 Python 模块 + 在 `__init__.py` 的 dict 里挂上**，其它层零改动。

### 4.2 工作区与共享状态

每个 run 一个目录 `workspace/runs/<run_id>/`，关键工件：

* `state.json` —— 流水线自身状态（当前 stage、round 序号、已耗时、当前 best）。
* `blackboard.json` —— 跨 stage 共享 KV，存 `hardware / baseline / best / latest_diagnosis / history / final_metrics / final_summary` 等。
* `events.jsonl` / `leaderboard.jsonl` —— 审计流，分别记录阶段事件与每轮 tuning 的候选评测结果。
* `inputs/`、`candidates/candidate_NNN/`、`best/` —— 输入张量、候选源码、当前最优。
* `agent_trace.log` —— ReAct 完整轨迹，FINALIZE 时被嵌进 `output.md` 作为 Phase-2 推理证据。

Blackboard 是一个**带白名单**的 KV：每个 LLM 角色只能写自己负责的 key（`hardware_profiler` 只能写 `hardware`，`analyst` 只能写 `latest_diagnosis`，`summary` 只能写 `final_summary`），从机制上避免越权。

### 4.3 确定性资源

* `resources/baseline.py` —— 生成各形状输入、用 `cudaEvent` 测 PyTorch 参考延迟。
* `resources/evaluation.py` —— 候选编译 + 校验。给 LLM 的快速反馈（单形状）和给编排器的最终评测（多形状）共用编译路径。
* 评测时 PyTorch reference 永远在子进程内现算、不缓存：避免主进程 TF32 状态与 Phase-2 评测脚本漂移。

### 4.4 五个 Agent 角色

| 角色 | 用到的工具 | 输出去向 |
| --- | --- | --- |
| `hardware_profiler` | CUDA 探针、ncu、环境探查 | `blackboard["hardware"]` |
| `optimizer_cold` | `write_candidate` / `submit_candidate` | 首个能跑的候选 |
| `analyst` | ncu / nsys / torch 测时 | `blackboard["latest_diagnosis"]`（bottleneck 假设 + 调参提示） |
| `optimizer` | `write_candidate` / `submit_candidate` | 针对 diagnosis 的下一个候选 |
| `summary` | 只读 + `write_blackboard("final_summary")` | 最终叙事段 |

终止方式统一为"显式调用工具结束循环"——前三类调通用 `terminate`，optimizer 系列调 `submit_candidate`（把 `candidate_id` 作为 payload 交给编排器）。LLM 没显式终止 = 没干完 = stage 失败。

### 4.5 自研 4 个工具类

* `ReadBlackboardTool` —— 读单个 key。
* `WriteBlackboardTool` —— 写单个 key，按角色实例化不同白名单，不终止循环。
* `WriteCandidateTool` —— 分配新候选目录、写 `.cu`、同步跑单形状编译+校验，立刻把结果回传给 LLM 让它迭代。
* `SubmitCandidateTool` —— 冻结一个 candidate_id 并终止循环，把控制权交回编排器做多形状 benchmark。

### 4.6 状态机与编排器

前面的零件（契约 / 工作区 / 资源 / Agent / 工具）需要一个调度者把它们装配成一条流水线。这就是 `PipelineOrchestrator` + 5 个 Stage 状态机做的事。

**Stage 集合**——一条线性流水：

| Stage | 谁来跑 | 做什么 |
| --- | --- | --- |
| `HARDWARE_PROFILE` | `hardware_profiler` agent | 描述 GPU（SM / DRAM bw / L2 / clock） |
| `BENCHMARK_BASELINE` | 纯代码 | 生成各形状输入，测 PyTorch 参考延迟（speedup 分母） |
| `INITIAL_CANDIDATE` | `optimizer_cold` agent | 拿到首个 compile + correctness 通过的候选，让 `./optimized_lora.cu` 一定存在 |
| `TUNING_LOOP` | `RoundRunner` 循环 | 反复 `analyst → optimizer → 多形状 benchmark → 可能 promote`，直到时间预算用尽 |
| `FINALIZE` | `summary` agent + 编排器 | 收尾，渲染最终报告、`output.md` 与 root 文件再同步 |

**转移规则**——`transitions.next_stage` 是个纯函数，**所有判断都基于磁盘工件的存在性**（`hardware_profile.json` 在不在、`baseline.json` 在不在、当前 `best_candidate_id` 是不是 `None` 等）。这就是断点续跑的基础：启动 = 看磁盘上缺什么、做缺的那段。当剩余预算少于一轮 tuning 所需的最小时间（约 300 秒）时直接收尾。

**运行时流程**——`PipelineOrchestrator.run()`：

1. **启动 / 续跑**：读 `state.json`（不存在就新建），把算子契约信息 seed 进黑板。新 run 与续跑走同一路径，目录已存在则原地接续。
2. **主循环**：反复 `next_stage → 分发到对应 runner → tick_elapsed + 保存 state.json`，每次进入 stage 记一条 `stage_enter` 事件。`MAX_LOOP_ITERATIONS = 200` 兜底防死循环。
3. **HARDWARE_PROFILE**：跑 agent → 把 `blackboard["hardware"]` 镜像到 `hardware_profile.json`（下次启动的存在性谓词）。
4. **BENCHMARK_BASELINE**：纯代码——落盘各形状输入，测 PyTorch 参考延迟，结果同时写黑板与 `baseline.json`。
5. **INITIAL_CANDIDATE**：跑 `optimizer_cold` 拿首个候选 → 编排器跑多形状 benchmark → 只要 compile + correctness 过就直接 promote（**即使 speedup < 1 也提**，保证 `./optimized_lora.cu` 一定存在）。
6. **TUNING_LOOP**：每轮 `round_index += 1`，`RoundRunner` 跑 analyst → optimizer → 多形状 benchmark → 比较 geomean speedup 决定是否 promote。每步往 `blackboard["history"]` 追加记录，benchmark 结果落 `leaderboard.jsonl`。
7. **FINALIZE**：先写 `blackboard["final_metrics"]`（真实的 best / baseline 数字），再跑 `summary` agent（**只能读、不能编造**），最后渲染 `final_report.json` / `summary.md` / `output.md` 并兜底再同步一次 root 文件。

**硬不变量**：

* `_promote_to_best` 是流水线里**唯一**允许动 `best/` 和 `./optimized_lora.cu` 的入口：拷贝源、写 `best_result.json`、更新黑板、追加 `best_promoted` 事件、同步 root 文件——五件事全在一处。
* `best/` 只允许编排器写；agent 只能写到自己的候选目录。
* `optimized_lora.cu` 同步点只有三处：INITIAL_CANDIDATE 通过、每次 `_promote_to_best`、FINALIZE 收尾。LLM 永远碰不到这个文件。

## 5. 失败与恢复

* Stage agent 出错或非正常终止 —— 记 `stage_failed` 事件，按状态机决定下一步（通常继续推进或直接 FINALIZE）。
* 候选编译/校验失败 —— `write_candidate` 把失败原因回送，LLM 迭代；多形状 benchmark 失败则不 promote。
* 子进程挂死（编译卡 lock）—— 单独的 `INFRASTRUCTURE_TIMEOUT` 错误码，提示 LLM 是基础设施问题、别再简化 kernel。
* 时间预算到 —— `next_stage` 第一条规则直接切 FINALIZE。
* 撞死循环上限 —— 写 `loop_guard_tripped` 后收尾。

## 6. 设计要点小结

1. 编排（状态机）与生成（LLM）严格分离：LLM 只能"提候选 + 写笔记"，promote 与同步全部由编排器掌握。
2. 算子无关靠 `OperatorContract + OperatorOps` 实现，新增算子是一次纯加法改动。
3. 断点续跑靠"所有 stage 谓词都是磁盘工件存在性"。
4. 框架层（`mls_agent`）与业务层（`operator_opt_pipe`）解耦，框架自身无算子知识，可单独复用。

---
