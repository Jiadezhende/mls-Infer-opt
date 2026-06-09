# Agent for MLS — 系统设计报告（Phase 3）

23302010024 印伟辰

把「LLM 推理引擎调优」组织成「模型训练」的自动化框架：给定 decoder-only 的模型结构信息，经
agent 多轮迭代，产出优化后的 `engine.py`，并留下 `runs/`、`results.log`、`output3.json` 供审计。

报告分两部分：第 1–7 节是本阶段（Phase 3）的系统设计；第 8 节回到 Phase 1、Phase 2，提炼三个
阶段在「agent 做最优化问题探索的反馈循环」上累积出的经验。

---

## 1. 目标与结构类比

输入是只读的模型结构（`model_config.json` + `weights/`），输出是一份自包含纯 PyTorch 的
`workspace/engine.py`，评测器按固定 API 契约调用：`create_engine` / `Engine.prefill / decode /
remove`。

框架结构对应机器学习训练循环，每个调优动作映射到一个训练概念：

| 训练概念 | 本框架对应 |
| --- | --- |
| init | 装配上下文 + 落保守 baseline |
| forward / eval | 评测一个候选（gate + bench） |
| grad（梯度方向） | 分析反馈、定瓶颈、给下一步 |
| train step | 生成一份新候选 engine |
| loss / score | 逐 case 两列吞吐对 baseline 的等权几何平均 |
| keep-best / checkpoint | 发布更优候选为 best |

主循环：init → forward(评测) → grad(分析) → step(生成候选) → 重新评测 → 若更优则发布为 best，
循环直到判停。

---

## 2. 分层架构（建设顺序 = 依赖顺序）

开发顺序为先定契约、再建能力层（agent 起步、逐阶段铺开）、最后由 loop 编排反馈循环，依赖单向
向下：

```
契约层          跨阶段产出格式（最先搭，所有层共享）
搜索空间领域层   space + 冲突消解 + Policy 聚合（依赖只向下到契约层）
能力层（各阶段能力，agent 在需要的阶段接入）：
  agent         OpenAI Responses + function-tool + 读写（能力层起点）
  generate      train step：产候选 engine（Policy + 开发规范 → engine.py）
  evaluate      forward：正确性门 + 性能 benchmark（子进程隔离）
  analyze       grad：诊断瓶颈 + 给下一个 Policy
反馈循环层      确定性状态机，编排各能力 + 维护唯一主状态 + 发布 best
产出层          实时进度 + results.log / 验收块
```

契约层与产出层是横向支撑；搜索空间领域层是 analyze 与 generate 共享的领域模型（轴/选项/冲突消解/
Policy 聚合），两者都向下依赖它、彼此不再横向 import；能力层是被编排的各阶段，反馈循环层专门负责
把它们串成一个环。下文按此分层展开：第 3 节契约层，第 4 节逐个阐述能力层各阶段（含各自的效果
考量），第 5 节反馈循环，第 6 节产出。

---

## 3. 契约层

所有阶段之间交换的对象都是纯 dataclass 契约，先于业务逻辑搭建。契约按内存对象图设计——评测结果
直接长在候选身上，不走关系库/外键/并行表：

- **TaskContext**：一次会话的只读上下文（模型配置/设备/三类路径根/判停上限/环境快照）。派生路径
  按需算出，约束 `engine.py` 只能写到约定位置。
- **Policy**：搜索空间里的一个点，analyze 产出、generate 消费的共享契约。类型落在最底层契约层，
  analyze 与 generate 都能直接引用、互不横向 import（聚合逻辑在搜索空间领域层，见第 4 节）。
- **Candidate**：一次 train step 的产物节点。诞生即定的不可变事实（id/kind/parent）+ 后填的评测
  结果（gate/bench），直接挂在对象上。源码不进内存、落盘到候选目录。
- **GateResult / BenchResult**：正确性与性能物理分离（硬布尔门 vs 性能标量）。「只有过门候选才有
  性能」在赋值处校验。
- **LoopState**：整个 run 唯一的主状态实例；候选构成内容寻址表，事件流 append-only。候选当前处于
  哪一步由 gate/bench/best 的存在性派生，不存冗余字段。

效果考量：契约先行让四个阶段可以独立开发、独立测试（可注入 fake，不必真跑 torch/LLM）；对象图
而非外键，避免全量候选常驻内存与灌爆 report JSON；状态派生而非冗余存储，消除不一致来源。

---

## 4. 能力层

能力层覆盖 agent、generate、evaluate、analyze 四个阶段。建设从 agent 起步、再逐阶段接入；阶段之间
不互相 import，只通过契约层与搜索空间领域层交换/共享数据，由反馈循环层串联。各阶段连同自身的
效果考量分述如下。

### 4.1 agent

职责：封装 OpenAI Responses API，支持本地 function-tool 执行循环，并兼容单次 `generate(prompt)`
接口。它只提供调用、function call、读写工具，不含业务，由 generate / analyze 在需要 LLM 的阶段
接入。

效果考量：

- **可选增益**：LLM 缺 key / 没装 SDK / 调用失败都返回不可用，由上层走规则兜底，框架在无 LLM 时
  仍能完整运行。
- **不假死**：关掉 SDK 默认的超时静默重试（最坏约 timeout×3），重试交上层循环控制，避免长输出下
  放大成数分钟假死。
- **能力与业务解耦**：tool-loop 的每轮调用、工具执行、用量都有审计记录，但客户端本身不知道任何
  业务语义。

### 4.2 generate（train step）

职责：把 Policy + 开发规范渲成 prompt、产出一份完整 `engine.py` 候选。三个入口同一件事、条件
不同：bootstrap（不依赖 LLM，落 pristine baseline，永久兜底）、propose（按 Policy 产新候选）、
repair（按结构化报错定向修复）。

关键抽象 —— Policy 与多维搜索空间（定义在搜索空间领域层，generate 与 analyze 共享）：不让 agent
自由发挥，而是先把一次 forward 数据流经的层次拆成若干正交的轴，构成唯一的搜索空间真相源——16 条
轴、5 个组件分组（cache / batching / operator / precision / weight_layout）。三个约定：

- **baseline-first**：每条轴的首个选项等价于现有 baseline，「全默认轴」渲染出来就是 baseline
  本身，bootstrap 不需要特例。
- **轴名/选项名即契约**：同时是 policy 键、prompt 渲染、策略标签、analyze 引用策略的标识。
- **数值敏感标注**：🔴 改了会动 logits（可能顶破容差），🟢 正确实现即与 baseline 数学等价。

Policy 是在这些轴上各选一个选项加被激活轴的 knobs。从任意来源到一份合法 Policy 走固定四步管线：
normalize（填默认、丢非法轴/选项）→ resolve（消解轴间冲突）→ fill knobs（仅为被激活轴填 knob）→
Policy + 策略标签。generate 的输入即 Policy + 开发规范：把稳定的引擎契约（API 契约 / 自包含硬规则 /
正确性目标 / 输出格式）叠上 Policy 的非默认轴中文指令、model_config、参考代码、analyze 方向，拼成
prompt。

效果考量：

- **内层自检收敛**：generate 不是调一次 LLM，而是确定性的自检重试自闭环：出码 → 静态自包含早筛
  （语法/import 白名单/API arity）→ quick gate（子进程隔离、对 oracle 比 logits）→ 不过就把结构化
  错误回灌、转 repair 提示让模型修自己的码，循环到过门或耗自检预算（上限若干轮）。控制权全程在
  确定性代码里，不依赖模型自觉调工具。
- **只产候选、无发布权**：内层 quick 自检是 ephemeral、不进主状态；正确性由外层 evaluate 权威
  保证。LLM 不可用 / 失败 / 产垃圾 / 自检始终不过都返回空（反馈循环当作这轮没收益走兜底），不抛
  异常。
- **冲突消解可复现**：对「合批解码需 KV 缓存」「enable_gqa 需 SDPA」这类依赖采用依赖方让步——
  违反时把更激进的依赖轴退回 baseline 默认。退回默认不引入新风险，且对同一输入决定性。
- **内容寻址判重**：候选 id 由轮次与源码哈希派生，可 O(1) 判「这段代码是否已测过」从而跳过重复
  评测；policy 序列化字节级可复现，便于审计 diff。

### 4.3 evaluate（forward）

职责：给候选两层信号——正确性硬门 GateResult（syntax → api → correctness 三阶段）与性能
BenchResult（prefill/decode/mixed 三类工况，每类按真实 grader 的两列计量——整体 tokens/s 与
decode tokens/s——外加峰值显存）。只有过门候选才进入性能评测。决定性、不调 LLM。

效果考量：

- **外层 full gate 是唯一权威**：挂到候选上的 gate 只有反馈循环在 keep-best/发布前重跑的 full
  gate，不采用 generate 内层 quick 自检的 ephemeral 结果，也不信候选自报。正确性判定为对官方
  reference model 比 allclose（容差 1e-2），逐 case fail-fast。oracle 与候选喂完全相同的确定性
  事件流（同 seed 生成），否则比对无意义。
- **子进程隔离**：所有 torch / 风险代码只在子进程跑，父进程纯 stdlib、零 torch。坏候选崩溃/超时
  只死子进程，父进程把超时→杀进程、非零退出/非法 stdout 一律翻成结构化失败 gate，上层始终拿到
  结果，不抛异常。
- **泛化验证**：隐藏评测会变 batch / 长度 / 步数 / 顺序，因此两侧都防过拟合。正确性侧除照搬的
  标准 case，额外加变 batch/长度/顺序的泛化抽测；性能侧 full 工况设为不规则（ragged）——
  decode/mixed 每请求长度与停止步各异、跨多个 seed，逐 seed 取下半均值聚合。等长 lockstep 流会被
  分组策略过拟合，ragged 流更能反映 varlen 批处理 / KV-cache 的鲁棒性；跨 seed 的 decode tps
  离散度回喂 analyze 作抗不规则信号。
- **score 口径对齐真实 grader**：grader 逐 case 报两列——整体 tokens/s =(prefill+decode)/elapsed
  与 decode tokens/s = decode/elapsed——外加峰值显存，且不公布任何合成权重（含一个纯 prefill
  case，decode 列为 0）。因此内部 score 不再内置 decode/mixed/prefill 权重，改为把 grader 会计量
  的每一列对 baseline 求比值、平铺进**等权**几何平均（prefill 整体 + decode 整体/decode 列 +
  mixed 整体/decode 列）：整体与 decode 的相对话语权由项数自然给出，而非人为常数；scale-free，
  不被某一列量级绑架；任一项塌到 0 即把整体拉到近 0，对应该项失败即灾难。峰值显存 grader 每 case
  都报，但合成口径未公布，故当前只计量+展示、不入分，作护栏（不盲目拿显存换速度）。worker 侧只算
  临时自评，权威归一化在反馈循环层完成（见 5 节）。

### 4.4 analyze（grad）

职责：每轮看反馈，定位瓶颈，从当前 best 出发给下一步方向或判停。流程：先把主状态汇总成 ephemeral
的态势对象，再做①确定性硬判停（预算/轮数/连续无提升）②LLM 要方向（单次调用 + 确定性解析）③不可用
就退回 rule-based 贪心阶梯。

关键抽象 —— 基于 best 叠 delta（梯度叠加）：每轮 propose 的父代是当前 best 的源码加 best 的 Policy，
analyze 只产相对 best 要改动的轴（axes delta），经搜索空间领域层把 delta 合并进 best 的 Policy 得到
下一个完整 Policy。等价于把梯度叠加到当前参数上做局部搜索：复用已验证的稳收益，避免每轮从零重新
踩坑。叠加用的是共享的搜索空间领域层（merge / aggregate / 冲突消解），analyze 不横向 import
generate——这正是把搜索空间抽成独立层的目的。

效果考量：

- **在真实评分轴上推理**：给 LLM 的方向 prompt 显式写明 grader 的两列评分口径（整体吞吐 /
  decode 吞吐，显存只观测），并提示 prefill 不是边角料（独占一个 case，且每个 case 的整体列分母
  都含 prefill 时间），让它整体吞吐与 decode 吞吐都顾、不只盯单一指标。
- **LLM 不可靠时仍给合法方向**：rule-based 兜底是一条贪心阶梯，按优化主线先验排序：KV 增量缓存
  （收益最大）→ RoPE 预算表 → batched prefill → 合批解码 → SDPA → GQA → 权重融合 → bf16，前置
  依赖在前、低风险(🟢)优先、数值敏感(🔴)靠后。相同 best 给相同下一步，可复现可测。
- **never-throw**：内部任何异常（含建态势、构 Policy）都翻成一条 error 事件并返回空（= 停，best
  已是安全产物）。
- **无发布权、不写 stop_reason**：analyze 每轮只 emit 一条 analyze 事件，停因放进事件数据，由反馈
  循环读后落到主状态的 stop_reason；knob 只进 Policy 的 knobs，绝不碰 model_config 的结构字段。

---

## 5. 反馈循环

职责：外层确定性控制器，不问 LLM 做业务判断、不信候选自证、不产 engine 代码，只负责编排各能力、
维护唯一主状态、发布已验证 best。

```
bootstrap → 评测 → keep-best（冻结 baseline_score 作 speedup 锚点）
  └─ while best 存在 且 未判停：
       analyze(state)  ──► 下一个 Policy（带 rationale）或 空(停)
       propose(Policy, best_code) ──► 新 Candidate（基于当前 best）
       evaluate(Candidate) ──► 填 gate / bench
       keep_best?  ── 是 ─► 增量发布 engine.py + output3.json
                   └─ 否，且 gate 不过 ─► repair 重试（外层）
finalize → final gate 复核 → 权威终发
```

效果考量：

- **发布以正确性为硬门**：只有 gate 通过且分数严格更高的候选才提升为 best，赋值处再校验一遍；
  只有当前 best 能被发布。
- **score 归一化单一咽喉点**：父进程统一把候选在 grader 会计量的每一列上对 baseline 求比值、
  合成为等权几何平均加速比，是所有 score 消费者（keep-best 比较 / speedup 展示 / analyze）之前
  唯一的归一化处，口径一致且诚实。
- **抗中途 kill**：评测有约 30min 外墙，内部设约 28min 硬墙，循环到点后在下一轮开始前停 +
  finalize。每刷新一次 best 立刻把已过 gate 的 `engine.py` 拷到发布点并同步刷新 `output3.json`，
  `results.log` 逐条 append。任意时刻被 kill，盘上都是当前最优。
- **三级兜底链**：最优正确候选 → 已验证 baseline → 原始 pristine baseline。最外层 try/finally
  保证两条对外契约即便主循环崩溃也成立：始终 exit 0，`workspace/engine.py` 必在盘上。

---

## 6. 产出与可审计性

| 产物 | 内容 | 写入时机 |
| --- | --- | --- |
| `workspace/engine.py` | 唯一发布出口（已验证 best） | keep-best 增量发布 + finalize 终发 |
| `workspace/output3.json` | 摘要 + `result` 判定 + `rounds[]` 逐轮推理（best/score/speedup/正确性/停因/诊断→策略→评测→结论） | 随 best 同步刷新 + finalize |
| `runs/{run_id}/report.json` | 任务结果记录（内容同 output3.json） | finalize |
| `runs/{run_id}/final/results.log` | 逐条结构化事件日志（增量写、抗中途 kill） | 逐条实时 append + finalize 全量 |
| `runs/{run_id}/candidates/{id}/` | 每个候选的 `engine.py` + `policy.json` | 落盘即留 |
| `runs/{run_id}/final/` | 本次 run 终态留档：`engine.py` / `output3.json` / `results.log`（供复盘） | finalize |

产出层把每条事件渲染成可 grep 的表头 + 缩进数据块，analyze → generate → evaluate → keep_best
四类事件按诊断 → 策略 → 评测 → 结论的顺序排列，逐轮可观测。

---

## 7. 本阶段小结

各层职责单一、依赖单向：契约层定格式，能力层（agent/generate/evaluate/analyze）各管一段并自带
对应的正确性与鲁棒性保障，反馈循环层串成环并独占发布权与兜底责任。LLM 全程是可选增益，缺失或
失败时各层退回确定性路径，任意失败或中断下都满足始终 exit 0、engine.py 必在盘、产物反映当前最优。

---

## 8. 三阶段回顾：agent 做最优化问题探索的反馈循环

三个阶段的题面不同，但内核是同一件事：让 agent 在一个评测器定义的目标下自动探索，多轮反馈逼近
更优解。下面先并排看三次循环的形态，再提炼累积出的经验。

### 8.1 三次循环的形态

- **Phase 1（硬件探查，[report1.md](report1.md)）**：给定目标规格，agent 规划测量、写并编译 CUDA
  微基准、跑 Nsight、记录带证据的硬件数值。架构是 Planner → Worker Pool → Critic：每个 worker 跑
  自治 ReAct 循环（迭代修 CUDA、超时降负载、累积证据），Critic 对结果 accept / retry，熔断器在
  同类错误连续失败时切断。这阶段严格说是**测量/探查**问题，但反馈环的雏形已经完整：自治迭代 +
  外层裁决 + 防失控。

- **Phase 2（CUDA 算子优化，[report2.md](report2.md)）**：30 分钟内搜一个又对又快的 CUDA kernel，
  实时把当前最佳写到交付文件。架构是通用 ReAct 框架 + 业务流水线，五个 stage 线性串联，核心是
  调优循环：分析 → 优化 → 多形状 benchmark → 可能 promote，直到时间预算耗尽。这阶段第一次把它当
  **最优化**问题，并立了三条硬约束：算子无关、LLM 不许量自己、断点可续。

- **Phase 3（推理引擎调优，本报告）**：把整个循环显式映射成模型训练——forward/grad/step/checkpoint，
  在 16 条正交轴构成的有界搜索空间上做基于 best 的梯度叠加，LLM 全程可选。

并排对比：

| 维度 | Phase 1 探查 | Phase 2 算子优化 | Phase 3 引擎调优 |
| --- | --- | --- | --- |
| 问题类型 | 测量 / 探查 | 单算子 CUDA 最优化 | 引擎多轴最优化 |
| 反馈环 | ReAct worker + Critic accept/retry | 分析→优化→benchmark→promote | analyze(grad)→propose(step)→evaluate(forward)→keep-best |
| 搜索空间 | 工具 + 探针自由组合 | free-form CUDA 候选 | 16 轴有界 Policy（baseline-first） |
| LLM 角色 | 解读原始计数器、定策略（权力最大） | 提候选 + 写诊断 | 可选增益，全程有确定性兜底 |
| 数值/正确性权威 | LLM 解读原始输出 | 确定性代码（LLM 不许量自己） | 外层 full gate 唯一裁定 |
| 发布权 | 入口统一去重写出 | 仅编排器 promote/sync | 仅反馈循环层 keep-best 发布 |
| 抗 kill / 续跑 | 熔断器防失控 | 磁盘谓词断点续跑 | 增量发布 + 时间硬墙 + 三级兜底 |
| 防作弊 / 过拟合 | anti-hacking（live 测量） | 多形状 benchmark | ragged 工况 + 泛化抽测 + 跨 seed |

可以看到一条主线：**agent 的自由度在收缩，确定性骨架在加厚**，而循环本身的探索能力不降反升。

### 8.2 累积出的经验

**经验一：搜索空间比 agent 本身更值得投入。** Phase 1 让 agent 自由组合工具与探针，Phase 2 让它
自由写 CUDA 候选——空间无界，带来的代价是难判重、难复现、难为失败给确定性兜底。Phase 3 把一次
forward 数据流拆成 16 条正交、baseline-first、轴名即契约的轴。一旦空间有界且可枚举，就立刻获得
四样东西：rule-based 贪心阶梯可以无 LLM 兜底走完、内容寻址可 O(1) 判重、policy 字节级可复现、
轴间冲突可确定性消解。最优化的收益主要来自"把问题拆成好的搜索空间"，而不是"换更强的提示"。

**经验二：LLM 在环里只提方向，不当数值与正确性的权威。** 这是逐阶段收紧的一条线。Phase 1 的
LLM 直接解读原始 profiler 文本提取数值（等于让它量自己）；Phase 2 据此立规矩"LLM 不许量自己"，
编译结果、speedup 全由确定性代码算；Phase 3 进一步把 LLM 降为纯可选增益——缺 key / 失败 / 产垃圾
都有确定性兜底，正确性由外层 full gate 唯一裁定，候选自报一律不算数。结论是：在最优化反馈环里，
LLM 适合做高方差、可证伪的"提方向 / 提候选"，不适合做需要确定性、可复现、可审计的"测量 / 判对 /
发布"。把这两类职责物理分离，环才稳。

**经验三：编排（确定性状态机）与生成（LLM）严格分离，发布权独占一处。** Phase 1 的编排器是纯
调度、自身不调 LLM；Phase 2 规定只有编排器能 promote 和同步交付文件；Phase 3 的反馈循环层是
确定性状态机，独占 keep-best 与发布，analyze 与 generate 都没有发布权。规律是：谁能改动"对外
交付物"必须收敛到唯一一处、且是确定性代码；LLM 永远碰不到交付文件，从机制上杜绝越权和把环境
带歪。

**经验四：把探索当作带 checkpoint 的局部搜索，而非每轮从零。** Phase 1/2 的每轮候选相对独立
（Phase 2 的优化角色虽看诊断，但更接近重写）。Phase 3 显式引入 best + delta：analyze 只产相对
当前 best 的 axes delta，合并成下一个完整 Policy，等价于把梯度叠加到当前参数上。在 LLM 调用次数
和时间预算都有限的前提下，复用已验证的稳收益（在 best 上叠 delta）比每轮重新踩坑的样本效率高
得多，而 keep-best 的严格单调保证循环不退化。

**经验五：交付物必须随时可交、抗中途 kill——时间墙下要先有安全网再谈优化。** 评测随时可能在
30 分钟内被 kill。Phase 1 用熔断器防失控；Phase 2 让首个候选即使 speedup<1 也先 promote（保证
文件一定存在），并用磁盘工件存在性做断点续跑；Phase 3 进一步用永久兜底的 bootstrap baseline、
每刷新 best 即增量发布、内部时间硬墙、三级兜底链、始终 exit 0。共同的纪律是：任意时刻盘上都得是
当前最优且合法的产物；优化是增量收益，绝不能拿交付物去赌探索。

**经验六：内部评测要比隐藏评测更狠——从 anti-hacking 到 anti-overfitting。** Phase 1 的 anti-hacking
是 live 测量、不信厂商规格表，防的是被环境骗。Phase 3 把同样的警惕转向自己的优化器：隐藏评测会变
batch / 长度 / 步数 / 顺序，所以内部评测主动用 ragged 工况 + 泛化抽测 + 跨 seed，把过拟合点暴露在
内层；且 oracle 与候选喂完全相同的确定性事件流，否则比对无意义。同理，内部评分口径要对齐真实
grader——只看得到逐 case 两列（整体 / decode 吞吐）而未拿到最终聚合公式时，宁可用等权（相对话语权
由计量项数自然给出）也不自拍一套没验证的权重，免得优化器朝一个偏掉的代理指标使劲。优化器会榨干
你奖励的一切，内部 eval 必须至少和隐藏 eval 一样对抗、且打在真实评分轴上，否则优出来的是过拟合到
固定形状或错误代理的假收益。

**经验七：每个决策都要可审计——自治环必须可观测。** 三个阶段都坚持留痕：Phase 1 让 agent 主动
记录带证据的测量与异常事件，Phase 2 留审计事件流、候选榜与完整 ReAct 轨迹，Phase 3 用 append-only
事件、`output3.json` 里机读的 `rounds[]`（诊断 → 策略 → 评测 → 结论）、实时增量的 `results.log`。
一个自治优化环若不把"为什么走这一步"留痕，就既无法调试、也无法被信任、更无法复盘；停因和归一化
口径尤其要单一来源、诚实呈现。

### 8.3 收束

三个阶段是同一道题的三次加深：agent 负责在搜索空间里提出方向，确定性骨架负责测量、判对、发布与
兜底。越往后，探索的自由越是交给一个被精心定义的有界空间，结果的权威越是交给确定性代码。这套
分工是三阶段反馈循环得以收敛、可复现、抗中断的根本原因。
