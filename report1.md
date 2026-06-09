# Agent for MLS - Phase 1

23302010024 印伟辰

## Overview

This project implements an autonomous GPU hardware-probing agent for the MLSYS
course project. Given `/target/target_spec.json`, the agent plans measurement
work, writes and compiles CUDA microbenchmarks, runs Nsight profiling tools, and
records numeric hardware measurements with evidence. Output is written to
`/workspace/output.json` as a flat `{ "metric_name": numeric_value }` map.

## Agent architecture

The system has two layers: a single-agent ReAct loop per worker, and a
fixed-pipeline multi-agent orchestrator.

### Single-agent layer (AgentLoop)

Each hardware-probe worker runs an autonomous ReAct loop. It receives one
measurement step, reads strategy documents from `skills/`, and iteratively
decides which tool to call until it calls `submit_results`.

**Tool set available to `hardware_probe` workers:**

| Tool | Purpose |
| --- | --- |
| `list_skills` / `read_skill` | Read measurement strategy docs from `skills/*.md` |
| `run_cuda_probe` | Compile and run a CUDA C microbenchmark; primary tool for latency |
| `profile_with_ncu` | Run Nsight Compute counters; primary tool for throughput/clock |
| `profile_with_nsys` | CPU-GPU timeline via Nsight Systems; for launch overhead |
| `probe_environment` | Query GPU properties (SM count, memory freq, etc.) |
| `record_measurement` | Store a `Result` into worker memory with evidence |
| `flag_event` | Log anomalies, strategy pivots, and infrastructure errors |
| `submit_results` | Signal the worker loop to exit and return collected results |

The Executor is the only component that compiles CUDA source, spawns
subprocesses, truncates stdout, classifies errors (`user_code` / `infrastructure`
/ `timeout`), and manages workspace directories. Workers never call system tools
directly.

`profile_with_ncu` returns raw Nsight Compute text output directly; no
intermediate CSV reduction layer exists. The LLM interprets the raw counter
output and extracts values.

The message loop appends every assistant response and every tool result back into
the worker's context window, allowing the agent to iteratively fix CUDA source,
reduce workload on timeout, and accumulate evidence across iterations.

### Cross-agent layer (Orchestrator)

The top-level runtime follows a fixed **Planner → Worker Pool → Critic**
pipeline. The Orchestrator is pure scheduling code — no LLM calls of its own.

**Planner**: one LLM call with forced tool `assign_workers`. Routes target
metrics to worker steps; falls back to 1:1 mapping on failure.

**Worker Pool**: `ThreadPoolExecutor` with per-worker isolated `AgentContext`
and `CircuitBreaker`. Each worker looks up its `AgentDefinition` by
`step.worker`, receives injected tools from `ToolFactory`, and runs its
`AgentLoop` concurrently. Workers share one `LLMClient` and one `Executor`
(both thread-safe).

**Critic**: reviews all `WorkerOutput` objects after the worker pool finishes.
Issues `accept` or `retry` decisions. On retry, only the failing target subset
is re-assigned; previously accepted measurements in the same step are carried
forward. The retry loop repeats until all steps are accepted or
`AGENT_MAX_WORKER_RETRIES` is exhausted.

`main.py` performs the only final-file write: it deduplicates results by metric
(highest confidence wins), converts values to `int` where the float is whole,
and writes `/workspace/output.json`.

### Circuit breaker

A `CircuitBreaker` is attached to every worker (not globally shared). It
applies to **all** tools. After `AGENT_CB_THRESHOLD` (default 3) consecutive
failures of the same `(tool, error_kind)` pair, the breaker opens and
`dispatch()` returns `{"status": "circuit_open"}` without executing the tool.
After `AGENT_HALF_OPEN_TIMEOUT_S` (default 60 s) the breaker allows one probe;
success closes it, failure resets the timer. This prevents runaway retry loops
on infrastructure errors.

### LLM compatibility

`LLMClient` wraps the OpenAI SDK with support for any OpenAI-compatible
provider via `BASE_URL`. Provider-specific parameter handling:

| Model family | `temperature` | Token limit param |
| --- | --- | --- |
| GPT-4o, DeepSeek Chat, etc. | sent | `max_tokens` |
| OpenAI o-series (o1, o3 …) | omitted | `max_completion_tokens` |
| DeepSeek Reasoner / R1 | omitted | `max_tokens` |

Runtime fallback: on a 400 error mentioning `"temperature"` or
`"max_completion_tokens"`, the client adjusts its parameter set and retries
immediately. The adjustment is cached for all subsequent calls in the session,
so unrecognised model names self-correct on the first failed call.

## Measurement strategy library

`skills/*.md` files encode domain knowledge the agent reads via
`list_skills` / `read_skill`:

| Skill file | Coverage |
| --- | --- |
| `memory_hierarchy.md` | Pointer-chasing latency probes, L2 capacity sweep |
| `clock_environment.md` | Actual boost clock via `clock64()`, SM masking checks, API spoof detection |
| `throughput_resources.md` | Global / shared memory bandwidth, bank conflict penalties |
| `gpu_profiling_overview.md` | Tool routing index: which tool to use for which metric family |

## Anti-hacking methodology

The agent does not consult spec-sheet lookup tables. Every measurement uses
live CUDA execution against the active environment:

- **Clock locking**: `nvidia-smi` may lock clocks to arbitrary frequencies.
  The agent measures actual frequency with `clock64()` inside a running kernel,
  not from `cudaDeviceProp` or `nvidia-smi` output.
- **SM masking**: `CUDA_VISIBLE_DEVICES` or driver settings may restrict
  execution. SM count is measured empirically via `launch__sm_count` counter
  or a kernel occupancy sweep.
- **API interception**: `cudaGetDeviceProperties()` may return misleading
  values. Driver API results are treated as secondary evidence; kernel
  measurements take precedence.

## Representative output

The JSON file below shows a sample ncu counter snapshot collected during
development (not the final flat output format):

```json
{
  "dram__bytes_write.sum.per_second": 823058059968.966,
  "dram__bytes_read.sum.per_second":  868745654775.521,
  "device__attribute_max_gpu_frequency_khz": 1394950,
  "launch__sm_count": 82,
  "device__attribute_max_mem_frequency_khz": 9751000,
  "device__attribute_fb_bus_width": 384
}
```

Final `/workspace/output.json` is a flat map of metric names to numeric values,
e.g. `{"dram_latency_cycles": 250, "actual_boost_clock_mhz": 1395}`.

Representative submission output ID: `f6f78d4354e14c8ce2ff58a695e29046`
