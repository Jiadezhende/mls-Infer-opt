#!/usr/bin/env bash
# Phase 3 evaluator entrypoint.
# This file is intended to live at /workspace/run.sh, while the repo lives at
# /workspace/mls-infer-opt/.
set -u

REPO_DIR="/workspace/mls-infer-opt"
TARGET_DIR="/target"

export MLS_TARGET_DIR="$TARGET_DIR"
export MLS_RUNS_DIR="$REPO_DIR/runs"
export MLS_OUTPUT_DIR="/workspace"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 评测外墙约 30min；设 28min 内部硬墙，留 ~2min 给末轮收尾 + finalize 发布 best，
# 避免跑过点被外部 kill（届时只剩兜底 baseline，等于白跑）。
export MLS_TIME_BUDGET_S="${MLS_TIME_BUDGET_S:-1680}"

cd "$REPO_DIR"

# --- LLM 凭证 ---------------------------------------------------------------
# precedence 收敛在 config._merged_env 一处：本仓库 .env 对凭证（OPENAI_API_KEY /
# OPENAI_BASE_URL / MLS_LLM_MODEL 等）逐键权威，环境变量仅在 .env 未设置时兜底。
# 故评测容器注入的孤立 OPENAI_API_KEY（无 OPENAI_BASE_URL）不会与我们 .env 的 base_url
# 错配 401。这里不再 source .env，避免 shell 层与 Python 层两套 precedence 打架。
# .env 不入库（.gitignore），凭证不会落进 git。

# 优先用本仓库 venv 的解释器（editable 安装），否则退回 PATH 上的 python3。
PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

# --- 依赖自举 ---------------------------------------------------------------
# 评测容器（NGC 镜像）已自带 torch(2.3, CUDA) / numpy / pytest；生产代码顶层第三方
# 依赖只有 torch，openai 为 LLM 路径的懒加载可选依赖。这里只补 openai：
#   - 不重装 torch（会拉到非 CUDA 轮子，破坏 GPU）；
#   - 走清华镜像加速；openai 装不上时代码会优雅退回规则兜底，故失败仅告警、不中断评测。
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
"$PYTHON" -m pip install -q --no-input -i "$PIP_MIRROR" "openai>=1.40" 2>/dev/null || \
  echo "[run.sh] warning: openai 安装失败，LLM 路径将退回规则兜底" >&2
# ---------------------------------------------------------------------------

"$PYTHON" -m mls_infer_opt.loop
exit 0
