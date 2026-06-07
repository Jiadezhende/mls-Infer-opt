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

cd "$REPO_DIR"

# 优先用本仓库 venv 的解释器（editable 安装），否则退回 PATH 上的 python3。
PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

"$PYTHON" -m mls_infer_opt.loop
exit 0
