#!/usr/bin/env bash
# 阶段 A 入口：评测系统先跑本脚本，跑完后只 import workspace/engine.py。
# 进程外壳在 loop/__main__.py：始终 exit 0，且保证 workspace/engine.py 必在盘上。
set -u
cd "$(dirname "$0")"

# 优先用本仓库 venv 的解释器（editable 安装），否则退回 PATH 上的 python3。
PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

"$PYTHON" -m mls_infer_opt.loop
exit 0
