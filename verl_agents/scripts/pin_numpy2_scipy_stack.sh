#!/usr/bin/env bash
# 与 PyVision RL + vLLM + numba 兼容的 NumPy/SciPy 栈（修复 numpy1+scipy 的 dtype 递归等问题）
#
# 是否需要执行？
# - 不必每次训练前都跑。当前机器若已按该组合装好依赖，直接 bash run_train.sh / run_pyvision_image_single_gpu.sh 即可。
# - 在「新机器 / 新 conda 环境 / 重装过 numpy scipy 后又出现 RecursionError 或 GenerationMixin 导入失败」时再执行一次。
#
# 用法（在 verl_agents 目录下）: bash scripts/pin_numpy2_scipy_stack.sh

set -euo pipefail
PY="${PYTHON:-/root/miniconda3/bin/python3}"
exec "$PY" -m pip install --upgrade --no-cache-dir \
  "numpy>=2.0.0,<2.1.0" \
  "scipy>=1.14.0,<1.19" \
  "scikit-learn>=1.5.0,<1.7" \
  "tifffile>=2022.8.12"
