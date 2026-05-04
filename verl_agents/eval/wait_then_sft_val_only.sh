#!/bin/bash
# 等待给定 PID 退出后，再跑 SFT 起点的 val-only（与当前验证集、脚本参数一致）。
# 用法（在 verl_agents 下）：
#   bash eval/wait_then_sft_val_only.sh 89458
# 不传 PID 则立即跑 SFT：
#   bash eval/wait_then_sft_val_only.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

if [ -n "${1:-}" ]; then
  _WAIT_PID="$1"
  echo "[wait_then_sft] 等待 PID ${_WAIT_PID} 结束（当前 GRPO 验证）..."
  while kill -0 "${_WAIT_PID}" 2>/dev/null; do sleep 30; done
  echo "[wait_then_sft] PID ${_WAIT_PID} 已结束，开始 SFT val-only"
fi

cd "${_VERL_AGENTS_ROOT}"
export VAL_SFT_BASELINE=1
bash eval/run_pyvision_val_only_2gpu.sh 2>&1 | tee "${_VERL_AGENTS_ROOT}/logs/pyvision-sft-val-only-after-grpo.log"
