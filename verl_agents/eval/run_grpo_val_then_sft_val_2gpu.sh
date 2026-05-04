#!/bin/bash
# 先 GRPO checkpoint 验证，再 SFT 起点验证（同一验证集与脚本参数）。
# GRPO 失败则不会启动 SFT（需 pipefail，勿用裸 `cmd | tee` 链）。
#
# 用法（在 verl_agents 目录下）：
#   bash eval/run_grpo_val_then_sft_val_2gpu.sh
#   nohup bash eval/run_grpo_val_then_sft_val_2gpu.sh >> logs/grpo_then_sft_chain.log 2>&1 &
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${_VERL_AGENTS_ROOT}"
mkdir -p logs

echo "========== 1/2 GRPO 微调 checkpoint 验证 =========="
bash eval/run_pyvision_val_only_2gpu.sh 2>&1 | tee logs/pyvision-grpo-val-only-chain.log

echo "========== 2/2 SFT 起点验证 =========="
VAL_SFT_BASELINE=1 bash eval/run_pyvision_val_only_2gpu.sh 2>&1 | tee logs/pyvision-sft-val-only-after-grpo-chain.log

echo "========== 全部完成 =========="
