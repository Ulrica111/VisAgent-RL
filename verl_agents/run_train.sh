#!/bin/bash
#SBATCH --job-name=pv-rl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=eaigc1_t
#SBATCH --quotatype=reserved

# export LLM_AS_A_JUDGE_BASE="http://10.140.60.133:18901/v1" # 10-140-1-174
# export no_proxy='10.140.60.133:18901'
export LLM_AS_A_JUDGE_CONFIG_PATH="/root/autodl-tmp/PyVision-RL/judge_config.json"

# 节点数（单机多卡保持 1）
export WORLD_SIZE=1

# 7B：双卡见 run_pyvision_image_2gpu.sh（避免单卡 FSDP + vLLM OOM）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
bash /root/autodl-tmp/PyVision-RL/verl_agents/examples/agent/run_pyvision_image_2gpu.sh
# 单卡可改用：bash .../run_pyvision_image_single_gpu.sh

# config for 32B
# bash examples/agent/final_merged_v1v8_thinklite_32b.sh