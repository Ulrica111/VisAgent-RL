#!/usr/bin/env bash
# 启动合并后的 PyVision-RL HF checkpoint 供 memory_agent / OpenAI 兼容客户端使用。
#
# 不要直接用: python -m vllm.entrypoints.openai.api_server
# 那样默认 multiprocessing=fork，易触发:
#   RuntimeError: Cannot re-initialize CUDA in forked subprocess
#
# 推荐: vllm CLI（会设置 VLLM_WORKER_MULTIPROC_METHOD=spawn），或下面显式 export。

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/PyVision-RL/ckpt/PyVision-Image-7B-GRPO/pyvision-rl-hf}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-pyvision-rl}"

exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --max-model-len 4096 \
  "$@"
