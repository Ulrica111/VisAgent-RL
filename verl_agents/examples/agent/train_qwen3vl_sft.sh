#!/bin/bash
# SFT training script for Qwen3-VL-8B on PyVision-Image-SFT-Data
# Hardware: 2x A800 80G
# Framework: LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory)
#
# Usage:
#   bash train_qwen3vl_sft.sh
#
#   Weights & Biases (optional):
#     export WANDB_API_KEY=xxxxxxxx        # https://wandb.ai/authorize
#     export WANDB_PROJECT=pyvision-sft    # project name in W&B (optional)
#     export RUN_NAME=my-run               # optional run name override
#     REPORT_TO=none bash train_qwen3vl_sft.sh   # disable W&B / TensorBoard
#
#   Data preview noise (optional):
#     export LLAMAFACTORY_QUIET_DATA=1           # 不打印预处理后的一条 training example
#     export LLAMAFACTORY_DATA_PREVIEW_CHARS=800 # 解码预览最大字符数（默认 2000）
#
# Prerequisites:
#   pip install llamafactory wandb  (or clone + pip install -e ".[metrics]")
#   Adjust the path variables below before running.

set -e

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
MODEL_PATH=/root/autodl-tmp/PyVision-RL/base_model/Qwen3-VL-2B-Instruct   # local HF checkpoint
DATA_ROOT=/root/autodl-tmp/PyVision-Image-SFT-Data
DATA_JSON=${DATA_ROOT}/pyvision_image_sft_data.json
IMAGE_ROOT=${DATA_ROOT}/images
OUTPUT_DIR=/root/autodl-tmp/PyVision-RL/ckpt/qwen3vl_2b_sft_ckpt
LLAMAFACTORY_DIR=/root/autodl-tmp/LLaMA-Factory   

# ---------------------------------------------------------------------------
# Experiment logging (LLaMA-Factory / HuggingFace Trainer -> W&B)
# ---------------------------------------------------------------------------
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-qwen3vl_pyvision_sft}"
export WANDB_PROJECT="${WANDB_PROJECT:-pyvision-sft}"
if [[ "${REPORT_TO}" == "wandb" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: REPORT_TO=wandb but WANDB_API_KEY is unset. Set it or use REPORT_TO=none." >&2
fi

if [[ ! -f "${DATA_JSON}" ]]; then
  echo "Error: missing ${DATA_JSON}（应与 dataset_info 中 pyvision_image_sft.file_name 一致）." >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "Error: missing image root ${IMAGE_ROOT}（JSON 里 images 为 mmpr/... 等相对路径，相对此目录）." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Training（默认单卡调试：不启 DeepSpeed；多卡时再取消下面 YAML 里 deepspeed 注释并用 FORCE_TORCHRUN=1）
# 数据集：dataset_info.json 中 pyvision_image_sft；JSON 内 images 如 mmpr/xxx.png 相对 ${DATA_ROOT}/images
# ---------------------------------------------------------------------------
echo "=== Starting SFT training ==="

CONFIG_FILE=$(mktemp /tmp/qwen3vl_sft_XXXXXX.yaml)
trap "rm -f ${CONFIG_FILE}" EXIT

cat > "${CONFIG_FILE}" <<YAML
### model
model_name_or_path: ${MODEL_PATH}

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: pyvision_image_sft
dataset_dir: ${LLAMAFACTORY_DIR}/data
media_dir: ${IMAGE_ROOT}
template: qwen2_vl
cutoff_len: 8192
max_samples: 999999
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 200
save_total_limit: 3
overwrite_output_dir: true
report_to: ${REPORT_TO}
run_name: ${RUN_NAME}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
# Qwen3-VL 视觉塔在 flash_attention_2 + grad checkpoint 下会触发 transformers 对 s_aux=None 的 bug（AttributeError）
flash_attn: sdpa
gradient_checkpointing: true
dataloader_num_workers: 2

### deepspeed — 多卡时再取消注释，并改为: FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli ...
# deepspeed: ${LLAMAFACTORY_DIR}/examples/deepspeed/ds_z2_config.json
YAML

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" llamafactory-cli train "${CONFIG_FILE}"
