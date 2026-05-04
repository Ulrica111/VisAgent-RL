#!/bin/bash
# 仅验证：在 PyVision processed 验证集上 rollout + 打分后退出（不训练）。
# 与 EVALUATION.md 中 eval_vstar.py（独立 vLLM 服务 + V* 数据）不同，本脚本走 verl 训练同源的 agent 与 reward。
#
# 用法（在 verl_agents 目录下）：
#   # 从已有 GRPO FSDP checkpoint 恢复再验证（默认）
#   bash eval/run_pyvision_val_only_2gpu.sh
#   # 仅 SFT 起点（不加载 GRPO，actor 权重即 actor_rollout_ref.model.path）
#   VAL_SFT_BASELINE=1 bash eval/run_pyvision_val_only_2gpu.sh
set -x

PROJECT_NAME="pyvision-rl"

# GRPO 实验的 default_local_dir（含 global_step_*，与训练时一致）
GRPO_CKPT_PARENT="/root/autodl-tmp/PyVision-RL/ckpt/PyVision-Image-7B-GRPO/pyvision-image-0426-222451/ckpt/pyvision-rl/pyvision-image-0426-222451"
# 注意：FSDP 存盘时 actor/huggingface 仅含 config + tokenizer/processor（无 model.safetensors），不能作 model.path。
# model.path 必须是含完整权重的 HF 目录（通常为训练用 SFT）；RL 权重由 resume=auto 时加载 FSDP 分片覆盖。

if [ "${VAL_SFT_BASELINE:-0}" = "1" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME_SFT:-pyvision-sft-baseline-val-only}"
  RESUME_MODE="disable"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME_GRPO:-pyvision-image-0426-222451-val-only}"
  RESUME_MODE="auto"
fi

export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/root/autodl-tmp/PyVision-RL/ckpt/PyVision-Image-7B-GRPO/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_BASE_DIR"

export TMPDIR="$HOME/tmp/ray"
mkdir -p "$TMPDIR"

export HYDRA_FULL_ERROR=1
export LLM_AS_A_JUDGE_CONFIG_PATH="/root/autodl-tmp/PyVision-RL/judge_config.json"

# processed 验证集（含 prompt / mm_hint）；可由原始 val + transfer_to_rl_form_image_w_mm_hint 生成
VAL_JSON="/root/autodl-tmp/PyVision-Image-RL-Data/vstar_pv_form_image_val_dataset_processed.json"
TRAIN_JSON="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data_processed.json"
PYVISION_MM_HINT_IMAGE_ROOT="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data"

# 含完整权重的 HF 目录（须真实存在，否则会误走 Hub 校验报 HFValidationError）
REF_MODEL_PATH="${REF_MODEL_PATH:-/root/PyVision-Image-7B-SFT}"

NGPUS=2
WORLD_SIZE=1

gen_batch_size=8
max_video_gen_batch_size=8
gen_batch_size_align_method="up_resample_image"
train_batch_size=8
ppo_mini_batch_size=8
rollout_num=1
max_vllm_images=8
max_turn=5
max_turn_in_val=10
tool_using_cumulative_reward_per_turn=0.1
concurrent_workers=1
max_prompt_len=8192
max_response_len=4096
rollout_max_model_len=12800
vllm_gpu_mem_util=0.35
vllm_max_num_batched_tokens=3072
min_pixels=3136
max_pixels=501760

enable_filter_groups=True
std_sort_enable=True
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'
max_num_gen_batches=0
norm_adv_by_std_in_grpo=False
with_mm_hint=True

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
PROMPT_TEMPLATE_PATH="${_VERL_AGENTS_ROOT}/verl/utils/dataset/rl_system_prompt_template.json"

SAVE_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/ckpt"
ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/rollouts"
FIRST_ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/first_rollouts"
VAL_METRICS_JSON_DIR="${OUTPUT_BASE_DIR}/val_metrics"

if [ "${VAL_SFT_BASELINE:-0}" = "1" ]; then
  DEFAULT_LOCAL_DIR="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
  mkdir -p "${DEFAULT_LOCAL_DIR}"
else
  DEFAULT_LOCAL_DIR="${GRPO_CKPT_PARENT}"
fi

mkdir -p "${_VERL_AGENTS_ROOT}/logs"

cd "${_VERL_AGENTS_ROOT}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${TRAIN_JSON}] \
    data.val_files=[${VAL_JSON}] \
    data.max_train_samples=8 \
    data.max_val_samples=0 \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_response_len} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.with_mm_hint=${with_mm_hint} \
    data.mm_hint_image_root=${PYVISION_MM_HINT_IMAGE_ROOT} \
    data.min_pixels=${min_pixels} \
    data.max_pixels=${max_pixels} \
    +data.prompt_template_path=${PROMPT_TEMPLATE_PATH} \
    +data.gen_batch_size=${gen_batch_size} \
    +data.max_video_gen_batch_size=${max_video_gen_batch_size} \
    +data.gen_batch_size_align_method=${gen_batch_size_align_method} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.std_sort_enable=${std_sort_enable} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=[${filter_groups_metric}] \
    +algorithm.filter_groups.end_reason_filter_reserve_names=[${end_reason_filter_reserve_names}] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_num_batched_tokens} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_mem_util} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.max_vllm_images=${max_vllm_images} \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=${max_response_len} \
    actor_rollout_ref.rollout.agent.max_turns=${max_turn} \
    actor_rollout_ref.rollout.agent.concurrent_workers=${concurrent_workers} \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn=${tool_using_cumulative_reward_per_turn} \
    +actor_rollout_ref.rollout.val_kwargs.max_turn_in_val=${max_turn_in_val} \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.the_first_batch_rollout_data_dir=${FIRST_ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    +trainer.val_metrics_json_dir=${VAL_METRICS_JSON_DIR} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 \
    2>&1 | tee "${_VERL_AGENTS_ROOT}/logs/${EXPERIMENT_NAME}.log"
