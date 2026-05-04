#!/bin/bash
set -x

##################################################################################################
#  双卡训练（2× GPU）：vLLM 与 FSDP 各以 TP/分片 使用两卡，避免单卡 FSDP + vLLM wake_up OOM
#  要求：real_train_batch_size = train_batch_size * rollout.n 能被 GPU 数整除（2 卡、n=1 时 train_batch_size 取偶数即可）
##################################################################################################
PROJECT_NAME="pyvision-rl"
EXPERIMENT_NAME="pyvision-image"

current_time=$(date '+%m%d-%H%M%S')
EXPERIMENT_NAME="${EXPERIMENT_NAME}-${current_time}"

# 输出根目录：data 盘下按实验名分子目录。覆盖默认：export OUTPUT_BASE_DIR=/your/path/${EXPERIMENT_NAME}
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/root/autodl-tmp/PyVision-RL/ckpt/PyVision-Image-7B-GRPO/${EXPERIMENT_NAME}}"

mkdir -p "$OUTPUT_BASE_DIR"
cp "$0" "$OUTPUT_BASE_DIR/"

# 7B+optimizer+hf 常需数十 GB；空间不足时清理本目录下旧实验或换 OUTPUT_BASE_DIR
echo "[disk] output mount:" && df -h "$OUTPUT_BASE_DIR" 2>/dev/null | tail -1

export TMPDIR="$HOME/tmp/ray"
mkdir -p "$TMPDIR"

export WANDB_MODE=online
export WANDB_RUN_ID=$EXPERIMENT_NAME
export WANDB_RESUME="allow"
export WANDB_DIR=$OUTPUT_BASE_DIR

SAVE_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/ckpt"
ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/rollouts"
FIRST_ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/first_rollouts"
VAL_METRICS_JSON_DIR="${OUTPUT_BASE_DIR}/val_metrics"

export HYDRA_FULL_ERROR=1

export LLM_AS_A_JUDGE_CONFIG_PATH="/root/autodl-tmp/PyVision-RL/judge_config.json"

PYVISION_IMAGE_RL_DATA="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data_processed.json"
PYVISION_MM_HINT_IMAGE_ROOT="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data"

REF_MODEL_PATH="${REF_MODEL_PATH:-/root/PyVision-Image-7B-SFT}"

# 与 train_batch 对齐；2 卡要求 (train_batch_size * rollout.n) % 2 == 0
gen_batch_size=8
max_video_gen_batch_size=8
gen_batch_size_align_method="up_resample_image"
train_batch_size=8
ppo_mini_batch_size=8

rollout_num=1
# 双卡仍 OOM 时主要调：rollout_max_model_len（KV 与 cumem 池）、vllm_gpu_mem_util、max_num_batched_tokens
# limit 过大时 vLLM 侧 MM/KV 预算偏高易二次 wake OOM；实测峰值约 5 张，取 8 留余量（不足再调到 10～12，勿一次拉满 16）
max_vllm_images=8
max_turn=5
max_turn_in_val=10

tool_using_cumulative_reward_per_turn=0.1
concurrent_workers=1

max_prompt_len=8192
max_response_len=4096
# 第一步训练后 FSDP+碎片下第二次 wake_up 易 OOM；须压低 max_model_len 与显存池比例（仍 < 8k+4k 的极限长对话时可能截断，可再微调到 14k+）
rollout_max_model_len=12800
vllm_gpu_mem_util=0.35
vllm_max_num_batched_tokens=3072

prompt_template_path="./verl_agents/verl/utils/dataset/rl_system_prompt_template.json"
min_pixels=3136
max_pixels=501760

enable_filter_groups=True
std_sort_enable=True
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'
max_num_gen_batches=0

norm_adv_by_std_in_grpo=False
with_mm_hint=True

NGPUS=2
WORLD_SIZE=1
total_epochs=1

# 只使用训练 JSON 中前 N 条（按数组顺序；加载后再 shuffle）。0 或留空=全量
max_train_samples=1000
max_val_samples=0

TRAIN_DATA_JSON_TOTAL="${PYVISION_IMAGE_RL_DATA}"

echo "============================"
echo "2-GPU training: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "From CKPT: ${REF_MODEL_PATH}"
echo "Datasets: ${TRAIN_DATA_JSON_TOTAL} (max_train_samples=${max_train_samples})"
echo "Output: ${OUTPUT_BASE_DIR}"
echo "============================"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${_VERL_AGENTS_ROOT}/logs"

# 主进程用满可见的两张卡（需 export CUDA_VISIBLE_DEVICES=0,1 或作业分配好）
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${TRAIN_DATA_JSON_TOTAL}] \
    data.val_files=[] \
    data.max_train_samples=${max_train_samples} \
    data.max_val_samples=${max_val_samples} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_response_len} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.with_mm_hint=${with_mm_hint} \
    data.mm_hint_image_root=${PYVISION_MM_HINT_IMAGE_ROOT} \
    data.min_pixels=${min_pixels} \
    data.max_pixels=${max_pixels} \
    +data.prompt_template_path=${prompt_template_path} \
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
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=100000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    +trainer.val_metrics_json_dir=${VAL_METRICS_JSON_DIR} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=${total_epochs} 2>&1 | tee "${_VERL_AGENTS_ROOT}/logs/${EXPERIMENT_NAME}.log"
