#!/bin/bash
set -x

##################################################################################################
#                                        基本配置                                               #
##################################################################################################
PROJECT_NAME="pyvision-rl"
EXPERIMENT_NAME="pyvision-image"

current_time=$(date '+%m%d-%H%M%S')
EXPERIMENT_NAME="${EXPERIMENT_NAME}-${current_time}"

export OUTPUT_BASE_DIR="/root/autodl-tmp/PyVision-RL/ckpt/PyVision-Image-7B-GRPO/${EXPERIMENT_NAME}"

mkdir -p "$OUTPUT_BASE_DIR"
cp "$0" "$OUTPUT_BASE_DIR/"  # 保存本次使用的训练脚本副本

export TMPDIR="$HOME/tmp/ray"
mkdir -p "$TMPDIR"

##################################################################################################
#                                       WandB 配置                                              #
##################################################################################################
export WANDB_MODE=online    # 在线上报到 wandb.ai，需提前 wandb login 或设置 WANDB_API_KEY
export WANDB_RUN_ID=$EXPERIMENT_NAME
export WANDB_RESUME="allow"
export WANDB_DIR=$OUTPUT_BASE_DIR

##################################################################################################
#                                       路径配置                                                #
##################################################################################################
export SAVE_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/ckpt"
ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/rollouts"
FIRST_ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/first_rollouts"

export HYDRA_FULL_ERROR=1
# 勿设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True：vLLM CuMem 与 PyTorch 可扩展段不兼容会 AssertionError

# LLM-as-a-Judge 配置文件（硅基流动 DeepSeek-V3）
export LLM_AS_A_JUDGE_CONFIG_PATH="/root/autodl-tmp/PyVision-RL/judge_config.json"

##################################################################################################
#                                      数据路径配置                                             #
##################################################################################################
# RL 训练数据（PyVision-Image-RL-Data）
PYVISION_IMAGE_RL_DATA="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data_processed.json"
# JSON 里 hint_path 为相对路径（如 deepeyes/xxx.png）时，与此目录拼接
PYVISION_MM_HINT_IMAGE_ROOT="/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data"

# 验证集（暂不使用，若有 vstar 验证集可取消注释并填路径）
# VSTAR_BENCH="/path/to/vstar_pv_form_image_val_dataset.json"

##################################################################################################
#                               SFT 起始模型路径                                                #
##################################################################################################
# 默认：本机 AutoDL 已存在的 Qwen2.5-VL SFT（避免不存在的 /mnt/petrelfs/... 触发 HFValidationError）
# 集群上可：export REF_MODEL_PATH=/mnt/petrelfs/.../sft 后运行本脚本
REF_MODEL_PATH="${REF_MODEL_PATH:-/root/PyVision-Image-7B-SFT}"

##################################################################################################
#                              单卡 A800 80G 参数说明                                           #
##################################################################################################
# 原始配置为 8 卡设计，以下参数已针对单卡 80G 显存调整
# 若仍 OOM：需再降本段长度/多图/像素，或上双卡 split（rollout 与 train 分 GPU）
# wake_up 时 vLLM 会映射 KV+权重，与 FSDP 同卡：max_model_len 与 gpu_mem_util 不能同时过大

gen_batch_size=1
max_video_gen_batch_size=1
gen_batch_size_align_method="up_resample_image"

rollout_num=1
# 同卡时 vLLM 的 KV 预算 = 总显存*util - peak(FSDP+推理剖析)；FSDP 先加载故 peak 大，需较高 util
# 多轮下同一次请求可累计多图，须 >= 实际峰值（常见 5+）；显存紧可酌减
max_vllm_images=16
max_turn=5
max_turn_in_val=10

tool_using_cumulative_reward_per_turn=0.1
concurrent_workers=1

# vLLM v1: available_kv = total*util - peak（含 FSDP）。fsdp_workers 在 init vLLM 前会暂卸载 FSDP（param_offload 时），一般 util 0.6～0.7 即可
# 无 offload 时仍可能需提高 util 或双卡
max_prompt_len=8192
max_response_len=4096
rollout_max_model_len=20480
vllm_gpu_mem_util=0.65
vllm_max_num_batched_tokens=6144

prompt_template_path="./verl_agents/verl/utils/dataset/rl_system_prompt_template.json"
min_pixels=3136
max_pixels=501760

enable_filter_groups=True
std_sort_enable=True
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'
max_num_gen_batches=0

norm_adv_by_std_in_grpo=False
with_mm_hint=True                       # PyVision-Image 用 True

WORLD_SIZE=1

# 只使用训练 JSON 中前 N 条（0=全量）
max_train_samples=1000
max_val_samples=0

TRAIN_DATA_JSON_TOTAL="${PYVISION_IMAGE_RL_DATA}"

echo "============================"
echo "Launched Training: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "From CKPT: ${REF_MODEL_PATH}"
echo "Datasets: ${TRAIN_DATA_JSON_TOTAL} (max_train_samples=${max_train_samples})"
echo "Output: ${OUTPUT_BASE_DIR}"
echo "============================"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${_VERL_AGENTS_ROOT}/logs"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${TRAIN_DATA_JSON_TOTAL}] \
    data.val_files=[] \
    data.max_train_samples=${max_train_samples} \
    data.max_val_samples=${max_val_samples} \
    data.train_batch_size=1 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_num_batched_tokens} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_mem_util} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
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
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=100000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=5 2>&1 | tee "${_VERL_AGENTS_ROOT}/logs/${EXPERIMENT_NAME}.log"
