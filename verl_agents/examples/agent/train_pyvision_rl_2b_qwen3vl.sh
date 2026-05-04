#!/bin/bash
##################################################################################################
# Qwen3-VL-2B RL（GRPO）：配置尽量与已跑通的 run_pyvision_image_2gpu.sh 对齐。
# - 对照脚本基于 Qwen2.5-VL-7B + 双卡；本脚本为 2B 基座与 checkpoint，长度/显存可调得更松。
# - 必须从 verl_agents 包根目录调用（与双卡脚本相同），例如：
#     cd /root/autodl-tmp/PyVision-RL/verl_agents
#     bash examples/agent/train_pyvision_rl_2b_qwen3vl.sh
##################################################################################################

set -x

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERL_AGENTS_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${_VERL_AGENTS_ROOT}/logs"

# NaiveRewardManager：JSON + 可选内网 BASE 覆盖 base_url（见 naive.py）
export LLM_AS_A_JUDGE_CONFIG_PATH="${LLM_AS_A_JUDGE_CONFIG_PATH:-/root/autodl-tmp/PyVision-RL/judge_config.json}"
# 若仅用 judge_config.json 里的 base_url，可注释下行。
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://10.140.66.34:18901/v1}"
export no_proxy="${no_proxy:-10.140.66.34:18901}"
export NO_PROXY_IP="${NO_PROXY_IP:-10.140.66.34:18901}"

export TMPDIR="${TMPDIR:-$HOME/tmp/ray}"
mkdir -p "$TMPDIR"

export HYDRA_FULL_ERROR=1

# 多模态对齐诊断（仅排错时开启；平时勿设，否则大量 WARNING 刷屏）
# export VERL_QWEN3_MM_DEBUG=1
# 单条轨迹里图很多时（如 mm_hint 8 张）序列会极长、首步很慢或显存顶满，可逐项收紧：
#   max_pixels（如 358400）、max_vllm_images（如 4）、gen_batch_size（如 8）、train_batch_size（如 8）

PROJECT_NAME="${PROJECT_NAME:-pyvision-rl-qwen3vl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl-2b-pyvision-grpo}"

# 续跑同一目录：export RESUME_RUN=1 且 export EXPERIMENT_NAME=完整目录名（如 ...-0504-140619），不再追加时间戳；
# 或直接 export OUTPUT_BASE_DIR=.../ckpt/qwen3vl_2b_rl_runs/<同名目录>（仍会拼接 EXPERIMENT_NAME，故通常需二者一致）。
if [[ "${RESUME_RUN:-0}" == "1" ]]; then
    if [[ -z "${EXPERIMENT_NAME:-}" ]] || [[ "$EXPERIMENT_NAME" == "qwen3vl-2b-pyvision-grpo" ]]; then
        echo "ERROR: RESUME_RUN=1 时请 export EXPERIMENT_NAME=已有实验完整名称（含日期时间后缀）。" >&2
        exit 1
    fi
    echo "[resume] RESUME_RUN=1, EXPERIMENT_NAME=${EXPERIMENT_NAME}（不追加时间戳）"
else
    current_time="$(date '+%m%d-%H%M%S')"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-${current_time}"
fi

# 对齐双卡脚本：OUTPUT_BASE_DIR 下挂 ckpt / rollouts / first_rollouts
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/root/autodl-tmp/PyVision-RL/ckpt/qwen3vl_2b_rl_runs}/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_BASE_DIR"
cp "$0" "$OUTPUT_BASE_DIR/" 2>/dev/null || true

echo "[disk] output:" && df -h "$OUTPUT_BASE_DIR" 2>/dev/null | tail -1

SAVE_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/ckpt"
ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/rollouts"
FIRST_ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/first_rollouts"
# 首次正常训练且配置 the_first_batch_rollout_data_dir 时，会在其子目录下写入 debug_train_batch.pkl（完整 DataProto）。
# 之后可跳过耗时的 rollout，只复现 old_log_prob/PPO 更新，例如额外传：
#   trainer.debug_train_load_rollout_path=/绝对路径/.../first_rollouts/项目名/实验名/debug_train_batch.pkl
VAL_METRICS_JSON_DIR="${OUTPUT_BASE_DIR}/val_metrics"
mkdir -p "$SAVE_CHECKPOINT_DIR" "$ROLLOUT_SAVE_DIR_PATH" "$FIRST_ROLLOUT_SAVE_DIR_PATH" "$VAL_METRICS_JSON_DIR"

# W&B（不需要可 export WANDB_MODE=offline）
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$EXPERIMENT_NAME}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_DIR="$OUTPUT_BASE_DIR"

# ---------------------------------------------------------------------------
# 数据路径（可与双卡脚本相同；默认用小子集便于试跑）
# ---------------------------------------------------------------------------
PYVISION_RL_TRAIN_JSON="${PYVISION_RL_TRAIN_JSON:-/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data_processed_subset2000.json}"
PYVISION_MM_HINT_IMAGE_ROOT="${PYVISION_MM_HINT_IMAGE_ROOT:-/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data}"

# 与 run_pyvision_image_2gpu 一致：相对 train_batch 的生成批大小与对齐方式
gen_batch_size="${gen_batch_size:-16}"
max_video_gen_batch_size="${max_video_gen_batch_size:-16}"
gen_batch_size_align_method="${gen_batch_size_align_method:-up_resample_image}"

train_batch_size="${train_batch_size:-16}"
ppo_mini_batch_size="${ppo_mini_batch_size:-16}"

# 约束：real_train_batch_size = train_batch_size * rollout.n 必须能被 (NGPUS * nnodes) 整除
: "${WORLD_SIZE:=1}"
# NGPUS 未设置或为空时按 nvidia-smi 探测本机 GPU 数（避免 1 卡环境仍申请 2 卡）。需要双卡时请 export NGPUS=2
if [[ -z "${NGPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
    [[ -z "$NGPUS" || "$NGPUS" == "0" ]] && NGPUS=1
  else
    NGPUS=1
  fi
fi
echo "[cuda] using NGPUS=${NGPUS} (set NGPUS explicitly to override)"
rollout_num="${rollout_num:-1}"

# 2B 可比 7B 双卡脚本略放开；显存紧时按 run_pyvision_image_2gpu 注释调低
max_prompt_len="${max_prompt_len:-12288}"
max_response_len="${max_response_len:-6144}"
rollout_max_model_len="${rollout_max_model_len:-18432}"
vllm_gpu_mem_util="${vllm_gpu_mem_util:-0.65}"
vllm_max_num_batched_tokens="${vllm_max_num_batched_tokens:-8192}"

max_vllm_images="${max_vllm_images:-8}"
max_turn="${max_turn:-5}"
max_turn_in_val="${max_turn_in_val:-10}"
tool_using_cumulative_reward_per_turn="${tool_using_cumulative_reward_per_turn:-0.1}"
concurrent_workers="${concurrent_workers:-1}"

PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-${_VERL_AGENTS_ROOT}/verl/utils/dataset/rl_system_prompt_template.json}"
min_pixels="${min_pixels:-3136}"
max_pixels="${max_pixels:-501760}"

enable_filter_groups="${enable_filter_groups:-True}"
std_sort_enable="${std_sort_enable:-True}"
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'
max_num_gen_batches="${max_num_gen_batches:-0}"
norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo:-False}"
with_mm_hint="${with_mm_hint:-True}"

max_train_samples="${max_train_samples:-0}"
max_val_samples="${max_val_samples:-0}"

# Qwen3-VL-2B SFT checkpoint
REF_MODEL_PATH="${REF_MODEL_PATH:-/root/autodl-tmp/PyVision-RL/ckpt/qwen3vl_2b_sft_ckpt/checkpoint-860}"

total_epochs="${total_epochs:-1}"

# Debug：跳过 rollout，直接加载已保存的 DataProto（见 trainer.debug_train_load_rollout_path）
DEBUG_TRAIN_BATCH="${DEBUG_TRAIN_BATCH:-}"
DEBUG_EXTRA_HYDRA=()
if [[ -n "$DEBUG_TRAIN_BATCH" ]]; then
    if [[ ! -f "$DEBUG_TRAIN_BATCH" ]]; then
        echo "ERROR: DEBUG_TRAIN_BATCH is not a file: $DEBUG_TRAIN_BATCH" >&2
        exit 1
    fi
    DEBUG_EXTRA_HYDRA+=(trainer.debug_train_load_rollout_path="${DEBUG_TRAIN_BATCH}")
    echo "[debug] Will load batch from: ${DEBUG_TRAIN_BATCH}"
fi
# 可选：只跑少数 optimizer step（例：export DEBUG_TRAINER_TOTAL_TRAINING_STEPS=1）
if [[ -n "${DEBUG_TRAINER_TOTAL_TRAINING_STEPS:-}" ]]; then
    DEBUG_EXTRA_HYDRA+=(trainer.total_training_steps="${DEBUG_TRAINER_TOTAL_TRAINING_STEPS}")
    echo "[debug] trainer.total_training_steps=${DEBUG_TRAINER_TOTAL_TRAINING_STEPS}"
fi

echo "============================"
echo "Qwen3-VL-2B RL (${NGPUS} GPU(s)/node × ${WORLD_SIZE} node)"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Aligned with: run_pyvision_image_2gpu.sh (Qwen2.5-VL-7B ref)"
echo "From CKPT: ${REF_MODEL_PATH}"
echo "Data: ${PYVISION_RL_TRAIN_JSON}"
echo "Output: ${OUTPUT_BASE_DIR}"
echo "============================"

# 须在 verl_agents 根下执行以保证 python -m verl...
cd "${_VERL_AGENTS_ROOT}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=["${PYVISION_RL_TRAIN_JSON}"] \
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
    trainer.val_metrics_json_dir="${OUTPUT_BASE_DIR}/val_metrics" \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=${total_epochs} \
    "${DEBUG_EXTRA_HYDRA[@]}" \
    2>&1 | tee "${_VERL_AGENTS_ROOT}/logs/${EXPERIMENT_NAME}.log"
