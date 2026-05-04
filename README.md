<div align="center">

## VisAgent-RL: Agentic Visual Reasoning with Qwen3-VL via RL

</div>

## Overview

**VisAgent-RL** is a reproduction and extension of [PyVision-RL](https://github.com/agents-x-project/PyVision-RL) that replaces the Qwen2.5-VL-7B backbone with **Qwen3-VL-8B**. The project covers the full training pipeline: supervised fine-tuning (SFT) followed by reinforcement learning (RL) via GRPO, built on top of the [verl](https://github.com/volcengine/verl) framework.

The core RL mechanism inherits PyVision-RL's oversampling–filtering–ranking rollout strategy and accumulative tool reward, adapted for the Qwen3-VL architecture.

## Contents

- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Training](#training)
  - [Stage 1: SFT](#stage-1-sft)
  - [Stage 2: RL (GRPO)](#stage-2-rl-grpo)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Data Preparation

This project uses the datasets from PyVision-RL:

| Split | Dataset |
|-------|---------|
| SFT   | [Agents-X/PyVision-Image-SFT-Data](https://huggingface.co/datasets/Agents-X/PyVision-Image-SFT-Data) |
| RL    | [Agents-X/PyVision-Image-RL-Data](https://huggingface.co/datasets/Agents-X/PyVision-Image-RL-Data) |

### Preprocess RL data

The RL dataset must be converted into the verl-compatible format before training. Update the paths inside `preprocess_rl_data.py` to match your local setup, then run:

```bash
python preprocess_rl_data.py
```

This calls `transfer_to_rl_form_image_w_mm_hint` to inject the system prompt template and outputs a processed JSON file (e.g. `pyvision_image_rl_data_processed.json`) that the RL training script reads.

## Installation

Follow the environment setup from [DeepEyes](https://github.com/Visual-Agent/DeepEyes).

Key version requirements:
```
transformers==4.54.0
vllm==0.9.1
```

Install interaction environment dependencies:
```bash
pip install -r pv_requirements.txt
```

Install the verl package:
```bash
cd verl_agents
pip install -e .
```

### LLM-as-a-Judge setup

The RL reward uses an external LLM judge (e.g. Qwen2.5-72B-Instruct). Configure its endpoint:

```json
// judge_config.json
{
    "api_key": "[EMPTY]",
    "base_url": "http://<your-server-ip>:<port>/v1",
    "model_name": "<model-name>"
}
```

Test the connection:
```bash
cd verl_agents
python test_call_qwen_serve.py
```

## Training

### Stage 1: SFT

SFT uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for full fine-tuning on the PyVision-Image SFT dataset.

Update the path variables at the top of the script, then run:

```bash
bash verl_agents/examples/agent/train_qwen3vl_sft.sh
```

Key configuration (editable inside the script):

| Parameter | Default |
|-----------|---------|
| Base model | `Qwen3-VL-8B-Instruct` |
| Batch size per device | 1 |
| Gradient accumulation | 8 |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Flash attention | sdpa (avoids Qwen3-VL grad-checkpoint bug) |

Multi-GPU: uncomment the `deepspeed` line in the script and prefix with `FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1`.

### Stage 2: RL (GRPO)

Make sure the RL data has been preprocessed (see [Data Preparation](#data-preparation)) and the judge service is running.

```bash
cd verl_agents
bash examples/agent/train_pyvision_rl_2b_qwen3vl.sh
```

The script auto-detects available GPUs. Key environment variables you can override:

```bash
# Point to your SFT checkpoint
export REF_MODEL_PATH=/path/to/qwen3vl_2b_sft_ckpt/checkpoint-860

# Point to preprocessed RL data
export PYVISION_RL_TRAIN_JSON=/path/to/pyvision_image_rl_data_processed.json

# Resume an existing run (skip timestamp suffix)
export RESUME_RUN=1
export EXPERIMENT_NAME=qwen3vl-2b-pyvision-grpo-0504-140619

# Multi-GPU
export NGPUS=2
```

Key RL hyperparameters (defaults in the script):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `rollout_num` | 1 | Samples per prompt |
| `max_turn` | 5 | Max agent turns per rollout |
| `train_batch_size` | 16 | |
| `max_prompt_len` | 12288 | |
| `max_response_len` | 6144 | |
| `vllm_gpu_mem_util` | 0.65 | Reduce if OOM |
| `tool_using_cumulative_reward_per_turn` | 0.1 | Accumulative tool reward |

Logs are written to `verl_agents/logs/<experiment_name>.log`.

## Evaluation

```bash
bash verl_agents/scripts/run_merge.sh   # merge FSDP shards to HF format
```

For benchmark evaluation, see [PyVision-RL-Eval](https://github.com/agents-x-project/PyVision-RL-Eval).

## Acknowledgements

This project is built on top of:

- [PyVision-RL](https://github.com/agents-x-project/PyVision-RL) — original framework, datasets, and RL training design
- [verl](https://github.com/volcengine/verl) — distributed RL training infrastructure
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — base multimodal model
