# Evaluation for DeepEyes

We provide a evaluation demo for assess your model on V* benchmark with the bbox processing. 

## Evaluating Model
You can use the `eval_vstar.py` to evalate the model with the auto bbox processing. It is worth noting that we firstly deploy model using VLLM. If you want to use transformers to implement your model, you should modify the code and the evaluation process will be slow.

Here is a sample of the evaluation command：
```
python eval_vstar.py \
    --model_name MODEL_NAME \
    --api_key API_KEY \
    --api_url API_URL\
    --vstar_bench_path PATH_TO_VSTAR \
    --save_path PATH_TO_SAVE_DIR \
    --eval_model_name MODEL_NAME_VLLM \
    --num_workers NUM_WORKERS
```
`MODEL_NAME` is the name of saving, and the evaluation results will be saved at `PATH_TO_SAVE_DIR/MODEL_NAME`. `MODEL_NAME_VLLM` is the name of VLLM server. you can set `MODEL_NAME_VLLM` as None, and will be detected automatically. `API_URL` is the VLLM server port, such as 'http://10.39.19.140:8000/v1'.


## Score Calculate
We use the combination of ruled-based evaluation and llm-judge assessment to calculate score. You can use the following command to calculate your results:

```
python judge_result.py \
    --model_name MODEL_NAME \
    --api_key API_KEY \
    --api_url API_URL\
    --vstar_bench_path PATH_TO_VSTAR \
    --save_path PATH_TO_SAVE_DIR \
    --eval_model_name MODEL_NAME_VLLM \
    --num_workers NUM_WORKERS
```
We use Qwen2.5 72B deployed by VLLM as judge model, so `API_URL` is the address of judge model VLLM server. 


## Visualization
We also provide the code `watch_demo.ipynb` to visualize the result. You should modify the `root_path` to the V* bench path and `json_path` to the result jsonl path. Desides, you can modify `line_id` or `tosee_img` to change the case to be visualized.  

## Evaluate HRBench
The evaluation of HRBench is similar to that of V*.

## PyVision-Image RL：从训练 checkpoint 做验证（verl `val_only`）

仓库内 **`eval_vstar.py`** 面向 **单独拉起 vLLM 服务 + V\* 官方数据** 的评测流程（见上文）。

若你要在 **与 GRPO 训练完全一致的环境**（FSDP actor、同 agent、同 `judge_config`）下，对 **自建的 PyVision 格式验证集** 跑一遍指标，请使用：

```bash
cd /path/to/verl_agents
bash eval/run_pyvision_val_only_2gpu.sh
```

**前置条件简述：**

- 验证集须为与训练相同的 **processed JSON**（含 `prompt`、`mm_hint` 等）。可先用 `verl.utils.dataset.rl_dataset.transfer_to_rl_form_image_w_mm_hint` 把原始字段（`question` / `answer` / `image_path`）转成 processed；脚本内默认路径为  
  `PyVision-Image-RL-Data/vstar_pv_form_image_val_dataset_processed.json`（可按需改脚本变量 `VAL_JSON`）。
- **`CKPT_PARENT`**：指向训练时 `trainer.default_local_dir`（内含 `global_step_*` 与 `latest_checkpointed_iteration.txt`）。脚本默认对接 `pyvision-image-0426-222451` 的 step 120；换实验时请改脚本中的 `CKPT_PARENT` / `EXPERIMENT_NAME`。
- 需要与保存 checkpoint 时 **相同 GPU 数**（默认 2 卡）；并设置 **`LLM_AS_A_JUDGE_CONFIG_PATH`**（与训练一致）。
- **`actor_rollout_ref.model.path`**：须为 **含完整权重** 的 HF 目录（`model.safetensors` 或 `pytorch_model.bin` 等），用于 `from_pretrained` 建图；一般为 **SFT 起点**。当前 verl FSDP 保存的 **`global_step_*/actor/huggingface` 只有 config + tokenizer，没有权重文件**，不能直接当 `model.path`。**`resume_mode=auto`** 会在初始化后再加载 **FSDP 分片**，得到 RL 训练后的参数。
- 日志：`verl_agents/logs/<EXPERIMENT_NAME>.log`；指标在终端中的 `validation metric dict` / `data src2acc`。
