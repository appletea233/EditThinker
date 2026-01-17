RUN_NAME=edit_thinker_qwen3vl_rl
# Please replace with your own paths
SAVE_DIR=/path/to/your/checkpoints  # Replace with your checkpoint save directory
PROJECT_PATH=/path/to/Edit-Thinker  # Replace with your project path

# Please replace with your own conda environment
source /path/to/conda/bin/activate /path/to/your/conda/env  # Replace with your conda environment path

# 禁用 NCCL 的 GPU P2P 通信
# export NCCL_P2P_DISABLE=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_CONSOLE=off

# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export TQDM_DISABLE=1                 # 全局禁用 tqdm
export TQDM_USE_WRITELN=0             # 防止用 '\r' 回车式刷新
export VLLM_USE_V1=1

# Ray 日志（把后端日志降到错误级别，并避免回显到 stderr）
# export RAY_BACKEND_LOG_LEVEL=ERROR
# export RAY_LOG_TO_STDERR=0
# export TRANSFORMERS_VERBOSITY=ERROR


WORKING_DIR=${PROJECT_PATH}/train/EasyR1
cd $WORKING_DIR
CURRPWD=${WORKING_DIR}

project_name=EasyR1-qwen3-vl
exp_name=edit_thinker_qwen3vl_rl

# Please replace with your SFT checkpoint path
MODEL_PATH=/path/to/your/sft/checkpoint  # Replace with your SFT model checkpoint path
TRAIN_FILE=${PROJECT_PATH}/data/rl_train.jsonl
TEST_FILE=$TRAIN_FILE
IMAGE_DIR=${PROJECT_PATH}/data

ROLLOUT_BS=16
GLOBAL_BS=8
MB_PER_UPDATE=1
MB_PER_EXP=1
TP_SIZE=4
N_GPUS_PER_NODE=8
NNODES=1

export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH:-}"

which python3

python3 -m verl.trainer.main \
  config=${WORKING_DIR}/examples/edit_thinker_config.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.image_dir="${IMAGE_DIR}" \
  data.format_prompt=${PROJECT_PATH}/inference/prompt_template.txt \
  data.rollout_batch_size="${ROLLOUT_BS}" \
  data.answer_key="evaluation" \
  worker.actor.global_batch_size="${GLOBAL_BS}" \
  worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.strategy=adamw_bf16 \
  worker.actor.optim.lr=2e-6 \
  worker.rollout.tensor_parallel_size="${TP_SIZE}" \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=50 \
  trainer.save_checkpoint_path="${SAVE_DIR}" \
  worker.reward.reward_function=${WORKING_DIR}/verl/reward_function/edit_thinker_reward.py:compute_score