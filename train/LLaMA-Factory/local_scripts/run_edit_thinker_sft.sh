RUN_NAME=edit_thinker_qwen3vl_sft

# Please replace with your own conda environment
source /path/to/conda/bin/activate /path/to/your/conda/env  # Replace with your conda environment path

PROJECT_PATH="/path/to/Edit-Thinker-submit"  # Replace with your project path

CONFIG_YAML="$PROJECT_PATH/train/LLaMA-Factory/examples/train_full/edit_thinker_qwen3vl_sft.yaml"

cd $PROJECT_PATH/train/LLaMA-Factory

llamafactory-cli train $CONFIG_YAML


