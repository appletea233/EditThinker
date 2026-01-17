#!/bin/bash
# Please set your own environment paths
source /opt/rh/devtoolset-8/enable  # Optional: if using devtoolset
source /path/to/conda/bin/activate /path/to/your/conda/env  # Replace with your conda environment path
export HF_HOME=/path/to/your/huggingface/cache  # Replace with your HuggingFace cache directory

export OPENAI_API_KEY=your_api_key_here  # Replace with your OpenAI API key

# ==================== Benchmark Selection ====================
# Options: gedit, imgedit, kris, rise
BENCHMARK=${1:-"gedit"}  # Default to gedit if no argument provided

# ==================== Common Configuration ====================
REWRITE_MODEL_NAME=gpt-4.1
EDIT_API_ENDPOINT="http://your-ip:8080"  # Replace with your image editing API endpoint
MODEL_NAME="longcat-image-edit"

MAX_RETRIES=100
MAX_EDIT_TURNS=5
NUM_WORKERS=128 

# ==================== Directory Configuration ====================
# Benchmark data directory (where datasets are stored)
# You can customize this path to point to your benchmark data location
# Please replace with your own benchmark and result directories
BENCHMARK_DIR=/path/to/your/benchmarks  # Replace with your benchmark data directory

# Result output directory (where inference results will be saved)
# You can customize this path to save results in a different location
RESULT_BASE_DIR=/path/to/your/results/${MODEL_NAME}/expert/${REWRITE_MODEL_NAME}  # Replace with your result output directory

# ==================== Benchmark-Specific Configuration ====================
case ${BENCHMARK} in
    gedit)
        echo "🎯 Running GEdit-Bench benchmark"
        DATASET_FORMAT="gedit"
        DATASET_NAME="stepfun-ai/GEdit-Bench"
        IMAGE_PATH="${BENCHMARK_DIR}/GEdit/images"
        RESULT_DIR="${RESULT_BASE_DIR}/gedit"
        METADATA_FILE=""
        EXTRA_ARGS="--skip_first_turn_rewrite"
        ;;
    
    imgedit)
        echo "🎯 Running ImgEdit benchmark"
        DATASET_FORMAT="imgedit"
        DATASET_NAME=""
        IMAGE_PATH="${BENCHMARK_DIR}/Benchmark/singleturn"
        RESULT_DIR="${RESULT_BASE_DIR}/imgedit"
        METADATA_FILE="${BENCHMARK_DIR}/Benchmark/singleturn/singleturn.json"
        EXTRA_ARGS="--skip_first_turn_rewrite"
        ;;
    
    kris)
        echo "🎯 Running KRIS benchmark"
        DATASET_FORMAT="kris"
        DATASET_NAME=""
        IMAGE_PATH="${BENCHMARK_DIR}/Kris_Bench/KRIS_Bench"
        RESULT_DIR="${RESULT_BASE_DIR}/kris"
        METADATA_FILE="${BENCHMARK_DIR}/Kris_Bench/final_data.json"
        EXTRA_ARGS="--skip_first_turn_rewrite"
        ;;
    
    rise)
        echo "🎯 Running RISE benchmark"
        DATASET_FORMAT="rise"
        DATASET_NAME=""
        IMAGE_PATH="${BENCHMARK_DIR}/RISEBench/data"
        RESULT_DIR="${RESULT_BASE_DIR}/rise"
        METADATA_FILE="${BENCHMARK_DIR}/RISEBench/datav2_total_w_subtask.json"
        EXTRA_ARGS="--skip_first_turn_rewrite"
        ;;
    
    *)
        echo "❌ Error: Unknown benchmark '${BENCHMARK}'"
        echo "Usage: $0 [gedit|imgedit|kris|rise]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Benchmark: ${BENCHMARK}"
echo "Dataset Format: ${DATASET_FORMAT}"
echo "Benchmark Data Directory:${BENCHMARK_DIR}"
echo "Input Image Path:${IMAGE_PATH}"
echo "Result Output Directory:${RESULT_DIR}"
if [ -n "${METADATA_FILE}" ]; then
    echo "Metadata File:${METADATA_FILE}"
fi
echo "=========================================="

CMD_ARGS=(
    --api_endpoint "${EDIT_API_ENDPOINT}"
    --model_name "${MODEL_NAME}"
    --rewrite_model_name "${REWRITE_MODEL_NAME}"
    --result_dir "${RESULT_DIR}"
    --max_retries ${MAX_RETRIES}
    --max_edit_turns ${MAX_EDIT_TURNS}
    --num_workers ${NUM_WORKERS}
    --dataset_format "${DATASET_FORMAT}"
    --image_path "${IMAGE_PATH}"
)

# Add dataset name if using HuggingFace
if [ -n "${DATASET_NAME}" ]; then
    CMD_ARGS+=(--dataset_name "${DATASET_NAME}")
fi

# Add metadata file if specified
if [ -n "${METADATA_FILE}" ]; then
    CMD_ARGS+=(--metadata_file "${METADATA_FILE}")
fi

# Add extra arguments
if [ -n "${EXTRA_ARGS}" ]; then
    CMD_ARGS+=(${EXTRA_ARGS})
fi

# Execute
python3 run_edit_thinker_inference.py "${CMD_ARGS[@]}"

echo ""
echo "✅ Inference completed for ${BENCHMARK} benchmark"
