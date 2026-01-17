#!/bin/bash

# Please set your own paths and API keys
export BENCH_DIR=/path/to/your/benchmarks/RISEBench  # Replace with your benchmark directory
export BASE_URL=https://your-endpoint.com/v1/openai/native  # Replace with your API endpoint
export OPENAI_API_KEY=your_api_key_here  # Replace with your OpenAI API key

output_dir=$1

python -m rise.gpt_eval \
    --data $BENCH_DIR/datav2_total_w_subtask.json \
    --input $BENCH_DIR/data \
    --output $output_dir \
    --nproc 128