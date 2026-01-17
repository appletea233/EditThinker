#!/bin/bash

# Please set your own paths and API keys
export BENCH_DIR=/path/to/your/benchmarks/Benchmark/singleturn  # Replace with your benchmark directory
export BASE_URL=https://your-endpoint.com/v1/openai/native  # Replace with your API endpoint
export OPENAI_API_KEY=your_api_key_here  # Replace with your OpenAI API key

output_dir=$1

python -m imgedit.basic_bench \
    --result_folder $output_dir \
    --edit_json $BENCH_DIR/singleturn.json \
    --origin_img_root $BENCH_DIR \
    --num_processes 128 \
    --prompts_json $BENCH_DIR/judge_prompt.json

# summarize score
python -m imgedit.step1_get_avgscore \
    --result_json $output_dir/result.json \
    --average_score_json $output_dir/average_score.json

python -m imgedit.step2_typescore \
    --average_score_json  $output_dir/average_score.json \
    --edit_json $BENCH_DIR/singleturn.json \
    --typescore_json $output_dir/typescore.json