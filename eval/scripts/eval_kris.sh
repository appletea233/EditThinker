#!/bin/bash

output_dir=$1

# Please set your own paths and API keys
export OPENAI_API_KEY=your_api_key_here  # Replace with your OpenAI API key
export BENCH_DIR=/path/to/your/benchmarks/Kris_Bench/KRIS_Bench  # Replace with your benchmark directory
export BASE_URL=https://your-endpoint.com/v1/openai/native  # Replace with your API endpoint

n_proc=64
python -m kris.metrics_common \
    --results_dir $output_dir \
    --max_workers ${n_proc} 

python -m kris.metrics_knowledge \
    --results_dir $output_dir \
    --max_workers ${n_proc} 

python -m kris.metrics_multi_element \
    --results_dir $output_dir \
    --max_workers ${n_proc} 

# python -m kris.metrics_temporal_prediction \
#     --results_dir $output_dir \
#     --max_workers ${n_proc} 

python -m kris.metrics_view_change \
    --results_dir $output_dir \
    --max_workers ${n_proc} 


# summarize score
python -m kris.summarize \
    --results_dir $output_dir
