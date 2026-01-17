#!/bin/bash

output_dir=$1

# Please set your own paths and API keys
export HF_HOME=/path/to/your/huggingface/cache  # Replace with your HF_HOME path
export BASE_URL=https://your-endpoint.com/v1/openai/native  # Replace with your API endpoint
export OPENAI_API_KEY=your_api_key_here  # Replace with your OpenAI API key

python -m gedit.test_gedit_score \
--save_path ${output_dir} \
--azure_endpoint $BASE_URL \
--gpt_keys $OPENAI_API_KEY \
--max_workers 64

python -m gedit.calculate_statistics \
--save_path ${output_dir} \
--language en