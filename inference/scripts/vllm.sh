model_path=$1

source /path/env/bin/activate

# get server ip
main_ip=""
if command -v ip >/dev/null 2>&1; then
    main_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
fi
echo "main_ip: $main_ip"    

vllm serve $model_path \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.6