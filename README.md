
<h1 align="left">
  <img src="images/logo.png" height="55" style="vertical-align: middle; margin-right: 10px;">
  EditThinker: Unlocking Iterative Reasoning for Any Image Editor
</h1>

Official repository for the paper "[EditThinker: Unlocking Iterative Reasoning for Any Image Editor](https://arxiv.org/abs/2512.05965)".

[[🌍 Project Page](https://appletea233.github.io/think-while-edit/)] [[📖 Paper](https://arxiv.org/abs/2512.05965)] [[🤗 ThinkEdit-140K Dataset](https://huggingface.co/datasets/appletea2333/ThinkEdit)] [[🤗 EditThinker-8B](https://huggingface.co/appletea2333/EditThinker-8B)] [[🤗 EditThinker-8B-Flux.1-Kontext-Dev](https://huggingface.co/appletea2333/EditThinker8B-Flux.1-Kontext-Dev)]

## 💥 News
- **[2026.01.17]** Updated models, datasets, training, inference, and evaluation code. 🎉
- **[2025.12.08]** The paper **EditThinker** is released on arXiv. 🚀

## 💭🎨 Think-while-Edit

Instruction-based image editing has emerged as a prominent research area. Benefiting from image generation foundation models, it has achieved high aesthetic quality, making instruction-following capability the primary challenge. Existing approaches improve instruction adherence via supervised or reinforcement learning, yet single-turn success rates remain limited due to inherent stochasticity and a lack of deliberation.

In this work, we propose a deliberative editing framework to "think" while they edit, which simulates the human cognitive loop by iteratively executing a Think-while-Edit cycle: Critiquing results and Refining instructions, followed by Repeating the generation until satisfactory. Specifically, we train a single MLLM, EditThinker, to act as the reasoning engine of this framework, which jointly produces the critique score, reasoning process, and refined instructions. We employ reinforcement learning to align the EditThinker's thinking with its editing, thereby generating more targeted instruction improvements.


<p align="center">
  <img src="images/demo1.gif" width="90%">
</p>
<p align="center"><i>The animation illustrates the multi-turn Think-while-Edit process. The model iteratively critiques the current generation, reasons about the gap, and refines the prompt until the stop signal is triggered.</i></p>



## 🌟 Performance

Our approach shows large gains for existing editing methods (FLUX.1 Kontext, OmniGen2, Qwen-Image-Edit) across four image editing benchmarks (ImgEdit-Bench, GEdit-Bench, RISE-Bench, Kris-Bench).


<p align="center">
  <img src="images/performance.png" width="90%">
</p>



## 🚀 Framework

EditThinker is a multi-round instruction iterative refinement framework. In the first round, the original image $I_{src}$ and instruction $T_s$ are fed into an editor. The output is fed into EditThinker, which generates the edit score $S_t$, refined prompt $T_t$, and reasoning process $R_t$.

<p align="center">
  <img src="images/fig2.png" width="90%">
</p>

## 📊 ThinkEdit-140K

To train EditThinker, we constructed ThinkEdit-140k. We employ reinforcement learning (RL) on difficult, high-variance trajectories to align EditThinker's reasoning with actual editing outcomes, while using SFT on stable trajectories for foundational capabilities.


<p align="center">
  <img src="images/fig3.png" width="90%">
</p>




## 🖼️ Visualizations

### **1. The Thinking Process**

<p align="center">
  <img src="images/vis_reason.png" width="90%">
</p>



### **2. More Visualizations**

<p align="center">
  <img src="images/vis_all.png" width="90%">
</p>




## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/appletea233/EditThinker.git
cd EditThinker

# build inference environment
conda create -n editthinker python=3.11
conda activate editthinker

pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# build SFT environment
conda create -n llamafactory python=3.11 
conda activate llamafactory
cd train/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# build RL environment
conda create -n easyr1 python=3.11 
conda activate easyr1
cd train/EasyR1
pip install -e .

```

For more details for the SFT and RL environment installation, please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),  [EasyR1](https://github.com/hiyouga/EasyR1)

Then, download the training datasets [[🤗 ThinkEdit](https://huggingface.co/datasets/appletea2333/ThinkEdit)] and unzip all the data.

The `sft_train.json` is for SFT cold start while `rl_train.jsonl` file is for RL training. For SFT data, you need to modify the image path to abs path and place it into `train/LLaMA-Factory/data`.

### 🔧 Setup Image Editing API Service

Before running inference, you need to start the image editing API service. We support multiple editing models including **FLUX.1-Kontext**, **OmniGen2**, and **Qwen-Image-Edit**.

```bash
# Navigate to the server directory
cd inference/server

# Activate the inference environment 
conda activate editthinker

# Start the FLUX.1-Kontext editing service (Service 6, Port 8005)
bash start_service_with_ip.sh 6 ip.txt

# Or start OmniGen2 service (Service 9, Port 8007)
# bash start_service_with_ip.sh 9 ip.txt

# The service will automatically save its IP to ip.txt
# Service will be available at http://<your-ip>:<port>
```

**Service Options:**
- `1` - Qwen-Image Generation (Port: 8000)
- `2` - Qwen-Image Lightning Generation (Port: 8001)
- `3` - FLUX.1-Krea-dev Generation (Port: 8002)
- `4` - Qwen-Image-Edit (Port: 8003)
- `5` - Qwen-Image-Edit Lightning (Port: 8004)
- `6` - FLUX.1-Kontext-dev Edit (Port: 8005)
- `7` - FLUX.1-Fill-dev Fill (Port: 8006)
- `8` - LongCat-Image-Edit (Port: 8010)
- `9` - OmniGen2-Image-Edit (Port: 8007)

#### Multi-Node Deployment with Load Balancing

For production deployment with high throughput requirements, you can deploy multiple service nodes and use Nginx for load balancing.

**Step 1: Start Multiple Service Nodes**

```bash
cd inference/server

# On Node 1 (e.g., 192.168.1.101)
conda activate editthinker
bash start_service_with_ip.sh 6 ip.txt

# On Node 2 (e.g., 192.168.1.102)
conda activate editthinker
bash start_service_with_ip.sh 6 ip.txt

# On Node 3 (e.g., 192.168.1.103)
conda activate editthinker
bash start_service_with_ip.sh 6 ip.txt

# All nodes will automatically save their IPs to ip.txt
```

**Step 2: Setup Nginx Load Balancer**

Use the provided script to automatically configure and start a user-level Nginx load balancer (no sudo required):

```bash
cd inference/server

# Setup Nginx load balancer
# Parameters:
#   -i: IP file path (contains all backend server IPs)
#   -d: Base directory for Nginx logs/config
#   -b: Backend port (the port your services are running on)
#   -p: Proxy port (the port Nginx will listen on)

bash setup_user_nginx.sh \
  -i ip.txt \
  -d /tmp/nginx_edit_service \
  -b 8005 \
  -p 8080
```

The script will automatically:
- ✅ Read all server IPs from `ip.txt`
- ✅ Generate Nginx configuration with load balancing
- ✅ Start user-level Nginx instance (no sudo required)
- ✅ Create management scripts (start/stop)
- ✅ Display access URL and management commands

**Step 3: Verify and Use**

```bash
# Check backend health
curl http://<nginx-server-ip>:8080/health

# Use the load balancer endpoint in your inference
EDIT_API_ENDPOINT="http://<nginx-server-ip>:8080"
```

### 📊 Inference

We provide two inference modes: **vLLM** (faster, for deployment) and **Expert** (more flexible, for research).

#### Method 1: Using vLLM Server (Recommended for Deployment)

```bash
cd inference

# Step 1: Start vLLM server for EditThinker model (in a separate terminal)
bash scripts/run_edit_thinker_vllm.sh

# Step 2: Run inference on benchmark
# BENCHMARK can be: gedit, imgedit, kris, rise
BENCHMARK=gedit  # Change to your target benchmark
VLLM_SERVER_IP=<your-vllm-server-ip>

bash scripts/run_edit_thinker_vllm.sh ${BENCHMARK} ${VLLM_SERVER_IP}
```

#### Method 2: Using Expert Mode (Recommended for Research)

```bash
cd inference

# Run inference on benchmark
# BENCHMARK can be: gedit, imgedit, kris, rise
BENCHMARK=gedit  # Change to your target benchmark

bash scripts/run_edit_thinker_expert.sh ${BENCHMARK}
```

**Supported Benchmarks:**
- `gedit` - GEdit-Bench
- `imgedit` - ImgEdit-Bench
- `kris` - KRIS-Bench
- `rise` - RISE-Bench



### 🎓 Training


#### Supervised Fine-Tuning (SFT)

```bash
bash train/LLaMA-Factory/local_scripts/run_edit_thinker_sft.sh
```

#### Reinforcement Learning (RL)

**Step 1: Configure Reward Function**

Edit the configuration in `train/EasyR1/verl/reward_function/edit_thinker_reward.py`:

```python
# Configure your GPT-4.1 API for evaluation
GPT41_KEY_PATHS: List[str] = ["your_api_key_here"]
GPT41_AZURE_ENDPOINT: str = "https://your-endpoint.com/v1/openai/native"

EDIT_API_ENDPOINT: Optional[str] = "http://your-nginx-ip:8080"  # Use FLUX.1-Kontext service
EDIT_MODEL_NAME: str = "flux-kontext-dev"  # Options: "omnigen2", "flux-kontext-dev", "qwen-image-edit"
```

**Step 2: Run RL Training**

Configure and run the training script:

```bash
bash train/EasyR1/local_scripts/run_edit_thinker_rl.sh
```

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [EasyR1](https://github.com/hiyouga/EasyR1), [verl](https://github.com/volcengine/verl),  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


## ✔️ Citation

Please cite us if you find this project helpful:

```bibtex
@article{li2025editthinker,
  title={Editthinker: Unlocking iterative reasoning for any image editor},
  author={Li, Hongyu and Zhang, Manyuan and Zheng, Dian and Guo, Ziyu and Jia, Yimeng and Feng, Kaituo and Yu, Hao and Liu, Yexin and Feng, Yan and Pei, Peng and others},
  journal={arXiv preprint arXiv:2512.05965},
  year={2025}
}
```

