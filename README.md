
<h1 align="left">
  <img src="images/logo.png" height="55" style="vertical-align: middle; margin-right: 10px;">
  EditThinker: Unlocking Iterative Reasoning for Any Image Editor
</h1>

Official repository for the paper "[EditThinker: Unlocking Iterative Reasoning for Any Image Editor](https://arxiv.org/abs/)".

[[🌍 Project Page](https://appletea233.github.io/think-while-edit/)] [[📖 Paper](https://arxiv.org/abs/)] [[🤗 ThinkEdit-140K Dataset](https://github.com/appletea233/EditThinker)] [[🤗 EditThinker-8B](https://github.com/appletea233/EditThinker)]

## 💥 News
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




## ✔️ Citation

Please cite us if you find this project helpful:

```bibtex
@article{li2026editthinker,
  title={EditThinker: Unlocking Iterative Reasoning for Any Image Editor},
  author={Li, Hongyu and Zhang, Manyuan and Zheng, Dian and Guo, Ziyu and Jia, Yimeng and Feng, Kaituo and Yu, Hao and Liu, Yexin and Feng, Yan and Pei, Peng and Cai, Xunliang and Huang, Linjiang and Li, Hongsheng and Liu, Si},
  year={2025}
}

```

