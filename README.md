<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="RecursiveMAS" src="assets/logo.png" width=300>
  </picture>
</p>

<h3 align="center">
Recursive Multi-Agent Systems
</h3>

---

## 📰 News

- **[2026.04.28]** RecursiveMAS paper released! [Code](https://github.com/RecursiveMAS/RecursiveMAS) and [Checkpoints](https://huggingface.co/RecursiveMAS/models) are available now!
- **[2026.04.28]** Project page is available at [recursivemas.github.io](https://recursivemas.github.io).

## 🌟 Introduction

**RecursiveMAS** is a recursive multi-agent framework that scales agent collaboration through **latent-space recursion**.

Instead of treating each LLM agent as an isolated module, RecursiveMAS casts the entire multi-agent system as a **unified recursive computation**. Heterogeneous agents are connected through lightweight RecursiveLink modules, allowing agents to iteratively exchange, refine, and evolve their latent states across recursion rounds.

<p align="center">
  <img src="assets/overview.png" width="95%" alt="RecursiveMAS Overview">
</p>

## ✨ Key Features of RecursiveMAS

- **System-level recursion**: RecursiveMAS organizes the entire multi-agent system as a recursive loop, where agents repeatedly refine the shared latent information flow across recursion rounds.

- **Inner and Outer RecursiveLink**: RecursiveMAS uses lightweight residual projection modules to support both agent-level latent-thoughts generation and cross-agent latent information transfer.

- **Generalizable collaboration patterns**: RecursiveMAS can be seamlessly adapted to diverse MAS structures, including Sequential-style, Mixture-style, Distillation-style, and Deliberation-style collaboration.

- **Strong and efficient performance**: Across 9 benchmarks, RecursiveMAS achieves an average accuracy improvement of 8.3%, together with 1.2×–2.4× end-to-end inference speedup and 34.6%–75.6% token usage reduction.


---

## 📊 Experiments

### 🚀 RecursiveMAS Scales Performance and Generalization

RecursiveMAS demonstrates a clear scaling trend across both training-time and inference-time recursion depths. Additionally, RecursiveMAS also generalizes to diverse MAS collaboration patterns.

<p align="center">
  <img src="assets/hero_fig.png" width="95%" alt="RecursiveMAS Main Results">
</p>


---

### ⚡ Superior Efficiency of RecursiveMAS

Compared with Recursive-TextMAS under the same MAS structure and recursion budget, RecursiveMAS achieves increasing inference-time speedup as the recursion depth becomes larger.

<p align="center">
  <img src="assets/efficiency.png" width="95%" alt="RecursiveMAS Efficiency">
</p>


---

## 🛠️ Experiment Setup

This repository provides the code for running RecursiveMAS under different multi-agent collaboration styles. 

To begin with, we recommend creating a new conda environment:

```bash
conda create -n recursivemas python=3.10 -y
conda activate recursivemas
```

Install the required packages:

```bash
pip install -r requirements.txt
```

You also need to login to huggingface to download our released checkpoints.

```bash
huggingface-cli login
```

Optionally, you may set your Hugging Face cache directory:

```bash
export HF_HOME=/path/to/your/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

---

## 💥 Quick Start

### 🔍 Clone the Repository

First, clone our repository and enter the project directory:

```bash
git clone https://github.com/RecursiveMAS/RecursiveMAS.git
cd RecursiveMAS
```

The current repository is organized as follows:

```text
RecursiveMAS/
├── README.md
├── __init__.py
├── run.py
├── load_from_repo.py
├── hf_resolver.py
├── modeling.py
├── prompts.py
├── requirements.txt
├── assets/
├── dataset/
└── inference_utils/
    ├── __init__.py
    ├── answer_utils.py
    ├── lcb_utils.py
    ├── reflection_tool_notes.py
    ├── inference_mas.py
    ├── inference_mas_mixture.py
    ├── inference_mas_distill.py
    └── inference_mas_deliberation.py
```

The key components are:

- `run.py`: the unified entry point for running RecursiveMAS inference.
- `load_from_repo.py`: maps each MAS style to our released Hugging Face checkpoints and dataset defaults.
- `hf_resolver.py`: resolves and load the Hugging Face checkpoints.
- `modeling.py`: implements RecursiveLink modules.
- `prompts.py`: stores prompts for different MAS collaboration styles.
- `inference_utils/`: contains inference pipelines and evaluation utilities for different MAS structures.

---

### ⚙️ Running Sequential RecursiveMAS at Different Scales

We provide Sequential-style RecursiveMAS under both lightweight and scaled settings.

- **Sequential-style (Light)** uses lightweight agents for efficient recursive collaboration.
```bash
python run.py --style sequential_light --dataset math500 --seed 42 --batch_size 16 --temperature 0.6 --top_p 0.95 --trust_remote_code 1 --device cuda
```

- **Sequential-style (Scaled)** uses stronger LLM agents to further improve reasoning performance.
```bash
python run.py --style sequential_scaled --dataset math500 --seed 42 --batch_size 16 --temperature 0.6 --top_p 0.95 --trust_remote_code 1 --device cuda
```
---

### 🧩 Exploring Various Collaboration Patterns

RecursiveMAS can also be adapted to different MAS collaboration patterns beyond the sequential setting.

- **Mixture-style RecursiveMAS** coordinates multiple domain-specialized agents and aggregates their information through a summarizer.
```bash
python run.py --style mixture --dataset math500 --seed 42 --batch_size 16 --temperature 0.6 --top_p 0.95 --trust_remote_code 1 --device cuda
```

- **Distillation-style RecursiveMAS** enables a larger Expert and a smaller Learner to interact recursively, improving the Learner while retaining better efficiency.
```bash
python run.py --style distillation --dataset math500 --seed 42 --batch_size 16 --temperature 0.6 --top_p 0.95 --trust_remote_code 1 --device cuda
```

- **Deliberation-style RecursiveMAS** supports recursive coordination between a Reflector and a Tool-Caller for tool-integrated reasoning.
```bash
python run.py --style deliberation --dataset math500 --seed 42 --batch_size 16 --temperature 0.6 --top_p 0.95 --trust_remote_code 1 --device cuda
```

---

## 🙏 Acknowledgements

This project is built upon the excellent open-source ecosystem of large language models. We sincerely thank the developers and maintainers of the following libraries and resources:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for providing flexible and widely used model loading and generation utilities.
- [vLLM](https://github.com/vllm-project/vllm) for supporting efficient LLM inference and serving.
- [Hugging Face Hub](https://huggingface.co/) for hosting model checkpoints and enabling convenient checkpoint distribution.

---
<!-- 
## 🚀 Contributing

We welcome discussions and contributions to RecursiveMAS. If you would like to suggest improvements, please feel free to contact us.

- [Xiyuan Yang](mailto:xiyuany4@illinois.edu)
- [Jiaru Zou](mailto:jiaru@stanford.edu)

--- -->

## 📚 Citation

@article{recursivemas,
  title={Recursive Multi-Agent Systems},
  author={Yang, Xiyuan and Zou, Jiaru and Pan, Rui and Qiu, Ruizhong and Lu, Pan and Diao, Shizhe and Jiang, Jindong and Tong, Hanghang and Zhang, Tong and Buehler, Markus J. and He, Jingrui and Zou, James},
  year={2026}
}
