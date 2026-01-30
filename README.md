<div align="center">

# 🍨 Parfait

**P**arameter-**A**ligned **R**egime-**F**ollowing **AI** for **T**ime-series

[![Generic badge](https://img.shields.io/badge/Status-Research_Preview-purple.svg)](https://github.com/yourusername/parfait)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b.svg)](https://arxiv.org)

*A World Model for Quantitative Finance.*

[**Abstract**](#abstract) | [**Methodology**](#methodology) | [**Installation**](#installation) | [**Usage**](#usage) | [**Citation**](#citation)

</div>

---

## 📜 Abstract

**Parfait** is a novel framework designed to bridge the gap between macroscopic semantic instructions and microscopic financial time-series generation. Unlike traditional generative models that rely solely on historical data distributions, Parfait introduces a Text-to-SDE-to-Diffusion paradigm.

It leverages Large Language Models (LLMs) to reason about "Black Swan" events, translating natural language descriptions into Stochastic Differential Equation (SDE) parameters. These parameters guide a Conditional Diffusion Model to synthesize high-fidelity, physically consistent OHLCV market data. Parfait enables the training of Reinforcement Learning (RL) agents in counterfactual scenarios (e.g., "World War III breaks out") that have no historical precedent.

---

## 🧠 Methodology

Parfait operates on a hierarchical architecture, decomposing market generation into three distinct layers:

### 1. The Semantic Layer
A fine-tuned LLM (Qwen2-7B) acts as the semantic reasoning engine. It accepts natural language instructions (e.g., *"The central bank announces an unexpected 50bps rate hike"*).
Instead of hallucinating prices directly, it estimates the probability distribution of future market regimes, outputting a set of SDE parameters:
*   Drift ($\mu$) & Volatility ($\sigma$)
*   Jump Intensity ($\lambda$) & Mean Reversion ($\kappa$)
*   Hurst Exponent ($H$)

### 2. The Theoretical Layer
We solve the corresponding SDEs (e.g., Geometric Brownian Motion with Jumps or Heston Model) to generate Guide Curves. These curves provide a "coarse" trajectory that enforces the macroscopic constraints defined by the LLM.

$$ dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S + (J-1)S_t dN_t $$

### 3. The Microstructure Layer
A 1D Conditional U-Net serves as the microstructure generator. It takes the coarse Guide Curve as a hard constraint and "inpaints" the realistic market noise (ticks, bid-ask bounce effects, volume clusters). The model is conditioned on historical context encoded by a Transformer Backbone.

<div align="center">
<pre>
[ Instruction ] 
      ⬇
[ LLM (Qwen2) ] ➡ { μ, σ, Jump_Prob } (Multi-Modal Scenarios)
      ⬇
[ SDE Solver ] ➡ [ Guide Curve ] (Coarse Trajectory)
      ⬇
[ Diffusion U-Net ] ⬅ [ History Encoder ]
      ⬇
[ Synthetic OHLCV Market Data ]
</pre>
</div>

---

## 🛠 Installation

Parfait requires a GPU environment (RTX Pro 6000 recommended for training) with CUDA support.

```bash
https://github.com/Chunjiang-Intelligence/Parfait
cd Parfait

conda create -n parfait python=3.12
conda activate parfait

pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Processing & Label Generation
Transform raw CSV market data into a dataset labeled with financial math parameters (Hurst, Volatility, Drift).

```bash
python scripts/01_generate_labels.py
```

### 2. SFT: Training the Semantic Layer
Fine-tune the Qwen2 LLM to align natural language instructions with SDE parameter distributions using **Knowledge Distillation**.

```bash
# Ensure OPENAI_API_KEY is set for distillation if needed
export OPENAI_API_KEY="sk-..."
python scripts/02_train_llm.py
```

### 3. Training the Diffusion Model
Train the Conditional U-Net to generate microstructure consistent with both history and SDE guide curves.

```bash
python scripts/03_train_diffusion.py
```

### 4. Inference: Running the Simulation
Generate diverse market scenarios based on a counterfactual instruction. This pipeline executes the full **Text $\to$ SDE $\to$ Diffusion** flow.

```python
# scripts/04_run_simulation.py
from src.inference.pipeline import InferencePipeline

instruction = "Global conflict escalates, leading to a liquidity crisis and gold price surge."
pipeline.run(instruction, history_data, total_simulations=100)
```

---

## 📝 Citation

If you use Parfait in your research or trading agent training, please cite:

```bibtex
@article{parfait2025,
  title={Parfait: Bridging the Semantic-Microstructure Gap in Financial World Modeling},
  author={Chunjiang Intelligence},
  year={2025}
}
```

---

<div align="center">
  <sub>Built with 🍦 and 📉 by the Chunjiang Research Team.</sub>
</div>
