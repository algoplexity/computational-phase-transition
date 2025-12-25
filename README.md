# 🌌 The Computational Phase Transition

[![Part of](https://img.shields.io/badge/Algoplexity-Horizon%201-purple.svg)](https://github.com/algoplexity/algoplexity) [![SSRN](https://img.shields.io/badge/SSRN-Preprint-blue.svg)](https://ssrn.com/abstract=XXXXXX) [![Data](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/algoplexity/computational-phase-transitions-data)

**Decoding the Cognitive State of Financial Markets via Algorithmic Information Dynamics.**

This repository contains the official implementation of **Horizon 1** of the [Algoplexity Research Program](https://github.com/algoplexity/algoplexity). It introduces the **AIT Physicist**, a Transformer-based diagnostic instrument that functions as an EEG for the market's meta-cognition.

---

## 📄 Abstract
Financial crises are traditionally modeled as statistical outliers. We propose a fundamental paradigm shift: treating the market as a **Strategic Complex Adaptive System (CAS)** undergoing **Computational Phase Transitions**.

Building on **Algorithmic Information Dynamics (AID)** and **Universal Artificial Intelligence (UAI)**, we solve the **Inverse Problem** of mapping the *temporal topology* of noisy financial time series back to their underlying generative rules. Unlike statistical models ("stochastic parrots"), the AIT Physicist acts as a **Sensory Organ** for an intelligent agent, collapsing the infinite superposition of market states into a tractable signal.

**Key Findings:**
1.  **The 1.0000 AUC Mandate:** In controlled laboratory environments, the instrument achieves **perfect binary separation (1.0 AUC)** by measuring the **Joint Compression Failure ($CIv7$)** of the causal chain using Next-Token Prediction.
2.  **Generalization Inversion:** The model detects regime shifts with a mean lead time of **-29.95%** on *unseen* out-of-sample data, outperforming its training metrics.
3.  **Taxonomy of Cognitive Failure:**
    *   **Cognitive Saturation** (Rule 54 / Colliding Solitons): The market "thinks itself into a corner" via hyper-rigid internal logic.
    *   **Cognitive Overload** (Rule 60 / Fractal Shattering): The external shock outruns the system's mixing time.

---

## 🛠️ Theoretical Framework & Methodology

This work operationalizes the "Perception" layer of the Algoplexity Agent.

### 1. The Architectural Pivot: "Less is More"
Following the falsification of statistical multivariate models in our preliminary study (**[The Somatic Marker, 2025a](https://github.com/algoplexity/Coherence-Meter)**), we adopt a **Univariate Topological** architecture. 
*   **Minimal Information Loss:** We reject lossy statistical operators (mean, variance, std). Following **Protocol CIv21**, change is detected exclusively via **Bit-Summation** of the Information Surplus.

### 2. The Engine: The AIT Physicist (TRM v3)
*   **Model:** Tiny Recursive Model (TRM v3) / Transformer.
*   **Training:** Pre-trained on the **Wolfram Computational Universe** (Class 4 Rules). 
*   **Intelligence:** Following **Zhang et al. (Yale, 2024)**, the model is trained on **Next-Token Prediction (NTP)**, achieving **97.9% accuracy** on Rule 110. This creates an internal representation capable of simulating universal logic.
*   **Signal:** **Integrated Information Surplus ($K$)**. We measure the "Somatic Agony" of the model—the total bits of surprise generated when the environment violates the model's hereditary intellect.

---

## 📂 Repository Structure

*   `ait_lib/`: The core library containing the TRM architecture and MILS encoding logic.
*   `weights/`: Contains `trm_intellect_v3.pth` (The 97.9% accurate universal observer).
*   `notebooks/`:
    *   `01_Yale_Replication.ipynb`: Training the Intellectual Observer on the Edge of Chaos.
    *   `02_The_Golden_Baseline.ipynb`: Proof of **1.0000 AUC** using the "Dense Agony" operator.
    *   `03_The_Gauntlet.ipynb`: Systematic Validation (Source of the -29.95% stat).
    *   `04_The_Showdown.ipynb`: Historical Case Studies (Phase Portraits).

---

## 📊 Reproducibility
The datasets used in this research are hosted as an immutable scientific artifact:
**[Algoplexity Structural Break Benchmark](https://huggingface.co/datasets/algoplexity/computational-phase-transitions-data)**

To replicate the 1.0000 AUC Baseline:
1.  Clone this repo.
2.  Install requirements: `pip install -r requirements.txt`
3.  Run `notebooks/02_The_Golden_Baseline.ipynb`.

---

## 🚀 ADIA Lab Deployment (`infer.py`)

The final production logic uses **Conditional Path Integration**. It measures the bit-cost of explaining the future using the hereditary memory of the past.

```python
def get_causal_agony(series, model, device):
    # S1: Dense 4-bit Sensation
    # Logic: Integrated Information Surplus (I)
    # No Statistics allowed (CIv21)
    # ... (Implementation of the 1.0000 AUC functional) ...
    return np.sum(surprise_bits)
```

---

## 🔗 Citation

```bibtex
@article{algoplexity2025computational,
  title={The Computational Phase Transition: Decoding the Cognitive State of Financial Markets via Algorithmic Information Dynamics},
  author={Mak, Yeu Wen},
  journal={arXiv preprint},
  year={2025}
}
```
