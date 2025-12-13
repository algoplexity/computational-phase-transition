# The Computational Phase Transition

[![Part of](https://img.shields.io/badge/Algoplexity-Horizon%201-purple.svg)](https://github.com/algoplexity/algoplexity) [![SSRN](https://img.shields.io/badge/SSRN-Preprint-blue.svg)](https://ssrn.com/abstract=XXXXXX) [![Data](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/algoplexity/computational-phase-transitions-data)

**Decoding the Cognitive State of Financial Markets via Algorithmic Information Dynamics.**

This repository contains the official implementation of **Horizon 1** of the [Algoplexity Research Program](https://github.com/algoplexity/algoplexity). It introduces the **AIT Physicist**, a Transformer-based diagnostic instrument that functions as an EEG for the market's meta-cognition.

---

## 📄 Abstract
Financial crises are traditionally modeled as statistical outliers. We propose a fundamental paradigm shift: treating the market as a **Strategic Complex Adaptive System (CAS)** undergoing **Computational Phase Transitions**.

Building on **Algorithmic Information Dynamics (AID)** and **Universal Artificial Intelligence (UAI)**, we solve the **Inverse Problem** of mapping the *temporal topology* of noisy financial time series back to their underlying generative rules. Unlike statistical models ("stochastic parrots"), the AIT Physicist acts as a **Sensory Organ** for an intelligent agent, collapsing the infinite superposition of market states into a tractable signal.

**Key Findings:**
1.  **Generalization Inversion:** The model detects regime shifts with a mean lead time of **-29.95%** on *unseen* out-of-sample data, outperforming its training metrics.
2.  **Taxonomy of Cognitive Failure:**
    *   **2008 (GFC):** **Cognitive Saturation** (Rule 54 / Colliding Solitons). The market "thinks itself into a corner" via hyper-rigid internal logic.
    *   **2020 (COVID):** **Cognitive Overload** (Rule 60 / Fractal Shattering). The external shock outruns the system's mixing time.

---

## 🛠️ Theoretical Framework & Methodology

This work operationalizes the "Perception" layer of the Algoplexity Agent.

### 1. The Architectural Pivot: "Less is More"
Following the falsification of statistical multivariate models in our preliminary study (**[The Somatic Marker, 2025a](https://github.com/algoplexity/Coherence-Meter)**), we adopt a **Univariate Topological** architecture.
*   *Why not Graphs?* While markets are networks, **Grattarola et al. (2021)** proved that Neural Networks can learn CA rules on any topology. We deliberately isolate the **Temporal Dimension (1D Lattice)** to perfect the detection mechanism before scaling to **Spatial Graphs (GNCA)** in Horizon 3.

### 2. The Engine: The AIT Physicist
*   **Model:** Tiny Recursive Model (TRM) / Transformer.
*   **Training:** Pre-trained on the **Wolfram Computational Universe** (The "Prime 9" Rules).
*   **Signal:** **Entropic Ambiguity**. We measure the *confusion* of the model ($H$) to detect phase transitions.

---

## 📂 Repository Structure

*   `ait_lib/`: The core library containing the TRM architecture and MILS encoding logic.
*   `notebooks/`:
    *   `01_The_Training_Ground.ipynb`: Pre-training the Physicist on Synthetic ECA data.
    *   `02_The_Gauntlet.ipynb`: Systematic Validation (Source of the -29.95% stat).
    *   `03_The_Showdown.ipynb`: Historical Case Studies (Source of the Phase Portraits).
*   `results/`: High-resolution figures of the Generalization Inversion and Phase Portraits.

---

## 📊 Reproducibility
The datasets used in this research are hosted as an immutable scientific artifact:
**[Algoplexity Structural Break Benchmark](https://huggingface.co/datasets/algoplexity/computational-phase-transitions-data)**

To replicate the findings:
1.  Clone this repo.
2.  Install requirements: `pip install -r requirements.txt`
3.  Run `notebooks/02_The_Gauntlet.ipynb`.

---

## 🔗 Citation

```bibtex
@article{algoplexity2025computational,
  title={The Computational Phase Transition: Decoding the Cognitive State of Financial Markets via Algorithmic Information Dynamics},
  author={Mak, Yeu Wen},
  journal={arXiv preprint},
  year={2025}
}
