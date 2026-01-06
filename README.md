# 🚀 FedALA-SA: Structure-Aware Federated Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**FedALA-SA** is a cutting-edge framework extending **FedALA** with Structure-Aware capabilities, designed to tackle the heterogeneity in Federated Graph Learning (FGL). By integrating topological insights and privacy-preserving mechanisms, FedALA-SA offers a robust solution for decentralized graph data analysis.

---

## 🌟 Key Features

*   **Structure-Aware Aggregation**: Leverages graph topology (Degree & Layer-wise) to optimize model updates.
*   **Privacy-First Design**: Integrated Differential Privacy (DP) mechanisms (Gaussian Noise) for secure collaboration.
*   **Scalable Architecture**: Tested and validated on scalable client setups (10-20+ clients).
*   **Automated Benchmarking**: One-click reproduction of baseline vs. innovation scenarios.
*   **Visual Analytics**: Automatic generation of convergence curves and ablation study plots.

---

## 🛠️ Installation

Get started in minutes! Ensure you have Python 3.10+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/mertcoskuner/FedALA-SA.git
cd FedALA-SA
```

### 2. Install Dependencies
We have prepared a `requirements.txt` with all necessary packages (including CUDA-enabled PyTorch).

```bash
pip install -r requirements.txt
```

> **Note:** If you are using a specific CUDA version (e.g., CUDA 11.8), the `requirements.txt` already handles the extra index URL for PyTorch.

---

## 🚀 Usage

### Run the Main Benchmark
Execute the comprehensive benchmark script to reproduce the paper's results (Baseline vs. FedALA-SA vs. Privacy):

```bash
python run_fedala_sa.py
```

**What this does:**
*   Runs 3 critical scenarios:
    1.  **Baseline**: Vanilla FedALA
    2.  **Innovation**: FedALA-SA (Structure-Aware)
    3.  **Privacy**: FedALA-SA + Differential Privacy
*   Tests scalability (10 vs 20 Clients).
*   💾 Saves results to `fedala_sa_final_results.csv`.
*   📊 Generates plots (`fedala_sa_ablation_study.png`, `fedala_sa_convergence_*.png`).

---

## 📊 project Structure

```
FedALA-SA/
├── 📂 data/               # Dataset storage
├── 📂 openfgl/            # Core library files
├── 📄 run_fedala_sa.py    # Main reproduction script
├── 📄 plots.py            # Plotting utilities
├── 📄 requirements.txt    # Dependency list
└── 📄 README.md           # This file
```

---

