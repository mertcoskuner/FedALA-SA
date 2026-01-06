# FedALA-SA Implementation

This project implements **FedALA-SA** (Structure-Aware), an enhanced version of FedALA for OpenFGL benchmark.

## Implementation Details

We implemented a new algorithm module `fedala_sa` in `openfgl/flcore/fedala_sa`.
It supports 3 options as requested:

### Option 1: Degree-based Scaling (Server-side)
- **Concept:** Clients with denser subgraphs (higher average degree) contribute more to the global model.
- **Implementation:** 
    - Client calculates `avg_degree` of its local subgraph.
    - Server aggregates updates using `weight = num_samples * avg_degree` instead of just `num_samples`.
- **Code:** `FedALAClient.send_message` sends `avg_degree`, `FedALAServer.execute` uses it.

### Option 2: Proximal Term (Client-side)
- **Concept:** Prevent drastic shift from global model due to non-IID subgraphs.
- **Implementation:**
    - Adds a proximal term to the local loss function: `Loss = TaskLoss + (mu / 2) * ||w - w_global||^2`.
    - `mu` defaults to 0.01.
- **Code:** `FedALAClient.train_prox` implements the custom training loop.

### Option 3: Classifier-only Adaptive Aggregation (Client-side)
- **Concept:** Feature extraction layers are generalized, classifier layers are personalized.
- **Implementation:**
    - `ALAModule` is initialized only for classifier layers (layers containing "classifier" or "fc" in name, or last layers).
    - Other layers use standard global model parameters without adaptive aggregation.
- **Code:** `ALAModule` accepts `target_layers`. `FedALAClient` identifies classifier layers.

## Prerequisites

Ensure you have the following additional libraries installed for plotting:
```bash
pip install seaborn matplotlib pandas
```

## Scripts

### 1. `run_reproduction.py`
- Reproduces vanilla FedALA results (Table 7 baseline).
- Datasets: Cora, CiteSeer, PubMed, Photo, Computers, Chameleon, Actor.
- Trials: 3.
- Output: `fedala_reproduction_results.csv`.

**Run:**
```bash
python run_reproduction.py
```

### 2. `run_fedala_sa.py`
This is the main benchmark script for the proposed method. It runs a comprehensive set of experiments:

**Scenarios:**
1. **Baseline**: Standard FedALA.
2. **Options 1, 2, 3**: The three proposed Structure-Aware strategies.
3. **Privacy**: Option 2 combined with Differential Privacy (Gaussian Mechanism).
4. **Scalability**: Runs the above comparisons for both **10 clients** and **20 clients**.

**Outputs:**
- **CSV Results**: `fedala_sa_final_results.csv` (Contains Mean±Std for all scenarios).
- **Plots**:
    - `fedala_sa_ablation_study.png`: Bar chart comparing Baseline vs Opt 2 vs Privacy.
    - `fedala_sa_convergence_{dataset}.png`: Learning curves (Accuracy vs Rounds) for all methods.
    - `graph_full_scalability.png`: Performance comparison between 10 and 20 clients.

**Run:**
```bash
python run_fedala_sa.py
```

### 3. `run_all.sh`
- Helper script to run both experiments sequentially.
- Automatically handles package installation and execution.

**Run:**
```bash
bash run_all.sh
```

## Expected Results
- `fedala_sa_final_results.csv` will summarize the performance.
- Check the generated `.png` files for visual analysis of convergence and scalability.
