# 🔬 Federated Learning Baselines — Multi-Round Experiments

This repository contains controlled implementations of classical **multi-round Federated Learning (FL)** baselines used for rigorous comparison with one-shot methods (e.g., GH-OFL).

The goal is strict experimental fairness across methods.

---

## 📌 Implemented Methods

- **FedAvg**
- **FedProx (μ = 0.01)**
- **SCAFFOLD (weighted, fair implementation)**

---

## 📊 Evaluated Datasets

- **CIFAR-10**
- **CIFAR-100**
- **SVHN**

All experiments use **Dirichlet client partitions** with:

```
α ∈ {0.5, 0.1, 0.05}
```

to simulate different levels of statistical heterogeneity.

---

## 📂 Repository Structure

```
SCAFFOLD_baseline/
│
├── fedavg_c10_fast.py
├── fedavg_c100.py
├── fedavg_svhn.py
│
├── fedprox_c10.py
├── fedprox_c100.py
├── fedprox_svhn.py
│
├── scaffold.py
│
├── visual.py              # Generates plots (results are hardcoded)
│
├── *.pdf                  # Generated plots
└── data/                  # Automatically downloaded datasets
```

---

## ⚙️ Experimental Configuration (IDENTICAL across methods)

All baselines share the same hyperparameters to guarantee strict comparability:

- **Backbone:** ResNet-18 (ImageNet pretrained)
- **Clients:** 10
- **Local epochs:** 1
- **Batch size:** 256
- **Rounds:** 50
- **Optimizer:** SGD (momentum = 0.9)
- **Learning rate:** 0.001
- **Seed:** 42

This ensures that performance differences arise from the **algorithm**, not tuning.

---

## ❗ Why LOCAL_EPOCHS = 1?

This choice is intentional.

It:

- Ensures direct comparability with one-shot methods
- Forces multi-round FL into a constrained regime
- Highlights communication efficiency differences
- Reveals behavior under minimal local training

Under this setting, drift-correction methods (especially SCAFFOLD) operate in a limited regime.

---

## ▶️ How to Run Experiments

Example:

```bash
python fedavg_c100.py
python fedprox_svhn.py
python scaffold.py
```

Datasets are automatically downloaded into `./data`.

---

## 📈 Plot Generation

`visual.py` generates plots by hardcoding the baseline results.

It does **not** run training — it only produces figures from stored values.

Example:

```bash
python visual.py
```

---

## 🔍 Implementation Notes

- FedAvg and FedProx use **weighted aggregation**.
- SCAFFOLD uses weighted updates for both model parameters and control variates.
- Architectures are dataset-consistent.
- No hyperparameter advantages are given to any method.
- Strict fairness is enforced across all baselines.

---

## 🧪 Research Context

These baselines are designed for controlled comparison with one-shot federated methods.  
They intentionally operate under constrained local training to highlight:

- Communication cost differences  
- Sensitivity to non-IID heterogeneity  
- Convergence stability  
- Algorithmic robustness  

---

**Maintainer:** Fabio Turazza  
PhD Student — Federated Learning & One-Shot Methods
