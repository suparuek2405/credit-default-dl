# Credit Card Default Prediction 🏦
> End-to-end Machine Learning & Deep Learning project — from EDA to TabNet

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![TabNet](https://img.shields.io/badge/TabNet-Attentive%20DL-orange)](https://arxiv.org/abs/1908.07442)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org)

---

## 🎯 Project Summary

Built a complete ML/DL pipeline to predict credit card default risk on the UCI dataset
(30,000 customers, Taiwan 2005). Progressed through 4 notebooks: EDA, baseline ML,
custom deep learning, and TabNet — systematically improving AUC at each stage.

**Final result: TabNet achieved AUC 0.7744, beating all traditional ML baselines.**

---

## 📊 Results Progression

| Stage | Model | AUC | Approach |
|---|---|---|---|
| NB02 | Logistic Regression | 0.7451 | Baseline |
| NB02 | XGBoost | 0.7387 | Baseline |
| NB02 | LinearSVC | 0.7456 | Baseline |
| NB02 | **Ensemble (soft vote)** | **0.7598** | **Baseline best** |
| NB03 | Custom MLP + Attention | 0.7597 | Deep Learning |
| NB04 | **TabNet** | **0.7744** | **Final model** |

> **+0.0146 AUC improvement** over best baseline with no resampling required.

---

## 🔍 What This Project Covers

### 01 — Exploratory Data Analysis
- Class imbalance analysis (22% default rate)
- Payment status patterns across 6 months
- Feature correlation heatmap
- Credit limit and bill amount distributions
- Domain-specific insights informing feature engineering

### 02 — Baseline Models
- **Feature engineering:** 17 domain-specific financial features
- **Resampling:** NearMiss v3 undersampling
- **Tuning:** Optuna TPE sampler (50 trials per model)
- **Ensemble:** Soft voting (LR + XGBoost + LinearSVC)
- **Proper evaluation:** val set for tuning, test set for final score

### 03 — Custom Deep Learning
- 4 PyTorch architectures compared (BaselineMLP, ResNet, AttentionMLP, AttentionMultiScale)
- Iterative training improvements (dropout, LR scheduler, gradient clipping)
- Optuna tuning (100 trials, expanded search space)
- **Key finding:** NearMiss + complex models = overfitting on boundary samples

### 04 — TabNet (Final Model)
- Switched to TabNet: purpose-built for tabular data (Arik & Pfister, 2021)
- Dropped NearMiss: used class weights instead → full 17,850 training samples
- Optuna tuning (50 trials)
- Built-in sequential attention: each step specializes in different risk dimensions
- **AUC 0.7744 on unseen test set**

---

## 🧠 Final Model: TabNet Architecture

TabNet uses sequential sparse attention steps, each focusing on different
feature subsets — similar to how a loan officer reviews different aspects
of a credit application.

**Decision flow:**

| Step | Focus | Top Features |
|---|---|---|
| Step 1 | Payment behavior trends | `pay_trend`, `PAY_0` |
| Step 2 | Bill amounts and history | `avg_bill`, `PAY_2` |
| Step 3 | Recent transaction details | `BILL_AMT1`, `PAY_AMT2` |
| Step 4 | Overall risk profile | `max_pay_status`, `LIMIT_BAL` |

**Top 5 most important features:**

| Feature | Type | Importance |
|---|---|---|
| PAY_0 | Raw | 0.160 |
| max_pay_status | Engineered ✨ | 0.135 |
| LIMIT_BAL | Raw | 0.101 |
| pay_trend | Engineered ✨ | 0.063 |
| BILL_AMT1 | Raw | 0.061 |

> 3 of top 10 features are engineered — confirming domain knowledge adds real signal.

**Best hyperparameters (Optuna, 50 trials):**

| Parameter | Value | Meaning |
|---|---|---|
| n_d / n_a | 128 | Decision step width |
| n_steps | 4 | Sequential attention steps |
| gamma | 1.114 | Feature reuse coefficient |
| lambda_sparse | 1.99e-04 | Sparsity regularization |
| learning rate | 3.47e-02 | Adam optimizer LR |
| batch size | 1024 | Training batch size |

---

## 💡 Key Technical Decisions & Lessons

**1. NearMiss vs Class Weights**
NearMiss reduced training data from 17,850 → 6,600 samples. Complex models
overfit these boundary samples in <10 epochs. Switching to class weights kept
all training data and eliminated the overfitting problem entirely.

**2. Feature Engineering Impact**
17 engineered features covering payment trends, utilization ratios, and risk
scores. `max_pay_status` ranked #2 in TabNet importance (0.135), validating
that domain knowledge adds signal beyond raw features.

**3. Architecture vs Data Quality**
4 custom DL architectures all converged to similar AUC (~0.757). TabNet with
full data immediately achieved 0.7784 baseline. Data quality > architecture
complexity for tabular problems.

**4. Proper Train/Val/Test Split**
Val set used exclusively for hyperparameter tuning (Optuna).
Test set used once per notebook for final honest evaluation.
Prevents test set leakage from tuning decisions.

---

## 📁 Project Structure

    credit-default-dl/
    ├── notebooks/
    │   ├── 01_eda.ipynb                  # EDA and domain insights
    │   ├── 02_baseline_models.ipynb      # LR, SVM, XGBoost + Optuna ensemble
    │   ├── 03_deep_learning.ipynb        # Custom PyTorch architecture search
    │   └── 04_tabnet.ipynb               # TabNet final model
    ├── src/
    │   ├── dataset.py                    # Data loading and splitting
    │   ├── features.py                   # Feature engineering pipeline
    │   ├── models.py                     # TabNet model + best params
    │   └── train.py                      # Training and evaluation functions
    └── results/
        ├── metrics.md                    # All experiment results tracked
        └── figures/                      # All charts and visualizations

---

## ⚙️ Quickstart

    git clone https://github.com/suparuek2405/credit-default-dl.git
    cd credit-default-dl
    pip install -r requirements.txt

**Run in Google Colab:**
Open any notebook in `notebooks/` directly in Colab via the GitHub link.

---

## 📦 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| PyTorch | 2.0+ | Deep learning framework |
| pytorch-tabnet | 4.0+ | TabNet architecture |
| scikit-learn | 1.3+ | Preprocessing and baselines |
| XGBoost | 2.0+ | Gradient boosting baseline |
| imbalanced-learn | 0.11+ | NearMiss resampling (NB02-03) |
| Optuna | 3.0+ | Hyperparameter tuning |
| pandas / numpy | latest | Data manipulation |
| matplotlib / seaborn | latest | Visualization |

---

## 📂 Dataset

[UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- 30,000 credit card holders · Taiwan · April–September 2005
- 23 original features · 40 after feature engineering
- 22% default rate (class imbalanced)

---

## 📚 References

- Arik, S. O., & Pfister, T. (2021). **TabNet: Attentive Interpretable Tabular Learning.**
  AAAI 2021. https://arxiv.org/abs/1908.07442
- Wang, S., & Zhang, X. (2024). **Research on Credit Default Prediction Model Based on
  TabNet-Stacking.** Entropy, 26(10), 861. https://doi.org/10.3390/e26100861

---

## 👤 Author

**Suparuek Wattananupan**
Data Scientist (AVP) · TTB Bank · Bangkok, Thailand

Specializing in wealth analytics, deep learning, and financial ML.

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/suparuek2405)
