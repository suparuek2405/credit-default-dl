# Credit Card Default Prediction 🏦
> Deep Learning project built with PyTorch — predicting credit card default using the UCI dataset

---

## 📊 Final Results

| Model | AUC | Recall | FP | FN |
|---|---|---|---|---|
| Logistic Regression | 0.7451 | 58.4% | 696 | 414 |
| XGBoost | 0.7387 | 70.8% | 970 | 291 |
| LinearSVC | 0.7456 | 59.4% | 714 | 404 |
| Baseline Ensemble | 0.7598 | 66.5% | 831 | 333 |
| DL Custom MLP (NB03) | 0.7597 | 68.6% | 989 | 312 |
| **TabNet (NB04)** | **0.7744** | **66.3%** | **870** | **335** |

> TabNet beats baseline ensemble by **+0.0146 AUC** with no resampling.

---

## 📌 Project Overview
End-to-end deep learning project predicting credit card default.
Full pipeline: EDA → Baseline ML → Custom DL → TabNet.

### Key Findings
- Feature engineering (17 new features) was the single biggest signal driver
- `PAY_0`, `max_pay_status`, and `late_pay_count` are the strongest predictors
- TabNet's sequential attention naturally handles imbalanced tabular data
- Class weights outperform NearMiss undersampling for deep learning
- TabNet attention steps each specialize in different risk dimensions

---

## 🧠 Architecture: TabNet

Purpose-built tabular deep learning model (Arik & Pfister, 2021).
Uses sequential sparse attention steps — each step focuses on
different feature subsets, mimicking how a loan officer reviews
an application.

**Decision flow:**

1. Input (40 features)
2. Step 1 — Payment behavior trends: `pay_trend`, `PAY_0`
3. Step 2 — Bill amounts and history: `avg_bill`, `PAY_2`
4. Step 3 — Recent transaction details: `BILL_AMT1`, `PAY_AMT2`
5. Step 4 — Overall risk profile: `max_pay_status`, `LIMIT_BAL`
6. Final prediction → P(default)

**Best hyperparameters (Optuna, 50 trials):**

| Parameter | Value |
|---|---|
| n_d / n_a | 128 |
| n_steps | 4 |
| gamma | 1.114 |
| learning rate | 3.47e-02 |
| batch size | 1024 |

---

## 📁 Project Structure

    credit-default-dl/
    ├── notebooks/
    │   ├── 01_eda.ipynb                  # Exploratory data analysis
    │   ├── 02_baseline_models.ipynb      # LR, SVM, XGBoost + Optuna
    │   ├── 03_deep_learning.ipynb        # Custom DL architecture search
    │   └── 04_tabnet.ipynb               # TabNet final model
    ├── src/
    │   ├── dataset.py                    # Data loading and splitting
    │   ├── features.py                   # Feature engineering pipeline
    │   ├── models.py                     # TabNet model + best params
    │   └── train.py                      # Training and evaluation functions
    └── results/
        ├── metrics.md                    # All experiment results
        └── figures/                      # All charts and plots

---

## ⚙️ Setup

    git clone https://github.com/suparuek2405/credit-default-dl.git
    cd credit-default-dl
    pip install -r requirements.txt

---

## 📦 Stack

| Tool | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| pytorch-tabnet | TabNet architecture |
| scikit-learn | Preprocessing and baselines |
| XGBoost | Gradient boosting baseline |
| imbalanced-learn | NearMiss resampling (NB02-03) |
| Optuna | Hyperparameter tuning |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |

---

## 📂 Dataset
[UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- 30,000 credit card holders
- Taiwan, April–September 2005
- 22% default rate (class imbalanced)

---

## 📚 References
- Arik, S. O., & Pfister, T. (2021). TabNet: Attentive Interpretable
  Tabular Learning. AAAI 2021. https://arxiv.org/abs/1908.07442
- Wang, S., & Zhang, X. (2024). Credit Default Prediction Based on
  TabNet-Stacking. Entropy, 26(10). https://doi.org/10.3390/e26100861

---

## 👤 Author
**Suparuek Wattananupan**
Data Scientist · PyTorch · Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)

---
*Project complete — all notebooks and results available above.*
