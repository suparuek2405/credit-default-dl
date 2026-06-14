# Credit Card Default Prediction 🏦
> End-to-end Machine Learning & Deep Learning project — from EDA to Foundation Models

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![TabNet](https://img.shields.io/badge/TabNet-Attentive%20DL-orange)](https://arxiv.org/abs/1908.07442)
[![TabPFN](https://img.shields.io/badge/TabPFN--3-Foundation%20Model-purple)](https://github.com/PriorLabs/TabPFN)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.5-blue)](https://auto.gluon.ai)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org)

---

## 🎯 Project Summary

Built a complete ML/DL pipeline to predict credit card default risk on the UCI dataset
(30,000 customers, Taiwan 2005). Progressed through 6 notebooks: EDA, baseline ML,
custom deep learning, TabNet, TabPFN-3, and AutoGluon — systematically improving AUC at each stage.

**Final result: AutoGluon best_quality_v150 achieved AUC 0.7903 — the new project best.**

---

## 📊 Results Progression

| Stage | Model | AUC | Default Recall | Approach |
|---|---|---|---|---|
| NB02 | Logistic Regression | 0.7451 | — | Baseline |
| NB02 | XGBoost | 0.7387 | — | Baseline |
| NB02 | LinearSVC | 0.7456 | — | Baseline |
| NB02 | **Ensemble (soft vote)** | **0.7598** | 0.665 | **Baseline best** |
| NB03 | Custom MLP + Attention | 0.7597 | 0.686 | Deep Learning |
| NB04 | TabNet | 0.7744 | 0.663 | Attentive DL |
| NB05 | TabPFN-3 | 0.7889 | 0.570 | Foundation Model |
| NB06 | **AutoGluon best_quality_v150** | **0.7903** | **0.598** | **Automated Ensemble** |

> **+0.0305 AUC improvement** over best baseline · **+0.0159** over TabNet · AutoGluon is the final best model.

![Results Table](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/23_results_table_nb06.png)

---

## 🔍 The Story

### Chapter 1 — Understanding the Problem

The dataset has a significant class imbalance: only 22% of customers defaulted.
This means a naive model that always predicts "no default" would be 78% accurate
but completely useless for catching real defaulters.

![Class Distribution](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/01_class_distribution.png)

### Chapter 2 — The Strongest Signal

EDA revealed that payment status is the single most predictive feature.
Customers who were 2 or more months late in September (the most recent month) are
overwhelmingly likely to default. Older months show weaker separation,
confirming that recency matters more than payment history.

![Payment Status Analysis](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/04_payment_status.png)

This directly motivated engineering features like `late_pay_count`,
`max_pay_status`, and `pay_trend` — which later ranked in TabNet's top 5.

### Chapter 3 — Establishing Baselines

Three models were tuned with Optuna (50 trials each) and combined into a
soft voting ensemble. The ensemble achieved AUC 0.7598 on the unseen test set,
the target for all deep learning experiments to beat.

![Baseline ROC Curves](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/09_baseline_roc.png)

### Chapter 4 — What TabNet Learned

TabNet's built-in feature importance confirms the EDA hypothesis:
payment behavior dominates. `PAY_0` (most recent payment status) ranks first,
and 3 of the top 10 features are domain-engineered, validating that
financial domain knowledge adds real signal beyond raw features.

![TabNet Feature Importance](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/17_tabnet_feature_importance.png)

### Chapter 5 — How TabNet Makes Decisions

Most models give you a prediction and nothing else. TabNet actually shows its work.

It processes features in 4 sequential decision steps, each one focusing on a
different dimension of credit risk. Think of it like a loan officer reviewing
an application from multiple angles before making a final call.

![TabNet Attention Steps](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/18_tabnet_attention_steps.png)

Step 1 asks: is this customer's payment behavior getting worse? (`pay_trend`, `PAY_0`)

Step 2 asks: how much do they owe and is it growing? (`avg_bill`, `PAY_2`)

Step 3 asks: what happened most recently? (`BILL_AMT1`, `PAY_AMT2`)

Step 4 asks: what is the overall risk picture? (`max_pay_status`, `LIMIT_BAL`)

Only after all 4 steps does it make a final prediction. Each step builds on
the previous one, progressively refining its understanding of the customer.
This is why TabNet is particularly well-suited for financial data — the model
is not just accurate, it is explainable.

### Chapter 6 — TabPFN-3: A Fundamentally Different Approach

TabPFN-3 requires no training loop at all. Instead of learning from your data through
gradient descent, it was pre-trained on millions of synthetic datasets and uses
in-context learning: your training rows become the model's input context,
and it makes predictions in a single forward pass — like a large language model
performing few-shot inference.

`.fit()` stores your data. `.predict()` reads it as a transformer prompt.

Result: AUC 0.7889 with zero configuration — beating TabNet without a single
hyperparameter decision.

![TabPFN-3 ROC](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/19_tabpfn3_roc.png)

> ⚠️ **License:** TabPFN-3 model weights are released under the TABPFN-3.0 License
> (non-commercial use only). This project is a personal portfolio and used accordingly.
> For commercial deployment, contact [PriorLabs](https://priorlabs.ai).

### Chapter 7 — AutoGluon: Automated Ensemble Takes the Lead

AutoGluon best_quality_v150 trains XGBoost, LightGBM, CatBoost, and neural networks
with 8-fold bagging, then stacks them in a 2-layer weighted ensemble. No manual
hyperparameter decisions required. Apache 2.0 licensed — fully commercial safe.

CatBoost was the strongest single model (AUC 0.7906), but the weighted ensemble
(AUC 0.7903) is used as the default predictor for better generalization.

The standout result: AutoGluon achieved **half the false positives of TabPFN-3**
(551 vs 1,044) at a higher recall — a more balanced outcome for a bank use case.

![AutoGluon ROC](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/21_autogluon_roc.png)

Permutation feature importance confirms PAY_0 as the dominant signal (3.5x more
important than any other feature), with 5 of the top 10 features being
domain-engineered — validating the feature engineering pipeline across all model families.

![AutoGluon Feature Importance](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/22_autogluon_importance.png)

---

## 💡 Key Technical Decisions & Lessons

**1. NearMiss vs Class Weights**
NearMiss reduced training data from 17,850 to 6,600 samples. Complex models
overfit these boundary samples in fewer than 10 epochs. Switching to class weights kept
all training data and eliminated the overfitting problem entirely.

**2. Feature Engineering Impact**
17 engineered features covering payment trends, utilization ratios, and risk
scores. `max_pay_status` ranked second in TabNet importance (0.135), and 5 of
AutoGluon's top 10 features are engineered — validating domain knowledge across
all model families.

**3. Architecture vs Data Quality**
4 custom DL architectures all converged to similar AUC (~0.757). TabNet with
full data immediately achieved 0.7784 baseline. Data quality beats architecture
complexity for tabular problems.

**4. Foundation Models on Tabular Data**
TabPFN-3 (zero tuning) beat TabNet (50 Optuna trials) by +0.0145 AUC.
AutoGluon then beat TabPFN-3 by +0.0014, while also halving false positives.
For this dataset, automated ensembles of classical models outperform
in-context learning — but TabPFN wins on simplicity.

**5. Proper Train/Val/Test Split**
Val set used exclusively for hyperparameter tuning (Optuna) and early stopping.
Test set (4,500 rows) used once per notebook for final honest evaluation.
Prevents test set leakage from tuning decisions.

---

## 🧠 Model Summary

### TabNet (NB04)

| Step | Question | Top Features |
|---|---|---|
| Step 1 | Is payment behavior getting worse? | `pay_trend`, `PAY_0` |
| Step 2 | How much do they owe and is it growing? | `avg_bill`, `PAY_2` |
| Step 3 | What happened most recently? | `BILL_AMT1`, `PAY_AMT2` |
| Step 4 | What is the overall risk picture? | `max_pay_status`, `LIMIT_BAL` |

### AutoGluon (NB06) — Top Features

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | PAY_0 | Raw | 0.0245 |
| 2 | late_pay_count | Engineered ✨ | 0.0069 |
| 3 | LIMIT_BAL | Raw | 0.0040 |
| 4 | bill_std | Engineered ✨ | 0.0040 |
| 5 | util_ratio | Engineered ✨ | 0.0038 |

> PAY_0 is 3.5x more important than the next feature across all model families.

---

## 📁 Project Structure

    credit-default-dl/
    ├── notebooks/
    │   ├── 01_eda.ipynb                  # EDA and domain insights
    │   ├── 02_baseline_models.ipynb      # LR, SVM, XGBoost + Optuna ensemble
    │   ├── 03_deep_learning.ipynb        # Custom PyTorch architecture search
    │   ├── 04_tabnet.ipynb               # TabNet final model
    │   ├── 05_tabpfn3.ipynb              # TabPFN-3 foundation model benchmark
    │   └── 06_autogluon.ipynb            # AutoGluon best_quality_v150 (final best)
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

---

## 📦 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| PyTorch | 2.0+ | Deep learning framework |
| pytorch-tabnet | 4.0+ | TabNet architecture |
| tabpfn | 8.0+ | TabPFN-3 foundation model |
| autogluon.tabular | 1.5+ | Automated ensemble (final best) |
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
- Hollmann, N. et al. (2025). **Accurate predictions on small data with a tabular foundation model.**
  Nature. https://doi.org/10.1038/s41586-024-08328-6
- Grinsztajn, L. et al. (2025). **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.**
  arXiv. https://arxiv.org/abs/2511.08667
- Erickson, N. et al. (2020). **AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.**
  ICML AutoML Workshop. https://arxiv.org/abs/2003.06505
- Wang, S., & Zhang, X. (2024). **Research on Credit Default Prediction Model Based on
  TabNet-Stacking.** Entropy, 26(10), 861. https://doi.org/10.3390/e26100861

---

## ⚠️ License Notes

- This project code is for personal portfolio use.
- **TabPFN-3** model weights (NB05) are under the TABPFN-3.0 License — non-commercial only.
- All other models (NB02–NB04, NB06 AutoGluon) are Apache 2.0 / MIT — commercial use permitted.

---

## 👤 Author

**Suparuek Wattananupan**
Data Scientist (AVP) · TTB Bank · Bangkok, Thailand

Specializing in wealth analytics, deep learning, and financial ML.

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Suparuek%20Wattananupan-blue?logo=linkedin)](https://www.linkedin.com/in/suparuek-wattananupan-7509aa181/)
