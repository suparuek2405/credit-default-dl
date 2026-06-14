# Credit Card Default Prediction 🏦
> End-to-end ML/DL project: from EDA to foundation models and automated ensembles

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![TabNet](https://img.shields.io/badge/TabNet-Attentive%20DL-orange)](https://arxiv.org/abs/1908.07442)
[![TabPFN](https://img.shields.io/badge/TabPFN--3-Foundation%20Model-purple)](https://github.com/PriorLabs/TabPFN)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.5-blue)](https://auto.gluon.ai)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org)

---

## Project Summary

This project builds a credit card default prediction pipeline on the UCI dataset (30,000 customers, Taiwan 2005). I worked through 6 notebooks progressively: EDA, baseline ML, custom deep learning, TabNet, TabPFN-3, and AutoGluon, each building on what the previous one taught.

**Best result: AutoGluon best_quality_v150 at AUC 0.7903.**

---

## Results

| Stage | Model | AUC | Default Recall | Approach |
|---|---|---|---|---|
| NB02 | Logistic Regression | 0.7451 | - | Baseline |
| NB02 | XGBoost | 0.7387 | - | Baseline |
| NB02 | LinearSVC | 0.7456 | - | Baseline |
| NB02 | **Ensemble (soft vote)** | **0.7598** | 0.665 | **Baseline best** |
| NB03 | Custom MLP + Attention | 0.7597 | 0.686 | Deep Learning |
| NB04 | TabNet | 0.7744 | 0.663 | Attentive DL |
| NB05 | TabPFN-3 | 0.7889 | 0.570 | Foundation Model |
| NB06 | **AutoGluon best_quality_v150** | **0.7903** | **0.598** | **Automated Ensemble** |

+0.0305 AUC over best baseline, +0.0159 over TabNet. AutoGluon is the final best model.

![Results Table](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/23_results_table_nb06.png)

---

## The Story

### Chapter 1: Understanding the Problem

The dataset has significant class imbalance: only 22% of customers defaulted. A naive model that always predicts "no default" would score 78% accuracy but catch zero real defaulters. The whole project is built around this tension.

![Class Distribution](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/01_class_distribution.png)

### Chapter 2: The Strongest Signal

EDA showed that payment status is the single most predictive feature. Customers who were 2+ months late in September (the most recent month) default at much higher rates. Older months show weaker separation, confirming that recency matters more than full payment history.

![Payment Status Analysis](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/04_payment_status.png)

This motivated engineering features like `late_pay_count`, `max_pay_status`, and `pay_trend`, which later ranked in TabNet's top 5 and AutoGluon's top 10.

### Chapter 3: Establishing Baselines

Three models were tuned with Optuna (50 trials each) and combined into a soft voting ensemble, hitting AUC 0.7598 on the test set. That became the number every later model had to beat.

![Baseline ROC Curves](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/09_baseline_roc.png)

### Chapter 4: What TabNet Learned

TabNet's feature importance confirmed the EDA finding: payment behavior dominates. `PAY_0` ranked first, and 3 of the top 10 features are engineered, showing that domain knowledge adds real signal on top of raw features.

![TabNet Feature Importance](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/17_tabnet_feature_importance.png)

### Chapter 5: How TabNet Makes Decisions

Most models give you a prediction with no explanation. TabNet processes features across 4 sequential decision steps, each focusing on a different dimension of credit risk. It's similar to how a loan officer reviews an application from multiple angles before making a call.

![TabNet Attention Steps](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/18_tabnet_attention_steps.png)

Step 1 asks: is payment behavior getting worse? (`pay_trend`, `PAY_0`)

Step 2 asks: how much do they owe and is it growing? (`avg_bill`, `PAY_2`)

Step 3 asks: what happened most recently? (`BILL_AMT1`, `PAY_AMT2`)

Step 4 asks: what is the overall risk picture? (`max_pay_status`, `LIMIT_BAL`)

Each step builds on the previous one, which is why TabNet works well on financial data where risk has multiple independent dimensions.

### Chapter 6: TabPFN-3 -- A Fundamentally Different Approach

TabPFN-3 has no training loop. Instead of learning through gradient descent, it was pre-trained on millions of synthetic datasets and uses in-context learning: your training rows become the model's input context, and it makes predictions in a single forward pass.

`.fit()` stores your data. `.predict()` reads it like a transformer prompt.

Result: AUC 0.7889 with zero configuration, beating TabNet by +0.0145 without a single hyperparameter decision.

![TabPFN-3 ROC](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/19_tabpfn3_roc.png)

> **License note:** TabPFN-3 weights are under the TABPFN-3.0 License (non-commercial only). This project is a personal portfolio. For commercial use, contact [PriorLabs](https://priorlabs.ai).

### Chapter 7: AutoGluon Takes the Lead

AutoGluon best_quality_v150 trains XGBoost, LightGBM, CatBoost, and neural networks with 8-fold bagging, then stacks them in a 2-layer weighted ensemble. No manual tuning required. Apache 2.0 licensed.

CatBoost was the strongest individual model (AUC 0.7906), but the weighted ensemble (AUC 0.7903) generalises better and is the default predictor.

The most interesting result was not just the AUC gain but the false positives: AutoGluon produced 551 vs TabPFN-3's 1,044, at higher recall. For a bank where false alarms have real costs, that matters more than a 0.0014 AUC difference.

![AutoGluon ROC](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/21_autogluon_roc.png)

Permutation feature importance confirmed PAY_0 as the dominant signal (3.5x more important than any other feature), with 5 of the top 10 being engineered features.

![AutoGluon Feature Importance](https://raw.githubusercontent.com/suparuek2405/credit-default-dl/main/results/figures/22_autogluon_importance.png)

---

## Key Decisions and Lessons

**NearMiss vs class weights.** NearMiss reduced training data from 17,850 to 6,600 samples. Complex models overfit those boundary samples quickly. Switching to class weights kept all training data and removed the problem entirely.

**Feature engineering impact.** 17 engineered features covering payment trends, utilization ratios, and risk scores. `max_pay_status` ranked second in TabNet (importance 0.135), and 5 of AutoGluon's top 10 are engineered. Domain knowledge consistently added value across all model families.

**Architecture vs data quality.** Four custom DL architectures all converged near AUC 0.757. TabNet with full data immediately hit 0.7784. Data quality beats architecture complexity for tabular problems.

**Foundation models on tabular data.** TabPFN-3 with zero tuning beat TabNet (50 Optuna trials) by +0.0145. AutoGluon then beat TabPFN-3 by +0.0014 while halving false positives. For this dataset, ensembles of classical models still win, but TabPFN is the better default when you need something fast with no setup.

**Train/val/test split discipline.** Val set used only for hyperparameter tuning and early stopping. Test set (4,500 rows) touched once per notebook for final evaluation. This is the only way to make AUC numbers across notebooks actually comparable.

---

## Model Summary

### TabNet (NB04)

| Step | Question | Top Features |
|---|---|---|
| Step 1 | Is payment behavior getting worse? | `pay_trend`, `PAY_0` |
| Step 2 | How much do they owe and is it growing? | `avg_bill`, `PAY_2` |
| Step 3 | What happened most recently? | `BILL_AMT1`, `PAY_AMT2` |
| Step 4 | What is the overall risk picture? | `max_pay_status`, `LIMIT_BAL` |

### AutoGluon (NB06) -- Top 5 Features

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | PAY_0 | Raw | 0.0245 |
| 2 | late_pay_count | Engineered | 0.0069 |
| 3 | LIMIT_BAL | Raw | 0.0040 |
| 4 | bill_std | Engineered | 0.0040 |
| 5 | util_ratio | Engineered | 0.0038 |

PAY_0 is 3.5x more important than the second feature across all model families.

---

## Project Structure

    credit-default-dl/
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_baseline_models.ipynb
    │   ├── 03_deep_learning.ipynb
    │   ├── 04_tabnet.ipynb
    │   ├── 05_tabpfn3.ipynb
    │   └── 06_autogluon.ipynb
    ├── src/
    │   ├── dataset.py
    │   ├── features.py
    │   ├── models.py
    │   └── train.py
    └── results/
        ├── metrics.md
        └── figures/

---

## Quickstart

    git clone https://github.com/suparuek2405/credit-default-dl.git
    cd credit-default-dl
    pip install -r requirements.txt

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| PyTorch | 2.0+ | Deep learning |
| pytorch-tabnet | 4.0+ | TabNet |
| tabpfn | 8.0+ | TabPFN-3 foundation model |
| autogluon.tabular | 1.5+ | Automated ensemble |
| scikit-learn | 1.3+ | Preprocessing and baselines |
| XGBoost | 2.0+ | Gradient boosting |
| imbalanced-learn | 0.11+ | NearMiss resampling (NB02-03) |
| Optuna | 3.0+ | Hyperparameter tuning |
| pandas / numpy | latest | Data manipulation |
| matplotlib / seaborn | latest | Visualization |

---

## Dataset

[UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

30,000 credit card holders, Taiwan, April to September 2005. 23 original features, 40 after engineering. 22% default rate.

---

## References

- Arik, S. O., & Pfister, T. (2021). **TabNet: Attentive Interpretable Tabular Learning.** AAAI 2021. https://arxiv.org/abs/1908.07442
- Hollmann, N. et al. (2025). **Accurate predictions on small data with a tabular foundation model.** Nature. https://doi.org/10.1038/s41586-024-08328-6
- Grinsztajn, L. et al. (2025). **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.** arXiv. https://arxiv.org/abs/2511.08667
- Erickson, N. et al. (2020). **AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.** ICML AutoML Workshop. https://arxiv.org/abs/2003.06505
- Wang, S., & Zhang, X. (2024). **Research on Credit Default Prediction Model Based on TabNet-Stacking.** Entropy, 26(10), 861. https://doi.org/10.3390/e26100861

---

## License Notes

This project code is for personal portfolio use. TabPFN-3 weights (NB05) are under the TABPFN-3.0 License (non-commercial only). All other models use Apache 2.0 or MIT.

---

## Author

**Suparuek Wattananupan**
Data Scientist (AVP), TTB Bank, Bangkok

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Suparuek%20Wattananupan-blue?logo=linkedin)](https://www.linkedin.com/in/suparuek-wattananupan-7509aa181/)
