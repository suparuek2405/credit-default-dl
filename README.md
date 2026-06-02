# Credit Card Default Prediction 🏦
> Deep Learning project built with PyTorch — predicting credit card default using the UCI dataset

---

## 📌 Project Overview
This project builds and compares multiple machine learning and deep learning models
to predict whether a credit card holder will default next month.

Full pipeline covered:
- Exploratory Data Analysis (class imbalance, correlations, distributions)
- Baseline models (Logistic Regression, SVM, XGBoost)
- Deep learning architecture search
- Feature engineering (domain-specific financial features)
- Handling class imbalance (NearMiss undersampling)

---

## 📁 Project Structure

    credit-default-dl/
    ├── notebooks/
    │   ├── 01_eda.ipynb                  # Exploratory data analysis
    │   ├── 02_baseline_models.ipynb      # LR, SVM, XGBoost benchmarks
    │   └── 03_deep_learning.ipynb        # DL experiments + architecture search
    ├── src/
    │   ├── dataset.py                    # PyTorch Dataset class
    │   ├── features.py                   # Feature engineering pipeline
    │   ├── models.py                     # All model architectures
    │   └── train.py                      # Training loop + early stopping
    └── results/
        ├── metrics.md                    # All experiment results tracked
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
| PyTorch | Model building and training |
| scikit-learn | Preprocessing and baselines |
| XGBoost | Gradient boosting baseline |
| imbalanced-learn | NearMiss resampling |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |

---

## 📂 Dataset
[UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- 30,000 credit card holders
- Taiwan, April–September 2005
- 22% default rate (class imbalanced)

---

## 👤 Author
**Suparuek Wattananupan**
Data Scientist · PyTorch · Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)

---
*Project status: Work in progress — updated regularly*
