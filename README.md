# Credit Card Fraud Detection — 🚧 Work in Progress

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

A machine learning project for detecting fraudulent credit card
transactions in a highly imbalanced dataset (~0.17% of transactions
are fraud). This repo is actively being built out — see
[Current status](#current-status) and [Next steps](#next-steps)
below for exactly what's done vs. still in progress.

## Problem

With fraud making up only ~0.17% of transactions, a model that
predicts "legitimate" every single time would score 99.8% accuracy
while catching zero fraud. The real challenge here isn't accuracy —
it's getting a usable **recall on the fraud class** without flooding
the system with false positives.

## Current status

- [x] Data loading + scaling (`Amount`, `Time`) — `src/preprocessing.py`
- [x] Stratified train/test split to preserve class ratio
- [x] SMOTE oversampling for the minority (fraud) class — `src/resampling.py`
- [x] Baseline models: Logistic Regression + Random Forest, compared
      on precision/recall/F1/ROC-AUC — `src/train_baseline.py`
- [x] Initial EDA notebook (class imbalance + correlation pass) —
      `notebooks/01_eda.ipynb`
- [ ] Gradient boosting comparison (XGBoost / LightGBM)
- [ ] Threshold tuning for the precision/recall trade-off
      (instead of the default 0.5 cutoff)
- [ ] Batch inference script for scoring a CSV of new transactions
- [ ] Streamlit app for interactive scoring
- [ ] Model explainability pass (SHAP) once a final model is picked

## Project structure

```
credit-card-fraud-detection/
├── notebooks/
│   └── 01_eda.ipynb          # exploratory analysis (in progress)
├── src/
│   ├── preprocessing.py      # loading + scaling
│   ├── resampling.py         # SMOTE / combined resampling
│   └── train_baseline.py     # baseline model training + evaluation
├── data/
│   └── README.md             # where to get the dataset (not committed)
├── models/                   # trained models land here (gitignored)
└── requirements.txt
```

## Getting started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get the dataset** — see `data/README.md` for the Kaggle link and
   where to place the CSV.

3. **Run the baseline**
   ```bash
   python src/train_baseline.py --data data/creditcard.csv
   ```
   This prints precision/recall/F1/ROC-AUC for both baseline models
   and saves whichever one scores higher on ROC-AUC.

4. **(Optional) Explore the data**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Why SMOTE here instead of just class weights?

Both are reasonable options for this dataset. I'm currently comparing
plain SMOTE oversampling against a combined SMOTE + undersampling
pipeline (`src/resampling.py`) to see which gives a better
precision/recall balance — this comparison isn't finalized yet, which
is part of why this repo is still marked in progress.

## Next steps

See the unchecked items in [Current status](#current-status) above —
the immediate priority is the XGBoost/LightGBM comparison and
threshold tuning, since the baseline models' default 0.5 decision
threshold is almost certainly not optimal for this level of class
imbalance.

## Disclaimer

Educational project. Not intended for use in a real fraud-detection
pipeline without significant further validation.

---
Developed by **Shubham Swaraj** — B.Tech CSE (AI & DS), IIIT Senapati, Manipur
