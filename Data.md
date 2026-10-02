# Data

This project uses the public **Credit Card Fraud Detection** dataset
(anonymized European cardholder transactions, ~285K rows, PCA-transformed
features `V1`–`V28` plus `Time` and `Amount`, with a binary `Class` label).

The raw CSV is **not included** in this repo (file size + it contains
real-world-shaped transaction data), so:

1. Download `creditcard.csv` from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place it here as `data/creditcard.csv`
3. Run the preprocessing + training scripts in `src/`

## Why this dataset is hard

- Only ~0.17% of transactions are fraudulent — a naive classifier that
  always predicts "legitimate" would already be 99.8% "accurate," which
  is exactly why accuracy alone is the wrong metric here.
- Features `V1`–`V28` are PCA components, so there's no direct
  real-world interpretation of individual features (only `Time` and
  `Amount` are raw).
