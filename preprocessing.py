"""
preprocessing.py
-----------------
Loading and preprocessing utilities for the credit card fraud dataset.

Status: working, but feature engineering is still minimal — see
README.md "Next steps" for what's planned next.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Class"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw transaction data."""
    return pd.read_csv(csv_path)


def scale_amount_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    V1-V28 already come PCA-scaled from the source dataset; only
    `Amount` and `Time` need explicit scaling before modeling.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_test_split_stratified(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split is important here — with ~0.17% positive class,
    a plain random split can easily leave the test set with almost
    no fraud examples at all.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
