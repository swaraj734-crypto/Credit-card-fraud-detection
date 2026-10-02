"""
train_baseline.py
-------------------
Trains and compares two baseline classifiers (Logistic Regression and
Random Forest) on the SMOTE-resampled training data, and reports the
metrics that actually matter for a fraud problem: precision, recall,
F1, and ROC-AUC — NOT accuracy.

Status: baseline only. This is intentionally simple for now so there's
a clear number to beat before trying heavier models (see README
"Next steps" for the planned XGBoost / LightGBM comparison).

Usage:
    python src/train_baseline.py --data data/creditcard.csv
"""

import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from preprocessing import (
    load_data,
    scale_amount_and_time,
    split_features_target,
    train_test_split_stratified,
)
from resampling import smote_resample


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))


def main(data_path: str, model_dir: str):
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = scale_amount_and_time(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    print("Applying SMOTE to the training set ...")
    X_train_res, y_train_res = smote_resample(X_train, y_train)
    print(f"Training set size after SMOTE: {X_train_res.shape[0]} rows")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ),
    }

    best_model = None
    best_name = None
    best_auc = -1

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_res, y_train_res)
        evaluate(name, model, X_test, y_test)

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print(f"\nBest baseline so far: {best_name} (ROC-AUC = {best_auc:.4f})")
    joblib.dump(best_model, f"{model_dir}/fraud_baseline_model.pkl")
    print(f"Saved best model to {model_dir}/fraud_baseline_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    main(args.data, args.model_dir)
