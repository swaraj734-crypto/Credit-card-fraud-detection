"""
resampling.py
--------------
Helpers for dealing with the extreme class imbalance in the fraud
dataset (~0.17% positive class).

Status: SMOTE oversampling is implemented and working. Still
experimenting with combining it against random undersampling of the
majority class (see README "Next steps") to see which gives a better
precision/recall trade-off without exploding training time.
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def smote_resample(X_train, y_train, random_state: int = 42):
    """Oversample the minority (fraud) class with SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def combined_resample(X_train, y_train, random_state: int = 42):
    """
    Combine SMOTE oversampling with light undersampling of the
    majority class. This is still being tuned — the ratios below are
    a first guess, not a finalized setting.
    """
    over = SMOTE(sampling_strategy=0.1, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
    pipeline = ImbPipeline(steps=[("over", over), ("under", under)])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    return X_res, y_res
