"""Metric registry: names, families and scoring on out-of-fold predictions.

All metric knowledge lives here; task.py only maps text to these names.
"""
import numpy as np
from sklearn import metrics as skm

REGRESSION = {"rmse", "mse", "mae", "mape", "rmsle", "r2"}
LABEL = {"f1_macro", "f1_micro", "f1_weighted", "f1", "accuracy", "balanced_accuracy"}
PROBA = {"roc_auc", "logloss", "pr_auc"}
CLASSIFICATION = LABEL | PROBA
GREATER_IS_BETTER = {"r2"} | LABEL | {"roc_auc", "pr_auc"}

MEDIAN_OPTIMAL = {"mae", "mape"}  # constant fallback uses median instead of mean
CLASS_BALANCED = {"f1_macro", "f1", "balanced_accuracy"}  # train with class_weight="balanced"

DEFAULT_METRIC = {"regression": "rmse", "binary": "roc_auc", "multiclass": "f1_macro"}


def score(metric: str, y_true, y_pred=None, proba=None, positive_index: int = 1) -> float:
    """y_pred: values or encoded labels; proba: (n, k) class probabilities."""
    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if metric == "mse":
        return float(skm.mean_squared_error(y_true, y_pred))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "rmsle":
        return float(np.sqrt(skm.mean_squared_log_error(y_true, np.clip(y_pred, 0, None))))
    if metric == "r2":
        return float(skm.r2_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(skm.f1_score(y_true, y_pred, average="macro"))
    if metric == "f1_micro":
        return float(skm.f1_score(y_true, y_pred, average="micro"))
    if metric == "f1_weighted":
        return float(skm.f1_score(y_true, y_pred, average="weighted"))
    if metric == "f1":
        if len(np.unique(y_true)) > 2:  # "F1" on a multiclass task means macro
            return float(skm.f1_score(y_true, y_pred, average="macro"))
        return float(skm.f1_score(y_true, y_pred, pos_label=positive_index))
    if metric == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y_true, y_pred))
    if metric == "roc_auc":
        return float(skm.roc_auc_score(y_true == positive_index, proba[:, positive_index]))
    if metric == "pr_auc":
        return float(skm.average_precision_score(y_true == positive_index, proba[:, positive_index]))
    if metric == "logloss":
        return float(skm.log_loss(y_true, proba, labels=np.arange(proba.shape[1])))
    raise ValueError(f"Unknown metric: {metric}")
