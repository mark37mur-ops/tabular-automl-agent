"""Метрики полигона и перевод в баллы по формуле из приложения 3 кейса."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, roc_auc_score

LOWER_IS_BETTER = {"rmse", "rmsle", "mae", "logloss"}


def compute(metric: str, y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "rmsle":
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None))) ** 2)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "logloss":
        return float(log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15)))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric == "f1":
        return float(f1_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Неизвестная метрика: {metric}")


def to_score(metric: str, value: float, baseline: float) -> float:
    """Балл 0–100: уровень бейзлайна и хуже даёт 0, идеальный ответ — 100."""
    if metric in LOWER_IS_BETTER:
        return 100.0 * max(0.0, 1.0 - value / baseline)
    return 100.0 * max(0.0, (value - baseline) / (1.0 - baseline))
