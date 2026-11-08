"""Constant fallback, validation and atomic writing of the answer file."""
import os

import numpy as np
import pandas as pd

from automl.metrics import MEDIAN_OPTIMAL
from automl.task import TaskSpec


def fallback_predictions(train: pd.DataFrame, n_rows: int, spec: TaskSpec):
    y = train[spec.target].dropna()
    if spec.task_type == "regression":
        value = y.median() if spec.metric in MEDIAN_OPTIMAL else y.mean()
    elif spec.pred_format == "proba":
        value = float((y == spec.positive_class).mean())
    else:
        value = y.mode().iloc[0]
    return np.full(n_rows, value, dtype=object if isinstance(value, str) else None)


def write_submission(test: pd.DataFrame, pred, spec: TaskSpec) -> None:
    """Validate and write; on any problem raise ValueError and leave the old file untouched."""
    pred = pd.Series(np.asarray(pred))
    if len(pred) != len(test):
        raise ValueError(f"{len(pred)} predictions for {len(test)} test rows")
    if pred.isna().any():
        raise ValueError("NaN in predictions")
    if spec.pred_format == "label":
        unknown = set(pred) - set(spec.classes)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown, key=str)[:5]}")
    else:
        pred = pred.astype(float)
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite predictions")
        if spec.pred_format == "proba":
            pred = pred.clip(0, 1)
    ids = test[spec.id_col] if spec.id_col in test.columns else pd.read_csv(spec.sample_path)[spec.id_col]
    if ids.duplicated().any():
        raise ValueError("Duplicate ids")
    out = pd.DataFrame({spec.id_col: ids.to_numpy(), spec.answer_col: pred.to_numpy()})
    tmp = spec.output_path.with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, spec.output_path)  # never leave a half-written file behind
