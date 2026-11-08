"""Feature preparation and K-fold LightGBM with a wall-clock deadline."""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from automl.metrics import CLASS_BALANCED, GREATER_IS_BETTER, score
from automl.task import TaskSpec

SEED = 42
N_SPLITS = 5
try:
    N_JOBS = max(1, min(8, int(os.environ.get("AGENT_N_JOBS", min(8, os.cpu_count() or 8)))))
except ValueError:
    N_JOBS = min(8, os.cpu_count() or 8)
DEFAULT_PARAMS = dict(n_estimators=5000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=SEED, n_jobs=N_JOBS, verbose=-1)


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec):
    """Same columns in train and test; strings become shared pandas categories.

    Columns are renamed to f0..fN (LightGBM rejects JSON characters in names);
    returns X, X_test, names (original name of each fi), dropped columns.
    """
    columns = [c for c in test.columns if c in train.columns and c not in (spec.id_col, spec.target)]
    X, X_test = train[columns].copy(), test[columns].copy()
    dropped = []
    for col in columns:
        both = pd.concat([X[col], X_test[col]], ignore_index=True)
        if both.nunique(dropna=False) <= 1:
            dropped.append(col)
        elif pd.api.types.is_bool_dtype(both):
            X[col], X_test[col] = X[col].astype(float), X_test[col].astype(float)
        elif not pd.api.types.is_numeric_dtype(both):
            if both.nunique() > max(1000, 0.5 * len(both)):  # free text / row ids: useless for trees
                dropped.append(col)
                continue
            categories = pd.Index(both.dropna().astype(str).unique())
            X[col] = pd.Categorical(X[col].astype("string"), categories=categories)
            X_test[col] = pd.Categorical(X_test[col].astype("string"), categories=categories)
        else:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
    X, X_test = X.drop(columns=dropped), X_test.drop(columns=dropped)
    names = list(X.columns)
    X.columns = X_test.columns = [f"f{i}" for i in range(len(names))]
    return X, X_test, names, dropped


def _model(spec: TaskSpec, params: dict | None):
    params = {**DEFAULT_PARAMS, **(params or {})}
    if spec.task_type == "regression":
        objective = {"mae": "regression_l1", "mape": "mape"}.get(spec.metric, "regression")
        return lgb.LGBMRegressor(objective=objective, **params)
    weight = "balanced" if spec.metric in CLASS_BALANCED else None
    return lgb.LGBMClassifier(class_weight=weight, **params)


def _stop_at(deadline: float):
    """LightGBM callback: stop boosting when the wall clock passes the deadline."""
    def callback(env):
        if time.time() > deadline:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    callback.order = 40
    return callback


def _folds(y: np.ndarray, classification: bool):
    """K-fold indices. Classes with fewer than N_SPLITS rows always stay in training folds:
    LightGBM cannot validate on a label it has not seen in training."""
    if not classification:
        return list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(y))
    counts = np.bincount(y)
    rare = np.flatnonzero(counts[y] < N_SPLITS)
    common = np.flatnonzero(counts[y] >= N_SPLITS)
    if len(common) < 2 * N_SPLITS:
        raise ValueError("too few rows of frequent classes for cross-validation")
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    return [(np.concatenate([common[tr], rare]), common[va])
            for tr, va in splitter.split(common, y[common])]


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec, deadline: float,
                      hard_deadline: float | None = None, params: dict | None = None,
                      log=print) -> dict:
    """K-fold LightGBM; test predictions are averaged over the folds that finished.

    deadline: absolute time.time() after which no new fold starts (the first fold always runs).
    hard_deadline: absolute time after which boosting is cut short (default: deadline).
    params: LightGBM params merged over DEFAULT_PARAMS.
    Returns {"pred", "oof", "cv_score", "folds"}: pred is in test row order and in answer
    format (values | labels | P(positive)); oof and cv_score are on the original target scale
    (oof holds class probabilities for classification, NaN for rows never validated).
    Score direction: metrics.GREATER_IS_BETTER.
    """
    train = train[train[spec.target].notna()].reset_index(drop=True)
    X, X_test, names, dropped = prepare_features(train, test, spec)
    log(f"features: {len(names)} used, dropped: {dropped}")
    y_raw = train[spec.target]
    classification = spec.task_type != "regression"

    if classification:
        classes = list(spec.classes)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).to_numpy(int)
        n_out = len(classes)
        oof, test_sum = np.full((len(y), n_out), np.nan), np.zeros((len(X_test), n_out))
    else:
        log_target = spec.metric == "rmsle" and y_raw.min() > -1
        y = np.log1p(y_raw.to_numpy(float)) if log_target else y_raw.to_numpy(float)
        oof, test_sum = np.full(len(y), np.nan), np.zeros(len(X_test))

    folds_done, fold_times = 0, []
    for fold, (tr, va) in enumerate(_folds(y, classification)):
        remaining = deadline - time.time()
        if folds_done and remaining < 1.5 * max(fold_times):
            log(f"time budget: stop after {folds_done} folds")
            break
        started = time.time()
        try:
            model = _model(spec, params)
            model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 _stop_at(hard_deadline or deadline)])
            if classification:
                cols = model.classes_  # a fold may miss a rare class: map back to full matrix
                proba_valid = np.zeros((len(va), n_out))
                proba_valid[:, cols] = model.predict_proba(X.iloc[va])
                proba_test = np.zeros((len(X_test), n_out))
                proba_test[:, cols] = model.predict_proba(X_test)
                if not np.isfinite(proba_valid).all() or not np.isfinite(proba_test).all():
                    raise ValueError("non-finite fold probabilities")
                oof[va] = proba_valid
                test_sum += proba_test
            else:
                pred_valid = model.predict(X.iloc[va])
                pred_test = model.predict(X_test)
                if not np.isfinite(pred_valid).all() or not np.isfinite(pred_test).all():
                    raise ValueError("non-finite fold predictions")
                oof[va] = pred_valid
                test_sum += pred_test
        except Exception as error:  # one broken fold must not cost the whole model
            log(f"fold {fold} failed: {error!r}")
            continue
        folds_done += 1
        fold_times.append(time.time() - started)
        log(f"fold {fold}: {fold_times[-1]:.1f}s, best_iteration={model.best_iteration_}")
    if folds_done == 0:
        raise RuntimeError("no fold finished")

    seen = ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])
    test_avg = test_sum / folds_done
    if classification:
        # For multiclass metrics positive_index is ignored.
        pos = classes.index(spec.positive_class) if spec.task_type == "binary" else 1
        if spec.pred_format == "proba":
            pred = test_avg[:, pos]
        else:
            pred = np.array(classes, dtype=object)[test_avg.argmax(1)]
        oof_pred, oof_proba = oof[seen].argmax(1), oof[seen]
    else:
        if log_target:
            oof, test_avg, y = np.expm1(oof), np.expm1(test_avg), y_raw.to_numpy(float)
        low, high = y_raw.min(), y_raw.max()  # trees cannot extrapolate; keep linear blends sane
        pred = np.clip(test_avg, low, high)
        oof = np.clip(oof, low, high)
        pos, oof_pred, oof_proba = 1, oof[seen], None

    try:
        cv = score(spec.metric, y[seen], oof_pred, proba=oof_proba, positive_index=pos)
        direction = "higher" if spec.metric in GREATER_IS_BETTER else "lower"
        log(f"CV {spec.metric} = {cv:.5f} ({direction} is better) on {seen.sum()} rows, "
            f"{folds_done} folds")
    except Exception as error:  # scoring is for logs only; never lose predictions over it
        cv = float("nan")
        log(f"CV scoring failed: {error!r}")
    return {"pred": pred, "oof": oof, "cv_score": cv, "folds": folds_done}
