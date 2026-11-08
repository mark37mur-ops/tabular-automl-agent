"""v2 ML core: one fixed validation scheme for every experiment, three model families.

Problem fixes the encoded target, the research/holdout split and the K folds once, so every
candidate (feature recipe, model, params) is compared on exactly the same rows.
"""
import os
import re
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from automl.metrics import CLASS_BALANCED, GREATER_IS_BETTER, PROBA, score
from automl.task import TaskSpec

SEED = 42
N_SPLITS = 5
try:
    N_JOBS = max(1, min(8, int(os.environ.get("AGENT_N_JOBS", min(8, os.cpu_count() or 8)))))
except ValueError:
    N_JOBS = min(8, os.cpu_count() or 8)
HOLDOUT_FRAC = 0.15
MIN_ROWS_FOR_HOLDOUT = 3000
MIN_TEMPORAL_TRAIN_ROWS = 100
MIN_TEMPORAL_VALID_ROWS = 20


@dataclass
class Problem:
    spec: TaskSpec
    y: np.ndarray            # encoded labels (int) or float target, all train rows
    y_raw: np.ndarray        # original target values (for scoring regression)
    classes: list | None
    research_idx: np.ndarray  # rows used for CV
    holdout_idx: np.ndarray   # rows never used for choices until the final check
    folds: list = field(default_factory=list)  # (train_pos, valid_pos) into research_idx
    scheme: str = "random"
    time_column: str | None = None
    oof_mask: np.ndarray | None = None  # expected coverage, positions into research_idx
    metadata: dict = field(default_factory=dict)

    @property
    def classification(self) -> bool:
        return self.spec.task_type != "regression"

    @property
    def n_out(self) -> int:
        return len(self.classes) if self.classification else 0

    @property
    def positive_index(self) -> int:
        if self.spec.task_type == "binary":
            return self.classes.index(self.spec.positive_class)
        return 1

    def better(self, a: float, b: float) -> bool:
        """True if score a is better than score b."""
        return a > b if self.spec.metric in GREATER_IS_BETTER else a < b

    def gain(self, new: float, old: float) -> float:
        """Positive when new is better than old."""
        return new - old if self.spec.metric in GREATER_IS_BETTER else old - new

    def score_pred(self, pred: np.ndarray, rows: np.ndarray, class_weights=None) -> float:
        """pred: values (regression, original scale) or class probabilities for `rows`."""
        if not self.classification:
            return score(self.spec.metric, self.y_raw[rows], pred)
        proba = pred if class_weights is None else pred * class_weights
        labels = proba.argmax(1)
        return score(self.spec.metric, self.y[rows], labels, proba=pred, positive_index=self.positive_index)


_TIME_WORDS = re.compile(r"\b(year|period|date|datetime|timestamp|time|год\w*|период\w*|дат[аы]\w*|врем\w*)\b", re.I)
_HISTORICAL_WORDS = re.compile(r"\b(birth\w*|built|construction|age|постро\w*|рожд\w*|возраст\w*)\b", re.I)


def _time_text(value) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).replace("_", " ")


def _future_time_column(train: pd.DataFrame, test: pd.DataFrame | None, spec: TaskSpec):
    """Conservative out-of-time evidence, using features only, never target/test labels.

    An unrelated monotonic row ID is insufficient: the name or dictionary must describe
    time, values must be numeric and complete, and every test value must be in the future.
    Small datasets retain the existing random policy rather than creating tiny holdouts.
    """
    if test is None or test.empty or len(train) < MIN_ROWS_FOR_HOLDOUT:
        return None
    descriptions = {}
    if spec.dict_path is not None:
        try:
            dictionary = pd.read_csv(spec.dict_path)
            key = next((c for c in ("feature", "column", "name") if c in dictionary.columns), None)
            if key is not None:
                for _, row in dictionary.iterrows():
                    descriptions[str(row[key])] = " ".join(
                        str(v) for c, v in row.items() if c != key and pd.notna(v))
        except (OSError, ValueError, pd.errors.ParserError):
            pass  # dictionary hints are optional
    candidates = []
    for col in train.columns:
        if col in (spec.id_col, spec.target) or col not in test.columns:
            continue
        meaning = _time_text(col) + " " + _time_text(descriptions.get(str(col), ""))
        if not _TIME_WORDS.search(meaning) or _HISTORICAL_WORDS.search(meaning):
            continue
        if not pd.api.types.is_numeric_dtype(train[col]) or not pd.api.types.is_numeric_dtype(test[col]):
            continue
        if pd.api.types.is_bool_dtype(train[col]) or pd.api.types.is_bool_dtype(test[col]):
            continue
        values = train[col].to_numpy(dtype=float, na_value=np.nan)
        future = test[col].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all() or not np.isfinite(future).all():
            continue
        periods = np.unique(values)
        if len(periods) < 4 or future.min() <= periods[-1]:
            continue
        # Prefer a repeated/coarser time period to a nearly unique timestamp.
        candidates.append((len(periods), str(col), col, values, periods, np.unique(future)))
    return min(candidates, key=lambda c: c[:2])[2:] if candidates else None


def _temporal_folds(problem: Problem, values: np.ndarray) -> list:
    times = values[problem.research_idx]
    folds = []
    for period in np.unique(times)[1:]:
        tr, va = np.flatnonzero(times < period), np.flatnonzero(times == period)
        if len(tr) < MIN_TEMPORAL_TRAIN_ROWS or len(va) < MIN_TEMPORAL_VALID_ROWS:
            continue
        if problem.classification and len(np.unique(problem.y[problem.research_idx[tr]])) < 2:
            continue
        folds.append((tr, va))
    return folds[-N_SPLITS:]  # bounded cost; earlier warm-up rows intentionally have no OOF


def make_problem(train: pd.DataFrame, spec: TaskSpec, holdout: bool = True,
                 test: pd.DataFrame | None = None) -> Problem:
    """Fix one validation policy; train must already have non-null targets.

    With explicit future-only test periods, reserve the last train period and use
    expanding-window CV on the preceding periods. Otherwise preserve random CV.
    All indices are positional even when the source DataFrame has a custom index.
    """
    y_raw = train[spec.target].to_numpy()
    classes = None
    if spec.task_type == "regression":
        y = y_raw.astype(float)
    else:
        classes = list(spec.classes)
        index = {c: i for i, c in enumerate(classes)}
        y = pd.Series(y_raw).map(index).to_numpy(int)
    rng = np.random.default_rng(SEED)
    n = len(y)
    hold = np.zeros(n, dtype=bool)
    temporal = _future_time_column(train, test, spec)
    if temporal is not None:
        col, values, periods, test_periods = temporal
        if holdout:
            hold = values == periods[-1]
        problem = Problem(spec=spec, y=y, y_raw=y_raw, classes=classes,
                          research_idx=np.flatnonzero(~hold), holdout_idx=np.flatnonzero(hold),
                          scheme="temporal", time_column=str(col))
        problem.folds = _temporal_folds(problem, values)
        if problem.folds and (not holdout or hold.sum() >= MIN_TEMPORAL_VALID_ROWS):
            problem.oof_mask = np.zeros(len(problem.research_idx), dtype=bool)
            for _, va in problem.folds:
                problem.oof_mask[va] = True
            problem.metadata = {"scheme": problem.scheme, "time_column": problem.time_column,
                                "train_periods": periods.tolist(), "test_periods": test_periods.tolist(),
                                "holdout_period": float(periods[-1]) if holdout else None,
                                "oof_rows": int(problem.oof_mask.sum()),
                                "research_rows": len(problem.research_idx),
                                "oof_coverage": float(problem.oof_mask.mean()), "folds": len(problem.folds)}
            return problem
        hold = np.zeros(n, dtype=bool)  # insufficient temporal support: retain established policy
    if holdout and n >= MIN_ROWS_FOR_HOLDOUT:
        if classes is None:
            hold[rng.permutation(n)[: int(HOLDOUT_FRAC * n)]] = True
        else:
            for k in range(len(classes)):
                rows = np.flatnonzero(y == k)
                if len(rows) >= 20:  # rare classes stay in research data
                    hold[rng.permutation(rows)[: int(HOLDOUT_FRAC * len(rows))]] = True
    problem = Problem(spec=spec, y=y, y_raw=y_raw, classes=classes,
                      research_idx=np.flatnonzero(~hold), holdout_idx=np.flatnonzero(hold))
    problem.folds = _folds(problem)
    problem.oof_mask = np.zeros(len(problem.research_idx), dtype=bool)
    for _, va in problem.folds:
        problem.oof_mask[va] = True
    problem.metadata = {"scheme": problem.scheme, "time_column": None,
                        "oof_rows": int(problem.oof_mask.sum()), "research_rows": len(problem.research_idx),
                        "oof_coverage": float(problem.oof_mask.mean()), "folds": len(problem.folds)}
    return problem


def make_subproblem(problem: Problem, rows: int, n_splits: int = 3) -> Problem:
    """A cheaper copy of the problem for screening ideas (RD-Agent's "debug on a sample"):
    a stratified subset of the research rows, fewer folds, no holdout."""
    rng = np.random.default_rng(SEED + 1)
    research = problem.research_idx
    if rows >= len(research):
        chosen = research
    elif not problem.classification:
        chosen = np.sort(rng.choice(research, size=rows, replace=False))
    else:
        frac = rows / len(research)
        parts = []
        for k in range(problem.n_out):
            members = research[problem.y[research] == k]
            take = len(members) if len(members) < 20 else max(n_splits, int(round(frac * len(members))))
            parts.append(rng.choice(members, size=min(take, len(members)), replace=False))
        chosen = np.sort(np.concatenate(parts))
    sub = Problem(spec=problem.spec, y=problem.y, y_raw=problem.y_raw, classes=problem.classes,
                  research_idx=chosen, holdout_idx=np.array([], dtype=int))
    sub.folds = _folds(sub, n_splits)
    return sub


def _folds(problem: Problem, n_splits: int = N_SPLITS) -> list:
    y = problem.y[problem.research_idx]
    if not problem.classification:
        return list(KFold(n_splits, shuffle=True, random_state=SEED).split(y))
    counts = np.bincount(y, minlength=problem.n_out)
    rare = np.flatnonzero(counts[y] < n_splits)
    common = np.flatnonzero(counts[y] >= n_splits)
    if len(common) < 2 * n_splits:
        raise ValueError("too few rows of frequent classes for cross-validation")
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return [(np.concatenate([common[tr], rare]), common[va]) for tr, va in splitter.split(common, y[common])]


# ---------------------------------------------------------------- models

LGBM_PARAMS = dict(n_estimators=5000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
LGBM_FAST = dict(LGBM_PARAMS, learning_rate=0.1)  # research experiments: same ranking, half the time
# Plain boosting, simple CTRs and fewer borders: ~4x faster on CPU, close in quality.
CATBOOST_PARAMS = dict(iterations=3000, learning_rate=0.08, depth=6, verbose=0, allow_writing_files=False,
                       boosting_type="Plain", max_ctr_complexity=1, border_count=128, one_hot_max_size=8)
XGB_PARAMS = dict(n_estimators=4000, learning_rate=0.05, max_depth=7, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", max_cat_to_onehot=1, max_cat_threshold=64)


def _deadline_reached(deadline: float) -> bool:
    return time.time() > deadline


def _sample_weights(y: np.ndarray, n_classes: int, balanced: bool | str):
    if not balanced:
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    if balanced == "sqrt":
        weights = np.sqrt(weights)
        weights /= np.mean(weights[y])  # preserve the overall loss/regularization scale
    return weights[y]


def _lgbm_class_weight(y: np.ndarray, n_classes: int, balanced: bool | str):
    if balanced != "sqrt":
        return "balanced" if balanced else None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    present = np.unique(y)
    weights = {int(k): float(np.sqrt(len(y) / (n_classes * counts[k]))) for k in present}
    scale = sum(weights[int(k)] * counts[k] for k in present) / len(y)
    return {k: weight / scale for k, weight in weights.items()}


def _catboost_class_weight(balanced: bool | str):
    return "SqrtBalanced" if balanced == "sqrt" else ("Balanced" if balanced else None)


def _fit_lgbm(problem, params, X_tr, y_tr, X_va, y_va, deadline, seed, balanced, objective):
    import lightgbm as lgb

    def stop(env):
        if _deadline_reached(deadline):
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    stop.order = 40
    p = {**LGBM_PARAMS, **(params or {}), "random_state": seed, "n_jobs": N_JOBS}
    if problem.classification:
        model = lgb.LGBMClassifier(class_weight=_lgbm_class_weight(y_tr, problem.n_out, balanced), **p)
    else:
        model = lgb.LGBMRegressor(objective=objective, **p)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False), stop])
    return model


def _to_catboost(X: pd.DataFrame):
    X = X.copy()
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        X[c] = X[c].astype("string").fillna("<NA>").astype(str)
    return X, cats


def _fit_catboost(problem, params, X_tr, y_tr, X_va, y_va, deadline, seed, balanced, objective):
    from catboost import CatBoostClassifier, CatBoostRegressor

    class Stop:
        def after_iteration(self, info):
            return not _deadline_reached(deadline)

    X_tr, cats = _to_catboost(X_tr)
    X_va, _ = _to_catboost(X_va)
    p = {**CATBOOST_PARAMS, **(params or {}), "random_seed": seed, "thread_count": N_JOBS,
         "cat_features": cats, "early_stopping_rounds": 100}
    if problem.classification:
        loss = "MultiClass" if problem.spec.task_type == "multiclass" else "Logloss"
        model = CatBoostClassifier(loss_function=loss, auto_class_weights=_catboost_class_weight(balanced), **p)
    else:
        model = CatBoostRegressor(loss_function="MAE" if objective == "regression_l1" else "RMSE", **p)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), callbacks=[Stop()])
    return model


def _fit_xgb(problem, params, X_tr, y_tr, X_va, y_va, deadline, seed, balanced, objective):
    import xgboost as xgb

    class Stop(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            return _deadline_reached(deadline)

    p = {**XGB_PARAMS, **(params or {}), "random_state": seed, "n_jobs": N_JOBS,
         "enable_categorical": True, "early_stopping_rounds": 100, "callbacks": [Stop()]}
    weight = None
    if problem.classification:
        weight = _sample_weights(y_tr, problem.n_out, balanced)
        model = xgb.XGBClassifier(**p)
    else:
        model = xgb.XGBRegressor(objective="reg:absoluteerror" if objective == "regression_l1"
                                 else "reg:squarederror", **p)
    model.fit(X_tr, y_tr, sample_weight=weight, eval_set=[(X_va, y_va)], verbose=False)
    return model


FITTERS = {"lgbm": _fit_lgbm, "catboost": _fit_catboost, "xgb": _fit_xgb}


def _predict(problem, kind, model, X):
    if kind == "catboost":
        X, _ = _to_catboost(X)
    if not problem.classification:
        return model.predict(X).astype(float)
    proba = np.zeros((len(X), problem.n_out))
    proba[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    return proba


@dataclass
class CVResult:
    name: str
    oof: np.ndarray        # predictions for research rows (original scale / probabilities)
    test: np.ndarray       # averaged over finished folds
    holdout: np.ndarray | None
    score: float
    seconds: float
    folds: int
    importance: pd.Series | None = None
    iterations: list[int] | None = None  # tree counts, normalised from each family's indexing


def _iteration_count(model, kind: str) -> int | None:
    """Return number of trees at the selected iteration, not a zero-based index."""
    try:
        if kind == "catboost":
            best = model.get_best_iteration()
            if best is not None and best >= 0:
                return int(best) + 1
            value = getattr(model, "tree_count_", None)
        elif kind == "xgb":
            try:
                return int(model.best_iteration) + 1
            except (AttributeError, ValueError):
                value = getattr(model, "n_estimators", None)
        else:
            value = getattr(model, "best_iteration_", None)
            if value is None or value <= 0:
                value = getattr(model, "n_estimators_", getattr(model, "n_estimators", None))
        return int(value) if value is not None and value > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def run_cv(problem: Problem, X: pd.DataFrame, X_test: pd.DataFrame, kind: str = "lgbm",
           params: dict | None = None, seed: int = SEED, deadline: float = float("inf"),
           target_log: bool = False, name: str | None = None, balanced: bool | str | None = None,
           log=print, offset: str | None = None) -> CVResult:
    """K-fold one model family on the problem's research rows with its fixed folds; the
    holdout rows (if any) and the test rows are predicted by every fold model.

    X holds all train rows (same order as problem.y). Returns predictions on the original
    target scale. No new fold starts after `deadline`; boosting is cut there as well.
    `params` is merged over the family defaults (LGBM_PARAMS / CATBOOST_PARAMS / XGB_PARAMS).
    `balanced` accepts False, True, or "sqrt" (gentler inverse-frequency weighting).
    """
    rows, folds = problem.research_idx, problem.folds
    spec = problem.spec
    columns = list(X.columns)
    renamed = {c: f"f{i}" for i, c in enumerate(columns)}  # LightGBM/XGB reject JSON chars
    Xr = X.iloc[rows].rename(columns=renamed).reset_index(drop=True)
    Xt = X_test.rename(columns=renamed)
    Xh = X.iloc[problem.holdout_idx].rename(columns=renamed) if len(problem.holdout_idx) else None

    y = problem.y[rows]
    if target_log:
        y = np.log1p(np.clip(y, 0, None))
    off_r, off_t, off_h = (_offset(frame, offset, problem) for frame in (X.iloc[rows], X_test,
                           X.iloc[problem.holdout_idx] if Xh is not None else None))
    if offset is not None:
        y = y - off_r
    if balanced is None:
        balanced = spec.metric in CLASS_BALANCED
    balanced = problem.classification and balanced
    objective = {"mae": "regression_l1", "mape": "regression_l1"}.get(spec.metric, "regression")
    shape = (len(rows), problem.n_out) if problem.classification else (len(rows),)
    oof = np.full(shape, np.nan)
    test_sum = np.zeros((len(Xt),) + shape[1:])
    hold_sum = np.zeros((len(Xh),) + shape[1:]) if Xh is not None else None
    importance = pd.Series(0.0, index=columns)
    started, done, fold_times = time.time(), 0, []
    iteration_counts = []
    for fold, (tr, va) in enumerate(folds):
        if done and time.time() + 1.2 * max(fold_times) > deadline:
            log(f"  {kind}: time budget, stop after {done} folds")
            break
        t0 = time.time()
        try:
            model = FITTERS[kind](problem, params, Xr.iloc[tr], y[tr], Xr.iloc[va], y[va],
                                  deadline, seed, balanced, objective)
            pred_va = _predict(problem, kind, model, Xr.iloc[va])
            pred_test = _predict(problem, kind, model, Xt)
            pred_hold = _predict(problem, kind, model, Xh) if hold_sum is not None else None
        except Exception as error:
            if done == 0 and fold == 0:  # first failure of this model: full traceback once
                log(f"  {kind} fold {fold} failed:\n{traceback.format_exc()}")
            else:
                log(f"  {kind} fold {fold} failed: {error!r}"[:300])
            continue
        if offset is not None:
            pred_va, pred_test = pred_va + off_r[va], pred_test + off_t
            pred_hold = pred_hold + off_h if pred_hold is not None else None
        oof[va] = pred_va
        test_sum += pred_test
        if hold_sum is not None:
            hold_sum += pred_hold
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and len(imp) == len(columns):
            importance += np.asarray(imp, dtype=float) / max(float(np.sum(imp)), 1e-12)
        done += 1
        iteration_counts.append(_iteration_count(model, kind))
        fold_times.append(time.time() - t0)
    if done == 0:
        raise RuntimeError(f"{kind}: no fold finished")
    test_pred, hold_pred = test_sum / done, (hold_sum / done if hold_sum is not None else None)
    if target_log:
        oof, test_pred = np.expm1(oof), np.expm1(test_pred)
        hold_pred = np.expm1(hold_pred) if hold_pred is not None else None
    seen = ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])
    if not problem.classification:  # trees cannot extrapolate; keep blends inside the target range
        low, high = np.nanmin(problem.y_raw.astype(float)), np.nanmax(problem.y_raw.astype(float))
        oof, test_pred = np.clip(oof, low, high), np.clip(test_pred, low, high)
        hold_pred = np.clip(hold_pred, low, high) if hold_pred is not None else None
    cv = problem.score_pred(oof[seen], rows[seen])
    return CVResult(name=name or kind, oof=oof, test=test_pred, holdout=hold_pred, score=cv,
                    seconds=time.time() - started, folds=done, importance=importance / done,
                    iterations=iteration_counts if all(i is not None for i in iteration_counts) else None)


def fit_full(problem: Problem, X: pd.DataFrame, X_test: pd.DataFrame, kind: str = "lgbm",
             params: dict | None = None, iterations: int | None = None, seed: int = SEED,
             deadline: float = float("inf"), target_log: bool = False,
             balanced: bool | str | None = None, log=print, offset: str | None = None) -> np.ndarray:
    """Refit a selected recipe on *all* labelled rows, without validation/early stopping.

    `iterations` is a positive tree count (typically the median of CVResult.iterations).
    `balanced` has the same None/False/True/"sqrt" policy as run_cv.
    Return test predictions on the original scale. A deadline-truncated refit raises, so
    the caller retains the already validated research-time predictions.
    """
    def check_time():
        if _deadline_reached(deadline):
            raise TimeoutError("full refit deadline reached")

    check_time()
    if len(X) != len(problem.y):
        raise ValueError("full refit requires every labelled row in original order")
    if kind not in FITTERS:
        raise ValueError(f"unknown model family {kind!r}")
    defaults = {"lgbm": LGBM_PARAMS, "catboost": CATBOOST_PARAMS, "xgb": XGB_PARAMS}[kind]
    p = {**defaults, **(params or {})}
    count_key = "iterations" if kind == "catboost" else "n_estimators"
    count = int(iterations if iterations is not None else p[count_key])
    if count < 1:
        raise ValueError("refit iterations must be a positive tree count")
    p[count_key] = count
    p.pop("early_stopping_rounds", None)
    renamed = {c: f"f{i}" for i, c in enumerate(X.columns)}
    X_all, Xt = X.rename(columns=renamed), X_test.rename(columns=renamed)
    y = problem.y.copy()
    if target_log:
        if problem.classification:
            raise ValueError("log target is only supported for regression")
        y = np.log1p(np.clip(y, 0, None))
    if offset is not None:
        y = y - _offset(X, offset, problem)
    if balanced is None:
        balanced = problem.spec.metric in CLASS_BALANCED
    balanced = problem.classification and balanced
    objective = "regression_l1" if problem.spec.metric in ("mae", "mape") else "regression"
    if kind == "lgbm":
        import lightgbm as lgb

        def stop(env):
            check_time()
        stop.order = 40
        p.update(random_state=seed, n_jobs=N_JOBS)
        model = (lgb.LGBMClassifier(class_weight=_lgbm_class_weight(y, problem.n_out, balanced), **p)
                 if problem.classification else lgb.LGBMRegressor(objective=objective, **p))
        model.fit(X_all, y, callbacks=[stop])
    elif kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        class Stop:
            def after_iteration(self, info):
                check_time()
                return True
        X_all, cats = _to_catboost(X_all)
        p.update(random_seed=seed, thread_count=N_JOBS, cat_features=cats, use_best_model=False)
        if problem.classification:
            loss = "MultiClass" if problem.spec.task_type == "multiclass" else "Logloss"
            model = CatBoostClassifier(loss_function=loss, auto_class_weights=_catboost_class_weight(balanced), **p)
        else:
            model = CatBoostRegressor(loss_function="MAE" if objective == "regression_l1" else "RMSE", **p)
        model.fit(X_all, y, callbacks=[Stop()])
    else:
        import xgboost as xgb

        class Stop(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                check_time()
                return False
        p.update(random_state=seed, n_jobs=N_JOBS, enable_categorical=True, callbacks=[Stop()])
        weight = None
        if problem.classification:
            weight = _sample_weights(y, problem.n_out, balanced)
            model = xgb.XGBClassifier(**p)
        else:
            model = xgb.XGBRegressor(objective="reg:absoluteerror" if objective == "regression_l1"
                                    else "reg:squarederror", **p)
        model.fit(X_all, y, sample_weight=weight, verbose=False)
    check_time()
    pred = _predict(problem, kind, model, Xt)
    if not problem.classification:
        if offset is not None:
            pred = pred + _offset(X_test, offset, problem)
        if target_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, np.nanmin(problem.y_raw.astype(float)), np.nanmax(problem.y_raw.astype(float)))
    if not np.isfinite(pred).all():
        raise ValueError("non-finite full refit predictions")
    check_time()
    log(f"refit: {kind} trained on all {len(y)} rows, {count} trees")
    return pred


def _offset(frame: pd.DataFrame | None, column: str | None, problem: Problem):
    if frame is None or column is None:
        return None
    if problem.classification:
        raise ValueError("offset is only supported for regression")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def answer_from(problem: Problem, pred: np.ndarray, class_weights=None) -> np.ndarray:
    """Predictions in the format the answer file expects."""
    if not problem.classification:
        return pred
    if problem.spec.pred_format == "proba":
        return pred[:, problem.positive_index]
    proba = pred if class_weights is None else pred * class_weights
    return np.array(problem.classes, dtype=object)[proba.argmax(1)]


def uses_probabilities(problem: Problem) -> bool:
    return problem.spec.metric in PROBA
