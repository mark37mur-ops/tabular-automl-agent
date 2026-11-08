"""Blending candidates on out-of-fold predictions and tuning class weights for label metrics."""
import numpy as np

from automl.cv import Problem

LABEL_TUNED = {"f1_macro", "f1", "balanced_accuracy", "f1_weighted"}


def greedy_blend(problem: Problem, oofs: list[np.ndarray], rows: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Caruana ensemble selection with replacement; returns weights summing to 1.

    oofs are predictions for `rows` (no NaN). Starts from the single best candidate.
    """
    scores = [problem.score_pred(p, rows) for p in oofs]
    best = int(np.argmax(scores) if problem.better(1, 0) else np.argmin(scores))
    counts = np.zeros(len(oofs))
    counts[best] = 1
    current, current_score = oofs[best].copy(), scores[best]
    for _ in range(iterations):
        n = counts.sum()
        trial = [(current * n + p) / (n + 1) for p in oofs]
        trial_scores = [problem.score_pred(t, rows) for t in trial]
        pick = int(np.argmax(trial_scores) if problem.better(1, 0) else np.argmin(trial_scores))
        if not problem.better(trial_scores[pick], current_score) and counts.sum() > 1:
            break
        counts[pick] += 1
        current, current_score = trial[pick], trial_scores[pick]
    return counts / counts.sum()


def blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * p for w, p in zip(weights, preds) if w > 0)


def tune_class_weights(problem: Problem, proba: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    """Per-class multipliers on probabilities that maximise a label metric (macro F1 etc.).

    Coordinate search over a log grid, two passes. None when the metric does not need it.
    """
    if not problem.classification or problem.spec.metric not in LABEL_TUNED \
            or problem.spec.pred_format == "proba":
        return None
    k = proba.shape[1]
    weights = np.ones(k)
    best = problem.score_pred(proba, rows, weights)
    grid = np.exp(np.linspace(np.log(0.25), np.log(6.0), 25))
    for _ in range(2):
        for c in range(k):
            for g in grid:
                trial = weights.copy()
                trial[c] = g
                s = problem.score_pred(proba, rows, trial)
                if problem.better(s, best):
                    best, weights = s, trial
    return weights
