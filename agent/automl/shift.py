"""Label shift: the test may hold classes in other proportions than train (e.g. more normal
traffic). EM re-estimates test class priors from the model's own test probabilities
(Saerens, Latinne & Decaestecker, 2002) and turns them into per-class weights on probabilities.

The "source prior" is the mean OOF probability, i.e. the prior the model's probabilities
actually reflect (class weighting during training shifts it away from raw class counts).
Switch: AGENT_LABEL_SHIFT (default on). Applied only for label answers, when the estimated
shift is noticeable (total variation >= MIN_TOTAL_VARIATION). Weights are clipped to
[1/MAX_WEIGHT, MAX_WEIGHT], close to the class-weight grid (0.25..6): a class with almost no OOF mass
must not get an extreme weight from a noisy estimate.
"""
import numpy as np

MIN_TOTAL_VARIATION = 0.02
MAX_ITER = 500
MAX_WEIGHT = 4.0
TOL = 1e-7


def em_prior_weights(source_prior: np.ndarray, test_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (weights = test_prior / source_prior clipped to MAX_WEIGHT, test_prior, iterations)."""
    src = np.clip(np.asarray(source_prior, float), 1e-6, None)
    src = src / src.sum()
    proba = np.clip(np.asarray(test_proba, float), 1e-12, None)
    proba = proba / proba.sum(axis=1, keepdims=True)
    prior = src.copy()
    for iteration in range(1, MAX_ITER + 1):
        adjusted = proba * (prior / src)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new = adjusted.mean(axis=0)
        if np.abs(new - prior).max() < TOL:
            prior = new
            break
        prior = new
    return np.clip(prior / src, 1 / MAX_WEIGHT, MAX_WEIGHT), prior, iteration


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())
